# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/huggingface/transformers.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GatedDeltaNet (ported from LoongForge to AIAK-Megatron / mcore 0.12.0rc0)."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformer_engine.pytorch import Linear as TE_Linear

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.models.huggingface import HuggingFaceModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import deprecate_inference_params

from .rmsnorm import Qwen3NextRMSNorm, Qwen3NextRMSNormGated

# ADAPTATION: our mcore 0.12.0rc0 has no megatron.core.utils.nvtx_range_push/pop.
# Provide thin local no-op wrappers (they only bracket kernels for profiling).
def nvtx_range_push(*args, **kwargs):
    """No-op nvtx range push (adapted for AIAK-Megatron)."""
    return None


def nvtx_range_pop(*args, **kwargs):
    """No-op nvtx range pop (adapted for AIAK-Megatron)."""
    return None


try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    FusedRMSNormGated = None
    causal_conv1d = None
    chunk_gated_delta_rule = None
    l2norm = None

    HAVE_FLA = False


@dataclass
class GatedDeltaNetSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """
    in_proj_qkvz: Union[ModuleSpec, type] = IdentityOp
    in_proj_ba: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaNet(HuggingFaceModule):
    """Gated Delta Net (GDN) layer class
    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (0, 16),
        **kwargs
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
        """
        super().__init__(config)
        assert config.context_parallel_size == 1, "GatedDeltaNet currently does not support context parallelism."
        assert HAVE_FLA, "GatedDeltaNet requires FLA support."

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        # ADAPTATION: no ProcessGroupCollection; derive TP world size from parallel_state.
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.sequence_parallel = config.sequence_parallel

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        assert self.activation in ["silu", "swish"], f"Only silu and swish are supported, but got {self.activation}"
        self.conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)
        self.key_head_dim = getattr(config, "linear_key_head_dim", 128)
        self.value_head_dim = getattr(config, "linear_value_head_dim", 128)
        self.num_key_heads = getattr(config, "linear_num_key_heads", 16)
        self.num_value_heads = getattr(config, "linear_num_value_heads", 32)
        # Head-parallel TP (--gdn-head-parallel): each TP rank owns num_key_heads/tp key heads (and
        # their value heads) instead of a full replica. in_proj_qkvz/in_proj_ba are laid out
        # key-head-major, so a contiguous row split hands every rank whole heads. All per-head
        # sizes below become rank-local; the core math is unchanged.
        self.head_tp = bool(getattr(config, "gdn_head_parallel", False)) and self.tp_size > 1
        if self.head_tp:
            assert self.num_key_heads % self.tp_size == 0 and self.num_value_heads % self.tp_size == 0, (
                f"--gdn-head-parallel needs linear heads ({self.num_key_heads} key / "
                f"{self.num_value_heads} value) divisible by TP={self.tp_size}")
            self.num_key_heads //= self.tp_size
            self.num_value_heads //= self.tp_size
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads

        # Conv1d for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim,
            padding=self.conv_kernel_dim - 1,
        )

        self.in_proj_qkvz_dim = self.qk_dim * 2 + self.v_dim * 2
        self.in_proj_ba_dim = self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_qkvz_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                f"in_proj_qkvz output tensor ({self.in_proj_qkvz_dim}) must be a multiple of {fp8_align_size}."
            )
            assert self.in_proj_ba_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                f"in_proj_ba output tensor ({self.in_proj_ba_dim}) must be a multiple of {fp8_align_size}."
            )
        # In head-TP mode these are plain local linears holding this rank's rows (dims above are
        # already rank-local); the TP communication is explicit in _forward_head_tp.
        self.in_proj_qkvz = TE_Linear(self.hidden_size, self.in_proj_qkvz_dim, bias=False)
        self.in_proj_ba = TE_Linear(self.hidden_size, self.in_proj_ba_dim, bias=False)

        # dt_bias parameter
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        A = torch.empty(self.num_value_heads).uniform_(self.A_init_range[0], self.A_init_range[1])
        self.A_log = nn.Parameter(torch.log(A))
        if self.tp_size > 1:
            setattr(self.dt_bias, "average_gradients_across_tp_domain", True)
            setattr(self.A_log, "average_gradients_across_tp_domain", True)

        self.out_norm = (
            Qwen3NextRMSNormGated(
                self.value_head_dim, eps=self.config.layernorm_epsilon
            ) if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.value_head_dim,
                eps=self.config.layernorm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
        )

        self.out_proj = TE_Linear(self.v_dim, self.hidden_size, bias=False)

        # SP-FIX (2026-08-26): these params are REPLICATED across TP but consume
        # TP-rank-local token shards under sequence-parallel (TP0 first half of each
        # sequence, TP1 second half), so their grads MUST be summed across the TP
        # group. Mark them sequence_parallel so finalize_model_grads'
        # _allreduce_layernorm_grads picks them up (the old
        # average_gradients_across_tp_domain flag has no consumer anywhere).
        # Verified missing-reduce by comparing TP0 vs TP1 grads (rel diff 0.2-2.9); for old checkpoints
        # see tools/convert_checkpoint/custom/llava_onevision2/ov25_tp_consistency.py.
        if self.head_tp:
            # Everything is head-sharded except out_norm (one [value_head_dim] weight shared by
            # all heads): each rank's grad covers only its own heads, so it must be summed over TP.
            # finalize_model_grads reduces `sequence_parallel` params when SP or qk_layernorm is on.
            assert self.config.sequence_parallel or getattr(self.config, "qk_layernorm", False), (
                "--gdn-head-parallel: out_norm grads are TP-reduced only with sequence_parallel or qk_layernorm")
            setattr(self.out_norm.weight, "sequence_parallel", True)
            # Sharded params must carry TP attributes, otherwise grad-norm/clipping treats them
            # as TP duplicates and counts only rank 0's shard.
            # (TE's local Linear already tags its weight tensor_model_parallel=False; overwrite.)
            for _p, _dim in ((self.in_proj_qkvz.weight, 0), (self.in_proj_ba.weight, 0),
                             (self.out_proj.weight, 1), (self.dt_bias, 0), (self.A_log, 0),
                             *((_q, 0) for _q in self.conv1d.parameters())):
                _p.tensor_model_parallel, _p.partition_dim, _p.partition_stride = True, _dim, 1
            return
        for _p in (
            self.in_proj_qkvz.weight,
            self.in_proj_ba.weight,
            self.out_proj.weight,
            self.dt_bias,
            self.A_log,
            self.out_norm.weight,
            *self.conv1d.parameters(),  # SP-FIX2: conv1d was missed initially (defined earlier in __init__)
        ):
            setattr(_p, "sequence_parallel", self.config.sequence_parallel)

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_key_heads,
            2 * self.key_head_dim + 2 * self.value_head_dim * self.num_value_heads // self.num_key_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_key_heads, 2 * self.num_value_heads // self.num_key_heads
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.key_head_dim,
            self.key_head_dim,
            (self.num_value_heads // self.num_key_heads * self.value_head_dim),
            (self.num_value_heads // self.num_key_heads * self.value_head_dim),
        ]
        split_arg_list_ba = [self.num_value_heads // self.num_key_heads, self.num_value_heads // self.num_key_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.value_head_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.value_head_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_value_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_value_heads)
        return query, key, value, z, b, a

    def apply_mask_to_padding_states(self, hidden_states, attention_mask):
        """
        Tunes out the hidden states for padding tokens according to the attention mask
        """
        # NOTE: attention mask is a 2D boolean tensor
        if attention_mask is not None and attention_mask.shape[1] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return hidden_states

    def _gdn_core(self, projected_states_qkvz, projected_states_ba, cu_seqlens, cu_seqlens_cpu):
        """Projections -> gated-norm output: conv1d, gated delta rule, gated RMSNorm.

        Split out of forward() so it can be activation-checkpointed as one unit.
        """
        query, key, value, z, beta, alpha = self.fix_query_key_value_ordering(
            projected_states_qkvz,
            projected_states_ba
        )
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        _conv_w = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel_dim]
        _conv_b = self.conv1d.bias                # None when conv_bias=False

        nvtx_range_push(suffix="conv1d")
        query = causal_conv1d(
            x=query,
            weight=_conv_w[:self.qk_dim],
            bias=_conv_b[:self.qk_dim] if _conv_b is not None else None,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )[0]
        key = causal_conv1d(
            x=key,
            weight=_conv_w[self.qk_dim:self.qk_dim * 2],
            bias=_conv_b[self.qk_dim:self.qk_dim * 2] if _conv_b is not None else None,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )[0]
        value = causal_conv1d(
            x=value,
            weight=_conv_w[self.qk_dim * 2:],
            bias=_conv_b[self.qk_dim * 2:] if _conv_b is not None else None,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )[0]
        nvtx_range_pop(suffix="conv1d")
        query = query.reshape(query.shape[0], query.shape[1], -1, self.key_head_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.key_head_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.value_head_dim)

        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = self.out_norm(core_attn_out, z)
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(z_shape_og)
        norm_out = norm_out.reshape(norm_out.shape[0], norm_out.shape[1], -1)
        return norm_out

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type=None,  # accepted & ignored: base TransformerLayer always passes it
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the GDN module.
        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[Any]): Inference context that manages KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.
        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if self.head_tp:
            return self._forward_head_tp(hidden_states, attention_mask, packed_seq_params)
        if self.sequence_parallel and self.tp_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, group=None)

        cu_seqlens = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q
        # PackedSeqParams (aiak mcore 0.12) has no cu_seqlens_cpu field; fla needs a CPU
        # copy of the segment boundaries for its varlen chunk math. Derive it from
        # cu_seqlens when the field is absent/None.
        cu_seqlens_cpu = getattr(packed_seq_params, "cu_seqlens_cpu", None) if packed_seq_params is not None else None
        if cu_seqlens_cpu is None and cu_seqlens is not None:
            cu_seqlens_cpu = cu_seqlens.detach().to("cpu")
        hidden_states = hidden_states.transpose(0, 1).contiguous() # [S, B, D] -> [B, S, D]
        if attention_mask is not None:
            if attention_mask.dim() >= 3 and attention_mask.shape[2] > 1: # [B, 1, S, S]
                attention_mask = (~attention_mask).sum(dim=(1, 2)) > 0 # [B, S]
            else:
                attention_mask = ~(attention_mask.squeeze(1).squeeze(1)) # [B, S]
        hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        # Input projection
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)

        # GDN selective activation checkpointing: with recompute_granularity=selective,
        # keep only the two projection outputs and recompute conv / delta rule / gated
        # norm in backward. Uses the megatron checkpoint (main_grad-aware);
        # torch.utils.checkpoint breaks main_grad accumulation here.
        if getattr(self.config, "recompute_granularity", None) == "selective" and self.training:
            norm_out = tensor_parallel.checkpoint(
                self._gdn_core, False,
                projected_states_qkvz, projected_states_ba, cu_seqlens, cu_seqlens_cpu,
            )
        else:
            norm_out = self._gdn_core(
                projected_states_qkvz, projected_states_ba, cu_seqlens, cu_seqlens_cpu
            )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        out = out.transpose(0, 1).contiguous() # [B, S, D] -> [S, B, D]

        if self.sequence_parallel and self.tp_size > 1:
            out = reduce_scatter_to_sequence_parallel_region(out, group=None) / self.tp_size
        return out, None

    def _forward_head_tp(self, hidden_states, attention_mask, packed_seq_params):
        """Head-parallel forward: [s(/tp), b, h] -> [s(/tp), b, h], this rank's heads only.

        Same communication as the replicated path (one all-gather in, one reduce-scatter out
        under SP; one all-reduce without SP), but the projections, conv and delta rule run on
        num_heads/tp heads instead of all of them.
        """
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, group=None)  # [S, B, D]
        else:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states)  # bwd: all-reduce dgrad
        cu_seqlens = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q
        cu_seqlens_cpu = getattr(packed_seq_params, "cu_seqlens_cpu", None) if packed_seq_params is not None else None
        if cu_seqlens_cpu is None and cu_seqlens is not None:
            cu_seqlens_cpu = cu_seqlens.detach().to("cpu")

        qkvz = self.in_proj_qkvz(hidden_states)  # [S, B, local heads]
        ba = self.in_proj_ba(hidden_states)
        qkvz, ba = qkvz.transpose(0, 1), ba.transpose(0, 1)  # [B, S, local]
        if attention_mask is not None:
            # Zeroing padded input rows == zeroing their projections (bias-free linears).
            if attention_mask.dim() >= 3 and attention_mask.shape[2] > 1:
                attention_mask = (~attention_mask).sum(dim=(1, 2)) > 0
            else:
                attention_mask = ~(attention_mask.squeeze(1).squeeze(1))
            qkvz = self.apply_mask_to_padding_states(qkvz, attention_mask)
            ba = self.apply_mask_to_padding_states(ba, attention_mask)

        if getattr(self.config, "recompute_granularity", None) == "selective" and self.training:
            norm_out = tensor_parallel.checkpoint(self._gdn_core, False, qkvz, ba, cu_seqlens, cu_seqlens_cpu)
        else:
            norm_out = self._gdn_core(qkvz, ba, cu_seqlens, cu_seqlens_cpu)

        out = self.out_proj(norm_out.transpose(0, 1))  # partial sums over this rank's heads
        if self.sequence_parallel:
            out = reduce_scatter_to_sequence_parallel_region(out, group=None)
        else:
            out = reduce_from_tensor_model_parallel_region(out)
        return out, None

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        self._backward_in_proj_qkvz()
        self._backward_in_proj_ba()
        self._backward_out_proj()

    def _backward_in_proj_qkvz(self):
        """Computes weight gradients of in_proj_qkvz layer."""
        self.in_proj_qkvz.backward_dw()

    def _backward_in_proj_ba(self):
        """Computes weight gradients of in_proj_ba layer."""
        self.in_proj_ba.backward_dw()

    def _backward_out_proj(self):
        """Computes weight gradients of out_proj layer."""
        self.out_proj.backward_dw()
