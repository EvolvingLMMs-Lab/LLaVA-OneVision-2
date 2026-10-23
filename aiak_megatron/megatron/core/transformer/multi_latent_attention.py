# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""multi-latent attention"""

import math
from dataclasses import dataclass
from typing import NoReturn, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
    fused_mla_apply_rotary_pos_emb,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType, AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.tensor_parallel import RecomputeManager


try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear = None, None, None
    HAVE_TE = False


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str = None,
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
        )

        # only for debug info
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        # for fa within mla, we need to pad the v_head_dim to q_head_dim
        self.v_channels = self.config.v_head_dim
        self.padding_v_head_dim = False

        if self.q_head_dim > self.config.v_head_dim and self.config.attention_backend == AttnBackend.flash:
            self.v_channels = self.q_head_dim
            self.padding_v_head_dim = True

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.v_channels,
            cp_comm_type=cp_comm_type,
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_type=None,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
    ):
        """Forward pass for multi-latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        if self.config.mla_recompute:
            #print ("attention self.config.moe_mla_recompute")
            self.mla_recompute_manager = RecomputeManager()
            query, key, value = self.mla_recompute_manager.checkpoint(self.get_query_key_value_tensors,
                                                            False,
                                                            hidden_states,
                                                            key_value_states,
                                                            position_ids,
                                                            packed_seq_params,
                                                            inference_params,)
        else:
            query, key, value = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_params=inference_params,
            )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb=None, attn_mask_type=attn_mask_type
        )

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )

        if self.config.mla_recompute:
            self.mla_recompute_manager.discard_output()
            if core_attn_out.requires_grad:
                #print ("core_attn_out.requires_grad")
                core_attn_out.register_hook(self.mla_recompute_manager.recompute)


        # add for fa padding
        if self.padding_v_head_dim:
            _prefix = core_attn_out.shape[:-1]
            core_attn_out = core_attn_out.reshape(*_prefix, -1, self.v_channels)
            core_attn_out = core_attn_out[..., : self.config.v_head_dim].reshape(*_prefix, -1).contiguous()

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        
        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        if self.config.q_lora_rank is None:
            # Not projectiing query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
            )

        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_down_proj',
                skip_weight_param_allocation=False,
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
            )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            q, _ = self.linear_q_up_proj(self.q_layernorm(q_compressed))
        else:
            # hidden_states:[s, b, 2048], q: [s, b, n * 192]
            q, _ = self.linear_q_proj(hidden_states)

        q_len, bsz, _ = q.size()

        # q: [s, b, n, 192]
        q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

        # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
        q_no_pe, q_pos_emb = torch.split(
            q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)

        # ColumnParallelLinear
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        # Linear
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        # kv: [s, b, 2048]
        kv, _ = self.linear_kv_up_proj(self.kv_layernorm(kv_compressed))

        # kv: [s, b, n, 256]
        kv = kv.view(
            q_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_params, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        if self.config.rope_type == "rope":
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        if inference_params is not None:
            # add offset to the sequence start for inference
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + q_len
            rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
        else:
            # Shorten rotary_pos_emb to the seuqence length when inference_params
            # is not provided. This makes sure we can run forward directly with
            # any sequence length. During training, the sequence length is always
            # the full rotary_pos_emb length.
            rotary_pos_emb = rotary_pos_emb[0:q_len]

        # [s, b, 64] -> [s, b, 1, 64]
        k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

            # convert to thd
            q_no_pe = q_no_pe.squeeze(1)
            q_pos_emb = q_pos_emb.squeeze(1)           
            k_no_pe = k_no_pe.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)
            value = value.squeeze(1)

        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
        if self.config.apply_rope_fusion and fused_mla_apply_rotary_pos_emb is not None:
            q_pos_emb = fused_mla_apply_rotary_pos_emb(
                q_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, mscale=mscale
            )
            k_pos_emb = fused_mla_apply_rotary_pos_emb(
                k_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, mscale=mscale
            )
        else:
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, mscale=mscale
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, mscale=mscale
            )

        # query: [s, b, n, 192]
        query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

        # key: [s, b, n, 192]
        if packed_seq_params is not None:
            k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
        else:
            k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)

        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

        if self.padding_v_head_dim:
            value = F.pad(value, [0, self.q_head_dim - self.config.v_head_dim])

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        return query, key, value

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations"""
        try:
            self._backward_kv_proj()
            self._backward_q_proj()
            self._backward_output_proj()
        except Exception as e:
            raise RuntimeError(f"Error in MLASelfAttention backward_dw: {str(e)}")

    def _backward_kv_proj(self):
        """Update weights for KV projection layers"""
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Update weights for Q projection layers"""
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Update weights for output projection layer"""
        self.linear_proj.backward_dw()
