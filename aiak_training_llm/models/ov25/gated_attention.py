# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""GatedSoftmaxAttention (Qwen3NextSelfAttention) ported to AIAK-Megatron."""

from typing import Optional, Tuple, Union

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import build_module
import os
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
    _apply_rotary_pos_emb_bshd,
)

# BUGFIX(packed M-RoPE) gate: OV25RotaryEmbedding emits a PER-TOKEN freqs
# table (row i = token i; per-segment position restarts are already encoded
# by get_rope_index_packed). The stock thd rope path (mcore unfused AND the
# TE fused kernel) re-slices freqs[:seg_len] for EVERY segment, so in a
# packed bin only the first sample gets correct rope — all later samples get
# the first sample's frequency rows. The fix applies the per-token table
# elementwise over the whole packed row (same semantics as
# apply_rotary_pos_emb_vision in the vision tower). Enable with
# MROPE_PACKED_FIX=1; default OFF so long-running experiments keep their
# launch-time semantics across auto-restarts.
_MROPE_PACKED_FIX = os.environ.get("MROPE_PACKED_FIX", "0") == "1"
_MROPE_FIX_LOGGED = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
    )
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


class Qwen3NextSelfAttention(SelfAttention):
    """
    Qwen3Next gated (softmax) self-attention module.

    Adapted to the AIAK-Megatron ``SelfAttention`` API:
      * ``__init__`` matches our ``SelfAttention(config, submodules, layer_number,
        attn_mask_type=..., cp_comm_type=...)`` signature (no ``pg_collection``).
      * GATED qkv: ``linear_qkv`` outputs ``2 * query_projection + 2 * kv_projection``;
        ``get_query_key_value_tensors`` splits query|gate and key|value and returns the gate.
      * The output is gated by ``core_attn_out * sigmoid(gate)`` before ``linear_proj``.
      * The forward is based on OUR ``SelfAttention.forward`` (training path only).
    """

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        # ADAPTATION: rebuild linear_qkv with GATED sizing (2q + 2kv) and drop tp_group
        # (our build_module infers TP from parallel_state).
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            2 * self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type=None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_params=None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the gated attention module.

        Based on AIAK-Megatron ``SelfAttention.forward`` (training path) with two
        Qwen3Next additions: the ``get_query_key_value_tensors`` override returns a
        gate tensor, and the core attention output is gated by ``sigmoid(gate)``.
        """
        # ADAPTATION: optional no_rope per-layer flag (getattr, default off).
        no_rope_freq = getattr(self.config, "no_rope_freq", None)
        if no_rope_freq:
            # no_rope_freq is 0-indexed; self.layer_number is 1-indexed
            if no_rope_freq[self.layer_number - 1]:
                rotary_pos_emb = None

        # hidden_states: [sq, b, h]
        if self.config.flash_decode:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, Value, Gate
        # =====================
        query, key, value, gate = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attn_mask_type,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            # ADAPTATION: use the spec-provided apply_rotary_fn when present (our
            # SelfAttention path), otherwise fall back to the standard mcore fn.
            rotary_fn = self.apply_rotary_fn if self.apply_rotary_fn is not None else apply_rotary_pos_emb
            if (
                _MROPE_PACKED_FIX
                and cu_seqlens_q is not None
                and q_pos_emb is not None
                and q_pos_emb.shape[0] == query.shape[0]
                and k_pos_emb.shape[0] == key.shape[0]
            ):
                # per-token table -> elementwise over the whole packed row
                global _MROPE_FIX_LOGGED
                if not _MROPE_FIX_LOGGED:
                    import sys as _sys
                    _sys.stderr.write("[MROPE-PACKED-FIX] elementwise per-token rope ACTIVE (packed thd)\n")
                    _MROPE_FIX_LOGGED = True
                query = _apply_rotary_pos_emb_bshd(
                    query.unsqueeze(1), q_pos_emb,
                    rotary_interleaved=self.config.rotary_interleaved,
                ).squeeze(1)
                key = _apply_rotary_pos_emb_bshd(
                    key.unsqueeze(1), k_pos_emb,
                    rotary_interleaved=self.config.rotary_interleaved,
                ).squeeze(1)
            else:
                query = rotary_fn(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
                key = rotary_fn(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Gate + Output. [sq, b, h]
        # =================
        core_attn_out = core_attn_out * torch.sigmoid(gate.reshape_as(core_attn_out))
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives query, key, value, and gate tensors from the input hidden states.

        The gated qkv projection interleaves query and gate per head:
            linear_qkv output layout per group is
            [ (np/ng * 2 + 2) * hn ] -> query|gate (interleaved) | key | value
        """
        mixed_qgkv, _ = self.linear_qkv(hidden_states)

        new_tensor_shape = mixed_qgkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qgkv = mixed_qgkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng * 2 + 2) * hn]
            # --> [sq, b, ng, np/ng * 2 * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_gate, key, value) = SplitAlongDim(mixed_qgkv, 3, split_arg_list)
        else:
            (query_gate, key, value) = torch.split(mixed_qgkv, split_arg_list, dim=3)

        query_gate = query_gate.reshape(query_gate.size(0), query_gate.size(1), -1, self.hidden_size_per_attention_head)
        query = query_gate[:, :, ::2]
        gate = query_gate[:, :, 1::2]

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate
