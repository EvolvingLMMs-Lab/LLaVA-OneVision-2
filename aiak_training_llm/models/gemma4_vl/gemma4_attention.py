"""Gemma4 self-attention with hybrid head_dim, K=V tying, and parameter-free V LayerNorm.

This module subclasses Megatron's SelfAttention to implement Gemma4-specific
behaviors WITHOUT modifying megatron/core/transformer/attention.py:

  1. Hybrid per-layer head_dim and num_kv_heads (sliding vs global layers).
     Implemented by cloning TransformerConfig and overriding kv_channels +
     num_query_groups before super().__init__.

  2. K=V tying on global layers (the K projection is also reused as V).
     Implemented by rebuilding self.linear_qkv with reduced fan_out
     (q + 1*kv instead of q + 2*kv) and overriding get_query_key_value_tensors
     to perform single-KV split with value = key.

  3. Parameter-free V LayerNorm (F.layer_norm with no weight/bias).
     Applied at the end of get_query_key_value_tensors.

  4. Dual-RoPE dispatch: when ``rotary_pos_emb`` is a ``dict`` mapping
     layer-type strings ('sliding'|'global') to per-type RoPE tensors,
     this layer selects the tensor matching its own ``layer_pattern[layer_number-1]``
     entry before calling the parent forward. This lets ``Gemma4Model`` ship two
     independent ``RotaryEmbedding`` instances (sliding theta=1e4 vs global
     theta=1e6 + proportional zero-pad) through the unmodified ``TransformerBlock``
     pass-through path. Pre-existing single-tensor callers are unaffected.

  5. Sandwich post-attention RMSNorm: HF Gemma4DecoderLayer applies
     ``post_attention_layernorm`` between ``self_attn(...)`` output and the
     residual add (modeling_gemma4.py:1387). Megatron's stock pre-norm
     ``TransformerLayer`` has no slot for this norm, so it is owned here and
     applied after ``super().forward`` returns ``(output, bias)``. The state_dict
     key becomes ``self_attention.post_attention_layernorm.weight`` and is
     written by the HF→MG converter from HF ``post_attention_layernorm.weight``.
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from aiak_training_llm.models.dispatch import multiacc_modules
from aiak_training_llm.utils import is_te_min_version


try:
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None


def _resolve_per_layer_overrides(
    config: TransformerConfig, layer_number: int
) -> TransformerConfig:
    """Per-layer override of kv_channels / num_query_groups / window_size.

    Reads ``config.layer_pattern[layer_number - 1]`` ('sliding' or 'global') and
    selects per-layer kv_channels and num_query_groups from
    ``config.per_layer_kv_channels`` / ``config.per_layer_num_query_groups``.

    Critical contract with TE backend: also writes ``cloned.window_size``, which
    ``TEDotProductAttention`` reads (megatron/core/extensions/transformer_engine.py
    line 738) and forwards to TE FlashAttention as ``window_size=(left, right)``.
    HF's ``sliding_window=1024`` ("1024 tokens visible") maps to TE's ``(1023, 0)``
    because TE's window is inclusive: ``[i-1023, i+0]`` = exactly 1024 tokens.
    Global layers get ``None`` (pure causal). If ``layer_pattern`` is empty, the
    config is returned unchanged (no-op for non-Gemma4 models).

    Caller MUST always receive the cloned config (never early-return after pattern
    is set) — otherwise sliding layers silently fall back to global causal.
    """
    if not config.layer_pattern:
        return config
    idx = layer_number - 1
    if idx < 0 or idx >= len(config.layer_pattern):
        return config
    layer_type = config.layer_pattern[idx]
    new_kv_channels = (config.per_layer_kv_channels or {}).get(layer_type)
    new_num_query_groups = (config.per_layer_num_query_groups or {}).get(layer_type)
    sliding_w = getattr(config, "sliding_window", None)
    cloned = copy.copy(config)
    if new_kv_channels is not None:
        cloned.kv_channels = new_kv_channels
    if new_num_query_groups is not None:
        cloned.num_query_groups = new_num_query_groups
    cloned.window_size = (sliding_w - 1, 0) if (layer_type == "sliding" and sliding_w is not None) else None
    return cloned


class Gemma4SelfAttention(SelfAttention):
    """Self-attention layer for Gemma4 with hybrid head_dim, K=V tying, and V LN.

    Args mirror SelfAttention exactly. The class inspects `config.layer_pattern`
    and `config.kv_tied_layers` to decide per-layer behavior.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
    ):
        effective_config = _resolve_per_layer_overrides(config, layer_number)
        super().__init__(
            config=effective_config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )

        self.kv_tied = (layer_number - 1) in (config.kv_tied_layers or [])

        if self.kv_tied:
            self.linear_qkv = build_module(
                submodules.linear_qkv,
                self.config.hidden_size,
                self.query_projection_size + self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
            )

        norm_cls = multiacc_modules.TENorm if is_te_min_version("1.9.0") else multiacc_modules.LocalNorm
        self.post_attention_layernorm = norm_cls(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
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
        sequence_len_offset=None,
    ):
        if isinstance(rotary_pos_emb, dict):
            layer_type = self.config.layer_pattern[self.layer_number - 1]
            try:
                rotary_pos_emb = rotary_pos_emb[layer_type]
            except KeyError as exc:
                raise KeyError(
                    f"Gemma4SelfAttention(layer_number={self.layer_number}): "
                    f"layer_pattern entry '{layer_type}' missing from rotary_pos_emb "
                    f"dict keys {list(rotary_pos_emb.keys())}."
                ) from exc
        output, bias = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            key_value_states=key_value_states,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        if bias is not None:
            output = output + bias
            bias = None
        output = self.post_attention_layernorm(output)
        return output, bias

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        if self.kv_tied:
            new_tensor_shape = mixed_qkv.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 1)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_qkv = mixed_qkv.view(*new_tensor_shape)

            split_arg_list = [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
            ]

            if SplitAlongDim is not None:
                (query, key) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, key) = torch.split(mixed_qkv, split_arg_list, dim=3)

            # K=V: value shares the K tensor. No clone — downstream kernels (TE/flash-attn)
            # treat K and V as read-only. Saves one tensor allocation per layer.
            value = key
        else:
            new_tensor_shape = mixed_qkv.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_qkv = mixed_qkv.view(*new_tensor_shape)

            split_arg_list = [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            if SplitAlongDim is not None:
                (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        # Parameter-free V LayerNorm: normalize over the head_dim with weight=None bias=None.
        # Equivalent to LayerNorm with weight=1 and bias=0 that is NOT learned.
        # Applied AFTER any K=V tying so V normalization does not mutate K.
        if self.kv_tied:
            # Re-derive value as a normalized copy of key so K is untouched.
            value = F.layer_norm(key, [self.hidden_size_per_attention_head])
        else:
            value = F.layer_norm(value, [self.hidden_size_per_attention_head])

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value
