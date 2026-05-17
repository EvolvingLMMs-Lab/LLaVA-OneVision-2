"""Gemma4 vision multi-head attention.

Verbatim port of HF ``Gemma4VisionAttention`` and ``eager_attention_forward``
(modeling_gemma4 lines 779-936).

Three non-obvious design choices that MUST be preserved bit-for-bit with HF:

1. ``self.scaling = 1.0`` (NOT ``head_dim ** -0.5``). The standard
   1/sqrt(d) scale is *omitted* because ``q_norm`` and ``k_norm`` already
   normalize Q and K to unit-RMS. Adding the scale here would double-scale
   and break HF parity.

2. ``v_norm`` is a parameterless RMSNorm (``with_scale=False``) applied to
   value states *after* the V projection but *before* head transpose. This
   is unusual (most attention impls don't normalize V); do not remove it.

3. RoPE is applied via ``apply_multidimensional_rope`` to Q and K *before*
   the head transpose, with ``unsqueeze_dim=2`` to match the
   ``[B, N, H, D]`` layout. After RoPE, Q/K/V are transposed to
   ``[B, H, N, D]`` for attention.

The attention math is fp32 softmax (upcast inside ``eager_attention_forward``)
and uses standard GQA via ``repeat_kv``. We deliberately do NOT route this
through Megatron's fused TE attention because (a) vision sequences are short
(<=2240 patches) so the kernel speedup is marginal, and (b) TE's softmax
fp16/bf16 path diverges from HF's fp32 path at the 4th decimal which would
fail downstream consistency checks against the HF reference.
"""

from __future__ import annotations

import torch
from torch import nn

from .gemma4_vision_common import Gemma4ClippableLinear, Gemma4RMSNorm
from .gemma4_vision_rotary import apply_multidimensional_rope


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma4VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        use_clipped_linears: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = attention_dropout
        self.is_causal = False

        self.q_proj = Gemma4ClippableLinear(
            hidden_size, num_attention_heads * head_dim, use_clipped_linears=use_clipped_linears
        )
        self.k_proj = Gemma4ClippableLinear(
            hidden_size, num_key_value_heads * head_dim, use_clipped_linears=use_clipped_linears
        )
        self.v_proj = Gemma4ClippableLinear(
            hidden_size, num_key_value_heads * head_dim, use_clipped_linears=use_clipped_linears
        )
        self.o_proj = Gemma4ClippableLinear(
            num_attention_heads * head_dim, hidden_size, use_clipped_linears=use_clipped_linears
        )

        self.q_norm = Gemma4RMSNorm(dim=head_dim, eps=rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=head_dim, eps=rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(dim=head_dim, eps=rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
