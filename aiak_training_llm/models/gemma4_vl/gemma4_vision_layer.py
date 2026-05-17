"""Gemma4 vision encoder layer.

Verbatim port of HF ``Gemma4VisionEncoderLayer`` (modeling_gemma4 lines 939-980).

Sandwich normalization layout — this is the source of the layer's 4 RMSNorms
and the order is NOT the conventional pre-norm or post-norm transformer:

    residual = h
    h = input_layernorm(h)             # pre-attn norm
    h, _ = self_attn(h, ...)
    h = post_attention_layernorm(h)    # post-attn norm, BEFORE residual add
    h = residual + h

    residual = h
    h = pre_feedforward_layernorm(h)   # pre-mlp norm
    h = mlp(h)
    h = post_feedforward_layernorm(h)  # post-mlp norm, BEFORE residual add
    h = residual + h

The two "post_*_layernorm" calls happen *between* the sublayer output and the
residual sum, not after. Reordering to standard pre-norm
(``residual + sublayer(norm(h))``) breaks HF parity because the sandwich
construction normalizes the sublayer output before the skip-add, which changes
the variance arithmetic of every subsequent layer. This pattern is shared with
the LLM hybrid layers (see ``gemma4_transformer_layer.py``); see plan v5 for
why we do not collapse the four norms into two.
"""

from __future__ import annotations

import torch
from torch import nn

from .gemma4_vision_attention import Gemma4VisionAttention
from .gemma4_vision_common import Gemma4RMSNorm
from .gemma4_vision_mlp import Gemma4VisionMLP


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_activation: str = "gelu_pytorch_tanh",
        use_clipped_linears: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Gemma4VisionAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_dropout=attention_dropout,
            use_clipped_linears=use_clipped_linears,
        )
        self.mlp = Gemma4VisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation=hidden_activation,
            use_clipped_linears=use_clipped_linears,
        )
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
