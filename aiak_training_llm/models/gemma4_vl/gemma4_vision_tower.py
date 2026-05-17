"""Gemma4 vision tower.

Verbatim port of HF ``Gemma4VisionEncoder`` + ``Gemma4VisionModel``
(modeling_gemma4 lines 983-2020) bundled into a single Megatron-side
``VisionModule`` subclass.

Forward signature matches HF exactly:
    forward(pixel_values, pixel_position_ids) -> hidden_states
where:
    pixel_values: [B, num_patches, in_channels * patch_size**2] (flattened patches)
    pixel_position_ids: [B, num_patches, 2] with (-1, -1) marking padding patches.

The output is a *flat* tensor ``[total_valid_soft_tokens, hidden_size]`` where
padding has been stripped via ``hidden_states[pooler_mask]``. This matches HF
and is what ``Gemma4Adapter`` (= ``Gemma4MultimodalEmbedder``) expects. The
LLM-side scatter logic in Gemma4VL.forward must use the per-image patch counts
to slice this flat tensor back into per-sample chunks before injecting into
the LLM input embeddings.

Two design points worth flagging for the converter (Phase C):

1. ``std_bias`` and ``std_scale`` are registered ONLY when
   ``config.standardize=True``. Gemma4-26B-A4B-it has ``standardize=true``,
   so both buffers exist as real ckpt entries. The Megatron checkpoint MUST
   include them as buffer tensors (non-persistent=False), and the converter
   must round-trip them. They live on the VisionModel itself, NOT inside
   ``patch_embedder`` (a common misreading from earlier OV2 ports).

2. The encoder runs a non-causal additive attention mask derived from
   ``~padding_positions``. We build it here as a ``[B, 1, 1, N]`` broadcast
   mask (valid=0, pad=finfo.min), avoiding the full ``[B, 1, N, N]`` materialize
   that ``create_bidirectional_mask`` produces in HF. The arithmetic is
   identical because the Gemma vision encoder applies the mask on the
   key dimension only (no cross-sample masking, no per-q masking).
"""

from __future__ import annotations

import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from .gemma4_vision_layer import Gemma4VisionEncoderLayer
from .gemma4_vision_patch_embed import Gemma4VisionPatchEmbedder
from .gemma4_vision_pooler import Gemma4VisionPooler
from .gemma4_vision_rotary import Gemma4VisionRotaryEmbedding


class Gemma4VisionTower(VisionModule):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        patch_size: int,
        in_channels: int,
        position_embedding_size: int,
        pooling_kernel_size: int,
        rope_theta: float = 100.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_activation: str = "gelu_pytorch_tanh",
        use_clipped_linears: bool = False,
        standardize: bool = True,
    ) -> None:
        super().__init__(config=transformer_config)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.standardize = standardize

        self.patch_embedder = Gemma4VisionPatchEmbedder(
            patch_size=patch_size,
            hidden_size=hidden_size,
            position_embedding_size=position_embedding_size,
            in_channels=in_channels,
        )
        self.rotary_emb = Gemma4VisionRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
            ndim=2,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                    hidden_activation=hidden_activation,
                    use_clipped_linears=use_clipped_linears,
                    layer_idx=i,
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.pooler = Gemma4VisionPooler(hidden_size=hidden_size)

        if self.standardize:
            self.register_buffer(
                "std_bias", torch.empty(hidden_size), persistent=True
            )
            self.register_buffer(
                "std_scale", torch.empty(hidden_size), persistent=True
            )

    @staticmethod
    def _build_additive_mask(
        valid_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        neg_inf = torch.finfo(dtype).min
        additive = torch.zeros_like(valid_mask, dtype=dtype)
        additive = additive.masked_fill(~valid_mask, neg_inf)
        return additive[:, None, None, :]

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output_length = pixel_values.shape[-2] // (
            self.pooling_kernel_size * self.pooling_kernel_size
        )

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        valid_positions = ~padding_positions

        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        attention_mask = self._build_additive_mask(valid_positions, inputs_embeds.dtype)
        position_embeddings = self.rotary_emb(inputs_embeds, pixel_position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                attention_mask=attention_mask,
            )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        hidden_states = hidden_states[pooler_mask]

        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states
