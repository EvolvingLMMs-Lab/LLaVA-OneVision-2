"""Gemma4 vision spatial pooler.

Verbatim port of HF ``Gemma4VisionPooler`` (modeling_gemma4 lines 584-640).

Three subtleties that determine numerical parity with HF:

1. The pooling kernel size ``k`` is derived dynamically from the *runtime*
   ratio between input patch count and target soft-token count:
   ``k = int(sqrt(input_seq_len // output_length))``. There is no
   ``pooling_kernel_size`` field used at runtime — the config field of that
   name is only consulted at processor / config-validation time to compute
   ``num_soft_tokens``. So this module must accept ``output_length`` per call
   and infer ``k`` from shapes; do NOT hard-code ``k=3`` even though that is
   the value used by Gemma4 vision under default config.

2. Pooling is done as ``(one_hot(kernel_idx) / k^2)^T @ hidden_states``: this
   is mathematically equivalent to a 2D avg-pool by spatial-bucket index but
   stays in the autograd graph as plain matmul, which is what the HF
   reference uses to keep the operation TPU/XLA-friendly. Replacing it with
   ``F.avg_pool2d`` is *not* equivalent because patches arrive in raster /
   packed order, not as a dense [B, C, H, W] grid.

3. The final ``hidden_states *= sqrt(hidden_size)`` scaling is applied
   ALWAYS, even on the no-pool branch (when input length already equals
   output length). This compensates for the post-LN variance reduction that
   the soft-token consumer (Gemma4MultimodalEmbedder) was trained against.
   Removing or moving this scale silently changes the LM input statistics
   and breaks downstream consistency.

The output ``mask`` is a *new* padding mask produced from the pooling weight
matrix: a pooled bucket is considered padding iff every input patch that
maps to it was already padding. The caller MUST use this returned mask, not
the input ``padding_positions``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Gemma4VisionPooler(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.root_hidden_size = hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k * k
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {tuple(hidden_states.shape)} to {length}: "
                f"k={k}^2 times length={length} must equal input_seq_len={input_seq_len}."
            )

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are "
                f"patches ({hidden_states.shape[1]}). Change `num_soft_tokens` upstream."
            )

        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )

        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions
