"""Gemma4 vision patch embedder.

Verbatim port of HF ``Gemma4VisionPatchEmbedder`` (modeling_gemma4 lines 550-582).

Key non-obvious facts that callers MUST honor:
- Pre-projection scaling is ``2 * (pixel_values - 0.5)``, NOT a learned standardize
  pair (``std_bias``/``std_scale``). The standardize buffers live on
  ``Gemma4VisionTower`` and are applied *after* pooling, not before patch embed.
- ``position_embedding_table`` has shape ``[2, position_embedding_size, hidden_size]``:
  the leading ``2`` indexes the spatial axis (x first, y second). Two PE lookups
  per patch are summed; padding patches (position == -1) get zeroed out.
- Position embedding uses a one-hot @ matmul rather than ``nn.Embedding`` lookup
  so it stays within the autograd graph for fused backward when patches are
  shared across multiple soft-token outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        hidden_size: int,
        position_embedding_size: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.position_embedding_size = position_embedding_size
        self.in_channels = in_channels

        self.input_proj = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, position_embedding_size, hidden_size)
        )

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings
