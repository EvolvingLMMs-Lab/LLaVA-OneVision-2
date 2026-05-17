"""Gemma4 vision-to-LM adapter.

Verbatim port of HF ``Gemma4MultimodalEmbedder`` (modeling_gemma4 lines 2023-2047).

In the HF state_dict the adapter is stored under
``model.embed_vision.{embedding_pre_projection_norm,embedding_projection}.*``.
For Gemma4-26B-A4B-it specifically:
- ``embedding_pre_projection_norm`` is a ``Gemma4RMSNorm`` with
  ``with_scale=False`` (no learnable weight, ckpt has no entry for it).
- ``embedding_projection`` is ``nn.Linear(vision.hidden_size=1152,
  text.hidden_size=2816, bias=False)``; the only ckpt entry is
  ``model.embed_vision.embedding_projection.weight``.

So the entire adapter holds exactly ONE trainable tensor. Do not promote the
RMSNorm to ``with_scale=True`` even though it would seem more "complete" — HF
ckpt loading will fail on an unexpected ``embedding_pre_projection_norm.weight``
key. The naming uses HF attribute names so converter mapping stays trivial.
"""

from __future__ import annotations

import torch
from torch import nn

from .gemma4_vision_common import Gemma4RMSNorm


class Gemma4Adapter(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            vision_hidden_size, eps=rms_norm_eps, with_scale=False
        )
        self.embedding_projection = nn.Linear(
            vision_hidden_size, text_hidden_size, bias=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)
