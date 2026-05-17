"""Shared Gemma4 vision/adapter primitives: RMSNorm and ClippableLinear.

Verbatim port of HF ``Gemma4RMSNorm`` (lines 168-187) and ``Gemma4ClippableLinear``
(lines 139-167) from transformers ``modeling_gemma4``.

Two non-obvious facts that determine HF checkpoint compatibility and MUST NOT
change:

1. ``Gemma4ClippableLinear`` always wraps the real ``nn.Linear`` under attribute
   name ``.linear``. HF checkpoints store keys as ``*.<projection>.linear.weight``
   even when ``use_clipped_linears=False`` and the clamp branch is dead code.
   Renaming this attribute (e.g. inlining the linear) breaks every HF→Megatron
   converter mapping.

2. ``Gemma4RMSNorm.with_scale=False`` produces a parameterless module (no
   ``weight`` attribute at all). Both ``v_norm`` in vision attention and the
   adapter's RMSNorm rely on this; do NOT add a default ``weight`` for "API
   uniformity" because state_dict loading from HF will then fail with an
   unexpected key.

The HF clamp branch (``use_clipped_linears=True``) registers four buffers
(input/output min/max). Gemma4-26B-A4B-it ships with ``use_clipped_linears=False``
so we keep the branch for fidelity but it is exercised only by post-training
quantization configs.
"""

from __future__ import annotations

import torch
from torch import nn


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.rsqrt(mean_squared)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self._norm(hidden_states.float())
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(hidden_states)

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if self.with_scale else None
        return f"weight={weight_shape}, eps={self.eps}, with_scale={self.with_scale}"


class Gemma4ClippableLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_clipped_linears: bool = False,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states
