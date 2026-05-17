"""Gemma4 vision gated MLP.

Verbatim port of HF ``Gemma4VisionMLP`` (modeling_gemma4 lines 643-657).

The MLP shape is the SwiGLU-style ``down(act(gate) * up)`` (NOT
``down(gate * act(up))``); the activation wraps the gate projection only.
This matches Gemma's LLM MLP convention and is the order that HF checkpoints
were trained with — swapping the wrap target produces silently wrong outputs
because ``gate_proj`` and ``up_proj`` share the same in/out shapes.

``hidden_activation`` is ``gelu_pytorch_tanh`` for Gemma4 vision (config
default), which maps to ``torch.nn.functional.gelu(x, approximate="tanh")``.
The standard exact GELU diverges from HF outputs at the 5th decimal place,
which compounds across 27 layers and breaks consistency checks; do NOT
substitute the exact form.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .gemma4_vision_common import Gemma4ClippableLinear


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


_ACTIVATIONS = {
    "gelu_pytorch_tanh": _gelu_pytorch_tanh,
    "gelu": F.gelu,
    "silu": F.silu,
    "relu": F.relu,
}


class Gemma4VisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str = "gelu_pytorch_tanh",
        use_clipped_linears: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Gemma4ClippableLinear(
            hidden_size, intermediate_size, use_clipped_linears=use_clipped_linears
        )
        self.up_proj = Gemma4ClippableLinear(
            hidden_size, intermediate_size, use_clipped_linears=use_clipped_linears
        )
        self.down_proj = Gemma4ClippableLinear(
            intermediate_size, hidden_size, use_clipped_linears=use_clipped_linears
        )
        if hidden_activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown hidden_activation={hidden_activation!r}; "
                f"expected one of {sorted(_ACTIVATIONS)}."
            )
        self.act_fn = _ACTIVATIONS[hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
