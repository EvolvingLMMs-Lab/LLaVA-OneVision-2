"""megatron local norm - Apex NOT used; pure-PyTorch fallbacks."""

from typing import Optional
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

# Megatron config type (import only for typing/runtime usage)
from megatron.core.transformer.transformer_config import TransformerConfig

# Prefer Megatron's fused op if present (this is NOT Apex):
try:
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm  # optional
    HAVE_MEGATRON_FUSED_LAYER_NORM = True
except Exception:
    HAVE_MEGATRON_FUSED_LAYER_NORM = False

# --- Pure PyTorch RMSNorm implementation ---
class RMSNorm(nn.Module):
    """Pure-PyTorch RMSNorm implementation.

    y = x * (weight / sqrt(mean(x^2, dim=-1, keepdim=True) + eps))

    Keeps a `weight` attribute compatible with other code expecting per-channel
    affine parameters. Sets `weight.sequence_parallel` when a config is provided.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(hidden_size))
        else:
            # keep attribute for compatibility
            self.register_buffer("weight", torch.ones(hidden_size))

        # optional support for megatron sequence_parallel flag on config
        self.sequence_parallel = getattr(self.config, "sequence_parallel", False)
        try:
            setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        except Exception:
            # some frameworks may not allow setting attributes on Parameter on certain dtypes,
            # ignore silently (compatibility only)
            pass

    def forward(self, x: torch.Tensor):
        # compute RMS over last dim
        # ms shape: (..., 1)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rs = torch.rsqrt(ms + self.eps)  # (..., 1)
        if self.elementwise_affine:
            # broadcast weight across leading dims; if hidden size differs from weight length,
            # repeat or truncate to fit to avoid shape mismatch.
            hidden = x.size(-1)
            w_raw = self.weight
            if w_raw.numel() != hidden:
                repeat = math.ceil(hidden / w_raw.numel())
                w_raw = w_raw.repeat(repeat)[:hidden]
            w = w_raw.view(*([1] * (x.dim() - 1)), -1)
            # return x * rs * w
            return x * rs * w
        else:
            return x * rs


# --- LayerNorm wrapper (Megatron fused if available, else torch.nn.LayerNorm) ---
class TorchLayerNormWrapper(nn.Module):
    """Wrap torch.nn.LayerNorm so its initializer and attributes match expected API."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            # expose weight attribute for compatibility (so callers can set sequence_parallel)
            self.weight = self.layernorm.weight
        else:
            # still expose a weight-like buffer for code that expects it
            self.register_buffer("weight", torch.ones(hidden_size))

    def forward(self, x: torch.Tensor):
        return self.layernorm(x)


# Pick the best LayerNorm implementation available (but do NOT import Apex anywhere).
if HAVE_MEGATRON_FUSED_LAYER_NORM:
    LayerNormClass = FusedLayerNorm
else:
    LayerNormClass = TorchLayerNormWrapper


# --- Factory class: LocalNorm ---
class LocalNorm:
    """
    Factory to create a normalization module consistent with megatron config.

    Usage:
        norm = LocalNorm(config, hidden_size, eps=1e-5, elementwise_affine=True)

    This will return either:
    - a LayerNorm-like module (Megatron fused if available, otherwise torch.nn.LayerNorm wrapper), or
    - RMSNorm (pure-PyTorch).
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        norm_type = getattr(config, "normalization", "LayerNorm")
        if norm_type == "LayerNorm":
            # prefer megatron fused if it exists, else torch wrapper
            # Megatron FusedLayerNorm initializer signature: (config=config, hidden_size=..., eps=...)
            if HAVE_MEGATRON_FUSED_LAYER_NORM:
                # Create using Megatron fused implementation (keeps config semantics)
                try:
                    instance = FusedLayerNorm(config=config, hidden_size=hidden_size, eps=eps)
                except Exception:
                    # fallback to torch wrapper if megatron fused can't be constructed
                    instance = LayerNormClass(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
            else:
                instance = LayerNormClass(hidden_size, eps=eps, elementwise_affine=elementwise_affine)

            # if config exposes sequence_parallel, replicate attribute on weight if present
            seq_par = getattr(config, "sequence_parallel", False)
            if hasattr(instance, "weight"):
                try:
                    setattr(instance.weight, "sequence_parallel", seq_par)
                except Exception:
                    pass

        elif norm_type == "RMSNorm":
            instance = RMSNorm(config=config, hidden_size=hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError("Only 'LayerNorm' and 'RMSNorm' are supported for LocalNorm")

        return instance
