"""Gemma4 vision 2D rotary position embedding.

Verbatim port of HF ``Gemma4VisionRotaryEmbedding`` and ``apply_multidimensional_rope``
from transformers ``modeling_gemma4`` (lines 659-866). Kept as a standalone module so
that the Megatron port stays bit-identical to HF; do NOT route this through the
shared Megatron RoPE plumbing because Megatron's RoPE assumes a single 1-D position
sequence whereas Gemma4 vision splits the head dimension across two spatial axes
(x, y) and applies independent rope to each half.

Math contract:
- ``head_dim`` is even and divisible by ``2 * ndim`` (ndim=2 here).
- ``spatial_dim = head_dim // ndim`` (=36 for head_dim=72): each axis gets its own
  ``inv_freq`` table of length ``spatial_dim // 2`` (=18 sin/cos pairs per axis).
- ``forward(x, position_ids)`` returns ``(cos, sin)`` each of shape
  ``[B, N, head_dim]`` where the last dim is the concatenation of the per-axis
  cos/sin tables (axis-x first ``spatial_dim`` channels, axis-y last ``spatial_dim``).
- ``apply_multidimensional_rope`` splits ``x``, ``cos``, ``sin`` along the last axis
  into ``ndim`` equal parts and applies standard rope independently to each, then
  concatenates back. This matches the HF reference and is the only way to reproduce
  ``model.embed_vision.embedding_projection`` numerics.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply standard 1-D rotary position embedding to ``x``.

    ``cos`` / ``sin`` have shape ``[B, N, D]`` and are unsqueezed at
    ``unsqueeze_dim`` to broadcast against ``x`` whose layout is either
    ``[B, H, N, D]`` (unsqueeze_dim=1, post-transpose) or
    ``[B, N, H, D]`` (unsqueeze_dim=2, pre-transpose, used by Gemma4 vision).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply multi-dimensional RoPE to ``x``.

    Splits ``x`` along the last (channel) dimension into ``ndim`` parts
    (where ``ndim = position_ids.shape[-1]``), applies standard rotary embedding
    independently to each part using its own slice of ``cos`` / ``sin``, then
    concatenates the parts back. For Gemma4 vision ``ndim=2`` (x and y axes).

    Args:
        x: Tensor of shape ``[B, N, H, D]`` (queries / keys before transpose).
        cos: Tensor of shape ``[B, N, D]`` produced by ``Gemma4VisionRotaryEmbedding``.
        sin: Tensor of shape ``[B, N, D]`` produced by ``Gemma4VisionRotaryEmbedding``.
        position_ids: Tensor of shape ``[B, N, ndim]`` with per-axis patch coordinates.
        unsqueeze_dim: Broadcast axis when applying rope (defaults to 2 to match
            Gemma4VisionAttention call site, where ``x`` is pre-transpose ``[B,N,H,D]``).

    Returns:
        Tensor of shape ``[B, N, H, D]`` with rope applied.
    """
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "Invalid configuration: num_rotated_channels_per_dim must be > 0, got "
            f"{num_rotated_channels_per_dim} (num_input_channels={num_input_channels}, "
            f"ndim={ndim})"
        )

    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        apply_rotary_pos_emb(
            x=x_parts[k],
            cos=cos_parts[k],
            sin=sin_parts[k],
            unsqueeze_dim=unsqueeze_dim,
        )
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """Gemma4 vision 2-D rotary embedding.

    Computes per-axis ``cos`` / ``sin`` tables for x and y patch coordinates and
    concatenates them along the channel dim. The result is a single
    ``[B, N, head_dim]`` cos/sin pair that ``apply_multidimensional_rope`` knows
    how to split back into per-axis pieces.

    The frequency table is computed once per axis with
    ``spatial_dim = head_dim // ndim`` and ``base = rope_theta``; ``inv_freq`` is
    a non-persistent buffer so it is re-derived from config on load and never
    stored in checkpoints.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 100.0,
        ndim: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if head_dim % (2 * ndim) != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 2*ndim={2 * ndim} "
                "to support multidimensional rotary embedding."
            )
        self.head_dim = head_dim
        self.ndim = ndim
        self.rope_theta = rope_theta
        self.attention_scaling = 1.0

        spatial_dim = head_dim // ndim
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / spatial_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )

        all_cos: list[torch.Tensor] = []
        all_sin: list[torch.Tensor] = []
        for i in range(self.ndim):
            dim_position_ids = position_ids[:, :, i]
            dim_position_ids_expanded = dim_position_ids[:, None, :].float()

            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ dim_position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)

        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin
