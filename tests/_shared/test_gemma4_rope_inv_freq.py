"""Numerical equivalence tests for Gemma4 dual-RoPE inv_freq buffers.

Guards the two RoPE construction paths in :class:`Gemma4Model` against silent
drift from the HF reference. Run on CPU; no Megatron initialisation required.

  - Sliding layers use stock :class:`RotaryEmbedding` with ``head_dim=256``,
    ``rotary_base=1e4``, full rotation. HF formula:
        inv_freq[k] = 1 / 1e4 ** (2k / 256)   for k in [0, 128)

  - Global layers use :class:`Gemma4ProportionalRotaryEmbedding` with
    ``head_dim=512``, ``partial_rotary_factor=0.25``, ``rotary_base=1e6``.
    HF proportional formula:
        rope_angles = int(0.25 * 512 // 2)             # = 64
        inv_freq[k] = 1 / 1e6 ** (2k / 512)            for k in [0, 64)
        inv_freq[k] = 0                                for k in [64, 256)

A regression in either path silently corrupts attention logits across the
LLM. These tests catch the corruption at construction time, before any forward
pass or checkpoint conversion.
"""

from __future__ import annotations

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

from aiak_training_llm.models.gemma4_vl.gemma4_proportional_rotary import (
    Gemma4ProportionalRotaryEmbedding,
)


def _expected_default_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def _expected_proportional_inv_freq(
    head_dim: int, partial_rotary_factor: float, base: float
) -> torch.Tensor:
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq = torch.zeros(head_dim // 2, dtype=torch.float32)
    inv_freq[:rope_angles] = 1.0 / (
        base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
    )
    return inv_freq


def test_sliding_inv_freq_matches_hf_default() -> None:
    rope = RotaryEmbedding(
        kv_channels=256,
        rotary_percent=1.0,
        rotary_interleaved=False,
        rotary_base=10000,
        use_cpu_initialization=True,
    )
    expected = _expected_default_inv_freq(head_dim=256, base=10000.0)
    assert rope.inv_freq.shape == expected.shape
    assert torch.allclose(rope.inv_freq, expected, rtol=0.0, atol=0.0), (
        f"Sliding RoPE inv_freq mismatch: max abs diff "
        f"{(rope.inv_freq - expected).abs().max().item():.3e}"
    )


def test_global_inv_freq_matches_hf_proportional() -> None:
    rope = Gemma4ProportionalRotaryEmbedding(
        head_dim=512,
        partial_rotary_factor=0.25,
        rotary_base=1_000_000,
        rotary_interleaved=False,
        use_cpu_initialization=True,
    )
    expected = _expected_proportional_inv_freq(
        head_dim=512, partial_rotary_factor=0.25, base=1_000_000.0
    )

    assert rope.inv_freq.shape == expected.shape, (
        f"Global RoPE inv_freq shape {tuple(rope.inv_freq.shape)} != expected "
        f"{tuple(expected.shape)}"
    )

    rope_angles = int(0.25 * 512 // 2)
    assert torch.allclose(
        rope.inv_freq[:rope_angles], expected[:rope_angles], rtol=0.0, atol=0.0
    ), (
        f"Global RoPE rotated head ({rope_angles} angles) mismatch: max abs "
        f"diff {(rope.inv_freq[:rope_angles] - expected[:rope_angles]).abs().max().item():.3e}"
    )

    assert torch.equal(
        rope.inv_freq[rope_angles:], torch.zeros(512 // 2 - rope_angles)
    ), (
        f"Global RoPE non-rotated tail must be exactly zero (HF zero-pad); got "
        f"max abs {rope.inv_freq[rope_angles:].abs().max().item():.3e}"
    )


def test_global_forward_emits_head_dim_long_emb() -> None:
    rope = Gemma4ProportionalRotaryEmbedding(
        head_dim=512,
        partial_rotary_factor=0.25,
        rotary_base=1_000_000,
        rotary_interleaved=False,
        use_cpu_initialization=True,
    )
    seq_len = 16
    emb = rope(seq_len)
    assert emb.shape[-1] == 512, (
        f"Global RoPE emb last dim must equal head_dim=512; got {emb.shape[-1]}"
    )
