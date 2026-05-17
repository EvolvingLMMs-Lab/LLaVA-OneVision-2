"""Gemma4-VL multimodal attention mask with bidirectional vision-span overlay.

This module implements HuggingFace ``create_causal_mask_mapping`` semantics for
Megatron's ``TransformerBlock.forward(attention_mask=...)`` contract:

  1. Base causal mask (lower-triangular True-is-VISIBLE inside the helper, then
     inverted to Megatron True-is-MASKED at return).

  2. Vision overlay: tokens belonging to the same vision span (contiguous run of
     ``mm_token_type_ids in {1, 2}``) attend bidirectionally to one another.
     Mathematically ``mask = causal | (q_group == kv_group) & (q_group >= 0)``,
     mirroring HF ``token_type_ids_mask_function`` and ``or_masks`` composition
     in ``transformers.models.gemma4.modeling_gemma4``.

The helper returns a SINGLE BoolTensor ``[B, 1, S, S]`` (True = MASKED), shared
by every ``TransformerLayer``. Sliding layers further restrict this base mask at
runtime via TE's ``window_size=(sliding_window-1, 0)`` parameter, which is wired
per-layer by ``gemma4_attention.py:_resolve_per_layer_overrides``. Because each
Gemma4 vision span is bounded by the ViT grid (16x16 = 256 patches) and the
sliding window is 1024, the OR overlay is always strictly contained inside the
sliding window — TE's hard window cutoff therefore composes losslessly with the
overlay, recovering exact HF semantics WITHOUT a per-layer mask dict.

A runtime guard (``_assert_vision_spans_within_window``) enforces the
``max_vision_span <= sliding_window`` invariant and fails fast if a future video
packing scheme produces a span that would break the equivalence.

External callers: ``Gemma4VLModel.forward`` (P5.5) — invoked when the batch
contains vision tokens. Pure-text batches skip this helper and pass the stock
causal mask produced by the data path.
"""

from __future__ import annotations

import torch


VISION_TOKEN_TYPE_IDS: tuple[int, ...] = (1, 2)


def _compute_vision_group_ids(mm_token_type_ids: torch.Tensor) -> torch.Tensor:
    """Assign each token a vision-span group id (text tokens get -1).

    Mirrors HF ``get_block_sequence_ids_for_mask`` (modeling_gemma4.py:2078-2087):
    a new group starts whenever a vision token follows a non-vision token.
    """
    is_vision = torch.zeros_like(mm_token_type_ids, dtype=torch.bool)
    for token_id in VISION_TOKEN_TYPE_IDS:
        is_vision = is_vision | (mm_token_type_ids == token_id)

    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_starts.to(torch.long), dim=-1) - 1
    return torch.where(is_vision, vision_group_ids, torch.full_like(vision_group_ids, -1))


def _assert_vision_spans_within_window(
    vision_group_ids: torch.Tensor,
    sliding_window: int,
) -> None:
    """Fail fast if any vision span exceeds the sliding window.

    Gemma4 single-image spans are physically bounded by ViT grid (<=256 patches),
    so this guard only triggers on hypothetical future video packing. Without it,
    TE's hard sliding cutoff would silently truncate the OR overlay, producing
    NaN-equivalent silent divergence from HF rather than a fail-fast error.
    """
    valid_mask = vision_group_ids >= 0
    if not valid_mask.any():
        return
    flat_groups = vision_group_ids[valid_mask]
    span_lengths = torch.bincount(flat_groups)
    max_span = int(span_lengths.max().item())
    assert max_span <= sliding_window, (
        f"Gemma4-VL vision span length {max_span} exceeds sliding_window {sliding_window}; "
        "the OR-overlay no longer composes losslessly with TE per-layer window cutoff. "
        "Either disable sliding window for vision batches or shorten the span."
    )


def build_gemma4_mm_attention_mask(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    sliding_window: int,
) -> torch.Tensor:
    """Build a Megatron-format multimodal attention mask for Gemma4-VL.

    Args:
        input_ids: ``[B, S]`` token ids (only shape and device are used).
        mm_token_type_ids: ``[B, S]`` int tensor with values in
            ``{0=text, 1=image, 2=video}``.
        sliding_window: Sliding window size from ``Gemma4VLConfig.sliding_window``
            (1024 for 26B-A4B). Used only by the runtime guard; sliding layers
            apply the cutoff themselves via TE ``window_size``.

    Returns:
        BoolTensor ``[B, 1, S, S]`` with ``True = MASKED`` (Megatron convention).
        Layers select between the full and sliding behavior via TE's per-layer
        ``window_size``; this single mask serves both.
    """
    if input_ids.shape != mm_token_type_ids.shape:
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} does not match "
            f"mm_token_type_ids shape {tuple(mm_token_type_ids.shape)}"
        )
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D [B, S]; got shape {tuple(input_ids.shape)}")

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    causal_visible = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    ).unsqueeze(0).expand(batch_size, -1, -1)

    vision_group_ids = _compute_vision_group_ids(mm_token_type_ids.to(device))
    _assert_vision_spans_within_window(vision_group_ids, sliding_window)

    q_groups = vision_group_ids.unsqueeze(-1)
    kv_groups = vision_group_ids.unsqueeze(-2)
    vision_overlay_visible = (q_groups == kv_groups) & (q_groups >= 0)

    visible = causal_visible | vision_overlay_visible
    masked = ~visible
    return masked.unsqueeze(1)
