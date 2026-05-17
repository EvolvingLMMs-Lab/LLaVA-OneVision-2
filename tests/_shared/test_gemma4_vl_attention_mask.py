from __future__ import annotations

import pytest
import torch

from aiak_training_llm.models.gemma4_vl.gemma4_vl_attention_mask import (
    _compute_vision_group_ids,
    build_gemma4_mm_attention_mask,
)


def _make_inputs(mtti_row: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    mtti = torch.tensor([mtti_row], dtype=torch.long)
    input_ids = torch.zeros_like(mtti)
    return input_ids, mtti


def test_pure_text_returns_strict_causal_mask():
    input_ids, mtti = _make_inputs([0, 0, 0, 0, 0])
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)

    assert mask.shape == (1, 1, 5, 5)
    assert mask.dtype == torch.bool

    expected_visible = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    expected_masked = ~expected_visible
    assert torch.equal(mask[0, 0], expected_masked)


def test_single_image_span_attends_bidirectionally_inside_span():
    input_ids, mtti = _make_inputs([0, 1, 1, 1, 0])
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)
    visible = ~mask[0, 0]

    for q in (1, 2, 3):
        for kv in (1, 2, 3):
            assert visible[q, kv], f"expected vision token q={q} to see kv={kv}"


def test_vision_overlay_does_not_break_causality_outside_span():
    input_ids, mtti = _make_inputs([0, 1, 1, 1, 0, 0])
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)
    visible = ~mask[0, 0]

    assert not visible[0, 1]
    assert not visible[0, 4]
    assert not visible[4, 5]
    assert visible[4, 1]
    assert visible[5, 3]


def test_two_image_spans_do_not_cross_attend():
    input_ids, mtti = _make_inputs([1, 1, 0, 1, 1])
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)
    visible = ~mask[0, 0]

    assert visible[0, 1] and visible[1, 0]
    assert visible[3, 4] and visible[4, 3]
    assert not visible[0, 3]
    assert not visible[0, 4]
    assert visible[3, 0] and visible[3, 1]
    assert visible[4, 0] and visible[4, 1]


def test_video_token_type_id_2_treated_as_vision():
    input_ids, mtti = _make_inputs([0, 2, 2, 2, 0])
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)
    visible = ~mask[0, 0]

    for q in (1, 2, 3):
        for kv in (1, 2, 3):
            assert visible[q, kv]


def test_compute_vision_group_ids_matches_hf_block_seq_ids():
    mtti = torch.tensor([[0, 1, 1, 0, 2, 2, 2, 0, 1]], dtype=torch.long)
    group_ids = _compute_vision_group_ids(mtti)
    expected = torch.tensor([[-1, 0, 0, -1, 1, 1, 1, -1, 2]], dtype=torch.long)
    assert torch.equal(group_ids, expected)


def test_runtime_guard_fires_when_span_exceeds_sliding_window():
    seq_len = 10
    mtti = torch.ones((1, seq_len), dtype=torch.long)
    input_ids = torch.zeros_like(mtti)
    with pytest.raises(AssertionError, match="exceeds sliding_window"):
        build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=4)


def test_runtime_guard_passes_when_span_equals_sliding_window():
    seq_len = 8
    mtti = torch.ones((1, seq_len), dtype=torch.long)
    input_ids = torch.zeros_like(mtti)
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=8)
    assert mask.shape == (1, 1, 8, 8)


def test_batch_dimension_is_independent_per_sequence():
    mtti = torch.tensor(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    input_ids = torch.zeros_like(mtti)
    mask = build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)

    assert mask.shape == (2, 1, 5, 5)

    assert (~mask[0, 0])[1, 2] and (~mask[0, 0])[2, 1]
    assert not (~mask[1, 0])[1, 2]
    assert (~mask[1, 0])[2, 3] and (~mask[1, 0])[3, 2]


def test_shape_mismatch_raises_value_error():
    input_ids = torch.zeros((1, 5), dtype=torch.long)
    mtti = torch.zeros((1, 6), dtype=torch.long)
    with pytest.raises(ValueError, match="does not match"):
        build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)


def test_non_2d_input_raises_value_error():
    input_ids = torch.zeros((5,), dtype=torch.long)
    mtti = torch.zeros((5,), dtype=torch.long)
    with pytest.raises(ValueError, match="must be 2D"):
        build_gemma4_mm_attention_mask(input_ids, mtti, sliding_window=1024)
