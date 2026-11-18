from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aiak_training_llm"
    / "data"
    / "multimodal"
    / "canvas_timestamp_schema.py"
)
SPEC = importlib.util.spec_from_file_location("canvas_timestamp_schema", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VISION_END = MODULE.VISION_END
VISION_START = MODULE.VISION_START
CanvasTimestampError = MODULE.CanvasTimestampError
frozen_point_timestamp = MODULE.frozen_point_timestamp
frozen_range_timestamp = MODULE.frozen_range_timestamp
rewrite_for_sample = MODULE.rewrite_for_sample
rewrite_existing_canvas_wrappers = MODULE.rewrite_existing_canvas_wrappers
summarize_canvases = MODULE.summarize_canvases
timestamp_texts = MODULE.timestamp_texts

IMAGE_PAD = "<|image_pad|>"


def grid_positions(t_values, h=2, w=2):
    values = list(t_values)
    assert len(values) == h * w
    coordinates = [(y, x) for y in range(h) for x in range(w)]
    return np.asarray(
        [[t, y, x] for t, (y, x) in zip(values, coordinates)],
        dtype=np.int64,
    )


def wrappers(count_or_token_counts):
    token_counts = (
        [1] * count_or_token_counts
        if isinstance(count_or_token_counts, int)
        else count_or_token_counts
    )
    return "\n".join(
        VISION_START + IMAGE_PAD * token_count + VISION_END
        for token_count in token_counts
    )


class CanvasTimestampSchemaTest(unittest.TestCase):
    def test_frozen_point_and_range_format(self):
        self.assertEqual(frozen_point_timestamp(12.34, 1), "<12.3 seconds>")
        self.assertEqual(
            frozen_range_timestamp(12.34, 14.76, 1),
            "<12.3 to 14.8 seconds>",
        )
        self.assertEqual(
            frozen_range_timestamp(12.34, 12.34, 1),
            "<12.3 to 12.3 seconds>",
        )

    def test_ordinary_branch_is_byte_identical(self):
        original = "prefix\n" + wrappers(2) + "\nquestion"
        rewritten, audit = rewrite_for_sample(original)
        self.assertEqual(rewritten, original)
        self.assertEqual(audit, {"schema": "frozen_legacy_bypass", "changed": False})

    def test_each_canvas_uses_its_observed_temporal_span(self):
        flat = np.concatenate(
            [
                grid_positions([0, 0, 0, 0]),
                grid_positions([0, 72, 36, 18]),
                grid_positions([84, 156, 120, 96]),
            ]
        )
        grid = np.asarray([[1, 2, 2]] * 3)
        summaries = summarize_canvases(
            flat,
            grid,
            fps=30,
            spatial_merge_size=2,
        )
        self.assertEqual(
            [item.timestamp_kind for item in summaries],
            ["point", "range", "range"],
        )
        self.assertEqual(
            timestamp_texts(summaries, 1),
            [
                "<0.0 seconds>",
                "<0.0 to 2.4 seconds>",
                "<2.8 to 5.2 seconds>",
            ],
        )

    def test_uniform_frames_remain_point_timestamps(self):
        flat = np.concatenate([grid_positions([t] * 4) for t in (0, 3, 9, 12, 15)])
        grid = np.asarray([[1, 2, 2]] * 5)
        summaries = summarize_canvases(
            flat,
            grid,
            fps=3,
            spatial_merge_size=2,
        )
        self.assertEqual(
            [item.timestamp_kind for item in summaries],
            ["point"] * 5,
        )
        self.assertEqual(
            timestamp_texts(summaries, 1),
            [
                "<0.0 seconds>",
                "<1.0 seconds>",
                "<3.0 seconds>",
                "<4.0 seconds>",
                "<5.0 seconds>",
            ],
        )

    def test_nonmonotonic_canvas_uses_min_max(self):
        flat = np.concatenate(
            [grid_positions([0] * 4), grid_positions([9, 3, 6, 0])]
        )
        grid = np.asarray([[1, 2, 2]] * 2)
        summaries = summarize_canvases(
            flat,
            grid,
            fps=3,
            spatial_merge_size=2,
        )
        self.assertFalse(summaries[1].raw_temporal_monotonic)
        self.assertEqual(timestamp_texts(summaries, 1)[1], "<0.0 to 3.0 seconds>")

    def test_variable_grid_and_wrapper_rewrite(self):
        first = grid_positions([0] * 4)
        second = np.asarray(
            [
                [3, 0, 0],
                [6, 0, 1],
                [9, 0, 2],
                [12, 0, 3],
                [3, 1, 0],
                [6, 1, 1],
                [9, 1, 2],
                [12, 1, 3],
            ]
        )
        rewritten, audit = rewrite_for_sample(
            wrappers([1, 2]),
            patch_positions=np.concatenate([first, second]),
            image_grid_thw=np.asarray([[1, 2, 2], [1, 2, 4]]),
            fps=3,
            spatial_merge_size=2,
        )
        self.assertEqual(
            rewritten,
            "<0.0 seconds><|vision_start|><|image_pad|><|vision_end|>\n"
            "<1.0 to 4.0 seconds><|vision_start|>"
            "<|image_pad|><|image_pad|><|vision_end|>",
        )
        self.assertEqual(audit["merged_token_count"], 3)
        self.assertEqual(
            [item["raw_patch_count"] for item in audit["summaries"]],
            [4, 8],
        )

        with self.assertRaisesRegex(ValueError, "wrapper/grid token mismatch"):
            rewrite_for_sample(
                wrappers(2),
                patch_positions=np.concatenate([first, second]),
                image_grid_thw=np.asarray([[1, 2, 2], [1, 2, 4]]),
                fps=3,
                spatial_merge_size=2,
            )

    def test_single_time_canvas_never_uses_degenerate_range(self):
        flat = np.concatenate([grid_positions([0] * 4), grid_positions([3] * 4)])
        grid = np.asarray([[1, 2, 2]] * 2)
        summaries = summarize_canvases(flat, grid, fps=3, spatial_merge_size=2)
        self.assertEqual(timestamp_texts(summaries, 1), ["<0.0 seconds>", "<1.0 seconds>"])

    def test_invalid_timing_and_wrapper_shape_are_sample_errors(self):
        flat = np.concatenate([grid_positions([0] * 4), grid_positions([3] * 4)])
        grid = np.asarray([[1, 2, 2]] * 2)
        for fps in (None, 0, -1, float("nan")):
            with self.subTest(fps=fps), self.assertRaises(CanvasTimestampError):
                summarize_canvases(flat, grid, fps=fps, spatial_merge_size=2)
        with self.assertRaisesRegex(CanvasTimestampError, "patch/grid closure mismatch"):
            summarize_canvases(flat[:-1], grid, fps=3, spatial_merge_size=2)

        summaries = summarize_canvases(flat, grid, fps=3, spatial_merge_size=2)
        with self.assertRaisesRegex(CanvasTimestampError, "vision-wrapper/canvas mismatch"):
            rewrite_existing_canvas_wrappers(wrappers(1), summaries, decimals=1)

    def test_partial_timing_fails_closed(self):
        with self.assertRaisesRegex(CanvasTimestampError, "requires patch positions"):
            rewrite_for_sample(wrappers(1), patch_positions=grid_positions([0] * 4))

    def test_invalid_range_inputs(self):
        with self.assertRaises(ValueError):
            frozen_range_timestamp(3, 2, 1)
        with self.assertRaises(ValueError):
            frozen_range_timestamp(float("nan"), 2, 1)


if __name__ == "__main__":
    unittest.main()
