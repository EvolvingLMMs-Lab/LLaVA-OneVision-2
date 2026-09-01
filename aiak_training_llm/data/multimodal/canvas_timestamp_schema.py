"""Data-driven point/range timestamps for timed image canvases.

The point formatter intentionally mirrors the existing frame timestamp path.
Each canvas uses a point when all of its patches share one temporal coordinate,
and a range when its patches span multiple temporal coordinates.

This module changes prompt text only. It never changes pixels, grids, patch
positions, vision-token order, or M-RoPE.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


SCHEMA = "canvas_patch_span_point_range_v1"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


class CanvasTimestampError(ValueError):
    """One sample cannot be serialized from its explicit canvas timing data."""


@dataclass(frozen=True)
class CanvasTime:
    """Validated source-time information for one visual canvas."""

    canvas_index: int
    timestamp_kind: str
    point_frame: int
    start_frame: int
    end_frame: int
    point_seconds: float
    start_seconds: float
    end_seconds: float
    raw_patch_count: int
    merged_token_count: int
    grid_t: int
    grid_h: int
    grid_w: int
    raw_temporal_monotonic: bool


def frozen_decimals(value: object) -> int:
    """Apply the existing encoder's ``value or 1`` precision rule."""

    decimals = value or 1
    if isinstance(decimals, bool) or not isinstance(decimals, (int, np.integer)):
        raise CanvasTimestampError(f"timestamp_decimal must be an integer, got {value!r}")
    decimals = int(decimals)
    if not 0 <= decimals <= 6:
        raise CanvasTimestampError(f"timestamp_decimal must be in [0,6], got {decimals}")
    return decimals


def frozen_point_timestamp(seconds: object, decimals: object = 1) -> str:
    """Format the existing frame point timestamp."""

    precision = frozen_decimals(decimals)
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"point timestamp must be finite and non-negative, got {seconds!r}")
    value = round(value, precision)
    return f"<{value:.{precision}f} seconds>"


def frozen_range_timestamp(start: object, end: object, decimals: object = 1) -> str:
    """Format a canvas range with the same rounding and precision rules."""

    precision = frozen_decimals(decimals)
    start_value = float(start)
    end_value = float(end)
    if (
        not math.isfinite(start_value)
        or not math.isfinite(end_value)
        or start_value < 0
        or end_value < start_value
    ):
        raise ValueError(f"invalid timestamp range: start={start!r} end={end!r}")
    start_value = round(start_value, precision)
    end_value = round(end_value, precision)
    return f"<{start_value:.{precision}f} to {end_value:.{precision}f} seconds>"


def _integer_array(value: object, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise CanvasTimestampError(f"{name} must be numeric, got dtype={array.dtype}")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise CanvasTimestampError(f"{name} must contain finite integers")
    return array.astype(np.int64, copy=False)


def split_flat_canvases(
    patch_positions: object,
    image_grid_thw: object,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Split flat raw patch positions using authoritative per-canvas grids."""

    positions = _integer_array(patch_positions, "patch_positions")
    grid = _integer_array(image_grid_thw, "image_grid_thw")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise CanvasTimestampError(f"patch_positions must be [N,3], got {positions.shape}")
    if grid.ndim != 2 or grid.shape[1] != 3 or len(grid) == 0:
        raise CanvasTimestampError(f"image_grid_thw must be non-empty [canvas,3], got {grid.shape}")
    if (positions < 0).any():
        raise CanvasTimestampError("timed patch positions cannot contain negative coordinates")
    if (grid <= 0).any() or (grid[:, 0] != 1).any():
        raise CanvasTimestampError(
            "canvas timestamp serializer requires positive per-canvas rows with grid_t=1"
        )

    counts = [int(np.prod(row)) for row in grid]
    if sum(counts) != len(positions):
        raise CanvasTimestampError(
            f"patch/grid closure mismatch: positions={len(positions)} grid_patches={sum(counts)}"
        )
    cuts = np.cumsum(counts)[:-1]
    return list(np.split(positions, cuts)), grid


def summarize_canvases(
    patch_positions: object,
    image_grid_thw: object,
    *,
    fps: object,
    spatial_merge_size: int,
) -> list[CanvasTime]:
    """Build point/range inputs from each canvas's observed temporal span."""

    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise CanvasTimestampError(
            f"timed canvas sample requires finite positive fps, got {fps!r}"
        ) from exc
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise CanvasTimestampError(
            f"timed canvas sample requires finite positive fps, got {fps!r}"
        )

    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {merge}")

    canvases, grid = split_flat_canvases(patch_positions, image_grid_thw)
    merge_factor = merge * merge
    result: list[CanvasTime] = []
    for canvas_index, (canvas, row) in enumerate(zip(canvases, grid)):
        if len(canvas) % merge_factor:
            raise CanvasTimestampError(
                f"canvas {canvas_index} patch count {len(canvas)} is not divisible by "
                f"merge^2={merge_factor}"
            )

        temporal = canvas[:, 0]
        point_frame = int(temporal[0])
        start_frame = int(temporal.min())
        end_frame = int(temporal.max())
        result.append(
            CanvasTime(
                canvas_index=canvas_index,
                timestamp_kind="point" if start_frame == end_frame else "range",
                point_frame=point_frame,
                start_frame=start_frame,
                end_frame=end_frame,
                point_seconds=point_frame / fps_value,
                start_seconds=start_frame / fps_value,
                end_seconds=end_frame / fps_value,
                raw_patch_count=len(canvas),
                merged_token_count=len(canvas) // merge_factor,
                grid_t=int(row[0]),
                grid_h=int(row[1]),
                grid_w=int(row[2]),
                raw_temporal_monotonic=bool(np.all(temporal[1:] >= temporal[:-1])),
            )
        )
    return result


def timestamp_texts(summaries: Sequence[CanvasTime], decimals: object = 1) -> list[str]:
    """Render a point or range from each canvas's timestamp kind."""

    texts: list[str] = []
    for item in summaries:
        if item.timestamp_kind == "point":
            texts.append(frozen_point_timestamp(item.point_seconds, decimals))
        elif item.timestamp_kind == "range":
            texts.append(frozen_range_timestamp(item.start_seconds, item.end_seconds, decimals))
        else:
            raise ValueError(f"unknown canvas timestamp kind {item.timestamp_kind!r}")
    return texts


def rewrite_existing_canvas_wrappers(
    text: str,
    summaries: Sequence[CanvasTime],
    *,
    decimals: object = 1,
) -> tuple[str, dict]:
    """Prefix each existing canvas wrapper without rebuilding its body."""

    pattern = re.compile(
        re.escape(VISION_START)
        + r"(?:(?!"
        + re.escape(VISION_START)
        + r").)*?"
        + re.escape(VISION_END),
        re.DOTALL,
    )
    wrappers = list(pattern.finditer(text))
    if len(wrappers) != len(summaries):
        raise CanvasTimestampError(
            f"vision-wrapper/canvas mismatch: wrappers={len(wrappers)} canvases={len(summaries)}"
        )
    for item, wrapper in zip(summaries, wrappers):
        pad_count = wrapper.group(0).count(IMAGE_PAD)
        if pad_count != item.merged_token_count:
            raise CanvasTimestampError(
                f"canvas {item.canvas_index} wrapper/grid token mismatch: "
                f"wrapper={pad_count} grid={item.merged_token_count}"
            )

    texts = timestamp_texts(summaries, decimals)
    cursor = 0
    parts: list[str] = []
    for timestamp, match in zip(texts, wrappers):
        parts.extend((text[cursor : match.start()], timestamp, match.group(0)))
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), {
        "schema": SCHEMA,
        "canvas_count": len(summaries),
        "point_count": sum(item.timestamp_kind == "point" for item in summaries),
        "range_count": sum(item.timestamp_kind == "range" for item in summaries),
        "merged_token_count": sum(item.merged_token_count for item in summaries),
        "timestamp_texts": texts,
        "summaries": [asdict(item) for item in summaries],
    }


def rewrite_for_sample(
    text: str,
    *,
    patch_positions: object | None = None,
    image_grid_thw: object | None = None,
    fps: object | None = None,
    spatial_merge_size: int = 2,
    decimals: object = 1,
) -> tuple[str, dict]:
    """Dispatch while keeping untimed samples byte-identical."""

    timing_values = (patch_positions, image_grid_thw, fps)
    if all(value is None for value in timing_values):
        return text, {"schema": "frozen_legacy_bypass", "changed": False}
    if any(value is None for value in timing_values):
        raise CanvasTimestampError(
            "timed canvas sample requires patch positions, image grids, and fps together"
        )
    summaries = summarize_canvases(
        patch_positions,
        image_grid_thw,
        fps=fps,
        spatial_merge_size=spatial_merge_size,
    )
    return rewrite_existing_canvas_wrappers(text, summaries, decimals=decimals)
