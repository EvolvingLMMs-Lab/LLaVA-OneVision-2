# Data-driven canvas point/range timestamps

The encoder selects timestamp syntax from each canvas's explicit raw patch
positions. It requires no dataset marker, environment switch, allowlist, or
codec-group sidecar.

## Timestamp schema

For each visual canvas:

- if every patch has the same `patch_positions[:, 0]` value, use the existing
  point timestamp, for example `<12.3 seconds>`;
- if the canvas spans multiple temporal values, use the minimum and maximum
  source time, for example `<12.3 to 14.8 seconds>`;
- timestamps stay immediately before their matching vision wrapper;
- a single-time canvas never emits a degenerate range such as
  `<12.3 to 12.3 seconds>`.

This is deliberately not an anchor/update classifier. An anchor that actually
contains multiple temporal values receives a range, while an update containing
one temporal value receives a point.

## Activation and compatibility

The path activates automatically when an image-backed sample supplies all of:

- raw `patch_positions`;
- processor `image_grid_thw` rows for the canvases;
- a positive FPS value.

Uniformly sampled frame data naturally emits one point timestamp per frame
because each frame canvas has one temporal value. Samples without explicit FPS
continue through the existing untimed/legacy fallback.

The serializer rewrites prompt text only. Pixels, processor grids, raw/model
patch positions, visual token order, and M-RoPE behavior are unchanged.

## Validation and malformed samples

Canvas grids and raw patch positions must close exactly. Invalid FPS, malformed
patch positions, wrapper-count mismatches, and wrapper/token mismatches are
classified as sample-local `CanvasTimestampError` failures. The task encoder
turns those failures into `SkipSample`, logs the first five occurrences per
worker, and reports the cumulative signature again every 1000 occurrences.

Configuration and programmer errors are not converted into skips and still
fail immediately.
