#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s_make_trivial_bins.py

Build a "trivial" bins.pkl for the no-packing path.

Each manifest entry becomes its own bin (sample_count = 1). The bins file
preserves the same numpy structured dtype used by ``s3_bin_packing.py`` so
that the existing ``s4_bins_to_webdataset.py`` consumer works unchanged.

Additionally, this script can inject ``timestamp_decimal`` (and any other
per-sample fields) into the ``raw`` payload of the manifest, so that the
WebDataset payload built by s4 carries that field.

Usage::

    python s_make_trivial_bins.py \
        --manifest /path/to/manifest.jsonl \
        --bins-out /path/to/bins.pkl \
        --inject-timestamp-decimal 2

    # Optionally rewrite the manifest in place with the injected fields:
    python s_make_trivial_bins.py \
        --manifest /path/to/manifest.jsonl \
        --bins-out /path/to/bins.pkl \
        --inject-timestamp-decimal 2 \
        --rewrite-manifest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Any, Iterable, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Must mirror s3_bin_packing.BinPacker.BIN_DTYPE so that downstream readers
# (notably s4) keep working.
BIN_DTYPE = np.dtype(
    [
        ("w", "<u2"),
        ("l", "<u4"),
        ("name", "<U200"),
        ("p", "<u4"),
    ]
)


def _iter_manifest(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _maybe_inject_fields(
    entry: dict[str, Any],
    inject_timestamp_decimal: Optional[int],
) -> bool:
    """Inject fields into entry['raw']. Returns True if the entry was modified."""
    raw = entry.setdefault("raw", {})
    changed = False
    if inject_timestamp_decimal is not None:
        # Only set if not already present, to avoid clobbering anything already there.
        if raw.get("timestamp_decimal") != inject_timestamp_decimal:
            raw["timestamp_decimal"] = inject_timestamp_decimal
            changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl built by s1_build_manifest.py")
    parser.add_argument("--bins-out", required=True, help="Output path for the trivial bins.pkl")
    parser.add_argument(
        "--inject-timestamp-decimal",
        type=int,
        default=None,
        help="If set, inject this value as timestamp_decimal into each entry['raw']",
    )
    parser.add_argument(
        "--rewrite-manifest",
        action="store_true",
        help="Rewrite the manifest in place with the injected fields. "
             "If false (default), changes are kept only in memory while building bins.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    bins: list[np.ndarray] = []
    rewritten_lines: list[str] = []
    n_changed = 0

    t0 = time.time()
    for entry in _iter_manifest(args.manifest):
        name = entry.get("name")
        if not name:
            logger.warning("Skipping entry without 'name': %s", str(entry)[:120])
            continue

        if _maybe_inject_fields(entry, args.inject_timestamp_decimal):
            n_changed += 1

        # Build a 1-element structured array for this bin.
        # We don't have token / patch counts in the no-packing path; use 0.
        arr = np.zeros((1,), dtype=BIN_DTYPE)
        arr[0]["w"] = 0
        arr[0]["l"] = 0
        arr[0]["name"] = str(name)[:200]
        arr[0]["p"] = 0
        bins.append(arr)

        if args.rewrite_manifest:
            rewritten_lines.append(json.dumps(entry, ensure_ascii=False))

    n_bins = len(bins)
    if n_bins == 0:
        logger.error("Manifest produced 0 bins; aborting.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.bins_out) or ".", exist_ok=True)
    with open(args.bins_out, "wb") as f:
        pickle.dump(bins, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Wrote %d trivial bins to %s in %.1fs (injected_changes=%d)",
        n_bins, args.bins_out, time.time() - t0, n_changed,
    )

    if args.rewrite_manifest:
        tmp = args.manifest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(rewritten_lines))
            if rewritten_lines:
                f.write("\n")
        os.replace(tmp, args.manifest)
        logger.info("Rewrote manifest in place with injected fields: %s", args.manifest)


if __name__ == "__main__":
    main()
