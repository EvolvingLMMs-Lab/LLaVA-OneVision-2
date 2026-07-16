#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_build_manifest.py

Build a lightweight manifest (JSONL) from one or more raw JSON/JSONL data files.

This is the new step-1 of the offline packing pipeline. It replaces the old
``s1_split_json_to_samples.py`` (which physically split the input JSONL into a
huge directory of per-sample ``.json`` + copied images + copied ``.npy`` files).

Why a manifest instead of splitting?
    * The original step-1 doubled disk usage (TBs of image copies) and doubled
      IO (s1 writes images, s4 reads them again and writes them into tars).
    * Downstream stages (s2/s3/s4) only ever need three pieces of information
      per sample:
          1. A globally unique, deterministic sample ``name``;
          2. The raw sample payload (messages / conversations / captions etc.);
          3. Absolute paths to the image and ``.npy`` (patch_positions) files.
      All of these can be packed into a single JSONL file ~MB in size.

Manifest line schema (one JSON object per line)::

    {
        "name":          "IMG_1234__a1b2c3d4",         # globally unique id
        "raw":           { ... original sample json ... },
        "images_abs":    ["/abs/path/a.jpg", ...],     # may be empty
        "patch_pos_abs": ["/abs/path/a.npy", null],    # parallel to images_abs
        "orig_index":    1234,                          # row idx in source jsonl
        "source_jsonl":  "/abs/path/input.jsonl"        # for traceability
    }

Name strategy (Q1=B + uuid-style hash for collision safety + traceability)
    * with images:    ``<sanitized_basename_no_ext>__<8 hex>``
    * without images: ``noimg_<orig_index:08d>__<8 hex>``
    * The 8-hex tag is ``sha1(first_image_abs | orig_index | source_basename)[:8]``,
      so it is deterministic. If two records still collide (extremely rare or
      identical inputs), we append ``_1``, ``_2`` ... suffixes.
    * Bin-packing's numpy dtype caps name length to 200 chars, so the basename
      portion is truncated to keep the total within the limit.

Patch-positions fallback (Q4=A)
    * If the raw record provides a ``patch_positions`` field, use it as-is.
    * Otherwise, look for a co-located ``.npy`` next to each image (same dir,
      same stem). Missing ``.npy`` becomes ``null`` in the manifest.

Usage
    python s1_build_manifest.py \\
        -i /path/to/input.jsonl \\
        --image-root /path/to/images \\
        -o /path/to/output_base/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# s3 packs sample names into a numpy ``<U200`` field. Keep a safety margin.
MAX_NAME_LEN = 200
HEX_TAG_LEN = 8                       # length of the deterministic collision tag
SUFFIX_RESERVED = 6                   # reserve room for "__" + tag + optional "_99"
BASENAME_MAX_LEN = MAX_NAME_LEN - HEX_TAG_LEN - SUFFIX_RESERVED

# Characters that we replace in basenames to make them safe for tar/webdataset.
_BAD_CHARS_RE = re.compile(r'[^A-Za-z0-9._-]')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_basename(name: str) -> str:
    """Strip extension, replace unsafe chars, and truncate to a safe length."""
    stem = os.path.splitext(os.path.basename(name))[0]
    clean = _BAD_CHARS_RE.sub("_", stem).strip("._-")
    if not clean:
        clean = "sample"
    if len(clean) > BASENAME_MAX_LEN:
        clean = clean[:BASENAME_MAX_LEN]
    return clean


def _hex_tag(parts: Iterable[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\x1f")  # ASCII unit separator
    return h.hexdigest()[:HEX_TAG_LEN]


def _resolve_image_path(
    path: str,
    base_dir: str,
    image_root: Optional[str],
    rel_img: Optional[str],
) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    if rel_img:
        return os.path.normpath(os.path.join(base_dir, rel_img, path))
    if image_root:
        return os.path.normpath(os.path.join(image_root, path))
    return os.path.normpath(os.path.join(base_dir, path))


def _extract_image_list(item: dict[str, Any]) -> list[str]:
    """Pull raw image path list from an item; tolerates both ``image`` and ``images``."""
    raw = item.get("images") or item.get("image")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(p) for p in raw]
    return []


def _extract_patch_position_list(item: dict[str, Any]) -> Optional[list[Optional[str]]]:
    """Return raw patch_positions list from the item, or None if field absent."""
    raw = item.get("patch_positions")
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [None if (p is None or p == "") else str(p) for p in raw]
    return None


# ---------------------------------------------------------------------------
# Worker: turn one raw item into a manifest entry (or None if skipped)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolveContext:
    base_dir: str
    image_root: Optional[str]
    rel_img: Optional[str]
    source_jsonl: str
    source_basename: str  # used as a salt for the hex tag


# ---------------------------------------------------------------------------
# Conversation schema normalization (LLaVA/ShareGPT {from,value} -> {role,content})
# ---------------------------------------------------------------------------
# The packing pipeline (s2 jinja template, s4 webdataset writer) requires each
# conversation turn to be a dict with ``role``/``content`` keys. Many public SFT
# datasets ship LLaVA/ShareGPT turns with ``from``/``value`` keys instead;
# rendered through the s2 template those produce EMPTY role/content (silent data
# loss). Normalizing at manifest-build time gives every downstream stage a
# uniform schema. Safe + idempotent: turns already carrying ``role``/``content``
# are left untouched.
_FROMVALUE_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def _normalize_conversations(item: dict[str, Any]) -> None:
    """In-place convert ``{from,value}`` conversation turns to ``{role,content}``."""
    for key in ("conversations", "messages"):
        conv = item.get(key)
        if not isinstance(conv, list):
            continue
        changed = False
        new_conv = []
        for turn in conv:
            if (
                isinstance(turn, dict)
                and "role" not in turn
                and "content" not in turn
                and ("from" in turn or "value" in turn)
            ):
                role_raw = str(turn.get("from", "")).lower()
                role = _FROMVALUE_ROLE_MAP.get(role_raw, role_raw or "user")
                new_turn = {"role": role, "content": turn.get("value", "")}
                for k, v in turn.items():
                    if k not in ("from", "value"):
                        new_turn[k] = v
                new_conv.append(new_turn)
                changed = True
            else:
                new_conv.append(turn)
        if changed:
            item[key] = new_conv


def _process_one(item: dict[str, Any], orig_index: int, ctx: ResolveContext) -> Optional[dict[str, Any]]:
    """Resolve image/npy paths and build one manifest entry. Returns None to drop the sample."""
    _normalize_conversations(item)
    images_raw = _extract_image_list(item)
    images_abs = [_resolve_image_path(p, ctx.base_dir, ctx.image_root, ctx.rel_img) for p in images_raw]

    # Pre-check: missing image -> drop whole sample (keep parity with old s1)
    missing_imgs = [p for p in images_abs if not os.path.exists(p)]
    if missing_imgs:
        item_id = item.get("id", orig_index)
        logger.warning(
            "Skipping sample %s: %d/%d image(s) missing (first missing: %s)",
            item_id, len(missing_imgs), len(images_abs), missing_imgs[0],
        )
        return None

    # patch_positions resolution (Q4=A: support both explicit field and co-located npy fallback)
    explicit_pp = _extract_patch_position_list(item)
    patch_pos_abs: list[Optional[str]] = []
    if explicit_pp is not None:
        for idx in range(len(images_abs)):
            if idx < len(explicit_pp) and explicit_pp[idx]:
                p = _resolve_image_path(explicit_pp[idx], ctx.base_dir, ctx.image_root, ctx.rel_img)
                if not os.path.exists(p):
                    item_id = item.get("id", orig_index)
                    logger.warning(
                        "Skipping sample %s: declared patch_positions missing (%s)", item_id, p,
                    )
                    return None
                patch_pos_abs.append(p)
            else:
                patch_pos_abs.append(None)
    else:
        # Fallback: look for sibling <img_stem>.npy
        for img_abs in images_abs:
            candidate = os.path.splitext(img_abs)[0] + ".npy"
            patch_pos_abs.append(candidate if os.path.exists(candidate) else None)

    # Build deterministic name
    if images_abs:
        basename_src = images_abs[0]
        basename = _sanitize_basename(basename_src)
        tag = _hex_tag([basename_src, orig_index, ctx.source_basename])
        name = f"{basename}__{tag}"
    else:
        tag = _hex_tag(["", orig_index, ctx.source_basename])
        name = f"noimg_{orig_index:08d}__{tag}"

    # Strip ``_orig_index`` if present (some pipelines inject it); we keep our own field.
    item.pop("_orig_index", None)

    return {
        "name": name,
        "raw": item,
        "images_abs": images_abs,
        "patch_pos_abs": patch_pos_abs,
        "orig_index": orig_index,
        "source_jsonl": ctx.source_jsonl,
    }


# ---------------------------------------------------------------------------
# Multiprocess driver (Q5=B: keep multi-process to absorb slow NFS stats)
# ---------------------------------------------------------------------------

# Module-level globals set by the pool initializer (cheaper than pickling
# the context for every task).
_WORKER_CTX: Optional[ResolveContext] = None


def _worker_init(ctx_dict: dict[str, Any]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ResolveContext(**ctx_dict)


def _worker_map(args: tuple[int, dict[str, Any]]) -> Optional[dict[str, Any]]:
    orig_index, item = args
    assert _WORKER_CTX is not None
    try:
        return _process_one(item, orig_index, _WORKER_CTX)
    except Exception as exc:  # noqa: BLE001
        logger.error("Worker failed on orig_index=%d: %s", orig_index, exc)
        return None


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

def _iter_input_records(input_path: str) -> Iterable[dict[str, Any]]:
    """Stream records from a JSON-array or JSONL file."""
    if input_path.lower().endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be an array")
        yield from data


def _load_all_records(input_path: str, shuffle: bool, seed: int) -> list[dict[str, Any]]:
    data = list(_iter_input_records(input_path))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def build_manifest(
    input_path: str,
    output_path: str,
    *,
    image_root: Optional[str] = None,
    rel_img: Optional[str] = None,
    num_processes: Optional[int] = None,
    chunk_size: int = 256,
    shuffle: bool = True,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build the manifest. Returns a stats dict."""
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite)")

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    base_dir = os.path.dirname(input_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logger.info("Loading input file: %s", input_path)
    t0 = time.time()
    records = _load_all_records(input_path, shuffle=shuffle, seed=seed)
    total = len(records)
    logger.info("Loaded %d records in %.1fs", total, time.time() - t0)

    ctx_dict = {
        "base_dir": base_dir,
        "image_root": image_root,
        "rel_img": rel_img,
        "source_jsonl": input_path,
        "source_basename": os.path.basename(input_path),
    }

    n_proc = num_processes or max(1, min(mp.cpu_count(), 16))
    logger.info("Resolving paths with %d processes (chunk_size=%d)...", n_proc, chunk_size)

    # Stream records into the pool; collect in arrival order for memory friendliness.
    written = 0
    skipped = 0

    # Collision tracking across the whole run.
    used_names: dict[str, int] = {}

    t_resolve = time.time()
    with open(output_path, "w", encoding="utf-8") as out_f, \
         mp.Pool(processes=n_proc, initializer=_worker_init, initargs=(ctx_dict,)) as pool:

        tasks = ((i, rec) for i, rec in enumerate(records))
        for entry in pool.imap_unordered(_worker_map, tasks, chunksize=chunk_size):
            if entry is None:
                skipped += 1
                continue

            # De-collide globally (extremely rare with the 8-hex tag, but be safe).
            base_name = entry["name"]
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            if count > 0:
                # ensure final length still fits MAX_NAME_LEN
                suffix = f"_{count}"
                trimmed = base_name[: MAX_NAME_LEN - len(suffix)]
                entry["name"] = f"{trimmed}{suffix}"
                logger.debug("Collision resolved: %s -> %s", base_name, entry["name"])

            out_f.write(json.dumps(entry, ensure_ascii=False))
            out_f.write("\n")
            written += 1

            if (written + skipped) % 5000 == 0:
                pct = (written + skipped) * 100.0 / max(1, total)
                logger.info(
                    "Progress: %d/%d (%.1f%%) | written=%d skipped=%d",
                    written + skipped, total, pct, written, skipped,
                )

    elapsed = time.time() - t_resolve
    logger.info(
        "Manifest written: %s | written=%d skipped=%d in %.1fs (%.0f rec/s)",
        output_path, written, skipped, elapsed, (written + skipped) / max(1e-6, elapsed),
    )

    return {"total": total, "written": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight manifest.jsonl for the offline packing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSON or JSONL file path")
    parser.add_argument("-o", "--output", required=True, help="Output manifest.jsonl path")
    parser.add_argument("--image-root", default=None, help="Absolute image root directory")
    parser.add_argument("--rel-img", default=None, help="Image dir relative to input file")
    parser.add_argument("--processes", type=int, default=None, help="Worker processes (default: min(cpu, 16))")
    parser.add_argument("--chunk-size", type=int, default=256, help="imap chunksize")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle (default: shuffle with seed)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for reproducibility (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output manifest")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    stats = build_manifest(
        input_path=args.input,
        output_path=args.output,
        image_root=args.image_root,
        rel_img=args.rel_img,
        num_processes=args.processes,
        chunk_size=args.chunk_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print(
        f"\nManifest built: {args.output}\n"
        f"  total   = {stats['total']}\n"
        f"  written = {stats['written']}\n"
        f"  skipped = {stats['skipped']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
