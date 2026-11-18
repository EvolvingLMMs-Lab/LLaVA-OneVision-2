#!/usr/bin/env python3
"""Stamp a sample-level ``data_type`` onto an existing webdataset, in place of repacking.

Rewrites the per-sample ``json`` member of every shard, rebuilds each ``.tar.idx``
(the stamp changes member sizes, so the old byte offsets are wrong), and
regenerates ``.nv-meta/sample_loader.py`` so the loader passes the field through.

Two layouts are handled, detected from ``.nv-meta/dataset.yaml``:

  MultiMixQASample    one sample per json -> ``"data_type": "streaming"``
  PackedCaptioningSample  N sub-samples per bin -> ``"data_type": [...]`` with one
                      entry per sub-sample, matching how ``fps`` and
                      ``timestamp_decimal`` are already stored

The source is never modified: output goes to a new directory.

Usage:
  python stamp_data_type.py --src <wds dir> --dst <new wds dir> [--data-type streaming]
"""

import argparse
import io
import json
import os
import shutil
import sys
import tarfile

UNPACKED = "MultiMixQASample"
PACKED = "PackedCaptioningSample"


def detect_layout(meta_dir):
    """Read the sample class out of dataset.yaml without pulling in a yaml dep."""
    path = os.path.join(meta_dir, "dataset.yaml")
    if not os.path.isfile(path):
        raise SystemExit(f"no dataset.yaml in {meta_dir}")
    text = open(path, encoding="utf-8").read()
    for name in (PACKED, UNPACKED):
        if name in text:
            return name
    raise SystemExit(f"unrecognised sample_type in {path}:\n{text}")


def stamp_payload(payload, layout, value):
    """Add the marker, shaped the way this layout stores per-sample metadata."""
    if layout == UNPACKED:
        payload["data_type"] = value
        return 1
    count = payload.get("sample_count")
    if count is None:
        count = len(payload.get("images") or [])
    if not count:
        raise ValueError("packed bin has no sub-samples to mark")
    payload["data_type"] = [value] * count
    return count


def rewrite_shard(src_tar, dst_tar, layout, value):
    """Copy a shard member by member, replacing each json with its stamped form."""
    marked = subs = 0
    with tarfile.open(src_tar) as src, tarfile.open(dst_tar, "w") as dst:
        for member in src:
            if not member.isfile():
                continue
            blob = src.extractfile(member).read()
            if member.name.endswith(".json"):
                payload = json.loads(blob)
                subs += stamp_payload(payload, layout, value)
                blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                member.size = len(blob)
                marked += 1
            dst.addfile(member, io.BytesIO(blob))
    return marked, subs


def write_index(tar_path):
    """Rebuild the energon tar index: one byte offset per sample boundary."""
    from megatron.energon.epathlib import EPath
    from megatron.energon.flavors.webdataset.itar import TarIndexWriter

    offsets, key = [], None
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name.partition(".")[0]
            if name != key:
                offsets.append(member.offset)
                key = name
    offsets.append(os.path.getsize(tar_path))
    with TarIndexWriter(EPath(tar_path)) as writer:
        for offset in offsets:
            writer.append(offset)
    return len(offsets) - 1


def regenerate_loader(meta_dir, layout, repo_root):
    """Emit the sample_loader that carries data_type through, from the live converters."""
    if layout == UNPACKED:
        sys.path.insert(0, os.path.join(repo_root, "tools", "data_preprocess"))
        from convert_jsonl_to_webdataset_simple import sample_loader_template as gen
    else:
        sys.path.insert(0, os.path.join(repo_root, "offline_packing"))
        from s5_convert_to_webdataset import generate_sample_loader_code as gen
    code = gen()
    if "data_type" not in code:
        raise SystemExit("the generated sample_loader has no data_type passthrough; "
                         "is this repo missing the data_type changes?")
    with open(os.path.join(meta_dir, "sample_loader.py"), "w", encoding="utf-8") as f:
        f.write(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Existing webdataset directory")
    parser.add_argument("--dst", required=True, help="Output directory (must not exist)")
    parser.add_argument("--data-type", default="streaming", help='Marker value (default: "streaming")')
    parser.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    args = parser.parse_args()

    src_meta = os.path.join(args.src, ".nv-meta")
    if not os.path.isdir(src_meta):
        raise SystemExit(f"{args.src} has no .nv-meta; not an energon webdataset")
    if os.path.exists(args.dst):
        raise SystemExit(f"--dst already exists: {args.dst}")

    layout = detect_layout(src_meta)
    shards = sorted(f for f in os.listdir(args.src) if f.endswith(".tar"))
    if not shards:
        raise SystemExit(f"no .tar shards in {args.src}")
    print(f"layout     {layout}")
    print(f"shards     {len(shards)}")
    print(f"data_type  {args.data_type!r}\n")

    dst_meta = os.path.join(args.dst, ".nv-meta")
    os.makedirs(dst_meta)
    for name in os.listdir(src_meta):
        path = os.path.join(src_meta, name)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(dst_meta, name))

    total_bins = total_subs = 0
    for name in shards:
        dst_tar = os.path.join(args.dst, name)
        bins, subs = rewrite_shard(os.path.join(args.src, name), dst_tar, layout, args.data_type)
        indexed = write_index(dst_tar)
        if indexed != bins:
            raise SystemExit(f"{name}: indexed {indexed} samples but stamped {bins} json members")
        total_bins += bins
        total_subs += subs
        print(f"  {name}: {bins} bin(s) stamped, {subs} sub-sample(s), index rebuilt")

    regenerate_loader(dst_meta, layout, args.repo_root)
    print(f"\nstamped {total_bins} bin(s) / {total_subs} sub-sample(s)")
    print(f"sample_loader.py regenerated with data_type passthrough")
    print(f"-> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
