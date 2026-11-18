#!/usr/bin/env python3
"""
build_combined_manifest.py — concat fix part manifest + selected 47m manifest entries.

Inputs:
  --fix-manifest         existing pandas70m_part${i} manifest.jsonl   (kept verbatim)
  --selected             list of 47m sample names selected by topup
  --pool-manifest-glob   glob to gather 47m manifests, e.g. /data/sft_packed/47m_part*/manifest.jsonl

Output:
  --output      manifest_combined.jsonl  (fix entries first, then selected 47m entries)

Implementation note: 47m manifests are ~28 GB total / 14M lines. Doing json.loads
on every line is slow (~10+ minutes). Instead we leverage that every record
starts with `{"name": "..."}` so we can substring-match each line against the
selected name set after extracting just the first quoted token.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time

# Each manifest line starts with: {"name": "<NAME>", ...
_NAME_RE = re.compile(rb'\{\"name\"\s*:\s*\"([^\"]+)\"')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-manifest", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--pool-manifest-glob", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # 1) read selected names → set of bytes for fast comparison
    sel_b: set[bytes] = set()
    with open(args.selected, "rb") as f:
        for line in f:
            n = line.strip()
            if n: sel_b.add(n)
    print(f"[combine] selected 47m names: {len(sel_b):,}")

    # 2) scan pool manifests with bytes regex
    paths = sorted(glob.glob(args.pool_manifest_glob))
    print(f"[combine] scanning {len(paths)} pool manifests...")
    found: dict[bytes, bytes] = {}   # name_bytes -> raw line bytes
    t0 = time.time()
    for pp in paths:
        c = 0
        line_n = 0
        with open(pp, "rb") as f:
            for line in f:
                line_n += 1
                m = _NAME_RE.match(line)
                if not m: continue
                nm = m.group(1)
                if nm in sel_b and nm not in found:
                    found[nm] = line if line.endswith(b"\n") else (line + b"\n")
                    c += 1
                    if len(found) == len(sel_b):
                        break
        print(f"[combine]   {os.path.basename(pp)}: lines={line_n:,} matched={c:,} (cum={len(found):,})  +{time.time()-t0:.1f}s")
        if len(found) == len(sel_b):
            break

    missing = len(sel_b) - len(found)
    if missing:
        print(f"[combine] WARNING: {missing} selected names not found in pool manifests", file=sys.stderr)

    # 3) write combined manifest
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_fix = 0
    with open(args.output, "wb") as out, open(args.fix_manifest, "rb") as fix:
        for line in fix:
            out.write(line)
            n_fix += 1
        for nm in sel_b:
            ln = found.get(nm)
            if ln is not None:
                out.write(ln)

    print(f"[combine] wrote {args.output}: fix={n_fix:,} + 47m={len(found):,} = {n_fix + len(found):,} total")

if __name__ == "__main__":
    main()
