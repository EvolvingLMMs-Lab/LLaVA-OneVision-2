#!/usr/bin/env python3
"""
update_dataset_yaml_fix_with47m.py

Rewrite Family 2 of /ov21/mixed_6sets_packed_64k/dataset_fix_with47m_20260616.yaml:
  OLD path: /ov21/pandas70m_gt10min_packed_64k/pandas70m_part{0..19}/webdataset
  NEW path: /ov21/pandas70m_gt10min_packed_64k_fix_with47m/pandas70m_part{0..19}/webdataset

Bin count per part = #bins in the fix bins_64klen.pkl (top-up doesn't change bin count).
For parts whose fix bins_64klen.pkl is not yet ready, use a placeholder = mean of available parts.

Usage:
    python update_dataset_yaml_fix_with47m.py \
        [--yaml /ov21/mixed_6sets_packed_64k/dataset_fix_with47m_20260616.yaml]
        [--in-place]
"""
from __future__ import annotations
import argparse, os, pickle, sys
import re

YAML_PATH = "/ov21/mixed_6sets_packed_64k/dataset_fix_with47m_20260616.yaml"
FIX_ROOT  = "/ov21/pandas70m_gt10min_packed_64k_fix"
NEW_ROOT  = "/ov21/pandas70m_gt10min_packed_64k_fix_with47m"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=YAML_PATH)
    ap.add_argument("--out", default="")
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    # 1) gather current bin counts from fix part bins_64klen.pkl
    bins_count: dict[int, int] = {}
    for i in range(20):
        pkl = f"{FIX_ROOT}/pandas70m_part{i}/bins_64klen.pkl"
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f:
                    bins_count[i] = len(pickle.load(f))
            except Exception as e:
                print(f"[warn] failed to read {pkl}: {e}", file=sys.stderr)
    avail = sorted(bins_count.keys())
    if not avail:
        print("[fatal] no fix bins available; aborting.", file=sys.stderr)
        sys.exit(2)
    placeholder = sum(bins_count.values()) // len(bins_count)
    print(f"[update] available parts: {avail} (count={len(avail)})")
    print(f"[update] placeholder bins for missing parts: {placeholder}")

    # 2) build new bin_count map (use placeholder for missing parts)
    new_bins: dict[int, int] = {}
    is_estimated: dict[int, bool] = {}
    for i in range(20):
        if i in bins_count:
            new_bins[i] = bins_count[i]
            is_estimated[i] = False
        else:
            new_bins[i] = placeholder
            is_estimated[i] = True
    total_new = sum(new_bins.values())
    print(f"[update] total new pandas70m fix_with47m bins: {total_new:,}")

    # 3) rewrite Family 2 region
    with open(args.yaml, "r", encoding="utf-8") as f:
        text = f.read()

    # Find Family 2 region: from "# ===== Family 2:" line until the next "# ===== Family"
    family2_re = re.compile(
        r"(      # ===== Family 2:[^\n]*\n)"
        r"(?:.*?)"
        r"(?=      # ===== Family \d+:|# ===== Totals)",
        re.DOTALL
    )
    m = family2_re.search(text)
    if m is None:
        print("[fatal] could not locate Family 2 region in yaml", file=sys.stderr)
        sys.exit(3)

    # Build new Family 2 block
    family_header = (
        "      # ===== Family 2: pandas70m_gt10min_FIX_WITH47M (long video + 47m top-up,\n"
        "      #                  20 parts, ~%d bins, top-up to 100%% bin util) =====\n"
        "      # paths: /ov21/pandas70m_gt10min_packed_64k_fix_with47m/\n"
        "      # NOTE: replaces previous polluted /ov21/pandas70m_gt10min_packed_64k/\n"
    ) % total_new
    parts_lines = []
    for i in range(20):
        nb = new_bins[i]
        tag = "  # ESTIMATED" if is_estimated[i] else ""
        parts_lines.append(
            f"      - weight: {nb}{tag}\n"
            f"        path: {NEW_ROOT}/pandas70m_part{i}/webdataset\n"
            f"        subflavors:\n"
            f"          augmentation: false\n"
        )
    new_family2 = family_header + "".join(parts_lines)

    new_text = text[:m.start()] + new_family2 + text[m.end():]

    # 4) regenerate Totals footer with the new Family 2 number
    # Compute totals from existing weights
    weights_re = re.compile(r"^      - weight:\s*(\d+)", re.M)
    sums = {}
    # We need per-family totals. Easiest: keep the existing Family-specific totals
    # block but rewrite the Family 2 line and grand total.
    # Old totals (from header):
    fam_re = re.compile(r"# Family \d+:[^\n]+\n", re.M)
    # Replace Family 2 line in totals comment
    new_text = re.sub(
        r"# Family 2:[^\n]+\n",
        f"# Family 2: pandas70m_gt10min_FIX_WITH47M  = {total_new:6d} bins (recomputed)\n",
        new_text
    )

    # grand total: parse all weight lines and sum them
    weights = [int(x) for x in weights_re.findall(new_text)]
    grand = sum(weights)
    new_text = re.sub(
        r"# GRAND TOTAL\s*=\s*\d+\s*bins\n",
        f"# GRAND TOTAL                          ={grand:7d} bins\n",
        new_text
    )
    new_text = re.sub(
        r"# At GBS=16, 1 epoch = ceil\([\d/]+\) = \d+ iter",
        f"# At GBS=16, 1 epoch = ceil({grand}/16) = {(grand + 15) // 16} iter",
        new_text
    )

    # 5) write
    if args.in_place:
        out = args.yaml
    else:
        out = args.out or (args.yaml + ".new")
    with open(out, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[update] wrote {out}")
    print(f"[update] grand total bins now: {grand:,}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
