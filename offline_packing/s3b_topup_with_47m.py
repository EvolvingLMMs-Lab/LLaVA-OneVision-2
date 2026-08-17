#!/usr/bin/env python3
"""
s3b_topup_with_47m.py — fill the slack space of a pandas70m fix bin pickle
with 47m image samples (sft cleaned 64k pool).

Usage:
    python s3b_topup_with_47m.py \
        --bins-in /ov21/pandas70m_gt10min_packed_64k_fix/pandas70m_part0/bins_64klen.pkl \
        --bins-out /ov21/pandas70m_gt10min_packed_64k_fix_with47m/pandas70m_part0/bins_topup.pkl \
        --pool-tokeninfo-glob '/ov21/47m_sft_packed_64k/47m_part*/token_info.txt' \
        --pool-blocklist /tmp/47m_used_global.txt \
        --image-only \
        --capacity 64000 \
        --max-samples 200 \
        --max-patches 608822 \
        --topup-budget-samples 320000 \
        --seed 42

Outputs:
    bins_topup.pkl    — list[np.ndarray] same dtype as s3, with new 47m entries appended into existing bins
    selected.txt      — names of 47m samples consumed (for dedup across parts)

Rules:
    * Each existing bin gets new 47m image samples until any of:
        - tokens > capacity
        - patches > max_patches
        - sample count > max_samples
    * 47m pool is shuffled deterministically (seed) so cross-part splits are stable.
    * --pool-blocklist (optional input) lets you exclude names already used by earlier parts.
    * The script ONLY mutates the bin contents; it never moves pandas70m samples.
"""
from __future__ import annotations
import argparse, glob, os, pickle, random, re, sys, time
import numpy as np
from collections import Counter

DTYPE = np.dtype([("w","<u2"),("l","<u4"),("name","<U200"),("p","<u4")])

PATTERN = re.compile(r"^(.+?):(\d+)(?::(\d+))?$")

def load_token_info(path: str, image_only: bool, blocked: set[str]) -> list[tuple[str,int,int]]:
    """Read token_info.txt; return list of (name, token, patch). Filter image_only -> patch>0."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = PATTERN.match(line)
            if not m: continue
            name = m.group(1)
            tok = int(m.group(2))
            patch = int(m.group(3)) if m.group(3) is not None else 0
            if image_only and patch == 0:
                continue
            if name in blocked:
                continue
            out.append((name, tok, patch))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins-in", required=True)
    ap.add_argument("--bins-out", required=True)
    ap.add_argument("--pool-tokeninfo-glob", required=True,
        help="glob to gather pool token_info files, e.g. /ov21/47m_sft_packed_64k/47m_part*/token_info.txt")
    ap.add_argument("--pool-blocklist", default="",
        help="optional file listing already-consumed names (one per line)")
    ap.add_argument("--selected-out", default="",
        help="path to write names of newly-selected 47m samples (one per line)")
    ap.add_argument("--image-only", action="store_true", default=True)
    ap.add_argument("--capacity", type=int, default=64000)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-patches", type=int, default=608822)
    ap.add_argument("--topup-budget-samples", type=int, default=350000,
        help="upper bound on number of 47m samples to load (random subset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) load existing bins
    print(f"[topup] loading bins: {args.bins_in}")
    with open(args.bins_in, "rb") as f:
        bins = pickle.load(f)
    n_bins = len(bins)
    cur_tok  = np.array([int(b["l"].sum()) for b in bins], dtype=np.int64)
    cur_pat  = np.array([int(b["p"].sum()) for b in bins], dtype=np.int64)
    cur_cnt  = np.array([len(b) for b in bins], dtype=np.int32)
    rem_tok  = args.capacity - cur_tok
    rem_pat  = args.max_patches - cur_pat
    rem_cnt  = args.max_samples - cur_cnt
    print(f"[topup] bins={n_bins} sum_token_slack={int(rem_tok.sum()):,} sum_patch_slack={int(rem_pat.sum()):,}")

    # 2) load blocklist (already used names from previous parts)
    blocked: set[str] = set()
    if args.pool_blocklist and os.path.exists(args.pool_blocklist):
        with open(args.pool_blocklist, "r", encoding="utf-8") as f:
            for line in f:
                n = line.strip()
                if n: blocked.add(n)
        print(f"[topup] blocklist size: {len(blocked):,}")

    # 3) load 47m pool
    paths = sorted(glob.glob(args.pool_tokeninfo_glob))
    print(f"[topup] pool token_info files: {len(paths)}")
    pool: list[tuple[str,int,int]] = []
    t0 = time.time()
    for pp in paths:
        chunk = load_token_info(pp, args.image_only, blocked)
        pool.extend(chunk)
        print(f"[topup]   loaded {len(chunk):,} from {os.path.basename(pp)} (cum={len(pool):,})")
    print(f"[topup] pool ready: {len(pool):,} samples in {time.time()-t0:.1f}s")

    # 4) random shuffle + clip to budget
    rng.shuffle(pool)
    if len(pool) > args.topup_budget_samples:
        pool = pool[: args.topup_budget_samples]
        print(f"[topup] clipped pool to budget={args.topup_budget_samples:,}")

    # 5) sort pool by token DESC for first-fit-decreasing top-up
    pool.sort(key=lambda x: x[1], reverse=True)

    # 6) greedy top-up: for each pool sample, find the bin with smallest remaining
    #    token slack that can still fit (BFD-like). To keep it O(n log m), use
    #    sortedcontainers if available; else linear scan with cap-based filter.
    try:
        from sortedcontainers import SortedList
        # Index bins by remaining_token, value = bin_idx; keep a parallel array of (rem_tok, bin_idx)
        # We need to pop on update — easier: keep simple lists and do partial scan
        use_sorted = False
    except ImportError:
        use_sorted = False

    # Track per-bin appended items
    appended: list[list[tuple[int,int,str,int]]] = [[] for _ in range(n_bins)]
    selected_names: list[str] = []

    placed = 0
    skipped = 0
    t1 = time.time()
    for name, tok, patch in pool:
        if tok <= 0:
            skipped += 1; continue
        # Need: rem_tok >= tok, rem_pat >= patch, rem_cnt >= 1
        # BFD: pick the bin whose rem_tok is smallest but >= tok
        # Use vectorized boolean mask
        mask = (rem_tok >= tok) & (rem_pat >= patch) & (rem_cnt >= 1)
        if not mask.any():
            skipped += 1; continue
        # Among candidate bins, pick the one with smallest rem_tok
        cand_rem = np.where(mask, rem_tok, np.iinfo(np.int64).max)
        idx = int(np.argmin(cand_rem))
        # place
        appended[idx].append((0, tok, name, patch))
        rem_tok[idx] -= tok
        rem_pat[idx] -= patch
        rem_cnt[idx] -= 1
        selected_names.append(name)
        placed += 1
        if placed % 50000 == 0:
            print(f"[topup]   placed={placed:,} skipped={skipped:,} ({(time.time()-t1):.0f}s) "
                  f"rem_token_slack={int(rem_tok.sum()):,}")

    elapsed = time.time() - t1
    print(f"[topup] DONE placed={placed:,} skipped={skipped:,} in {elapsed:.0f}s")

    # 7) build new bins by concatenating appended items into existing arrays
    new_bins: list[np.ndarray] = []
    n_with_topup = 0
    for i, b in enumerate(bins):
        if not appended[i]:
            new_bins.append(b)
            continue
        n_with_topup += 1
        extra = np.array(appended[i], dtype=DTYPE)
        new_bins.append(np.concatenate([b, extra]))

    # 8) print final stats
    final_tok = np.array([int(b["l"].sum()) for b in new_bins], dtype=np.int64)
    final_pat = np.array([int(b["p"].sum()) for b in new_bins], dtype=np.int64)
    final_cnt = np.array([len(b) for b in new_bins], dtype=np.int32)
    util = float(final_tok.mean()) / args.capacity * 100
    print()
    print(f"[topup] === Final stats ===")
    print(f"        bins:               {len(new_bins):,}")
    print(f"        bins receiving 47m: {n_with_topup:,} ({n_with_topup/len(new_bins)*100:.1f}%)")
    print(f"        avg samples/bin:    {final_cnt.mean():.1f}  max={int(final_cnt.max())}")
    print(f"        avg tokens/bin:     {final_tok.mean():.0f}  ({util:.1f}% util)")
    print(f"        avg patches/bin:    {final_pat.mean():.0f}")
    print(f"        47m samples added:  {placed:,}")

    # 9) write outputs
    os.makedirs(os.path.dirname(args.bins_out), exist_ok=True)
    with open(args.bins_out, "wb") as f:
        pickle.dump(new_bins, f, protocol=4)
    print(f"[topup] wrote {args.bins_out}")

    if args.selected_out:
        with open(args.selected_out, "w", encoding="utf-8") as f:
            for n in selected_names:
                f.write(n + "\n")
        print(f"[topup] wrote {args.selected_out}  ({len(selected_names):,} names)")


if __name__ == "__main__":
    main()
