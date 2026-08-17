#!/usr/bin/env python3
"""Analyze a bins_*.pkl produced by s3_bin_packing.py.

Prints token / patch utilization stats. Patch counts are RAW ViT patches
(before spatial merge). LLM vision tokens = raw / merge_size**2 (9 for merge=3).

Usage:
    python analyze_bins.py /path/bins_32klen.pkl
"""
import sys, pickle, numpy as np

def main(pkl):
    bins = pickle.load(open(pkl, "rb"))
    toks = np.array([sum(int(it[1]) for it in b) for b in bins])
    # bin item layout (s3 BIN_DTYPE): (orig_idx, token_len, name, patch_count)
    pats = np.array([sum(int(it[3]) if len(it) > 3 else 0 for it in b) for b in bins])
    print(f"Bins: {len(bins)}")
    print(f"Token  mean={toks.mean():.0f}  std={toks.std():.0f}  "
          f"p10={np.percentile(toks,10):.0f}  max={toks.max()}")
    if pats.sum() > 0:
        cv = pats.std() / max(pats.mean(), 1)
        print(f"Patch  mean={pats.mean():.0f}  std={pats.std():.0f}  CV={cv:.3f}")
        for p in (10, 50, 90, 99):
            print(f"   p{p:02d}={np.percentile(pats, p):.0f}")
        print(f"   min={pats.min()}  max={pats.max()}  "
              f"p99/p50={np.percentile(pats,99)/max(np.percentile(pats,50),1):.2f}x")
        # Vision tokens (merge=3 default; pass --merge to override)
        merge = 3
        if "--merge" in sys.argv:
            merge = int(sys.argv[sys.argv.index("--merge")+1])
        vtok = pats / (merge ** 2)
        print(f"\nVision tokens (raw / {merge}**2):")
        print(f"   mean={vtok.mean():.0f}  max={vtok.max():.0f}  "
              f"p99={np.percentile(vtok,99):.0f}")

if __name__ == "__main__":
    main(sys.argv[1])
