#!/usr/bin/env python3
"""Split the giant llava_video_full_1fps384.jsonl into 20 evenly-sized shards."""
import os, sys, math
SRC = "/data/data_jsonl/llava_video_full/llava_video_full_1fps384.jsonl"
OUT_DIR = "/data/data_jsonl/llava_video_full/jsonl_done_split20"
N_PARTS = 20

os.makedirs(OUT_DIR, exist_ok=True)

print(f"counting lines in {SRC} ...", flush=True)
with open(SRC, "rb") as f:
    total = sum(1 for _ in f)
print(f"  total = {total:,}")

per = math.ceil(total / N_PARTS)
print(f"  per-part = {per:,}  (last part may be smaller)")

handles = []
for i in range(N_PARTS):
    p = f"{OUT_DIR}/llava_video_full_part{i:02d}.jsonl"
    handles.append(open(p, "wb"))
counts = [0] * N_PARTS

with open(SRC, "rb") as f:
    for lineno, line in enumerate(f):
        idx = min(lineno // per, N_PARTS - 1)
        handles[idx].write(line)
        counts[idx] += 1
        if (lineno+1) % 200000 == 0:
            print(f"  wrote {lineno+1:,} ...", flush=True)
for h in handles: h.close()

print("\n=== per-part line counts ===")
for i, n in enumerate(counts):
    p = f"{OUT_DIR}/llava_video_full_part{i:02d}.jsonl"
    sz = os.path.getsize(p)
    print(f"  part{i:02d}: {n:>8,d} records  size={sz/(1<<30):.2f} GB")
print(f"\nTOTAL: {sum(counts):,}")
