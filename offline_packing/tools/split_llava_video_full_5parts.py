#!/usr/bin/env python3
"""
Split llava_video_full_1fps384.jsonl into 5 parts that are balanced by
total frame count (not just line count), since per-record frame count
varies wildly (1..181, mean 59.6, median 36).

Strategy:
  - Pass 1: scan once, record (offset, frame_count) per line.
  - Greedy LPT: sort lines DESC by frames, assign each to the lightest part
    (heap of (frames, idx)). This typically balances within <0.05%.
  - Pass 2: write each part's lines in original order (to preserve any
    weak locality / no shuffling surprises).
"""
import os, sys, json, heapq, time
SRC = "/data/data_jsonl/llava_video_full/llava_video_full_1fps384.jsonl"
OUT_DIR = "/data/data_jsonl/llava_video_full/jsonl_done_split5"
N_PARTS = 5

os.makedirs(OUT_DIR, exist_ok=True)

# --- Pass 1: scan for frame counts and offsets ---
print(f"Pass 1: scanning {SRC} ...", flush=True)
t0 = time.time()
rec_frames = []        # frame count per record (index = lineno)
with open(SRC, "rb") as f:
    for lineno, line in enumerate(f):
        # cheap: count <image> tags in messages[0] content via byte search
        # (faster than full json parse, equal to len(images) per earlier check)
        # we still need a tiny json parse just to find content[0] — but '<image>'
        # bytes anywhere in the line is the same count (only msgs[0] contains it).
        nf = line.count(b'<image>')
        rec_frames.append(nf)
        if (lineno+1) % 200000 == 0:
            print(f"  scanned {lineno+1:,} ...", flush=True)
N = len(rec_frames)
total_frames = sum(rec_frames)
print(f"  N={N:,}  total_frames={total_frames:,}  ({time.time()-t0:.1f}s)")

# --- Plan: greedy LPT assignment ---
print(f"Planning: greedy LPT into {N_PARTS} parts...")
order = sorted(range(N), key=lambda i: -rec_frames[i])  # heaviest first
heap = [(0, p) for p in range(N_PARTS)]                  # (load, part_id)
heapq.heapify(heap)
assign = [0] * N
part_loads = [0] * N_PARTS
part_counts = [0] * N_PARTS
for i in order:
    load, p = heapq.heappop(heap)
    assign[i] = p
    part_loads[p] += rec_frames[i]
    part_counts[p] += 1
    heapq.heappush(heap, (part_loads[p], p))

print("  planned per-part totals (frames | records):")
for p in range(N_PARTS):
    pct = 100 * part_loads[p] / total_frames
    print(f"    part{p}: frames={part_loads[p]:>11,d} ({pct:>5.2f}%)  records={part_counts[p]:>8,d}")

# --- Pass 2: stream rewrite ---
print("Pass 2: writing 5 parts...", flush=True)
handles = [open(f"{OUT_DIR}/llava_video_full_part{p}.jsonl", "wb") for p in range(N_PARTS)]
with open(SRC, "rb") as f:
    for lineno, line in enumerate(f):
        handles[assign[lineno]].write(line)
        if (lineno+1) % 200000 == 0:
            print(f"  wrote {lineno+1:,} ...", flush=True)
for h in handles: h.close()

print("\n=== per-part final stats ===")
for p in range(N_PARTS):
    path = f"{OUT_DIR}/llava_video_full_part{p}.jsonl"
    sz = os.path.getsize(path)
    print(f"  part{p}: records={part_counts[p]:>8,d}  frames={part_loads[p]:>11,d}  size={sz/(1<<30):>5.2f} GB  -> {path}")
print(f"\nTOTAL: records={N:,}  frames={total_frames:,}  ({time.time()-t0:.1f}s)")
