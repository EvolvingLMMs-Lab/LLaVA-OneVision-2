#!/usr/bin/env python3
"""Split Molmo2-videoforsft_up10mins into 5 parts per category, balanced by total frame count.

For each of {short, medium, long}:
  - Read all 10 source jsonl from /ov21/Molmo2-videoforsft_up10mins/<cat>/output/
  - Greedy LPT split into 5 frame-balanced parts (heaviest record first).
  - Write to /ov21/Molmo2-videoforsft_up10mins/<cat>/jsonl_done_split5/molmo2_<cat>_part{0..4}.jsonl
"""
import os, sys, heapq, time

ROOT = "/ov21/Molmo2-videoforsft_up10mins"
N_PARTS = 5

for CAT in ("short", "medium", "long"):
    SRC_DIR = f"{ROOT}/{CAT}/output"
    OUT_DIR = f"{ROOT}/{CAT}/jsonl_done_split5"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*70}\n=== Category: {CAT} ===\n{'='*70}")

    parts_in = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".jsonl"))
    print(f"  Source parts: {len(parts_in)} ({parts_in[0]} .. {parts_in[-1]})")

    # --- Pass 1: scan frame counts (count <image> tags - matches len(images)) ---
    t0 = time.time()
    rec_frames = []   # frame count per record
    rec_loc    = []   # (src_part_idx, byte_offset, byte_length)
    print(f"  Pass 1: scanning frame counts ...", flush=True)
    for sp_idx, sp in enumerate(parts_in):
        fp = f"{SRC_DIR}/{sp}"
        with open(fp, "rb") as f:
            offset = 0
            for line in f:
                nf = line.count(b'<image>')
                rec_frames.append(nf)
                rec_loc.append((sp_idx, offset, len(line)))
                offset += len(line)
    N = len(rec_frames)
    total_frames = sum(rec_frames)
    print(f"    N={N:,}  total_frames={total_frames:,}  ({time.time()-t0:.1f}s)")

    if N == 0:
        print(f"    SKIP {CAT}: no records"); continue

    # --- Plan: greedy LPT ---
    print(f"  Plan: greedy LPT into {N_PARTS} parts...")
    order = sorted(range(N), key=lambda i: -rec_frames[i])
    heap = [(0, p) for p in range(N_PARTS)]
    heapq.heapify(heap)
    assign = [0] * N
    part_loads  = [0] * N_PARTS
    part_counts = [0] * N_PARTS
    for i in order:
        load, p = heapq.heappop(heap)
        assign[i] = p
        part_loads[p] += rec_frames[i]
        part_counts[p] += 1
        heapq.heappush(heap, (part_loads[p], p))

    print(f"    planned per-part totals:")
    for p in range(N_PARTS):
        pct = 100 * part_loads[p] / total_frames if total_frames else 0
        print(f"      part{p}: frames={part_loads[p]:>11,d} ({pct:>5.2f}%)  records={part_counts[p]:>7,d}")

    # --- Pass 2: stream rewrite using rec_loc ---
    print(f"  Pass 2: writing 5 balanced parts...", flush=True)
    out_paths = [f"{OUT_DIR}/molmo2_{CAT}_part{p}.jsonl" for p in range(N_PARTS)]
    out_handles = [open(op, "wb") for op in out_paths]
    # Open source files once
    src_handles = [open(f"{SRC_DIR}/{sp}", "rb") for sp in parts_in]
    for ridx, (sp_idx, off, ln) in enumerate(rec_loc):
        src_handles[sp_idx].seek(off)
        chunk = src_handles[sp_idx].read(ln)
        out_handles[assign[ridx]].write(chunk)
    for h in out_handles: h.close()
    for h in src_handles: h.close()

    print(f"  === per-part final stats ({CAT}) ===")
    for p in range(N_PARTS):
        sz = os.path.getsize(out_paths[p])
        print(f"    part{p}: records={part_counts[p]:>7,d}  frames={part_loads[p]:>11,d}  "
              f"size={sz/(1<<20):>7.2f} MB  -> {out_paths[p]}")
    print(f"  TOTAL ({CAT}): records={N:,}  frames={total_frames:,}  ({time.time()-t0:.1f}s)")

print(f"\n{'='*70}\nAll 3 categories split into 5 parts each.\n{'='*70}")
