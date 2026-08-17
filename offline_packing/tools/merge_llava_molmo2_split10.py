#!/usr/bin/env python3
"""Merge llava_video_full (1.4M records) + Molmo2 (23k records) into 10 balanced parts.

Strategy:
  * Gather all records from BOTH datasets: (name, tokens, patches, manifest_entry).
  * Namespace collision guard: llava keeps original name; Molmo2 gets 'm2_' prefix
    (Molmo2 is the smaller set; already-known 9-name-collision zone will vanish).
  * Greedy LPT balance by TOKENS (records sorted DESC → each to lightest bin).
  * 10 parts, ~140k records each, ~660M tokens each → ~10.3k bins @ 99% fill.
  * Write per-part {token_info.txt, manifest.jsonl} to:
      /data/llava_video_full_molmo2_merged_packed_64k/part{0..9}/
"""
import os, json, heapq, random, time

ROOT_LLAVA = '/data/llava_video_full_packed_64k'
ROOT_MOLMO = '/data/Molmo2-videoforsft_up10mins'
OUT_ROOT = '/data/llava_video_full_molmo2_merged_packed_64k'
N_PARTS = 10
MOLMO_PREFIX = 'm2_'

os.makedirs(OUT_ROOT, exist_ok=True)

# ---- Step 1: gather llava records ----
print(f'[1/4] Loading llava_video_full 5 parts...', flush=True)
t0 = time.time()
llava_records = []   # list of dict: {name, tokens, patches, entry}
for pi in range(5):
    ti = f'{ROOT_LLAVA}/llava_video_full_part{pi}/token_info.txt'
    mf = f'{ROOT_LLAVA}/llava_video_full_part{pi}/manifest.jsonl'
    name2tp = {}
    with open(ti) as f:
        for line in f:
            parts = line.rstrip('\n').split(':')
            if len(parts) < 2: continue
            try:
                name2tp[parts[0]] = (int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            except Exception:
                pass
    n_added = 0
    with open(mf) as f:
        for line in f:
            d = json.loads(line)
            name = d.get('name')
            if name in name2tp:
                tok, pat = name2tp[name]
                llava_records.append({'name': name, 'tokens': tok, 'patches': pat, 'entry': d})
                n_added += 1
    print(f'  llava_part{pi}: {n_added:,} records')
print(f'  llava total: {len(llava_records):,}  ({time.time()-t0:.1f}s)')

# ---- Step 2: gather Molmo2 records with 'm2_' prefix ----
print(f'\n[2/4] Loading Molmo2 3 cats × 5 parts...', flush=True)
t1 = time.time()
molmo_records = []
for cat in ('short','medium','long'):
    for pi in range(5):
        ti = f'{ROOT_MOLMO}/{cat}_packed_64k/molmo2_{cat}_part{pi}/token_info.txt'
        mf = f'{ROOT_MOLMO}/{cat}_packed_64k/molmo2_{cat}_part{pi}/manifest.jsonl'
        name2tp = {}
        with open(ti) as f:
            for line in f:
                parts = line.rstrip('\n').split(':')
                if len(parts) < 2: continue
                try:
                    name2tp[parts[0]] = (int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
                except Exception:
                    pass
        n_added = 0
        with open(mf) as f:
            for line in f:
                d = json.loads(line)
                orig = d.get('name')
                if orig in name2tp:
                    new_name = f'{MOLMO_PREFIX}{orig}'
                    d2 = dict(d); d2['name'] = new_name
                    tok, pat = name2tp[orig]
                    molmo_records.append({'name': new_name, 'tokens': tok, 'patches': pat, 'entry': d2})
                    n_added += 1
        print(f'  {cat}_part{pi}: {n_added:,} records')
print(f'  molmo total: {len(molmo_records):,}  ({time.time()-t1:.1f}s)')

# ---- Step 3: Greedy LPT split into N_PARTS balanced by TOKENS ----
print(f'\n[3/4] Greedy LPT: split into {N_PARTS} parts by total tokens...', flush=True)
all_records = llava_records + molmo_records
# Sort DESC by tokens (heaviest first)
all_records.sort(key=lambda r: -r['tokens'])
# Heap of (load, part_idx)
heap = [(0, p) for p in range(N_PARTS)]
heapq.heapify(heap)
part_records = [[] for _ in range(N_PARTS)]
part_loads   = [0]*N_PARTS
part_docs    = [0]*N_PARTS
part_llava   = [0]*N_PARTS
part_molmo   = [0]*N_PARTS

t2 = time.time()
for r in all_records:
    load, pid = heapq.heappop(heap)
    part_records[pid].append(r)
    part_loads[pid] += r['tokens']
    part_docs[pid] += 1
    if r['name'].startswith(MOLMO_PREFIX):
        part_molmo[pid] += 1
    else:
        part_llava[pid] += 1
    heapq.heappush(heap, (part_loads[pid], pid))
total_tokens = sum(part_loads)
print(f'  planned per-part load ({time.time()-t2:.1f}s):')
for p in range(N_PARTS):
    pct = 100 * part_loads[p] / total_tokens
    est_bins = part_loads[p] / (64000 * 0.99)
    print(f'    part{p}: docs={part_docs[p]:>8,d}  tokens={part_loads[p]:>13,d}  ({pct:>5.2f}%)  '
          f'llava={part_llava[p]:>7,d} molmo={part_molmo[p]:>5,d}  est_bins={est_bins:,.0f}')

# ---- Step 4: write per-part {token_info.txt, manifest.jsonl} ----
print(f'\n[4/4] Writing per-part outputs to {OUT_ROOT}/...', flush=True)
t3 = time.time()
for p in range(N_PARTS):
    d = f'{OUT_ROOT}/part{p}'
    os.makedirs(d, exist_ok=True)
    ti_path = f'{d}/token_info.txt'
    mf_path = f'{d}/manifest.jsonl'
    # Shuffle within-part (deterministic seed) so that llava/molmo are interleaved
    rng = random.Random(2026 + p)
    recs = part_records[p]
    rng.shuffle(recs)
    with open(ti_path, 'w') as tf, open(mf_path, 'w') as mf:
        for r in recs:
            tf.write(f"{r['name']}:{r['tokens']}:{r['patches']}\n")
            mf.write(json.dumps(r['entry'], ensure_ascii=False) + '\n')
    print(f'  part{p}: {len(recs):,} lines -> {d}')

print(f'\n[DONE]  {time.time()-t0:.1f}s total.')
print(f'Total records merged: {len(all_records):,}  '
      f'({len(llava_records):,} llava + {len(molmo_records):,} molmo)')
print(f'Total tokens merged:  {total_tokens:,}')
print(f'Estimated total bins: {total_tokens/(64000*0.99):,.0f}  (@ 99% fill)')
