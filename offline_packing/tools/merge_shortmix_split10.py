#!/usr/bin/env python3
"""Merge 3 short-video datasets (flash_non_tg + flash_v2 + molmo2_short) into 10 LPT-balanced parts.

Strategy (matches merge_llava_molmo2_split10.py):
  * Load each dataset's manifest.jsonl + token_info.txt.
  * Namespace prefixes to avoid rare hash collisions:
      flash_non_tg  → 'fntg_'
      flash_v2      → 'fv2_'
      molmo2_short  → 'm2s_'
  * Greedy LPT balance by TOKENS (records sorted DESC → each to lightest bin).
  * Write per-part {token_info.txt, manifest.jsonl} to:
      /data/shortmix_3sets_packed_64k/part{0..9}/
"""
import os, json, heapq, random, time

PREP_ROOT = '/data/shortmix_prep'
OUT_ROOT  = '/data/shortmix_3sets_packed_64k'
N_PARTS   = 10

DATASETS = [
    ('flash_non_tg', 'fntg_'),
    ('flash_v2',     'fv2_'),
    ('molmo2_short', 'm2s_'),
]

os.makedirs(OUT_ROOT, exist_ok=True)

# ---- Step 1: gather all records with prefixed names ----
print(f'[1/3] Loading token_info + manifest for all 3 datasets ...', flush=True)
t0 = time.time()
all_recs = []   # list of dict: {name, tokens, patches, entry}
for ds, pref in DATASETS:
    ti = f'{PREP_ROOT}/{ds}/token_info.txt'
    mf = f'{PREP_ROOT}/{ds}/manifest.jsonl'
    if not (os.path.exists(ti) and os.path.exists(mf)):
        raise SystemExit(f'MISSING: {ti} or {mf}')
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
                new_name = f'{pref}{orig}'
                d2 = dict(d); d2['name'] = new_name
                tok, pat = name2tp[orig]
                all_recs.append({'name': new_name, 'tokens': tok, 'patches': pat, 'entry': d2, 'ds': ds})
                n_added += 1
    print(f'  {ds:15s}: loaded {n_added:,} records (prefix={pref})')
print(f'  Combined: {len(all_recs):,} records  ({time.time()-t0:.1f}s)')

# ---- Step 2: Greedy LPT split by tokens ----
print(f'\n[2/3] Greedy LPT: split into {N_PARTS} parts by total tokens...', flush=True)
all_recs.sort(key=lambda r: -r['tokens'])
heap = [(0, p) for p in range(N_PARTS)]
heapq.heapify(heap)
part_records = [[] for _ in range(N_PARTS)]
part_loads   = [0]*N_PARTS
part_by_ds   = [{ds:0 for ds,_ in DATASETS} for _ in range(N_PARTS)]

t1 = time.time()
for r in all_recs:
    load, pid = heapq.heappop(heap)
    part_records[pid].append(r)
    part_loads[pid] += r['tokens']
    part_by_ds[pid][r['ds']] += 1
    heapq.heappush(heap, (part_loads[pid], pid))
total_tokens = sum(part_loads)
print(f'  planned per-part load  ({time.time()-t1:.1f}s):')
for p in range(N_PARTS):
    pct = 100 * part_loads[p] / total_tokens
    est_bins = part_loads[p] / (64000 * 0.99)
    breakdown = ' '.join(f'{ds}={part_by_ds[p][ds]:,}' for ds,_ in DATASETS)
    print(f'    part{p}: docs={len(part_records[p]):>7,d}  tokens={part_loads[p]:>13,d} ({pct:>5.2f}%)  '
          f'est_bins={est_bins:>6,.0f}  {breakdown}')

# ---- Step 3: write per-part outputs ----
print(f'\n[3/3] Writing per-part files to {OUT_ROOT}/...', flush=True)
t2 = time.time()
for p in range(N_PARTS):
    d = f'{OUT_ROOT}/part{p}'
    os.makedirs(d, exist_ok=True)
    ti_path = f'{d}/token_info.txt'
    mf_path = f'{d}/manifest.jsonl'
    rng = random.Random(2026 + p)
    recs = part_records[p]
    rng.shuffle(recs)  # deterministic interleave
    with open(ti_path, 'w') as tf, open(mf_path, 'w') as mf:
        for r in recs:
            tf.write(f"{r['name']}:{r['tokens']}:{r['patches']}\n")
            mf.write(json.dumps(r['entry'], ensure_ascii=False) + '\n')
    print(f'  part{p}: {len(recs):,} lines -> {d}')

print(f'\n[DONE]  {time.time()-t0:.1f}s total.')
print(f'Total records merged: {len(all_recs):,}')
print(f'Total tokens merged:  {total_tokens:,}')
print(f'Estimated total bins: {total_tokens/(64000*0.99):,.0f}  (@ 99% fill)')
