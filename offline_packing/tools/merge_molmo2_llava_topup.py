#!/usr/bin/env python3
"""Build merged token_info.txt + manifest.jsonl for Molmo2 top-up with llava_video_full fillers.

Strategy:
  * For each of 15 Molmo2 parts (3 cats × 5 parts), sample a slice of llava records
    sized to fill the mean gap (with 10% margin) and emit a merged token_info + manifest.
  * llava sample names are prefixed with ``lvf_`` to avoid 9 known hash collisions.
  * Merged files are written to /data/Molmo2-videoforsft_up10mins/<cat>_packed_64k_topup/
      molmo2_<cat>_part{i}/{token_info.txt, manifest.jsonl}
  * s3 will produce bins referencing both Molmo2 (native name) and llava (lvf_-prefixed).
  * s4 reads the merged manifest → reads the ORIGINAL images/npy paths from disk.
"""
import os, sys, json, hashlib, random, shutil

ROOT_MOLMO = '/data/Molmo2-videoforsft_up10mins'
ROOT_LLAVA = '/data/llava_video_full_packed_64k'
OUT_ROOT = '/data/Molmo2-videoforsft_up10mins'  # cat-scoped topup dirs go here
PREFIX = 'lvf_'  # prepended to llava names to avoid Molmo2 hash collisions

# Records per Molmo2 part × 5 (from previous inspection):
CATS = {
    'short':  {'parts': 5, 'recs_per_part': 3263, 'fillers_per_part': 18400},
    'medium': {'parts': 5, 'recs_per_part': 1141, 'fillers_per_part':  8100},
    'long':   {'parts': 5, 'recs_per_part':  287, 'fillers_per_part':  1100},
}
TOTAL_FILLERS_NEEDED = sum(v['parts'] * v['fillers_per_part'] for v in CATS.values())
print(f'Total llava fillers needed: {TOTAL_FILLERS_NEEDED:,}')

# ---- Step 1: gather ALL llava records: (name, tokens, patches, manifest_entry) ----
print('\n[1/3] Loading all llava_video_full token_info + manifest ...')
llava_all = []  # list of (tokens, name, patch, manifest_entry_dict)
for pi in range(5):
    ti = f'{ROOT_LLAVA}/llava_video_full_part{pi}/token_info.txt'
    mf = f'{ROOT_LLAVA}/llava_video_full_part{pi}/manifest.jsonl'
    name2tp = {}
    with open(ti) as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            try:
                name2tp[parts[0]] = (int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            except Exception:
                pass
    with open(mf) as f:
        for line in f:
            d = json.loads(line)
            name = d.get('name')
            if name in name2tp:
                tok, pat = name2tp[name]
                llava_all.append((tok, name, pat, d))
    print(f'  llava_part{pi}: loaded {len(name2tp):,} records; cumulative={len(llava_all):,}')

# Shuffle deterministically & pick first TOTAL_FILLERS_NEEDED
random.Random(2026).shuffle(llava_all)
llava_picked = llava_all[:TOTAL_FILLERS_NEEDED]
print(f'  Picked {len(llava_picked):,} fillers (of {len(llava_all):,} total)')

# ---- Step 2: split picked fillers across 15 Molmo2 parts ----
print('\n[2/3] Distribute fillers to 15 Molmo2 parts ...')
cursor = 0
distribution = {}  # (cat, part_idx) -> list of llava records
for cat, cfg in CATS.items():
    n = cfg['fillers_per_part']
    for p in range(cfg['parts']):
        distribution[(cat, p)] = llava_picked[cursor:cursor+n]
        cursor += n
        print(f'  {cat}_part{p}: {len(distribution[(cat,p)]):,} fillers  (cursor={cursor:,})')

# ---- Step 3: write merged token_info + manifest per part ----
print('\n[3/3] Writing merged files per part ...')
for (cat, p), slice_ in distribution.items():
    molmo_ti = f'{ROOT_MOLMO}/{cat}_packed_64k/molmo2_{cat}_part{p}/token_info.txt'
    molmo_mf = f'{ROOT_MOLMO}/{cat}_packed_64k/molmo2_{cat}_part{p}/manifest.jsonl'
    out_dir  = f'{OUT_ROOT}/{cat}_packed_64k_topup/molmo2_{cat}_part{p}'
    os.makedirs(out_dir, exist_ok=True)
    out_ti = f'{out_dir}/token_info.txt'
    out_mf = f'{out_dir}/manifest.jsonl'

    n_molmo = 0
    with open(out_ti, 'w') as tf, open(out_mf, 'w') as mf:
        # Molmo2 originals (no rename)
        with open(molmo_ti) as f:
            for line in f:
                tf.write(line); n_molmo += 1
        with open(molmo_mf) as f:
            for line in f:
                mf.write(line)
        # Llava fillers with lvf_ prefix
        for tok, name, pat, entry in slice_:
            new_name = f'{PREFIX}{name}'
            tf.write(f'{new_name}:{tok}:{pat}\n')
            entry2 = dict(entry); entry2['name'] = new_name
            mf.write(json.dumps(entry2, ensure_ascii=False) + '\n')
    n_total = n_molmo + len(slice_)
    print(f'  {cat}_part{p}: {n_molmo:,} molmo + {len(slice_):,} llava = {n_total:,} lines → {out_dir}')

print('\nDone.')
print(f'Total merged records: molmo2({sum(v["parts"]*v["recs_per_part"] for v in CATS.values()):,}) '
      f'+ llava_fillers({TOTAL_FILLERS_NEEDED:,}) = combined 15 parts')
