#!/usr/bin/env bash
# Auto-pipeline watcher for pandas70m_gt10min_fix:
#   Phase 1 (CURRENT): wait for all 10 schema-B extraction parts to finish
#   Phase 2: merge clean (Schema A no-noimg) + (Schema B done) → 20 parts
#   Phase 3: launch packing on 10 hosts × 2 rounds (parts 0-9 then 10-19)
#   Phase 4: validation (assert 0% noimg in resulting bins)
#
# Output:  /data/pandas70m_gt10min_packed_64k_fix/  (NEW; old _packed_64k untouched)
#
# Run as:
#   nohup bash watcher_pandas70m_fix.sh > /tmp/watcher_pandas70m_fix.log 2>&1 &
#   tail -f /tmp/watcher_pandas70m_fix.log

set -euo pipefail
PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
SCHEMAB_ROOT=/data/dataset_frame/pandas70m_gt10min_schemab
SCHEMAA_ROOT=/data/dataset_frame/pandas70m_gt10min
FIX_ROOT=/data/dataset_frame/pandas70m_gt10min_fix
PACK_OUT=/data/pandas70m_gt10min_packed_64k_fix
HOSTS=(192.0.2.19 192.0.2.25 192.0.2.27 192.0.2.28 192.0.2.30 192.0.2.31 192.0.2.32 192.0.2.33 192.0.2.34 192.0.2.35)

mkdir -p "${FIX_ROOT}/jsonl_done" "${PACK_OUT}"
SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR)

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

#============================================================
# Phase 1: wait for schema-B extraction (10 parts on 10 hosts)
#============================================================
log "Phase 1: wait for Schema B extraction to complete..."
while true; do
    n_done=0
    n_running=0
    n_dead=0
    for i in $(seq 0 9); do
        ip=${HOSTS[$i]}
        part=$(printf "part%02d" "$i")
        out_jsonl="${SCHEMAB_ROOT}/jsonl_done/pandas70m_gt10min_schemab_done_${part}.jsonl"
        # Done = jsonl_done file present AND process gone.
        proc=$(ssh "${SSH_OPTS[@]}" "$ip" "pgrep -f '[r]un_cut_frames.py.*pandas70m_gt10min_schemab_${part}\\.jsonl' >/dev/null && echo R || echo S" 2>/dev/null || echo "?")
        if [[ -f "$out_jsonl" && "$proc" == "S" ]]; then
            n_done=$((n_done+1))
        elif [[ "$proc" == "R" ]]; then
            n_running=$((n_running+1))
        else
            n_dead=$((n_dead+1))
        fi
    done
    log "Phase 1: done=${n_done}/10 running=${n_running} dead=${n_dead}"
    [[ "$n_done" == "10" ]] && break
    if [[ "$n_dead" -gt 0 && "$n_running" == "0" ]]; then
        log "ERROR: $n_dead parts died without producing output. Investigate."
        exit 2
    fi
    sleep 600
done
log "Phase 1: All 10 schema-B parts done."

#============================================================
# Phase 2: merge into clean fix jsonl_done (20 parts)
#============================================================
log "Phase 2: merge clean Schema A + new Schema B → 20 parts"
python3 -u <<PY
import json, os, glob

fix_dir = '${FIX_ROOT}/jsonl_done'
os.makedirs(fix_dir, exist_ok=True)

# 1) Schema A: read 20 old jsonl_done files, KEEP only records with images_source
a_files = sorted(glob.glob('${SCHEMAA_ROOT}/jsonl_done/pandas70m_gt10min_done_part*.jsonl'))
print(f'Schema A files found: {len(a_files)}')
clean_a = []
total_a, kept_a = 0, 0
for f in a_files:
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            total_a += 1
            if d.get('images_source'):
                clean_a.append(d)
                kept_a += 1
print(f'Schema A: total={total_a}, kept_with_images={kept_a}, dropped_noimg={total_a-kept_a}')

# 2) Schema B: read 10 new jsonl_done files
b_files = sorted(glob.glob('${SCHEMAB_ROOT}/jsonl_done/pandas70m_gt10min_schemab_done_part*.jsonl'))
print(f'Schema B files found: {len(b_files)}')
clean_b = []
for f in b_files:
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get('images_source'):
                clean_b.append(d)
print(f'Schema B: kept_with_images={len(clean_b)}')

combined = clean_a + clean_b
print(f'Combined clean records: {len(combined)}')
N = 20
fhs = [open(os.path.join(fix_dir, f'pandas70m_gt10min_fix_done_part{i:02d}.jsonl'), 'w') for i in range(N)]
for i, d in enumerate(combined):
    fhs[i % N].write(json.dumps(d, ensure_ascii=False) + '\n')
for fh in fhs:
    fh.close()
print('Wrote 20 fix parts.')
for i in range(N):
    p = os.path.join(fix_dir, f'pandas70m_gt10min_fix_done_part{i:02d}.jsonl')
    n = sum(1 for _ in open(p))
    print(f'  part{i:02d}: {n}')
PY
log "Phase 2: merge done."

#============================================================
# Phase 3: packing 20 parts on 10 hosts (2 rounds)
#============================================================
ROUND_PARTS=(0 1 2 3 4 5 6 7 8 9)        # round 1
ROUND2_PARTS=(10 11 12 13 14 15 16 17 18 19)  # round 2

run_round() {
    local round_idx="$1"; shift
    local parts=("$@")
    log "Phase 3 round=${round_idx}: launch parts: ${parts[*]}"
    for i in $(seq 0 9); do
        ip=${HOSTS[$i]}
        part=${parts[$i]}
        IMG_VAR="llava_megatron:25.12"
        # 5.49 docker proxy issue → use 26.05
        if [[ "$ip" == "192.0.2.35" ]]; then IMG_VAR="llava_megatron:26.05"; fi
        log "  → ${ip} part=${part} (IMG=${IMG_VAR})"
        ssh "${SSH_OPTS[@]}" "$ip" "
            cd ${PIPE}
            export IMG=${IMG_VAR}
            nohup bash tools/run_pandas70m_part_fix.sh ${part} \
                > ${PACK_OUT}/host_\$(hostname).log 2>&1 &
            disown
        " &
    done
    wait
    sleep 30  # let processes attach properly
    log "Phase 3 round=${round_idx}: all 10 launched. Now waiting for completion..."

    # Wait for each part's DONE.txt
    while true; do
        n_done=0
        for part in "${parts[@]}"; do
            done_file="${PACK_OUT}/pandas70m_part${part}/DONE.txt"
            [[ -f "$done_file" ]] && n_done=$((n_done+1))
        done
        log "  round=${round_idx}: ${n_done}/10 done"
        [[ "$n_done" == "10" ]] && break
        sleep 300
    done
    log "Phase 3 round=${round_idx}: complete."
}

run_round 1 "${ROUND_PARTS[@]}"
run_round 2 "${ROUND2_PARTS[@]}"

#============================================================
# Phase 4: validation
#============================================================
log "Phase 4: validate that NO bin contains noimg samples."
python3 -u <<PY
import pickle, glob
total_bins = 0
noimg_bins = 0
for bf in sorted(glob.glob('${PACK_OUT}/pandas70m_part*/bins_64klen.pkl')):
    with open(bf,'rb') as f:
        bins = pickle.load(f)
    total_bins += len(bins)
    # bins is list[list[name]]
    # We just check that the manifest behind these bins has only image samples.
    # Quick proxy: check token_info has all entries with non-zero patches.
    tinfo = bf.replace('bins_64klen.pkl','token_info.txt')
    bad = 0
    with open(tinfo) as fh:
        for line in fh:
            parts = line.strip().split(':')
            if len(parts) >= 3:
                # Format: name:total_tokens:patches:...
                if int(parts[2]) == 0:
                    bad += 1
    if bad > 0:
        noimg_bins += 1
        print(f'NOIMG in {bf}: {bad}')
print(f'Total parts: {total_bins} bins')
print(f'Parts with noimg: {noimg_bins}')
print('VALIDATION:', 'PASS' if noimg_bins == 0 else 'FAIL')
PY

#============================================================
# Final summary
#============================================================
log "All done. Pipeline complete."
log "Old polluted: /data/pandas70m_gt10min_packed_64k/    (8.1 TB, untouched)"
log "New FIX:      /data/pandas70m_gt10min_packed_64k_fix/ (NEW)"
