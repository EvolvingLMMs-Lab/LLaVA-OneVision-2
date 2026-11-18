#!/usr/bin/env bash
# Auto-pipeline watcher for pandas70m_gt10min_fix_recap (caption rewritten):
#   Source: /data/dataset_frame/pandas70m_gt10min_fix_recap/jsonl_done/
#   Output: /data/pandas70m_gt10min_packed_64k_fix_recap/
#   Layout: 10 hosts × 2 rounds (parts 0-9 then 10-19), same topology as fix.
#
# Run as:
#   nohup bash watcher_pandas70m_recap.sh > /tmp/watcher_pandas70m_recap.log 2>&1 &
#   tail -f /tmp/watcher_pandas70m_recap.log

set -euo pipefail
PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
SRC_ROOT=/data/dataset_frame/pandas70m_gt10min_fix_recap
PACK_OUT=/data/pandas70m_gt10min_packed_64k_fix_recap
HOSTS=(192.0.2.19 192.0.2.25 192.0.2.27 192.0.2.28 192.0.2.30 \
       192.0.2.31 192.0.2.32 192.0.2.33 192.0.2.34 192.0.2.35)

mkdir -p "${PACK_OUT}"
SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR)

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

#============================================================
# Phase 0: sanity check inputs
#============================================================
log "Phase 0: verify all 20 recap jsonl exist."
missing=0
for i in $(seq 0 19); do
    p=$(printf "%02d" $i)
    f="${SRC_ROOT}/jsonl_done/pandas70m_gt10min_fix_recap_done_part${p}.jsonl"
    if [[ ! -f "$f" ]]; then
        log "  MISSING: $f"
        missing=$((missing+1))
    fi
done
if (( missing > 0 )); then
    log "ERROR: ${missing} recap inputs missing. Abort."; exit 1
fi
log "Phase 0: all 20 recap inputs present."

#============================================================
# Phase 1: launch round 1 (parts 0-9)
#============================================================
ROUND1_PARTS=(0 1 2 3 4 5 6 7 8 9)
ROUND2_PARTS=(10 11 12 13 14 15 16 17 18 19)

run_round() {
    local round_idx="$1"; shift
    local parts=("$@")
    log "Phase ${round_idx}: launch parts: ${parts[*]}"
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
            nohup bash tools/run_pandas70m_part_recap.sh ${part} \
                > ${PACK_OUT}/host_\$(hostname).log 2>&1 &
            disown
        " &
    done
    wait
    sleep 30
    log "Phase ${round_idx}: all 10 launched. Waiting for completion..."

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
    log "Phase ${round_idx}: complete."
}

run_round 1 "${ROUND1_PARTS[@]}"
run_round 2 "${ROUND2_PARTS[@]}"

#============================================================
# Phase 3: validation
#============================================================
log "Phase 3: validate webdataset outputs."
python3 -u <<PY
import os, pickle, glob
total_bins = 0
parts_ok = 0
parts_bad = 0
for i in range(20):
    pdir = f"${PACK_OUT}/pandas70m_part{i}"
    bf = f"{pdir}/bins_64klen.pkl"
    wds = f"{pdir}/webdataset"
    if not os.path.exists(bf):
        print(f"  part{i}: MISSING bins"); parts_bad += 1; continue
    bins = pickle.load(open(bf, 'rb'))
    total_bins += len(bins)
    tars = sorted(glob.glob(f"{wds}/*.tar"))
    idxs = sorted(glob.glob(f"{wds}/*.tar.idx"))
    if not tars or len(tars) != len(idxs):
        print(f"  part{i}: BAD tars={len(tars)} idx={len(idxs)}"); parts_bad += 1; continue
    print(f"  part{i}: bins={len(bins):,}  tars={len(tars)}  idx={len(idxs)}")
    parts_ok += 1
print(f"\nSUMMARY: parts_ok={parts_ok}  parts_bad={parts_bad}  total_bins={total_bins:,}")
PY

log "All done. Pipeline complete."
log "Recap pack: ${PACK_OUT}"
