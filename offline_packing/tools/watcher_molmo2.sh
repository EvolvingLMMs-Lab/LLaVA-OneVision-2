#!/usr/bin/env bash
# Auto-pipeline watcher for Molmo2-videoforsft_up10mins (3 categories × 5 parts each).
#   Source: /data/Molmo2-videoforsft_up10mins/<cat>/jsonl_done_split5/molmo2_<cat>_part{0..4}.jsonl
#   Output: /data/Molmo2-videoforsft_up10mins/<cat>_packed_64k/molmo2_<cat>_part{0..4}/
#   Hosts:  5 picks from the 10-host pool, EXCLUDING 192.0.2.27 per user request.
#   Plan:   3 sequential rounds (one per category, each round 5 parts × 5 hosts in parallel).
#
# Run as:
#   nohup bash watcher_molmo2.sh > /tmp/watcher_molmo2.log 2>&1 &
#   tail -f /tmp/watcher_molmo2.log
set -euo pipefail

PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
ROOT=/data/Molmo2-videoforsft_up10mins
# 5 hosts (NO .40). Pick from {.32, .38, .41, .43, .44, .45, .46, .47, .49}.
HOSTS=(192.0.2.19 192.0.2.25 192.0.2.28 192.0.2.30 192.0.2.31)
N_PARTS=5
CATS=(short medium long)

SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR)

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

#============================================================
# Phase 0: sanity check ALL inputs (3 categories × 5 parts)
#============================================================
log "Phase 0: verify all $((${#CATS[@]} * N_PARTS)) split jsonl exist."
missing=0
for CAT in "${CATS[@]}"; do
    SRC_DIR=${ROOT}/${CAT}/jsonl_done_split5
    for i in $(seq 0 $((N_PARTS-1))); do
        f="${SRC_DIR}/molmo2_${CAT}_part${i}.jsonl"
        if [[ ! -f "$f" ]]; then
            log "  MISSING: $f"; missing=$((missing+1))
        fi
    done
done
if (( missing > 0 )); then
    log "ERROR: ${missing} inputs missing. Abort."; exit 1
fi
log "Phase 0: all $((${#CATS[@]} * N_PARTS)) inputs present."

#============================================================
# Phase 1-3: process each category in its own round
#============================================================
run_category() {
    local CAT="$1"
    local PACK_OUT=${ROOT}/${CAT}_packed_64k
    mkdir -p "${PACK_OUT}"
    log "=========================================="
    log "  Category=${CAT}: launch parts 0..$((N_PARTS-1)) on ${HOSTS[*]}"
    log "  Output: ${PACK_OUT}"
    log "=========================================="
    for i in $(seq 0 $((N_PARTS-1))); do
        ip=${HOSTS[$i]}
        part=$i
        IMG_VAR="llava_megatron:25.12"
        if [[ "$ip" == "192.0.2.35" ]]; then IMG_VAR="llava_megatron:26.05"; fi
        log "  → ${ip} cat=${CAT} part=${part} (IMG=${IMG_VAR})"
        ssh "${SSH_OPTS[@]}" "$ip" "
            cd ${PIPE}
            export IMG=${IMG_VAR}
            nohup bash tools/run_molmo2_part.sh ${CAT} ${part} \
                > ${PACK_OUT}/host_\$(hostname).log 2>&1 &
            disown
        " &
    done
    wait
    sleep 30
    log "  Category=${CAT}: all ${N_PARTS} launched. Waiting for completion..."

    while true; do
        n_done=0
        for i in $(seq 0 $((N_PARTS-1))); do
            done_file="${PACK_OUT}/molmo2_${CAT}_part${i}/DONE.txt"
            [[ -f "$done_file" ]] && n_done=$((n_done+1))
        done
        log "    cat=${CAT}: ${n_done}/${N_PARTS} done"
        [[ "$n_done" == "$N_PARTS" ]] && break
        sleep 180
    done
    log "  Category=${CAT}: COMPLETE."
}

for CAT in "${CATS[@]}"; do
    run_category "$CAT"
done

#============================================================
# Phase 4: validate and report fill rate (all 3 categories)
#============================================================
log "Phase 4: validate webdataset outputs and report fill rate."
python3 -u <<'PY'
import os, pickle, glob
ROOT = "/data/Molmo2-videoforsft_up10mins"
CATS = ("short", "medium", "long")
GLOBAL = {"bins": 0, "docs": 0, "tokens": 0, "capacity": 0, "parts_ok": 0, "parts_bad": 0}
for cat in CATS:
    print(f"\n=== {cat.upper()} ===")
    cat_bins = cat_docs = cat_tokens = cat_capacity = 0
    parts_ok = parts_bad = 0
    for i in range(5):
        pdir = f"{ROOT}/{cat}_packed_64k/molmo2_{cat}_part{i}"
        bf = f"{pdir}/bins_64klen.pkl"
        wds = f"{pdir}/webdataset"
        if not os.path.exists(bf):
            print(f"  part{i}: MISSING bins"); parts_bad += 1; continue
        bins = pickle.load(open(bf, 'rb'))
        bin_tokens = [sum(int(it[1]) for it in b) for b in bins]
        bin_docs   = [len(b) for b in bins]
        n_bins = len(bins)
        tars = sorted(glob.glob(f"{wds}/*.tar"))
        idxs = sorted(glob.glob(f"{wds}/*.tar.idx"))
        if not tars or len(tars) != len(idxs):
            print(f"  part{i}: BAD tars={len(tars)} idx={len(idxs)}"); parts_bad += 1; continue
        fill = 100*sum(bin_tokens)/(n_bins*64000) if n_bins else 0
        dpb  = sum(bin_docs)/n_bins if n_bins else 0
        print(f"  part{i}: bins={n_bins:>6,d}  docs={sum(bin_docs):>7,d}  "
              f"docs/bin={dpb:>5.2f}  fill={fill:>5.2f}%  "
              f"tars={len(tars)}  idx={len(idxs)}")
        parts_ok += 1
        cat_bins   += n_bins
        cat_docs   += sum(bin_docs)
        cat_tokens += sum(bin_tokens)
        cat_capacity += n_bins * 64000
    if cat_bins:
        print(f"  --- {cat.upper()} TOTAL ---")
        print(f"    parts_ok={parts_ok} bad={parts_bad}")
        print(f"    bins={cat_bins:,}  docs={cat_docs:,}  docs/bin={cat_docs/cat_bins:.2f}  "
              f"fill={100*cat_tokens/cat_capacity:.2f}%")
        GLOBAL["bins"]     += cat_bins
        GLOBAL["docs"]     += cat_docs
        GLOBAL["tokens"]   += cat_tokens
        GLOBAL["capacity"] += cat_capacity
        GLOBAL["parts_ok"] += parts_ok
        GLOBAL["parts_bad"] += parts_bad
print(f"\n=== GLOBAL (3 categories × 5 parts) ===")
print(f"  parts_ok       = {GLOBAL['parts_ok']}")
print(f"  parts_bad      = {GLOBAL['parts_bad']}")
print(f"  total bins     = {GLOBAL['bins']:,}")
print(f"  total docs     = {GLOBAL['docs']:,}")
print(f"  docs / bin     = {GLOBAL['docs']/GLOBAL['bins']:.2f}" if GLOBAL['bins'] else "  (no bins)")
print(f"  token fill rate= {100*GLOBAL['tokens']/GLOBAL['capacity']:.2f}%" if GLOBAL['capacity'] else "")
PY

log "All done. Pipeline complete."
log "Outputs:"
log "  ${ROOT}/short_packed_64k/"
log "  ${ROOT}/medium_packed_64k/"
log "  ${ROOT}/long_packed_64k/"
