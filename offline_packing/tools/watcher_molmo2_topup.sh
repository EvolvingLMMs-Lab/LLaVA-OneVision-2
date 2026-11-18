#!/usr/bin/env bash
# Auto-pipeline watcher for Molmo2 TOP-UP with llava fillers.
# 3 categories × 5 parts × 1 round (5 hosts each). s3+s4 only (fast, no s1/s2).
#
# Prereq: run tools/merge_molmo2_llava_topup.py FIRST to build merged token_info/manifest.
#
# Run as:
#   nohup bash watcher_molmo2_topup.sh > /tmp/watcher_molmo2_topup.log 2>&1 &
#   tail -f /tmp/watcher_molmo2_topup.log
set -euo pipefail

PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
ROOT=/ov21/Molmo2-videoforsft_up10mins
HOSTS=(172.16.5.32 172.16.5.38 172.16.5.41 172.16.5.43 172.16.5.44)
N_PARTS=5
CATS=(short medium long)

SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR)

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

#============================================================
# Phase 0: sanity check ALL merged inputs
#============================================================
log "Phase 0: verify all $((${#CATS[@]} * N_PARTS)) merged token_info+manifest exist."
missing=0
for CAT in "${CATS[@]}"; do
    for i in $(seq 0 $((N_PARTS-1))); do
        d=${ROOT}/${CAT}_packed_64k_topup/molmo2_${CAT}_part${i}
        [[ -f "$d/token_info.txt" ]] || { log "  MISSING: $d/token_info.txt"; missing=$((missing+1)); }
        [[ -f "$d/manifest.jsonl" ]] || { log "  MISSING: $d/manifest.jsonl"; missing=$((missing+1)); }
    done
done
if (( missing > 0 )); then
    log "ERROR: ${missing} inputs missing. Run merge_molmo2_llava_topup.py first."; exit 1
fi
log "Phase 0: all inputs present."

#============================================================
# Phase 1-3: process each category in its own round
#============================================================
run_category() {
    local CAT="$1"
    local PACK_OUT=${ROOT}/${CAT}_packed_64k_topup
    log "=========================================="
    log "  Category=${CAT}: launch topup parts 0..$((N_PARTS-1)) on ${HOSTS[*]}"
    log "  Output: ${PACK_OUT}"
    log "=========================================="
    for i in $(seq 0 $((N_PARTS-1))); do
        ip=${HOSTS[$i]}
        part=$i
        IMG_VAR="llava_megatron:25.12"
        if [[ "$ip" == "172.16.5.49" ]]; then IMG_VAR="llava_megatron:26.05"; fi
        log "  → ${ip} cat=${CAT} part=${part} (IMG=${IMG_VAR})"
        ssh "${SSH_OPTS[@]}" "$ip" "
            cd ${PIPE}
            export IMG=${IMG_VAR}
            nohup bash tools/run_molmo2_topup_part.sh ${CAT} ${part} \
                > ${PACK_OUT}/host_\$(hostname).log 2>&1 &
            disown
        " &
    done
    wait
    sleep 20
    log "  Category=${CAT}: all ${N_PARTS} launched. Waiting for completion..."

    while true; do
        n_done=0
        for i in $(seq 0 $((N_PARTS-1))); do
            done_file="${PACK_OUT}/molmo2_${CAT}_part${i}/DONE.txt"
            [[ -f "$done_file" ]] && n_done=$((n_done+1))
        done
        log "    cat=${CAT}: ${n_done}/${N_PARTS} done"
        [[ "$n_done" == "$N_PARTS" ]] && break
        sleep 120
    done
    log "  Category=${CAT}: COMPLETE."
}

for CAT in "${CATS[@]}"; do
    run_category "$CAT"
done

#============================================================
# Phase 4: validate + fill-rate report
#============================================================
log "Phase 4: validate topup webdataset outputs + report fill rate."
python3 -u <<'PY'
import os, pickle, glob
ROOT = "/ov21/Molmo2-videoforsft_up10mins"
CATS = ("short", "medium", "long")
GLOBAL = {"bins":0,"docs":0,"tokens":0,"capacity":0,"ok":0,"bad":0}
for cat in CATS:
    print(f"\n=== {cat.upper()} (topup) ===")
    cb=cd=ct=cc=0; ok=bad=0
    for i in range(5):
        pdir = f"{ROOT}/{cat}_packed_64k_topup/molmo2_{cat}_part{i}"
        bf = f"{pdir}/bins_64klen.pkl"
        wds = f"{pdir}/webdataset"
        if not os.path.exists(bf):
            print(f"  part{i}: MISSING bins"); bad += 1; continue
        bins = pickle.load(open(bf,'rb'))
        bin_tokens = [sum(int(it[1]) for it in b) for b in bins]
        bin_docs   = [len(b) for b in bins]
        n = len(bins)
        tars = sorted(glob.glob(f"{wds}/*.tar"))
        idxs = sorted(glob.glob(f"{wds}/*.tar.idx"))
        if not tars or len(tars) != len(idxs):
            print(f"  part{i}: BAD tars={len(tars)} idx={len(idxs)}"); bad += 1; continue
        fill = 100*sum(bin_tokens)/(n*64000) if n else 0
        dpb  = sum(bin_docs)/n if n else 0
        print(f"  part{i}: bins={n:>6,d}  docs={sum(bin_docs):>7,d}  docs/bin={dpb:>5.2f}  "
              f"fill={fill:>5.2f}%  tars={len(tars)}  idx={len(idxs)}")
        ok += 1; cb+=n; cd+=sum(bin_docs); ct+=sum(bin_tokens); cc+=n*64000
    if cb:
        print(f"  --- {cat.upper()} TOTAL --- ok={ok} bad={bad} "
              f"bins={cb:,} docs={cd:,} docs/bin={cd/cb:.2f} fill={100*ct/cc:.2f}%")
        for k,v in {"bins":cb,"docs":cd,"tokens":ct,"capacity":cc,"ok":ok,"bad":bad}.items():
            GLOBAL[k] += v
print(f"\n=== GLOBAL (Molmo2 topup 3×5) ===")
print(f"  parts_ok={GLOBAL['ok']} parts_bad={GLOBAL['bad']}")
if GLOBAL['bins']:
    print(f"  total bins  = {GLOBAL['bins']:,}")
    print(f"  total docs  = {GLOBAL['docs']:,}")
    print(f"  docs/bin    = {GLOBAL['docs']/GLOBAL['bins']:.2f}")
    print(f"  fill rate   = {100*GLOBAL['tokens']/GLOBAL['capacity']:.2f}%")
PY

log "All done."
log "Topup outputs: ${ROOT}/{short,medium,long}_packed_64k_topup/"
