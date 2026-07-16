#!/usr/bin/env bash
# Auto-pipeline watcher for merged (llava_video_full + Molmo2) → 10 parts × 10 hosts × single round.
# s3+s4 only (merge_llava_molmo2_split10.py already emitted token_info/manifest).
#
# Host pool: 10 machines. Per user: replace .40 with .86.
#   {.32, .38, .41, .43, .44, .45, .46, .47, .49, .86}
# .49 lacks docker image llava_megatron:25.12 → fall back to 26.05.
#
# Run as:
#   nohup bash watcher_llava_molmo2_merged.sh > /tmp/watcher_llava_molmo2_merged.log 2>&1 &
#   tail -f /tmp/watcher_llava_molmo2_merged.log
set -euo pipefail

PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
OUT=/data/llava_video_full_molmo2_merged_packed_64k
HOSTS=(192.0.2.19 192.0.2.25 192.0.2.28 192.0.2.30 192.0.2.31 \
       192.0.2.32 192.0.2.33 192.0.2.34 192.0.2.35 192.0.2.40)
N_PARTS=10

SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR)
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

#============================================================
# Phase 0: sanity check ALL merged inputs
#============================================================
log "Phase 0: verify all ${N_PARTS} merged token_info+manifest exist."
missing=0
for i in $(seq 0 $((N_PARTS-1))); do
    d=${OUT}/part${i}
    [[ -f "$d/token_info.txt" ]] || { log "  MISSING: $d/token_info.txt"; missing=$((missing+1)); }
    [[ -f "$d/manifest.jsonl" ]] || { log "  MISSING: $d/manifest.jsonl"; missing=$((missing+1)); }
done
if (( missing > 0 )); then
    log "ERROR: ${missing} inputs missing. Run merge_llava_molmo2_split10.py first."; exit 1
fi
log "Phase 0: all inputs present."

#============================================================
# Phase 1: launch all 10 parts on 10 hosts (single round)
#============================================================
log "Phase 1: launch parts 0..$((N_PARTS-1)) on ${HOSTS[*]}"
for i in $(seq 0 $((N_PARTS-1))); do
    ip=${HOSTS[$i]}
    part=$i
    IMG_VAR="llava_megatron:25.12"
    if [[ "$ip" == "192.0.2.35" ]]; then IMG_VAR="llava_megatron:26.05"; fi
    log "  → ${ip} part=${part} (IMG=${IMG_VAR})"
    ssh "${SSH_OPTS[@]}" "$ip" "
        cd ${PIPE}
        export IMG=${IMG_VAR}
        nohup bash tools/run_llava_molmo2_merged_part.sh ${part} \
            > ${OUT}/host_\$(hostname).log 2>&1 &
        disown
    " &
done
wait
sleep 30
log "Phase 1: all ${N_PARTS} launched. Waiting for completion..."

while true; do
    n_done=0
    for i in $(seq 0 $((N_PARTS-1))); do
        [[ -f "${OUT}/part${i}/DONE.txt" ]] && n_done=$((n_done+1))
    done
    log "  progress: ${n_done}/${N_PARTS} done"
    [[ "$n_done" == "$N_PARTS" ]] && break
    sleep 300
done
log "Phase 1: all ${N_PARTS} parts done."

#============================================================
# Phase 2: validate + fill-rate report
#============================================================
log "Phase 2: validate merged webdataset + report fill rate."
python3 -u <<PY
import os, pickle, glob, json
OUT = "${OUT}"
N_PARTS = ${N_PARTS}
GLOBAL = {"bins":0, "docs":0, "tokens":0, "capacity":0, "ok":0, "bad":0,
          "llava_docs":0, "molmo_docs":0}
for i in range(N_PARTS):
    pdir = f"{OUT}/part{i}"
    bf = f"{pdir}/bins_64klen.pkl"
    wds = f"{pdir}/webdataset"
    if not os.path.exists(bf):
        print(f"  part{i}: MISSING bins"); GLOBAL["bad"] += 1; continue
    bins = pickle.load(open(bf,'rb'))
    bin_tokens = [sum(int(it[1]) for it in b) for b in bins]
    bin_docs   = [len(b) for b in bins]
    n = len(bins)
    tars = sorted(glob.glob(f"{wds}/*.tar"))
    idxs = sorted(glob.glob(f"{wds}/*.tar.idx"))
    if not tars or len(tars) != len(idxs):
        print(f"  part{i}: BAD tars={len(tars)} idx={len(idxs)}"); GLOBAL["bad"] += 1; continue
    fill = 100*sum(bin_tokens)/(n*64000) if n else 0
    dpb  = sum(bin_docs)/n if n else 0
    # Count llava vs molmo docs (name-based)
    llava_c = molmo_c = 0
    for b in bins:
        for it in b:
            nm = str(it[0])
            if nm.startswith('m2_'): molmo_c += 1
            else: llava_c += 1
    print(f"  part{i}: bins={n:>7,d}  docs={sum(bin_docs):>8,d}  docs/bin={dpb:>5.2f}  "
          f"fill={fill:>5.2f}%  llava={llava_c:>7,d} molmo={molmo_c:>5,d}  tars={len(tars)}")
    GLOBAL["bins"] += n
    GLOBAL["docs"] += sum(bin_docs)
    GLOBAL["tokens"] += sum(bin_tokens)
    GLOBAL["capacity"] += n * 64000
    GLOBAL["llava_docs"] += llava_c
    GLOBAL["molmo_docs"] += molmo_c
    GLOBAL["ok"] += 1
print(f"\n=== GLOBAL (llava + molmo merged, {N_PARTS} parts) ===")
print(f"  parts_ok={GLOBAL['ok']} bad={GLOBAL['bad']}")
if GLOBAL['bins']:
    print(f"  total bins  = {GLOBAL['bins']:,}")
    print(f"  total docs  = {GLOBAL['docs']:,}")
    print(f"    llava     = {GLOBAL['llava_docs']:,}")
    print(f"    molmo     = {GLOBAL['molmo_docs']:,}")
    print(f"  docs/bin    = {GLOBAL['docs']/GLOBAL['bins']:.2f}")
    print(f"  fill rate   = {100*GLOBAL['tokens']/GLOBAL['capacity']:.2f}%")
PY

log "All done. Merged pack: ${OUT}/"
