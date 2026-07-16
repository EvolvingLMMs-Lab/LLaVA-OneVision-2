#!/usr/bin/env bash
# Auto-pipeline watcher for llava_video_full (5 parts × 5 hosts × single round).
#   Source: /data/data_jsonl/llava_video_full/jsonl_done_split5/
#   Output: /data/llava_video_full_packed_64k/
#   Hosts:  5 picks from the 10-host pool, EXCLUDING 192.0.2.27 per user request.
#
# Run as:
#   nohup bash watcher_llava_video_full.sh > /tmp/watcher_llava_video_full.log 2>&1 &
#   tail -f /tmp/watcher_llava_video_full.log
set -euo pipefail

PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
SRC_ROOT=/data/data_jsonl/llava_video_full/jsonl_done_split5
PACK_OUT=/data/llava_video_full_packed_64k
# 5 hosts (NO .40). Pick 5 from {.32, .38, .41, .43, .44, .45, .46, .47, .49}.
HOSTS=(192.0.2.19 192.0.2.25 192.0.2.28 192.0.2.30 192.0.2.31)
N_PARTS=5

mkdir -p "${PACK_OUT}"
SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR)

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

#============================================================
# Phase 0: sanity check inputs
#============================================================
log "Phase 0: verify all ${N_PARTS} split jsonl exist."
missing=0
for i in $(seq 0 $((N_PARTS-1))); do
    f="${SRC_ROOT}/llava_video_full_part${i}.jsonl"
    if [[ ! -f "$f" ]]; then
        log "  MISSING: $f"
        missing=$((missing+1))
    fi
done
if (( missing > 0 )); then
    log "ERROR: ${missing} inputs missing. Abort."; exit 1
fi
log "Phase 0: all ${N_PARTS} inputs present."

#============================================================
# Phase 1: launch all 5 parts on 5 hosts (single round)
#============================================================
log "Phase 1: launch parts 0..$((N_PARTS-1)) on ${HOSTS[*]}"
for i in $(seq 0 $((N_PARTS-1))); do
    ip=${HOSTS[$i]}
    part=$i
    IMG_VAR="llava_megatron:25.12"
    # 5.49 docker proxy issue → use 26.05 (kept here in case of future host swap)
    if [[ "$ip" == "192.0.2.35" ]]; then IMG_VAR="llava_megatron:26.05"; fi
    log "  → ${ip} part=${part} (IMG=${IMG_VAR})"
    ssh "${SSH_OPTS[@]}" "$ip" "
        cd ${PIPE}
        export IMG=${IMG_VAR}
        nohup bash tools/run_llava_video_full_part.sh ${part} \
            > ${PACK_OUT}/host_\$(hostname).log 2>&1 &
        disown
    " &
done
wait
sleep 30
log "Phase 1: all 5 launched. Waiting for completion..."

while true; do
    n_done=0
    for i in $(seq 0 $((N_PARTS-1))); do
        done_file="${PACK_OUT}/llava_video_full_part${i}/DONE.txt"
        [[ -f "$done_file" ]] && n_done=$((n_done+1))
    done
    log "  progress: ${n_done}/${N_PARTS} done"
    [[ "$n_done" == "$N_PARTS" ]] && break
    sleep 300
done
log "Phase 1: all ${N_PARTS} parts done."

#============================================================
# Phase 2: validate and report fill rate
#============================================================
log "Phase 2: validate webdataset outputs and report bin fill rate."
python3 -u <<PY
import os, pickle, glob
total_bins = 0
total_tokens = 0
total_capacity = 0
total_docs = 0
parts_ok = 0
parts_bad = 0
for i in range(${N_PARTS}):
    pdir = f"${PACK_OUT}/llava_video_full_part{i}"
    bf = f"{pdir}/bins_64klen.pkl"
    wds = f"{pdir}/webdataset"
    if not os.path.exists(bf):
        print(f"  part{i}: MISSING bins"); parts_bad += 1; continue
    bins = pickle.load(open(bf, 'rb'))
    # bins is List[List[ndarray]] (each item a numpy void with (col0, tokens, name, patches))
    bin_tokens = []
    bin_docs = []
    for b in bins:
        toks = sum(int(it[1]) for it in b)
        bin_tokens.append(toks)
        bin_docs.append(len(b))
    n_bins = len(bins)
    total_bins += n_bins
    total_tokens += sum(bin_tokens)
    total_capacity += n_bins * 64000
    total_docs += sum(bin_docs)
    tars = sorted(glob.glob(f"{wds}/*.tar"))
    idxs = sorted(glob.glob(f"{wds}/*.tar.idx"))
    if not tars or len(tars) != len(idxs):
        print(f"  part{i}: BAD tars={len(tars)} idx={len(idxs)}"); parts_bad += 1; continue
    fill = 100*sum(bin_tokens)/(n_bins*64000)
    docs_per_bin = sum(bin_docs)/n_bins
    print(f"  part{i}: bins={n_bins:>7,d}  docs={sum(bin_docs):>8,d}  "
          f"docs/bin={docs_per_bin:>5.2f}  fill={fill:>5.2f}%  "
          f"tars={len(tars)}  idx={len(idxs)}")
    parts_ok += 1
if total_bins:
    print(f"\n=== GLOBAL ===")
    print(f"  parts_ok       = {parts_ok}")
    print(f"  parts_bad      = {parts_bad}")
    print(f"  total bins     = {total_bins:,}")
    print(f"  total docs     = {total_docs:,}")
    print(f"  docs / bin     = {total_docs/total_bins:.2f}")
    print(f"  token fill rate= {100*total_tokens/total_capacity:.2f}%")
PY

log "All done. Pipeline complete."
log "Output: ${PACK_OUT}"
