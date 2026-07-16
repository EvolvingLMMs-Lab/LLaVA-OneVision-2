#!/bin/bash
# Re-run Step 4 (Bins -> WebDataset) inside the megatron docker image so that
# the energon/webdataset stack produces .tar.idx files alongside each .tar.
#
# The bare-metal run lacked the megatron-energon package, so .tar.idx files
# were missing. Bins/manifest from the earlier run are reused as-is.
#
# Usage:  bash run_47m_s4_docker.sh <part_idx>   (part_idx ∈ 1..7)
set -euo pipefail

IDX=${1:?"usage: $0 <part_idx 1..7>"}
IMG=${IMG:-llava_megatron:25.12}
OUT=/data/47m_sft_packed/47m_part${IDX}
PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33

BINS=${OUT}/bins_32klen.pkl
MANIFEST=${OUT}/manifest.jsonl
WDS=${OUT}/webdataset

[[ -f "$BINS" ]]     || { echo "MISSING: $BINS"; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "MISSING: $MANIFEST"; exit 2; }

# Sanity: webdataset/ must not collide with a previous run. If user left a
# stale dir, abort instead of mixing fresh tars with old residue.
# Exception: if --force is passed, wipe and re-run.
FORCE=${FORCE:-0}
if [[ -d "$WDS" ]] && ls "$WDS"/*.tar 1>/dev/null 2>&1; then
    if [[ "$FORCE" == "1" ]]; then
        echo "[part${IDX}] FORCE=1: wiping existing $WDS"
        rm -rf "$WDS"
    else
        echo "REFUSE: $WDS already contains *.tar — move it aside first (or FORCE=1)"
        exit 3
    fi
fi
mkdir -p "$WDS"

# DONE.txt from the earlier (broken, no-idx) run must be cleared so downstream
# pipelines don't mistake the half-baked state for a fresh complete one.
rm -f "${OUT}/DONE.txt"

LOG=${OUT}/s4_docker.log
echo "[part${IDX}] host=$(hostname) image=${IMG} -> ${WDS}" | tee "$LOG"

docker run --rm \
    --name "ov2_s4_part${IDX}_$$" \
    -v /vlm:/vlm -v /ov2:/ov2 -v /data:/data -v /rice_vl:/rice_vl \
    -w "${PIPE}" \
    --entrypoint bash \
    "${IMG}" -c "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=${PIPE}:\${PYTHONPATH:-}
        python s4_bins_to_webdataset.py \
            --bins-file '${BINS}' \
            --manifest  '${MANIFEST}' \
            --output-dir '${WDS}' \
            --workers 32 \
            --max-samples-per-shard 5000 \
            --shard-prefix '47m_part${IDX}' \
            --sample-class-name PackedCaptioningSample \
            --longest-edge 3072 \
            --jpeg-quality 92
    " 2>&1 | tee -a "$LOG"

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
echo "[part${IDX}] tars=${NTAR} idx=${NIDX}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[part${IDX}] DONE at $(date)" | tee -a "$LOG"
else
    echo "[part${IDX}] INCOMPLETE (tars=${NTAR} idx=${NIDX}); leaving DONE.txt absent" | tee -a "$LOG"
    exit 4
fi
