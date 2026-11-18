#!/bin/bash
# Re-pack for one of 10 parts of merged (llava_video_full + Molmo2) dataset.
# Skips s1/s2 (uses pre-merged token_info + manifest from merge_llava_molmo2_split10.py).
# Only runs s3 (BFD bin packing) and s4 (bins -> webdataset).
#
# Usage:  bash run_llava_molmo2_merged_part.sh <part_idx 0..9>
#
# Input:  /ov21/llava_video_full_molmo2_merged_packed_64k/part${idx}/
#           {token_info.txt, manifest.jsonl}   ← pre-merged
# Output: /ov21/llava_video_full_molmo2_merged_packed_64k/part${idx}/
#           {bins_64klen.pkl, webdataset/*.tar+*.tar.idx, DONE.txt}
set -euo pipefail
export PYTHONUNBUFFERED=1

IDX=${1:?"usage: $0 <part_idx 0..9>"}
OUT=/ov21/llava_video_full_molmo2_merged_packed_64k/part${IDX}
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
IMG=${IMG:-llava_megatron:25.12}
MAXLEN=64000

MANIFEST=${OUT}/manifest.jsonl
TOKEN_INFO=${OUT}/token_info.txt
BINS=${OUT}/bins_64klen.pkl
WDS=${OUT}/webdataset
STATUS=${OUT}/.status; mkdir -p "$STATUS"

[[ -f "$MANIFEST" ]]   || { echo "MISSING: $MANIFEST"; exit 2; }
[[ -f "$TOKEN_INFO" ]] || { echo "MISSING: $TOKEN_INFO"; exit 2; }
LOG=${OUT}/run.log
echo "[merged-part${IDX}] host=$(hostname) manifest=$MANIFEST lines=$(wc -l <"$MANIFEST")" | tee "$LOG"

# --- glibc / BLAS hygiene ---
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llava_main_xy

# ---------- Step 3: bin packing (from merged token_info) ----------
if [[ ! -f ${STATUS}/step3.done ]]; then
    echo "[merged-${IDX}] STEP 3: bin packing capacity=${MAXLEN}" | tee -a "$LOG"
    python ${PIPE}/s3_bin_packing.py \
        --input "$TOKEN_INFO" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 200 \
        --algorithm bfd --patch-slack 0.30 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step3.done
fi

# ---------- Step 4: webdataset (docker) ----------
mkdir -p "$WDS"
if [[ ! -f ${STATUS}/step4.done ]]; then
    echo "[merged-${IDX}] STEP 4: bins -> webdataset (docker)" | tee -a "$LOG"
    docker run --rm \
        --name "ov2_llava_molmo2_merged_part${IDX}_$$" \
        -v /vlm:/vlm -v /ov2:/ov2 -v /ov21:/ov21 -v /rice_vl:/rice_vl \
        -w "${PIPE}" --entrypoint bash \
        "${IMG}" -c "
            set -euo pipefail
            export PYTHONUNBUFFERED=1
            export PYTHONPATH=${PIPE}:\${PYTHONPATH:-}
            python s4_bins_to_webdataset.py \
                --bins-file '${BINS}' \
                --manifest  '${MANIFEST}' \
                --output-dir '${WDS}' \
                --workers 16 \
                --max-samples-per-shard 500 \
                --shard-prefix 'llava_molmo2_merged_part${IDX}_64k' \
                --sample-class-name PackedCaptioningSample \
                --longest-edge 3072 --jpeg-quality 92
        " 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step4.done
fi

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[merged-${IDX}] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[merged-${IDX}] DONE at $(date)" | tee -a "$LOG"
else
    echo "[merged-${IDX}] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
