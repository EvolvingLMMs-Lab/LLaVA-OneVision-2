#!/bin/bash
# Re-pack the 47M SFT dataset with 64k token cap, reusing the existing
# s2 stage0_persistent/ files from the 32k run (raw token counts are
# independent of max-len; the filter is applied at merge time).
#
# Steps:
#   1. merge stage0_chunk_*.txt -> token_info_64k.txt (filter > 64000)
#   2. s3 bin packing with capacity=64000
#   3. s4 bins -> webdataset inside docker (for .tar.idx)
#
# Usage: bash run_47m_64k.sh <part_idx 1..7>
set -euo pipefail
export PYTHONUNBUFFERED=1

IDX=${1:?"usage: $0 <part_idx 1..7>"}
IMG=${IMG:-llava_megatron:25.12}
OUT=/data/47m_sft_packed_8k/47m_part${IDX}
SRC_OUT=/data/47m_sft_packed/47m_part${IDX}
PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
MAXLEN=8000

[[ -d "${SRC_OUT}/stage0_persistent" ]] || { echo "MISSING: ${SRC_OUT}/stage0_persistent"; exit 2; }
[[ -f "${SRC_OUT}/manifest.jsonl" ]]   || { echo "MISSING: ${SRC_OUT}/manifest.jsonl"; exit 2; }

mkdir -p "$OUT"
LOG=${OUT}/run.log
echo "[part${IDX}-8k] host=$(hostname) out=$OUT" | tee "$LOG"

MANIFEST=${OUT}/manifest.jsonl
TOKEN_INFO=${OUT}/token_info.txt
BINS=${OUT}/bins_8klen.pkl
WDS=${OUT}/webdataset

# Symlink the manifest (8.6 GB; no need to copy).
[[ -L "$MANIFEST" || -f "$MANIFEST" ]] || ln -s "${SRC_OUT}/manifest.jsonl" "$MANIFEST"

# ---------- Step 2': re-merge stage0 with new cap ----------
# stage0 lines look like:  name:token_len:patch_count
# Sort by token_len ascending, keep only token_len <= MAXLEN.
if [[ ! -s "$TOKEN_INFO" ]]; then
    echo "[part${IDX}-8k] re-merging stage0 (cap=${MAXLEN})..." | tee -a "$LOG"
    # Concatenation order doesn't matter — awk filters and sort handles ordering.
    # Use LC_ALL=C for fast byte-wise sort. Sort by 2nd colon-separated field
    # (numeric ascending).
    LC_ALL=C awk -F: -v m=${MAXLEN} 'NF>=3 && $2+0<=m {print $0}' \
        ${SRC_OUT}/stage0_persistent/stage0_chunk_*.txt | \
        LC_ALL=C sort -t: -k2,2n > "$TOKEN_INFO"
    KEPT=$(wc -l < "$TOKEN_INFO")
    TOTAL=$(cat ${SRC_OUT}/stage0_persistent/stage0_chunk_*.txt | wc -l)
    echo "[part${IDX}-8k] token_info: kept=${KEPT}/${TOTAL} (filtered $((TOTAL-KEPT)))" | tee -a "$LOG"
fi

# ---------- Step 3: bin packing ----------
if [[ ! -f "$BINS" ]]; then
    echo "[part${IDX}-8k] STEP 3: bin packing capacity=${MAXLEN}" | tee -a "$LOG"
    source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate llava_main_xy 2>/dev/null || true
    python ${PIPE}/s3_bin_packing.py \
        --input "$TOKEN_INFO" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 200 \
        --algorithm bfd --patch-slack 0.30 2>&1 | tee -a "$LOG"
fi

# ---------- Step 4: bins -> webdataset (docker) ----------
mkdir -p "$WDS"
echo "[part${IDX}-8k] STEP 4: bins -> webdataset (docker)" | tee -a "$LOG"
docker run --rm \
    --name "ov2_8k_part${IDX}_$$" \
    -v /vlm:/vlm -v /ov2:/ov2 -v /data:/data -v /rice_vl:/rice_vl \
    -w "${PIPE}" --entrypoint bash \
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
            --shard-prefix '47m_part${IDX}_8k' \
            --sample-class-name PackedCaptioningSample \
            --longest-edge 3072 --jpeg-quality 92
    " 2>&1 | tee -a "$LOG"

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
echo "[part${IDX}-8k] tars=${NTAR} idx=${NIDX} size=$(du -sh $WDS|awk '{print $1}')" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[part${IDX}-8k] DONE at $(date)" | tee -a "$LOG"
else
    echo "[part${IDX}-8k] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
