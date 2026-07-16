#!/bin/bash
# Per-part top-up for fix_recap: inject 47m image samples into the slack space
# of fix_recap bins, then run s4 to write a new webdataset.
#
# Inputs:
#   /ov21/pandas70m_gt10min_packed_64k_fix_recap/pandas70m_part${IDX}/bins_64klen.pkl
#   /ov21/pandas70m_gt10min_packed_64k_fix_recap/pandas70m_part${IDX}/manifest.jsonl
# Outputs:
#   /ov21/pandas70m_gt10min_packed_64k_fix_recap_with47m/pandas70m_part${IDX}/
set -euo pipefail
export PYTHONUNBUFFERED=1

IDX=${1:?"usage: $0 <part_idx 0..19>"}
PIDX=$(printf "%02d" ${IDX})

PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
SRC=/ov21/pandas70m_gt10min_packed_64k_fix_recap/pandas70m_part${IDX}
OUT=/ov21/pandas70m_gt10min_packed_64k_fix_recap_with47m/pandas70m_part${IDX}
IMG=${IMG:-llava_megatron:25.12}
CAP=64000

[[ -f "${SRC}/bins_64klen.pkl" ]]  || { echo "MISSING bins: ${SRC}/bins_64klen.pkl"; exit 2; }
[[ -f "${SRC}/manifest.jsonl" ]]   || { echo "MISSING manifest: ${SRC}/manifest.jsonl"; exit 2; }
[[ -f "${SRC}/DONE.txt" ]]         || { echo "fix_recap part not finished: ${SRC}/DONE.txt missing"; exit 2; }

mkdir -p "${OUT}"
LOG=${OUT}/run.log
echo "[topup-recap-${IDX}] host=$(hostname) src=${SRC} out=${OUT}" | tee "${LOG}"

STATUS=${OUT}/.status; mkdir -p "${STATUS}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llava_main_xy

BINS_TOPUP=${OUT}/bins_topup.pkl
SELECTED=${OUT}/selected_47m.txt
COMBINED=${OUT}/manifest_combined.jsonl
BLOCKLIST=${OUT}/blocklist.txt
: > "${BLOCKLIST}"
# Seed differs from fix_with47m's seed (1000+IDX) to keep two top-up datasets distinct.
SEED=$((2000 + IDX))

# ---------- Step A: top-up bins ----------
if [[ ! -f ${STATUS}/stepA.done ]]; then
    echo "[topup-recap-${IDX}] STEP A: inject 47m image samples into bin slack" | tee -a "$LOG"
    python ${PIPE}/s3b_topup_with_47m.py \
        --bins-in "${SRC}/bins_64klen.pkl" \
        --bins-out "${BINS_TOPUP}" \
        --pool-tokeninfo-glob '/ov21/47m_sft_packed_64k/47m_part*/token_info.txt' \
        --pool-blocklist "${BLOCKLIST}" \
        --selected-out "${SELECTED}" \
        --image-only \
        --capacity ${CAP} \
        --max-samples 200 \
        --max-patches 608822 \
        --topup-budget-samples 600000 \
        --seed ${SEED} 2>&1 | tee -a "$LOG"
    touch ${STATUS}/stepA.done
fi

# ---------- Step B: build combined manifest (fix_recap + selected 47m) ----------
if [[ ! -f ${STATUS}/stepB.done ]]; then
    echo "[topup-recap-${IDX}] STEP B: build combined manifest" | tee -a "$LOG"
    python ${PIPE}/tools/build_combined_manifest.py \
        --fix-manifest "${SRC}/manifest.jsonl" \
        --selected "${SELECTED}" \
        --pool-manifest-glob '/ov21/47m_sft_packed/47m_part*/manifest.jsonl' \
        --output "${COMBINED}" 2>&1 | tee -a "$LOG"
    touch ${STATUS}/stepB.done
fi

# ---------- Step C: webdataset ----------
WDS=${OUT}/webdataset
mkdir -p "${WDS}"
if [[ ! -f ${STATUS}/stepC.done ]]; then
    echo "[topup-recap-${IDX}] STEP C: bins -> webdataset (docker)" | tee -a "$LOG"
    docker run --rm \
        --name "ov2_p70m_recap_topup_part${IDX}_$$" \
        -v /vlm:/vlm -v /ov2:/ov2 -v /ov21:/ov21 -v /rice_vl:/rice_vl \
        -w "${PIPE}" --entrypoint bash \
        "${IMG}" -c "
            set -euo pipefail
            export PYTHONUNBUFFERED=1
            export PYTHONPATH=${PIPE}:\${PYTHONPATH:-}
            python s4_bins_to_webdataset.py \
                --bins-file '${BINS_TOPUP}' \
                --manifest  '${COMBINED}' \
                --output-dir '${WDS}' \
                --workers 16 \
                --max-samples-per-shard 500 \
                --shard-prefix 'pandas70m_recap_topup_part${IDX}_64k' \
                --sample-class-name PackedCaptioningSample \
                --longest-edge 3072 --jpeg-quality 92
        " 2>&1 | tee -a "$LOG"
    touch ${STATUS}/stepC.done
fi

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[topup-recap-${IDX}] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[topup-recap-${IDX}] DONE at $(date)" | tee -a "$LOG"
else
    echo "[topup-recap-${IDX}] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
