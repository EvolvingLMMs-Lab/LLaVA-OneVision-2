#!/bin/bash
# Top-up gt20min-768 bins with shortmix_3sets fillers -> new webdataset (higher fill).
# gt20min part${PIDX} (2-digit) <- shortmix part${IDX} (1-digit), 1:1 mapping.
set -euo pipefail
export PYTHONUNBUFFERED=1
IDX=${1:?"usage: $0 <part_idx 0..9>"}
PIDX=$(printf "%02d" ${IDX})
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
SRC=/ov21/pandas70m_gt20min_768frames_packed_64k/part${PIDX}
POOL=/ov21/shortmix_3sets_packed_64k/part${IDX}
OUT=/ov21/pandas70m_gt20min_768frames_packed_64k_shortmixtopup/part${PIDX}
IMG=${IMG:-llava_megatron:25.12}
CAP=64000
[[ -f "${SRC}/bins_64klen.pkl" ]] || { echo "MISSING bins: ${SRC}"; exit 2; }
[[ -f "${SRC}/manifest.jsonl" ]]  || { echo "MISSING manifest: ${SRC}"; exit 2; }
[[ -f "${POOL}/token_info.txt" ]] || { echo "MISSING pool tokeninfo: ${POOL}"; exit 2; }
[[ -f "${POOL}/manifest.jsonl" ]] || { echo "MISSING pool manifest: ${POOL}"; exit 2; }
mkdir -p "${OUT}"; LOG=${OUT}/run.log
echo "[smtopup-${IDX}] host=$(hostname) src=${SRC} pool=${POOL} out=${OUT}" | tee "${LOG}"
STATUS=${OUT}/.status; mkdir -p "${STATUS}"
source /root/miniconda3/etc/profile.d/conda.sh; conda activate llava_main_xy
BINS_TOPUP=${OUT}/bins_topup.pkl; SELECTED=${OUT}/selected_shortmix.txt
COMBINED=${OUT}/manifest_combined.jsonl; BLOCKLIST=${OUT}/blocklist.txt; : > "${BLOCKLIST}"
SEED=$((3000 + IDX))
# STEP A: inject shortmix fillers into slack
if [[ ! -f ${STATUS}/stepA.done ]]; then
  echo "[smtopup-${IDX}] STEP A: topup" | tee -a "$LOG"
  python ${PIPE}/s3b_topup_with_47m.py     --bins-in "${SRC}/bins_64klen.pkl" --bins-out "${BINS_TOPUP}"     --pool-tokeninfo-glob "${POOL}/token_info.txt" --pool-blocklist "${BLOCKLIST}"     --selected-out "${SELECTED}" --image-only --capacity ${CAP}     --max-samples 200 --max-patches 608822 --topup-budget-samples 200000 --seed ${SEED} 2>&1 | tee -a "$LOG"
  touch ${STATUS}/stepA.done
fi
# STEP B: combined manifest
if [[ ! -f ${STATUS}/stepB.done ]]; then
  echo "[smtopup-${IDX}] STEP B: combine manifest" | tee -a "$LOG"
  python ${PIPE}/tools/build_combined_manifest.py     --fix-manifest "${SRC}/manifest.jsonl" --selected "${SELECTED}"     --pool-manifest-glob "${POOL}/manifest.jsonl" --output "${COMBINED}" 2>&1 | tee -a "$LOG"
  touch ${STATUS}/stepB.done
fi
# STEP C: webdataset (docker)
WDS=${OUT}/webdataset; mkdir -p "${WDS}"
if [[ ! -f ${STATUS}/stepC.done ]]; then
  echo "[smtopup-${IDX}] STEP C: webdataset" | tee -a "$LOG"
  docker run --rm --name "ov2_smtopup_part${IDX}_$$"     -v /vlm:/vlm -v /ov2:/ov2 -v /ov21:/ov21 -v /rice_vl:/rice_vl     -w "${PIPE}" --entrypoint bash "${IMG}" -c "
      set -euo pipefail; export PYTHONUNBUFFERED=1; export PYTHONPATH=${PIPE}:\${PYTHONPATH:-}
      python s4_bins_to_webdataset.py --bins-file '${BINS_TOPUP}' --manifest '${COMBINED}'         --output-dir '${WDS}' --workers 24 --max-samples-per-shard 500         --shard-prefix 'gt20min768_shortmixtopup_part${IDX}_64k'         --sample-class-name PackedCaptioningSample --longest-edge 3072 --jpeg-quality 92
    " 2>&1 | tee -a "$LOG"
  touch ${STATUS}/stepC.done
fi
NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l); NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[smtopup-${IDX}] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"
if (( NTAR > 0 )) && (( NIDX == NTAR )); then touch "${OUT}/DONE.txt"; echo "[smtopup-${IDX}] DONE" | tee -a "$LOG"; else echo "[smtopup-${IDX}] INCOMPLETE" | tee -a "$LOG"; exit 4; fi
