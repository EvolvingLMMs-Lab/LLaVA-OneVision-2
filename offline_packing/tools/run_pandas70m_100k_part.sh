#!/bin/bash
# Per-host packing for one pandas70m_gt10min_100k_10to20min_plus_gt20min_1fps_m768_320x320
# shard. Multi-image (up to 768-frame video at 1 fps) + shared patch_positions.npy.
# Frames are small (320x320 max) → patch budget per frame is tiny but frame count
# can be huge.
#
# Usage:  bash run_pandas70m_100k_part.sh <part_idx>     (part_idx in 0..11)
#
# Output:  /ov21/pandas70m_gt10min_100k_packed_64k/p70m100k_part${idx}/
#
# Anti-OOM hardening (1 TB / 128 core machines; the per-sample byte budget is
# similar to pandas70m_gt10min/384f even though there can be 768 frames, because
# each frame is only 320x320 vs 480x854 for the 384f set):
#   * S2_MAX_PROCESSES=16 (be cautious — up to 768 PIL handles per worker)
#   * S2_MEM_PER_PROC_GB=8 (double 180s baseline; lots of small PIL handles)
#   * mem_guard.sh runs alongside, kill PGID if MemAvail-real-used > 92% x3
#   * S2_MAXTASKS_PER_CHILD=3 (recycle workers more often than 180s)
#   * NO_NPY=0 (use the .npy directly via npy_first_dim path → cheap)
set -euo pipefail
export PYTHONUNBUFFERED=1

IDX=${1:?"usage: $0 <part_idx 0..11>"}
PIDX=$(printf "%02d" ${IDX})
SRC=/ov21/dataset_frame/pandas70m_gt10min_100k_10to20min_plus_gt20min_1fps_m768_320x320/jsonl_done/pandas70m_selected_done_part${PIDX}.jsonl
OUT=/ov21/pandas70m_gt10min_100k_packed_64k/p70m100k_part${IDX}
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
IMG=${IMG:-llava_megatron:25.12}
MAXLEN=64000

[[ -f "$SRC" ]] || { echo "MISSING: $SRC"; exit 2; }
mkdir -p "$OUT"
LOG=${OUT}/run.log
echo "[p70m100k-part${IDX}] host=$(hostname) src=$SRC out=$OUT lines=$(wc -l <"$SRC")" | tee "$LOG"

# Memory guard in background. mem_guard kills our PGID if memory pressure
# stays high for 3 consecutive 15s windows. We pass our own pattern + done_file
# so the guard tracks the right process and stops once token_info.txt exists.
nohup bash ${PIPE}/tools/mem_guard.sh ${IDX} 92 \
    "run_pandas70m_100k_part.sh ${IDX}" \
    "${OUT}/token_info.txt" \
    "${OUT}" \
    > ${OUT}/mem_guard.log 2>&1 &
GUARD_PID=$!
trap "kill ${GUARD_PID} 2>/dev/null || true" EXIT

# --- glibc / BLAS hygiene ---
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- s2 caps (very conservative; up to 768 small 320x320 frames per sample) ---
export S2_MAX_PROCESSES=${S2_MAX_PROCESSES:-16}
export S2_MAX_THREADS_PER_PROC=${S2_MAX_THREADS_PER_PROC:-4}
export S2_MEM_PER_PROC_GB=${S2_MEM_PER_PROC_GB:-8}
export S2_MAXTASKS_PER_CHILD=${S2_MAXTASKS_PER_CHILD:-3}
export S2_USE_SPAWN=${S2_USE_SPAWN:-1}
export FAST_TOKENIZER=${FAST_TOKENIZER:-1}
export FAST_IMG_META=${FAST_IMG_META:-1}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llava_main_xy

MANIFEST=${OUT}/manifest.jsonl
TOKEN_INFO=${OUT}/token_info.txt
BINS=${OUT}/bins_64klen.pkl
WDS=${OUT}/webdataset
STATUS=${OUT}/.status; mkdir -p "$STATUS"

# ---------- Step 1: build manifest (all paths absolute, no copying) ----------
if [[ ! -f ${STATUS}/step1.done ]]; then
    echo "[p70m100k-${IDX}] STEP 1: build manifest" | tee -a "$LOG"
    python ${PIPE}/s1_build_manifest.py \
        -i "$SRC" --image-root / -o "$MANIFEST" --overwrite 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step1.done
fi

# ---------- Step 2: compute token lengths ----------
if [[ ! -f ${STATUS}/step2.done ]]; then
    echo "[p70m100k-${IDX}] STEP 2: token lengths (cap=${MAXLEN})" | tee -a "$LOG"
    python ${PIPE}/s2_compute_token_lengths.py \
        --manifest "$MANIFEST" --output "$TOKEN_INFO" \
        --model-path "$MODEL" --max-len ${MAXLEN} --task-type sft \
        --chunk-size 500 --min-workers 4 --max-workers 16 --timeout 600 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step2.done
fi

# ---------- Step 3: bin packing ----------
if [[ ! -f ${STATUS}/step3.done ]]; then
    echo "[p70m100k-${IDX}] STEP 3: bin packing capacity=${MAXLEN}" | tee -a "$LOG"
    python ${PIPE}/s3_bin_packing.py \
        --input "$TOKEN_INFO" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 200 \
        --algorithm bfd --patch-slack 0.30 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step3.done
fi

# ---------- Step 4: webdataset (docker for energon/.idx) ----------
mkdir -p "$WDS"
[[ ! -f ${STATUS}/step4.done ]] && {
    echo "[p70m100k-${IDX}] STEP 4: bins -> webdataset (docker)" | tee -a "$LOG"
    docker run --rm \
        --name "ov2_p70m100k_part${IDX}_$$" \
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
                --shard-prefix 'pandas70m_100k_part${IDX}_64k' \
                --sample-class-name PackedCaptioningSample \
                --longest-edge 3072 --jpeg-quality 92
        " 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step4.done
}

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[p70m100k-${IDX}] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[p70m100k-${IDX}] DONE at $(date)" | tee -a "$LOG"
else
    echo "[p70m100k-${IDX}] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
