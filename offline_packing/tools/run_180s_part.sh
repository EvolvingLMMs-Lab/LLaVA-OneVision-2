#!/bin/bash
# Per-host packing for one all_180s_96w_aa_ao_1fps_m384/jsonl_done/*.jsonl shard.
# Multi-image (181-frame video) + shared patch_positions.npy schema.
#
# Usage:  bash run_180s_part.sh <part_idx>     (part_idx in 0..9)
#
# Output:  /ov21/all_180s_96w_packed_64k/180s_part${idx}/
#
# Anti-OOM hardening (these are 1 TB / 128 core machines, but we're cautious):
#   * S2_MAX_PROCESSES=32 (half of the 47M default; videos load 181x more bytes)
#   * S2_MEM_PER_PROC_GB=4 (double the 47M baseline; PIL of multi-image bigger)
#   * mem_guard.sh runs alongside, kill PGID if MemAvail-real-used > 92% x3
#   * S2_MAXTASKS_PER_CHILD=5 (recycle workers more often than 47M)
#   * NO_NPY=0 (use the .npy directly via npy_first_dim path → cheap)
set -euo pipefail
export PYTHONUNBUFFERED=1

IDX=${1:?"usage: $0 <part_idx 0..9>"}
PIDX=$(printf "%02d" ${IDX})
SRC=/ov21/dataset_frame/all_180s_96w_aa_ao_1fps_m384/jsonl_done/all_180s_96w_done_part${PIDX}.jsonl
OUT=/ov21/all_180s_96w_packed_64k/180s_part${IDX}
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
IMG=${IMG:-llava_megatron:25.12}
MAXLEN=64000

[[ -f "$SRC" ]] || { echo "MISSING: $SRC"; exit 2; }
mkdir -p "$OUT"
LOG=${OUT}/run.log
echo "[180s-part${IDX}] host=$(hostname) src=$SRC out=$OUT lines=$(wc -l <"$SRC")" | tee "$LOG"

# Memory guard in background. mem_guard kills our PGID if memory pressure
# stays high for 3 consecutive 15s windows. We pass our own pattern + done_file
# so the guard tracks the right process and stops once token_info.txt exists.
nohup bash ${PIPE}/tools/mem_guard.sh ${IDX} 92 \
    "run_180s_part.sh ${IDX}" \
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

# --- s2 caps (conservative for video data; 1 TB host but big PIL buffers) ---
export S2_MAX_PROCESSES=${S2_MAX_PROCESSES:-32}
export S2_MAX_THREADS_PER_PROC=${S2_MAX_THREADS_PER_PROC:-4}
export S2_MEM_PER_PROC_GB=${S2_MEM_PER_PROC_GB:-4}
export S2_MAXTASKS_PER_CHILD=${S2_MAXTASKS_PER_CHILD:-5}
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
    echo "[180s-${IDX}] STEP 1: build manifest" | tee -a "$LOG"
    python ${PIPE}/s1_build_manifest.py \
        -i "$SRC" --image-root / -o "$MANIFEST" --overwrite 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step1.done
fi

# ---------- Step 2: compute token lengths ----------
if [[ ! -f ${STATUS}/step2.done ]]; then
    echo "[180s-${IDX}] STEP 2: token lengths (cap=${MAXLEN})" | tee -a "$LOG"
    python ${PIPE}/s2_compute_token_lengths.py \
        --manifest "$MANIFEST" --output "$TOKEN_INFO" \
        --model-path "$MODEL" --max-len ${MAXLEN} --task-type sft \
        --chunk-size 500 --min-workers 4 --max-workers 32 --timeout 300 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step2.done
fi

# ---------- Step 3: bin packing ----------
if [[ ! -f ${STATUS}/step3.done ]]; then
    echo "[180s-${IDX}] STEP 3: bin packing capacity=${MAXLEN}" | tee -a "$LOG"
    python ${PIPE}/s3_bin_packing.py \
        --input "$TOKEN_INFO" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 200 \
        --algorithm bfd --patch-slack 0.30 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step3.done
fi

# ---------- Step 4: webdataset (docker for energon/.idx) ----------
mkdir -p "$WDS"
[[ ! -f ${STATUS}/step4.done ]] && {
    echo "[180s-${IDX}] STEP 4: bins -> webdataset (docker)" | tee -a "$LOG"
    docker run --rm \
        --name "ov2_180s_part${IDX}_$$" \
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
                --workers 32 \
                --max-samples-per-shard 1000 \
                --shard-prefix '180s_part${IDX}_64k' \
                --sample-class-name PackedCaptioningSample \
                --longest-edge 3072 --jpeg-quality 92
        " 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step4.done
}

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[180s-${IDX}] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[180s-${IDX}] DONE at $(date)" | tee -a "$LOG"
else
    echo "[180s-${IDX}] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
