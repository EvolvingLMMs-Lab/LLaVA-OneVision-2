#!/bin/bash
# Per-host packing for Timelens_temporal_grounding_merged_p16_f3.jsonl (19,466 records, 1 part).
# Schema: messages(2 turns) + images(17..128 frames, mean 123) + shared patch_positions.npy + fps.
# Very light per record (mean 6.6k tokens, max 8.1k), so ~2k bins @ 99% fill.
#
# Usage:  bash run_timelens.sh
#
# Output: /ov21/Timelens_temporal_grounding_packed_64k/
set -euo pipefail
export PYTHONUNBUFFERED=1

SRC=/ov21/data_jsonl/Timelens_temporal_grounding_merged_p16_f3.jsonl
OUT=/ov21/Timelens_temporal_grounding_packed_64k
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
IMG=${IMG:-llava_megatron:25.12}
MAXLEN=64000

[[ -f "$SRC" ]] || { echo "MISSING: $SRC"; exit 2; }
mkdir -p "$OUT"
LOG=${OUT}/run.log
echo "[timelens] host=$(hostname) src=$SRC out=$OUT lines=$(wc -l <"$SRC")" | tee "$LOG"

# Memory guard
nohup bash ${PIPE}/tools/mem_guard.sh 0 92 \
    "run_timelens.sh" \
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

# --- s2 caps (light data: mean 123 frames, but NPY fast-path applies) ---
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

# ---------- Step 1: build manifest ----------
if [[ ! -f ${STATUS}/step1.done ]]; then
    echo "[timelens] STEP 1: build manifest" | tee -a "$LOG"
    python ${PIPE}/s1_build_manifest.py \
        -i "$SRC" --image-root / -o "$MANIFEST" --overwrite 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step1.done
fi

# ---------- Step 2: compute token lengths ----------
if [[ ! -f ${STATUS}/step2.done ]]; then
    echo "[timelens] STEP 2: token lengths (cap=${MAXLEN})" | tee -a "$LOG"
    python ${PIPE}/s2_compute_token_lengths.py \
        --manifest "$MANIFEST" --output "$TOKEN_INFO" \
        --model-path "$MODEL" --max-len ${MAXLEN} --task-type sft \
        --chunk-size 500 --min-workers 4 --max-workers 16 --timeout 600 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step2.done
fi

# ---------- Step 3: bin packing ----------
if [[ ! -f ${STATUS}/step3.done ]]; then
    echo "[timelens] STEP 3: bin packing capacity=${MAXLEN}" | tee -a "$LOG"
    python ${PIPE}/s3_bin_packing.py \
        --input "$TOKEN_INFO" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 200 \
        --algorithm bfd --patch-slack 0.30 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step3.done
fi

# ---------- Step 4: webdataset (docker) ----------
mkdir -p "$WDS"
if [[ ! -f ${STATUS}/step4.done ]]; then
    echo "[timelens] STEP 4: bins -> webdataset (docker)" | tee -a "$LOG"
    docker run --rm \
        --name "ov2_timelens_$$" \
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
                --shard-prefix 'timelens_64k' \
                --sample-class-name PackedCaptioningSample \
                --longest-edge 3072 --jpeg-quality 92
        " 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step4.done
fi

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[timelens] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[timelens] DONE at $(date)" | tee -a "$LOG"
else
    echo "[timelens] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
