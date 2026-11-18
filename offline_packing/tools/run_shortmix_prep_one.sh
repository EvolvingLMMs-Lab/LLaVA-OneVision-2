#!/bin/bash
# Per-dataset s1+s2 preparation for the short-video mix (flash_non_tg + flash_v2 + molmo2_short).
# Writes manifest.jsonl + token_info.txt only (no bin packing, no webdataset).
# This lets us later merge all 3 token_info+manifest into 10 balanced parts.
#
# Usage:  bash run_shortmix_prep_one.sh <dataset_name>
#   dataset_name in {flash_non_tg, flash_v2, molmo2_short}
#
# Output: /data/shortmix_prep/<dataset_name>/{manifest.jsonl, token_info.txt, .status/}
set -euo pipefail
export PYTHONUNBUFFERED=1

DS=${1:?"usage: $0 <flash_non_tg|flash_v2|molmo2_short>"}
case "$DS" in
    flash_non_tg) SRC=/data/data_jsonl/flash_video_non_temporal_grounding_short_p16_f3.jsonl ;;
    flash_v2)     SRC=/data/data_jsonl/flash_video_v2_short_p16_f3.jsonl ;;
    molmo2_short) SRC=/data/data_jsonl/Molmo2_videoforsft_short_p16_f3.jsonl ;;
    *) echo "BAD DS: $DS"; exit 2 ;;
esac

OUT=/data/shortmix_prep/${DS}
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
MAXLEN=64000

[[ -f "$SRC" ]] || { echo "MISSING: $SRC"; exit 2; }
mkdir -p "$OUT"
LOG=${OUT}/run.log
echo "[shortmix-prep-${DS}] host=$(hostname) src=$SRC out=$OUT lines=$(wc -l <"$SRC")" | tee "$LOG"

# Memory guard
nohup bash ${PIPE}/tools/mem_guard.sh 0 92 \
    "run_shortmix_prep_one.sh ${DS}" \
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

# --- s2 caps: short videos are LIGHT (mean 21-32 frames, mean 1500-4000 tokens).
# Can use MORE processes since each is cheap.
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
STATUS=${OUT}/.status; mkdir -p "$STATUS"

# ---------- Step 1: build manifest ----------
if [[ ! -f ${STATUS}/step1.done ]]; then
    echo "[shortmix-prep-${DS}] STEP 1: build manifest" | tee -a "$LOG"
    python ${PIPE}/s1_build_manifest.py \
        -i "$SRC" --image-root / -o "$MANIFEST" --overwrite 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step1.done
fi

# ---------- Step 2: compute token lengths ----------
if [[ ! -f ${STATUS}/step2.done ]]; then
    echo "[shortmix-prep-${DS}] STEP 2: token lengths (cap=${MAXLEN})" | tee -a "$LOG"
    python ${PIPE}/s2_compute_token_lengths.py \
        --manifest "$MANIFEST" --output "$TOKEN_INFO" \
        --model-path "$MODEL" --max-len ${MAXLEN} --task-type sft \
        --chunk-size 1000 --min-workers 8 --max-workers 32 --timeout 600 2>&1 | tee -a "$LOG"
    touch ${STATUS}/step2.done
fi

echo "[shortmix-prep-${DS}] DONE at $(date)  manifest=$(wc -l <"$MANIFEST") token_info=$(wc -l <"$TOKEN_INFO")" | tee -a "$LOG"
touch "${OUT}/DONE.txt"
