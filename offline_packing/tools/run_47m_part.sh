#!/bin/bash
# Per-host launcher for one 47m_*.jsonl shard.
# Usage:  bash run_47m_part.sh <part_idx>
#   part_idx in 1..7  -> processes 47m_${part_idx}.jsonl
# Writes everything under /ov21/47m_sft_packed/47m_part${part_idx}/
set -euo pipefail
export PYTHONUNBUFFERED=1

# --- glibc / BLAS memory hygiene (huge effect on s2 RSS) ---
# Without these, each worker's RSS grows to ~12 GB; with them, ~4-6 GB.
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- s2 concurrency caps (read by s2_compute_token_lengths.py) ---
# Use :=defaults so caller can override via env (e.g. for A/B testing).
# Updated 2026-05-28: post FAST_TOKENIZER+spawn, worker RSS=270MB → CPU is
# the bottleneck, not memory. Bumped procs 16→64 to use the spare 75% CPU.
export S2_MAX_PROCESSES=${S2_MAX_PROCESSES:-64}
export S2_MAX_THREADS_PER_PROC=${S2_MAX_THREADS_PER_PROC:-4}
export S2_MEM_PER_PROC_GB=${S2_MEM_PER_PROC_GB:-2}
export S2_MAXTASKS_PER_CHILD=${S2_MAXTASKS_PER_CHILD:-10}
export S2_USE_SPAWN=${S2_USE_SPAWN:-1}
export FAST_TOKENIZER=${FAST_TOKENIZER:-1}
export FAST_IMG_META=${FAST_IMG_META:-1}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llava_main_xy

IDX=${1:?"usage: $0 <part_idx 1..7>"}
SRC=/ov2/chengzheng/ov2_wds/47m_sft_cleaned_64k_20w/jsonls/47m_${IDX}.jsonl
OUT=/ov21/47m_sft_packed/47m_part${IDX}
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing

[[ -f "$SRC" ]] || { echo "MISSING: $SRC"; exit 2; }
mkdir -p "$OUT"
echo "[part${IDX}] host=$(hostname) src=$SRC out=$OUT lines=$(wc -l <"$SRC")"

bash "${PIPE}/auto_pipe.sh" \
    -i "${SRC}" -r / -m "${MODEL}" -o "${OUT}" -L 32000 \
    -w 8 --direct-workers 32 \
    --shard-prefix "47m_part${IDX}" \
    --sample-class PackedCaptioningSample \
    --max-samples-per-shard 5000 \
    --longest-edge 3072 --jpeg-quality 92 \
    --no-npy \
    --patch-slack 0.30

touch "${OUT}/DONE.txt"
echo "[part${IDX}] DONE at $(date)"
