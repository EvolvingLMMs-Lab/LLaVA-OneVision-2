#!/bin/bash
# Smoke test for refactored offline_packing pipeline.
# - 10k samples from 47m_1.jsonl
# - L=32000 packing length
# - model: LLaVA-OneVision-2-4B-p16m33  (patch16, merge3 -> factor=48)
# - image preprocess: longest_edge=3072, jpeg q=92
# Target machine: 192.0.2.11

set -euo pipefail
export PYTHONUNBUFFERED=1

# Optional: CONDA_ENV=<name> activates that conda env first (needs transformers / webdataset).
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
echo "[smoke] using python: $(which python)"

SRC_JSONL=/data/jsonls/sft_part1.jsonl
SMOKE_BASE=/data/smoke_test
SAMPLE_JSONL=${SMOKE_BASE}/input_10k.jsonl
OUT_BASE=${SMOKE_BASE}/packed_10k_L32k
IMAGE_ROOT=/
MODEL_PATH=/models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE_DIR=/workspace/LLaVA-OneVision-2.5/offline_packing

mkdir -p "${SMOKE_BASE}"

if [[ ! -f "${SAMPLE_JSONL}" ]]; then
    echo "[smoke] sampling 10000 lines from ${SRC_JSONL} -> ${SAMPLE_JSONL}"
    head -n 10000 "${SRC_JSONL}" > "${SAMPLE_JSONL}"
fi
echo "[smoke] input lines: $(wc -l < ${SAMPLE_JSONL})"

echo "[smoke] launching auto_pipe.sh L=32000 ..."
bash "${PIPE_DIR}/auto_pipe.sh" \
    -i "${SAMPLE_JSONL}" \
    -r "${IMAGE_ROOT}" \
    -m "${MODEL_PATH}" \
    -o "${OUT_BASE}" \
    -L 32000 \
    -w 16 \
    --direct-workers 16 \
    --shard-prefix smoke \
    --sample-class PackedCaptioningSample \
    --max-samples-per-shard 5000 \
    --longest-edge 3072 \
    --jpeg-quality 92 \
    --no-npy

echo "[smoke] running verify_token_consistency.py ..."
BINS_FILE="${OUT_BASE}/bins_32klen.pkl"
python "${PIPE_DIR}/verify_token_consistency.py" \
    --token-info "${OUT_BASE}/token_info.txt" \
    --webdataset-dir "${OUT_BASE}/webdataset" \
    --bins-file "${BINS_FILE}" \
    --model-path "${MODEL_PATH}" \
    --n-samples 16 \
    --tolerance 2 || echo "[smoke] verify_token_consistency exited non-zero (review above)"

echo "[smoke] DONE. Outputs at ${OUT_BASE}"
ls -lh "${OUT_BASE}"/ "${OUT_BASE}/webdataset"/ 2>/dev/null | head -40
