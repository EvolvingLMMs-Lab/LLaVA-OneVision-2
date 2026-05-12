#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen-VL processor/model directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-${SCRIPT_DIR}/sample_text_only.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${SCRIPT_DIR}/images}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/output_smoke}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-4096}"

bash "${REPO_ROOT}/offline_packing/auto_pipe.sh" \
  -i "${INPUT_JSONL}" \
  -r "${IMAGE_ROOT}" \
  -m "${MODEL_PATH}" \
  -o "${OUTPUT_BASE}" \
  -L "${MAX_TOKEN_LEN}" \
  -M 4 \
  -w 4 \
  --direct-workers 2 \
  --shard-prefix smoke \
  --max-samples-per-shard 100 \
  --max-shard-size 500000000
