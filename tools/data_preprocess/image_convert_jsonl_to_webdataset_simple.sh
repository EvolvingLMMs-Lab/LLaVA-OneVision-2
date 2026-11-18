#!/bin/bash
# Simple JSONL -> WebDataset conversion (no packing)
#
# Usage:
#   bash tools/data_preprocess/convert_jsonl_to_webdataset_simple.sh \
#       /path/to/input.jsonl \
#       /path/to/output_dir
#
# Environment Variables:
#   AIAK_TRAINING_PATH: repo root (default: /workspace/LLaVA-OneVision-2.5)
#   AIAK_MEGATRON_PATH: aiak_megatron path (default: ${AIAK_TRAINING_PATH}/aiak_megatron)

set -e

usage() {
  cat <<'EOF'
Simple JSONL -> WebDataset conversion (no packing)

Usage:
  bash tools/data_preprocess/convert_jsonl_to_webdataset_simple.sh <jsonl> <output_dir>

Arguments:
  <jsonl>        Input JSONL file path.
  <output_dir>   Output directory for generated WebDataset shards.

Environment Variables:
  AIAK_TRAINING_PATH  Repo root (default: /workspace/LLaVA-OneVision-2.5)
  AIAK_MAGATRON_PATH  aiak_megatron path (default: ${AIAK_TRAINING_PATH}/aiak_megatron)

Examples:
  bash tools/data_preprocess/convert_jsonl_to_webdataset_simple.sh \
    /data/train.jsonl /data/webdataset_out

  AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5 \
  bash tools/data_preprocess/convert_jsonl_to_webdataset_simple.sh \
    /data/train.jsonl /data/webdataset_out
EOF
}

# ============================================================================
# Environment Setup
# ============================================================================

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

JSONL_PATH="${1:-}"
OUTPUT_DIR="${2:-}"
if [[ -z "${JSONL_PATH}" || -z "${OUTPUT_DIR}" ]]; then
  usage
  exit 1
fi

# ============================================================================
# Run Conversion
# ============================================================================

PYTHONPATH="${AIAK_MAGATRON_PATH}:${AIAK_TRAINING_PATH}:${PYTHONPATH}" \
  python "${AIAK_TRAINING_PATH}/tools/data_preprocess/convert_jsonl_to_webdataset_simple.py" \
  --jsonl "${JSONL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers 32
