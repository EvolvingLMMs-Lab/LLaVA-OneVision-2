#!/bin/bash
# LLaVA-OneVision2 Model Consistency Check Script
#
# This script compares the model outputs between HuggingFace and Megatron-LM
# implementations to verify layer-by-layer consistency for both vision encoder
# and language model.
#
# Usage:
#   bash examples/llava_onevision2/check_model_consistency.sh [TP] [PP]
#
# Arguments:
#   TP: Tensor parallel size (default: 1)
#   PP: Pipeline parallel size (default: 1)
#
# Environment Variables:
#   HF_MODEL_PATH: Path to HuggingFace model (default: /ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen3_2b_stage0)
#   MCORE_CHECKPOINT_PATH: Path to Megatron checkpoint (default: /ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen3_2b_stage0_mcore_tp1_pp1)
#   TEST_IMAGE_PATH: Path to test image (default: http://images.cocodataset.org/val2017/000000039769.jpg)

set -euo pipefail

# ============================================================================
# Environment Setup
# ============================================================================

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

TP="${1:-1}"
PP="${2:-1}"

MODEL_NAME="${MODEL_NAME:-llava-onevision2-2b}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen3_2b_stage0}"
MCORE_CHECKPOINT_PREFIX="${MCORE_CHECKPOINT_PREFIX:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen3_2b_stage0_mcore}"
MCORE_CHECKPOINT_PATH="${MCORE_CHECKPOINT_PATH:-${MCORE_CHECKPOINT_PREFIX}_tp${TP}_pp${PP}}"
PREPROCESSOR_PATH="${PREPROCESSOR_PATH:-/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5}"
TEST_IMAGE_PATH="${TEST_IMAGE_PATH:-${AIAK_TRAINING_PATH%/}/asset/performance.png}"
TEST_PROFILE="${TEST_PROFILE:-full}"

# Output settings
OUTPUT_DIR="outputs/model_consistency_check"
RUN_TAG="tp${TP}_pp${PP}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_DIR}/results_${RUN_TAG}.json"

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$HF_MODEL_PATH" ]]; then
    echo "Error: HF_MODEL_PATH does not exist: $HF_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$MCORE_CHECKPOINT_PATH" ]]; then
    echo "Error: MCORE_CHECKPOINT_PATH does not exist: $MCORE_CHECKPOINT_PATH"
    echo "Hint: set MCORE_CHECKPOINT_PATH explicitly, or set MCORE_CHECKPOINT_PREFIX so it resolves to *_tp${TP}_pp${PP}."
    exit 1
fi

if [[ ! -e "$TEST_IMAGE_PATH" ]]; then
    echo "Error: TEST_IMAGE_PATH does not exist: $TEST_IMAGE_PATH"
    exit 1
fi

AVAILABLE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
if [[ "$AVAILABLE_GPUS" -lt $((TP * PP)) ]]; then
    echo "Error: requested TP=${TP}, PP=${PP}, need $((TP * PP)) GPUs but only found ${AVAILABLE_GPUS}."
    exit 1
fi

# ============================================================================
# Node Configuration
# ============================================================================

declare -a list_ip=(
    "localhost"
)

CURRENT_IP=$(hostname -I | awk '{print $1}')

if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

NNODES=${#list_ip[@]}
MASTER_ADDR=${list_ip[0]}

if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1
    MASTER_ADDR=127.0.0.1
    NODE_RANK=0
    echo "--- Single-node mode ---"
else
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done

    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in the IP list."
        exit 1
    fi
    echo "--- Running on ${NNODES} nodes ---"
fi

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "Current Node Rank: ${NODE_RANK}"

# ============================================================================
# Distributed Configuration
# ============================================================================

GPUS_PER_NODE=$((TP * PP))
MASTER_PORT=${MASTER_PORT:-"26500"}

if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

# ============================================================================
# Model Arguments
# ============================================================================

MODEL_ARGS=(
    --model-name "$MODEL_NAME"

    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $HF_MODEL_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl

)

# ============================================================================
# Check Arguments
# ============================================================================

CHECK_ARGS=(
    --hf-model-path "$HF_MODEL_PATH"
    --preprocessor-path "$PREPROCESSOR_PATH"
    --output-path "$OUTPUT_PATH"
    --test-image-path "$TEST_IMAGE_PATH"
    --test-profile "$TEST_PROFILE"
)

# ============================================================================
# Training/Loading Arguments
# ============================================================================

TRAINING_ARGS=(
    --seq-length 4096
    --max-position-embeddings 32768
    --micro-batch-size 1
    --global-batch-size 1
    --bf16
    --load "$MCORE_CHECKPOINT_PATH"
    --ckpt-format torch
)

# ============================================================================
# Model Parallel Arguments
# ============================================================================

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --distributed-backend nccl
)

# ============================================================================
# Logging
# ============================================================================

echo "============================================================"
echo "LLaVA-OneVision2 Model Consistency Check"
echo "============================================================"
echo "HuggingFace Model:   ${HF_MODEL_PATH}"
echo "Megatron Checkpoint: ${MCORE_CHECKPOINT_PATH}"
echo "Preprocessor Path:   ${PREPROCESSOR_PATH}"
echo "Test Image:          ${TEST_IMAGE_PATH}"
echo "Output Path:         ${OUTPUT_PATH}"
echo "Test Profile:        ${TEST_PROFILE}"
echo "TP: ${TP}, PP: ${PP}"
echo "GPUs per node:       ${GPUS_PER_NODE}"
echo "============================================================"

# ============================================================================
# Run Check Script
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PYTHONPATH="ds/llavaonevision2:$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/examples/llava_onevision2/check_model_consistency.py" \
    "${MODEL_ARGS[@]}" \
    "${CHECK_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/check_${RUN_TAG}.log"

echo "============================================================"
echo "Check completed. Results saved to: ${OUTPUT_PATH}"
echo "============================================================"
