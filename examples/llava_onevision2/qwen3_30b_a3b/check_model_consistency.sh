#!/bin/bash

set -euo pipefail

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

HF_MODEL_PATH="${HF_MODEL_PATH:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto-model}"
MCORE_CHECKPOINT_PATH="${MCORE_CHECKPOINT_PATH:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/stage_0_tp1_pp1_ep8}"
PREPROCESSOR_PATH="${PREPROCESSOR_PATH:-$HF_MODEL_PATH}"
TEST_IMAGE_PATH="${TEST_IMAGE_PATH:-http://images.cocodataset.org/val2017/000000039769.jpg}"
TEST_PROFILE="${TEST_PROFILE:-low_vram}"

TP="${1:-1}"
PP="${2:-1}"
EP="${3:-8}"
CUSTOM_PIPELINE_LAYERS="${4:-}"

if [[ -n "$CUSTOM_PIPELINE_LAYERS" ]]; then
    IFS=',' read -r -a _custom_pp_layers <<< "$CUSTOM_PIPELINE_LAYERS"
    if [[ "${#_custom_pp_layers[@]}" -ne "$PP" ]]; then
        echo "Error: CUSTOM_PIPELINE_LAYERS='$CUSTOM_PIPELINE_LAYERS' has ${#_custom_pp_layers[@]} entries, but PP=$PP."
        exit 1
    fi
fi

if [[ ! -d "$HF_MODEL_PATH" ]]; then
    echo "Error: HF_MODEL_PATH does not exist: $HF_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$MCORE_CHECKPOINT_PATH" ]]; then
    echo "Error: MCORE_CHECKPOINT_PATH does not exist: $MCORE_CHECKPOINT_PATH"
    exit 1
fi

OUTPUT_DIR="outputs/model_consistency_check_30b_a3b"
OUTPUT_PATH="${OUTPUT_DIR}/results_$(date +%Y%m%d_%H%M%S).json"

mkdir -p "$OUTPUT_DIR"

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
        if [[ "${list_ip[$i]}" == "$CURRENT_IP" ]]; then
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

GPUS_PER_NODE=$((TP * PP * EP))
MASTER_PORT=${MASTER_PORT:-"26500"}
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$GPUS_PER_NODE

if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
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

MODEL_ARGS=(
    --model-name llava-onevision2-30b-a3b

    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$HF_MODEL_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

CHECK_ARGS=(
    --hf-model-path "$HF_MODEL_PATH"
    --preprocessor-path "$PREPROCESSOR_PATH"
    --output-path "$OUTPUT_PATH"
    --test-image-path "$TEST_IMAGE_PATH"
    --test-profile "$TEST_PROFILE"
)

TRAINING_ARGS=(
    --seq-length 4096
    --max-position-embeddings 40960
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --bf16
    --load "$MCORE_CHECKPOINT_PATH"
    --ckpt-format torch
)

if [[ -n "$CUSTOM_PIPELINE_LAYERS" ]]; then
    TRAINING_ARGS+=(
        --custom-pipeline-layers "$CUSTOM_PIPELINE_LAYERS"
    )
fi

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --expert-model-parallel-size "${EP}"
    --num-experts 128
    --distributed-backend nccl
)

MOE_ARGS=(
    --moe-router-topk 8
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
)

echo "============================================================"
echo "LLaVA-OneVision2 30B-A3B Model Consistency Check"
echo "============================================================"
echo "HuggingFace Model:   ${HF_MODEL_PATH}"
echo "Megatron Checkpoint: ${MCORE_CHECKPOINT_PATH}"
echo "Preprocessor Path:   ${PREPROCESSOR_PATH}"
echo "Test Image:          ${TEST_IMAGE_PATH}"
echo "Test Profile:        ${TEST_PROFILE}"
echo "Output Path:         ${OUTPUT_PATH}"
echo "TP: ${TP}, PP: ${PP}, EP: ${EP}"
echo "MBS: ${MICRO_BATCH_SIZE}, GBS: ${GLOBAL_BATCH_SIZE}"
if [[ -n "$CUSTOM_PIPELINE_LAYERS" ]]; then
    echo "Custom Pipeline:     ${CUSTOM_PIPELINE_LAYERS}"
fi
echo "GPUs per node:       ${GPUS_PER_NODE}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PYTHONPATH="ds/llavaonevision2:$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/examples/llava_onevision2/check_model_consistency.py" \
    "${MODEL_ARGS[@]}" \
    "${CHECK_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/check_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Check completed. Results saved to: ${OUTPUT_PATH}"
echo "============================================================"
