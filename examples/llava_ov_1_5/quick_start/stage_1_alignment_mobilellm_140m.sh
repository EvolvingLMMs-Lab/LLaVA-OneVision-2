#!/bin/bash
# Stage 1 Alignment Training for LLaVA-OneVision-1.5 with MobileLLM-R1-140M
# This script trains the projection adapter while keeping vision encoder and language model frozen

REPO_ROOT=/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5
echo "$REPO_ROOT"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_ROOT}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-$REPO_ROOT/aiak_megatron}"

# Model parallelism configuration
TP="${1:-1}"  # Tensor parallel
PP="${2:-1}"  # Pipeline parallel
SEQ_LEN="${3:-32768}"  # Sequence length (reduced for testing)
MBS="${4:-1}"  # Micro batch size
GBS="${5:-4}"  # Global batch size (for TP=1, PP=1, 8 GPUs -> DP=8)
NSTEP="${6:-100}"  # Number of training iterations (1 step with 4 examples)

# Data paths - UPDATE THESE FOR YOUR SETUP
DATA_PATH="${DATA_PATH:-"$REPO_ROOT/data/LLaVA-558K-Webdataset"}"
# MobileLLM tokenizer path - use facebook/MobileLLM-R1-140M from HuggingFace
TOKENIZER_PATH="${TOKENIZER_PATH:-"facebook/MobileLLM-R1-140M"}"
# Pretrained checkpoint (MobileLLM-R1-140M + FastViT merged)
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-"$REPO_ROOT/checkpoints/mobilellm-fastvit-merged-tp1-pp1"}"

echo "AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"

# Multi-node configuration
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
echo "Current Node IP: ${CURRENT_IP}"
echo "Current Node Rank: ${NODE_RANK}"
echo "Node Size: ${NNODES}"

# Output directories
SAVE_CKPT_PATH=$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-26000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}

if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
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

# ========================================
# MODEL CONFIGURATION - MobileLLM Backbone
# ========================================
MODEL_ARGS=(
    --model-name llava-ov-mobilellm-140m
)

# ========================================
# DATA CONFIGURATION
# ========================================
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0 # Data Splitting part 
    --num-workers 16
    
    # FastViT vision encoder configuration  
    # Using pretrained FastViT + MobileLLM-R1-140M (merged checkpoint)
    --use-fastvit
    --fastvit-image-size 1024  # Match pretrained checkpoint resolution
    --vision-tower-name mobileclip_l_1024  # MobileCLIP-L with 1024 resolution (from checkpoint)
    --image-aspect-ratio pad # Pad images to square aspect ratio
    
    # Load pretrained checkpoint (language model + vision encoder merged)
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
)

# ========================================
# TRAINING CONFIGURATION
# ========================================
TRAINING_ARGS=(
    --training-phase sft # supervised fine-tuning 
    --chat-template llama3  # MobileLLM uses Llama3 tokenizer
    --trainable-modules adapter  # Stage 1: Train only adapter (vision and language model fully frozen)
    --no-gradient-accumulation-fusion
    --seq-length "${SEQ_LEN}"
    --no-rope-fusion
    --training-rice-vl-max-answer-length 32768
    --transformer-impl local
    --max-position-embeddings 32768  # MobileLLM supports 32k context
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-4
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-05  # MobileLLM RMSNorm epsilon
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$SAVE_CKPT_PATH"
    --save "$SAVE_CKPT_PATH"
    # --save-interval 2000
    --save-interval "$SAVE_INTERVAL"
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 3  # Must divide evenly into 15 layers (15/3=5 chunks)
)

if [[ "${NO_LOAD_OPTIM_RNG:-0}" == "1" ]]; then
    TRAINING_ARGS+=(
        --no-load-optim
        --no-load-rng
    )
fi

# ========================================
# MODEL PARALLELISM CONFIGURATION
# ========================================
MODEL_PARALLEL_ARGS=(
    --attention-backend local
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

# ========================================
# LOGGING CONFIGURATION
# ========================================
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT:-llava-ov-mobilellm}"
        --wandb-exp-name "${WANDB_NAME:-stage1-mobilellm-140m}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKED_DATA='1'
export OFFLINE_PACKING_VQA='1'

# ========================================
# RUN TRAINING
# ========================================
echo "========================================="
echo "Starting Stage 1 Training"
echo "Model: LLaVA-OneVision-1.5 + MobileLLM-R1-140M"
echo "Vision: FastViT (MobileCLIP)"
echo "Training: Adapter only (vision + language frozen)"
echo "========================================="

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
