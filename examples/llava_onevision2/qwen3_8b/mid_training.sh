# =============================================================================
# LLaVA-OneVision2 8B (Qwen3-8B + 300M ViT) – Stage-2 Mid Training
# =============================================================================
#
# Recommended parallelism strategy
# ---------------------------------
# The model has two heterogeneous sub-models:
#   • ViT  (300 M params) : 24 layers, hidden_size=1024
#   • LLM  (8 B  params) : 36 transformer layers, hidden_size=4096
#
# Pipeline placement rule: the ViT encoder + MLP adapter always live on PP
# stage-0 (first pipeline rank).  Because stage-0 already carries the ViT
# workload, assigning it 0 LLM transformer layers keeps compute balanced.
#
# 36 LLM layers → stages 1-3 each receive  36/3 = 12  layers:
#   --pipeline-model-parallel-size 4  --custom-pipeline-layers 0,12,12,12
#
# Recommended configurations by GPU count
# ----------------------------------------
#  8  GPUs : TP=2  PP=4  --custom-pipeline-layers 0,12,12,12  (default below)
#  16 GPUs : TP=4  PP=4  --custom-pipeline-layers 0,12,12,12
#  32 GPUs : TP=4  PP=4  --custom-pipeline-layers 0,12,12,12  (2 nodes × 16 GPUs)
#
# For PP=1 (single-stage, no pipeline split) use the plain TP-only preset:
#   TP=8 PP=1   (no --custom-pipeline-layers needed)
# =============================================================================

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-32768}"
MBS="${4:-1}"
GBS="${5:-128}"

TOTAL_SAMPLES=9000000
NSTEP="${6:-$(((TOTAL_SAMPLES + GBS - 1) / GBS))}"
# When PP=4 the ViT sits on stage-0 with 0 LLM layers; stages 1-3 share 36
# layers evenly.  Override via environment variable for other PP values, e.g.:
# CUSTOM_PIPELINE_LAYERS=0,18,18        (PP=3)
# CUSTOM_PIPELINE_LAYERS=0,6,6,6,6,6,6 (PP=7)
# CUSTOM_PIPELINE_LAYERS="${CUSTOM_PIPELINE_LAYERS:-0,12,12,12}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/models/checkpoints_llava_onevision2_8b}"

DATA_PATH=${DATA_PATH:-"/mnt/data/models/llava_onevision2_8b/ax_85m_image_5m_video.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/data/models/llava_onevision2_8b/auto-model"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/data/models/llava_onevision2_8b/llava_onevision2_8b_stage1_mcore_tp1_pp1"}


#! /bin/bash
# The script needs to be run on at least 1 nodes.

# --- Multi-node configuration ---
# List of IP addresses for the nodes in the training cluster
declare -a list_ip=(
    "localhost"
)

# Get the primary IP of the current node
CURRENT_IP=$(hostname -I | awk '{print $1}')

if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

# Dynamically determine NNODES, MASTER_ADDR
NNODES=${#list_ip[@]}
MASTER_ADDR=${list_ip[0]}

if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1
    MASTER_ADDR=127.0.0.1
    NODE_RANK=0
    echo "--- Single-node mode ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
else
    # Find the rank of the current node
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done

    # Exit if the current IP is not in the list
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in the IP list."
        echo "Please run this script on a node with an IP in list_ip."
        exit 1
    fi

    echo "--- Running on ${NNODES} nodes ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
fi
# --- End of Multi-node configuration ---


SAVE_CKPT_PATH=$OUTPUT_DIR/$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

# CHECKPOINT_PATH=$SAVE_CKPT_PATH

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"${list_ip[0]}"}
MASTER_PORT=${MASTER_PORT:-"26000"}

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
    --model-name llava-onevision2-8b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 2000
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 4
)

# Build MODEL_PARALLEL_ARGS; only pass --custom-pipeline-layers when PP > 1
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

# if [[ $PP -gt 1 && -n "$CUSTOM_PIPELINE_LAYERS" ]]; then
#     MODEL_PARALLEL_ARGS+=(--custom-pipeline-layers "${CUSTOM_PIPELINE_LAYERS}")
# fi

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKING_BMR=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
