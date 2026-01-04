REPO_ROOT=/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5
echo "$REPO_ROOT"
AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_ROOT}" #Root of the LLaVA-OneVision training repo
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-$REPO_ROOT/aiak_megatron}" # Megatron-LM fork used (aiak_megatron) inside the training repo.
# TP="${1:-1}" # Tensor parallel
TP="${1:-1}"
PP="${2:-1}" # pipeline parallel
# Defaults: TP=1, PP=1.
# SEQ_LEN="${3:-32768}"
SEQ_LEN="${3:-512}" 
MBS="${4:-1}" # micro batch size 
# GBS="${5:-8}" # global batch size
GBS="${5:-8}"
# NSTEP="${6:-2500}" # number of training iterations
NSTEP="${6:-20}" # number of training iterations
# DATA_PATH=${DATA_PATH:-"/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/data/LLaVA-558K-Webdataset"}
# TOKENIZER_PATH=${TOKENIZER_PATH:-"/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/checkpoints/LLaVA-OneVision-1.5-4B-stage0"}
# CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/checkpoints/LLaVA-OneVision-1.5-4B-stage0_mcore_tp2_pp1"}
DATA_PATH="${DATA_PATH:-"$REPO_ROOT/data/LLaVA-558K-Webdataset"}"
TOKENIZER_PATH="${TOKENIZER_PATH:-"$REPO_ROOT/checkpoints/LLaVA-OneVision-1.5-4B-stage0"}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-"$REPO_ROOT/checkpoints/LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1"}"
echo "AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
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


SAVE_CKPT_PATH=$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

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
    --model-name llava-ov-1.5-4b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl # will change this chat template 
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules adapter
    --no-gradient-accumulation-fusion
    --seq-length "${SEQ_LEN}"
    --no-rope-fusion
    --training-rice-vl-max-answer-length 512
    --transformer-impl local 
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-4
    --min-lr 1.0e-6
    --clip-grad 1.0 # gradient norm clipping.
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH" # load initial checkpoint.
    --save "$SAVE_CKPT_PATH" # where to save new checkpoints.
    --save-interval 2000
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 4
)

MODEL_PARALLEL_ARGS=(
    # --attention-backend flash
    --attention-backend local
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

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

export OFFLINE_PACKED_DATA='1'
export OFFLINE_PACKING_VQA='1'
# export PYTORCH_ALLOC_CONF=max_split_size_mb:128
# export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.72
export PYTORCH_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.72


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