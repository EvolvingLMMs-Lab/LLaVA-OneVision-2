# =============================================================================
# Gemma4-VL 26B-A4B - Stage-1 Projector (Adapter-only) Training
# =============================================================================
# Plan v5 default recipe:
#   TP=4, PP=2, EP=4, EPP=1, SEQ_LEN=8192, MBS=1, GBS=128
#   --trainable-modules adapter (vision_model + LLM frozen)
#   --moe-router-topk 8, --num-experts 128, --moe-aux-loss-coeff 1e-3
#
# Smoke override (single 8x80GB node):
#   TP=2 PP=1 EP=4 SEQ_LEN=4096 MBS=1 GBS=8 NSTEP=100
#
# Usage:
#   bash stage1_projector.sh [TP] [PP] [EP] [SEQ_LEN] [MBS] [GBS] [NSTEP]
# =============================================================================

TP="${1:-4}"
PP="${2:-2}"
EP="${3:-4}"
SEQ_LEN="${4:-8192}"
MBS="${5:-1}"
GBS="${6:-128}"
NSTEP="${7:-500}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/ov2/xiangan/ckpts_gemma4_26b_a4b}"
DATA_PATH=${DATA_PATH:-"/workspace/dataset/LLaVA-558K-Webdataset"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ov2/pretrain_models/google/gemma-4-26B-A4B-it"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/LLaVA-OneVision-2/stage_0_gemma4_26b_a4b_release"}

#! /bin/bash
# The script needs to be run on at least 1 nodes.

# --- Multi-node configuration ---
declare -a list_ip=(
    127.0.0.1
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
else
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in list_ip."
        exit 1
    fi
fi
# --- End of Multi-node configuration ---

SAVE_CKPT_PATH=$OUTPUT_DIR/$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=8

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
    --model-name gemma4-26b-a4b-vl
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 4
)

TRAINING_ARGS=(
    --training-phase sft
    --chat-template gemma4
    --trainable-modules adapter
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
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
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 100
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 4
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --expert-model-parallel-size "${EP}"
    --encoder-pipeline-model-parallel-size 1
    --num-experts 128
    --use-distributed-optimizer
    --distributed-backend nccl
)

MOE_ARGS=(
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-router-dtype fp32
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
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_ep${EP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKING_BMR=1

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
