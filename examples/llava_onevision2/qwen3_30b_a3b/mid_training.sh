# =============================================================================
# LLaVA-OneVision2 30B-A3B – Stage-2 Mid Training
# =============================================================================
# Main recommended setup in this script:
#   TP=1, PP=1, EP=8
#
# Usage:
#   bash mid_training.sh [TP] [PP] [EP] [SEQ_LEN] [MBS] [GBS] [NSTEP]
# =============================================================================

TP="${1:-1}"
PP="${2:-2}"
EP="${3:-8}"
SEQ_LEN="${4:-32768}"
MBS="${5:-1}"
GBS="${6:-128}"

TOTAL_SAMPLES=9000000
NSTEP="${7:-$(((TOTAL_SAMPLES + GBS - 1) / GBS))}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

DATA_PATH=${DATA_PATH:-"dataset_mid/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-Packed-384332"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/LLaVA-OneVision-2/stage_1_alignment_llava_ov2_30b_a3b_release"}

#! /bin/bash
# The script needs to be run on at least 1 nodes.

# --- Multi-node configuration ---
declare -a list_ip=(
    "172.16.5.36"
    "172.16.5.37"
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
    --model-name llava-onevision2-30b-a3b
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
    --max-position-embeddings "${SEQ_LEN}"
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
    # --recompute-num-layers 4
    --custom-pipeline-recompute-layers 12,12 # ViT 24layers
    --custom-pipeline-layers 12,36
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --expert-model-parallel-size "${EP}"
    --num-experts 128
    --use-distributed-optimizer
    --distributed-backend nccl
)

MOE_ARGS=(
    --moe-router-topk 8
    --moe-aux-loss-coeff 0.01
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
