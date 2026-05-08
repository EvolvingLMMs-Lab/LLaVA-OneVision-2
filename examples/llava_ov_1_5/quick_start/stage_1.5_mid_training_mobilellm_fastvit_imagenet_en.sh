#!/bin/bash
# Stage 1.5 midtraining for the MobileLLM-R1-140M + FastViT/FastVLM pipeline.
#
# This launcher expects a small Energon/WebDataset slice prepared from:
#   mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M / imagenet / EN
# See tools/prepare_hf_caption_parquet_to_energon_wds.py.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5}"
AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_ROOT}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-$REPO_ROOT/aiak_megatron}"
TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-4096}"
MBS="${4:-1}"
GBS="${5:-4}"
NSTEP="${6:-1000}"

DATA_PATH="${DATA_PATH:-"$REPO_ROOT/data/midtraining_imagenet_en_part00_webdataset"}"
TOKENIZER_PATH="${TOKENIZER_PATH:-facebook/MobileLLM-R1-140M}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-"$REPO_ROOT/stage_1_alignment_mobilellm_140m_fastvlm_faithful"}"
SAVE_CKPT_PATH="${SAVE_CKPT_PATH:-"$REPO_ROOT/stage_1_5_midtraining_mobilellm_fastvit_imagenet_en"}"
TENSORBOARD_PATH="${TENSORBOARD_PATH:-"$SAVE_CKPT_PATH/tensorboard"}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-26015}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MIDTRAIN_TRAINABLE_MODULES="${MIDTRAIN_TRAINABLE_MODULES:-language_model adapter vision_model}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [ ! -f "$DATA_PATH/.nv-meta/dataset.yaml" ]; then
    echo "Missing prepared midtraining dataset: $DATA_PATH"
    echo "Create a small ImageNet-EN slice first, for example:"
    echo "  python tools/prepare_hf_caption_parquet_to_energon_wds.py \\"
    echo "    --data-files 'imagenet/EN/part00/*.parquet' \\"
    echo "    --max-samples 10000 \\"
    echo "    --output-dir '$DATA_PATH'"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]; then
    echo "Missing Stage 1 checkpoint tracker: $CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
    echo "Set CHECKPOINT_PATH to the completed 2500-step Stage 1 alignment checkpoint directory."
    exit 1
fi

mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"

declare -a list_ip=("localhost")
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

NNODES=${#list_ip[@]}
MASTER_ADDR=${MASTER_ADDR:-${list_ip[0]}}

if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1
    MASTER_ADDR=127.0.0.1
    NODE_RANK=0
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --master_port "$MASTER_PORT"
    )
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
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

MODEL_ARGS=(
    --model-name llava-ov-mobilellm-140m
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers "$NUM_WORKERS"
    --chat-template llama3

    --use-fastvit
    --fastvit-image-size 1024
    --vision-tower-name mobileclip_l_1024
    --image-aspect-ratio pad
)

read -r -a TRAINABLE_MODULES_ARRAY <<< "$MIDTRAIN_TRAINABLE_MODULES"

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules "${TRAINABLE_MODULES_ARRAY[@]}"
    --no-gradient-accumulation-fusion
    --seq-length "$SEQ_LEN"
    --no-rope-fusion
    --training-rice-vl-max-answer-length "$SEQ_LEN"
    --transformer-impl local
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --lr "${LR:-1.0e-5}"
    --min-lr "${MIN_LR:-1.0e-6}"
    --clip-grad 1.0
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-05
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.002}"
    --initial-loss-scale 65536
    --bf16
    --finetune
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval "$SAVE_INTERVAL"
    --ckpt-format torch
    --dataloader-save "$SAVE_CKPT_PATH/dataloader"
    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 3
)

if [[ "${NO_SAVE_OPTIM_RNG:-0}" == "1" ]]; then
    TRAINING_ARGS+=(
        --no-save-optim
        --no-save-rng
    )
fi

MODEL_PARALLEL_ARGS=(
    --attention-backend local
    --pipeline-model-parallel-size "$PP"
    --tensor-model-parallel-size "$TP"
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval "$LOG_INTERVAL"
    --tensorboard-dir "$TENSORBOARD_PATH"
    --log-timers-to-tensorboard
)

if [[ "${WANDB_ENABLE:-0}" == "1" || -n "${WANDB_API_KEY:-}" ]]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT:-llava-ov-mobilellm}"
        --wandb-exp-name "${WANDB_NAME:-stage1_5-mobilellm-fastvit-imagenet-en}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="$SAVE_CKPT_PATH/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKED_DATA=1
export OFFLINE_PACKING_VQA=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.72,max_split_size_mb:128}"

echo "========================================="
echo "Starting Stage 1.5 Midtraining"
echo "Model: MobileLLM-R1-140M + FastViT/FastVLM"
echo "Data: $DATA_PATH"
echo "Load Stage 1: $CHECKPOINT_PATH"
echo "Save Stage 1.5: $SAVE_CKPT_PATH"
echo "Trainable modules: ${TRAINABLE_MODULES_ARRAY[*]}"
echo "GPUs: $GPUS_PER_NODE | SEQ_LEN: $SEQ_LEN | MBS: $MBS | GBS: $GBS | ITERS: $NSTEP"
echo "Log interval: $LOG_INTERVAL | Save interval: $SAVE_INTERVAL"
if [[ "${WANDB_ENABLE:-0}" == "1" || -n "${WANDB_API_KEY:-}" ]]; then
    echo "W&B: mode=$WANDB_MODE entity=${WANDB_ENTITY:-default} project=${WANDB_PROJECT:-llava-ov-mobilellm} name=${WANDB_NAME:-stage1_5-mobilellm-fastvit-imagenet-en}"
else
    echo "W&B: disabled"
fi
echo "========================================="

if [[ "${PRINT_DATA_SAMPLE:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/tools/preview_caption_wds_sample.py" "$DATA_PATH" \
        --sample-index "${DATA_SAMPLE_INDEX:-0}" \
        --max-chars "${DATA_SAMPLE_MAX_CHARS:-700}" || true
fi

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}" \
"$TORCHRUN" "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"

{
    echo "========================================="
    echo "Stage 1.5 training completed"
    echo "Checkpoint root: $SAVE_CKPT_PATH"
    if [[ -f "$SAVE_CKPT_PATH/latest_checkpointed_iteration.txt" ]]; then
        echo "Latest checkpoint iteration: $(cat "$SAVE_CKPT_PATH/latest_checkpointed_iteration.txt")"
    fi
    echo "Training log: $logfile"
    echo "========================================="
} | tee -a "$logfile"
