#!/usr/bin/env bash
# =============================================================================
# Baseline single-node SFT / 4B-p16m3 / 1 node / TP=8 PP=1 / GBS=8 / SEQ=8k
#
# Single-node mirror of the 16-node mixed-6sets-packed-64k production run:
#   examples/llava_onevision2/quick_start_4b/
#       date0612_ax_stage_1p5_mixed_6sets_packed_64k_16n_tp8_gbs16_NoSP_from_epoch2_iter20480.sh
#
# Key deltas vs. 16n production:
#   - LIST_IP = (localhost)               -> single-node torchrun
#   - DATA_PATH = 47m_sft_packed_8k       -> single family, 7 parts, 1,128,813 bins
#   - SEQ_LEN   = 8192   (was 102400)     -> matches 8k packed bins
#   - GBS       = 8      (was 16)         -> DP_SIZE=1 (8 GPUs / TP=8 / PP=1)
#   - NSTEP     = 20     (was 39207)      -> startup validation only
#   - LR_WARMUP_ITERS = 10 (was 200)      -> short warmup for short run
#   - MASTER_PORT = 26020 (was 26032)     -> avoid clash with 16n run
#   - SAVE_INTERVAL = 0   (was 1000)      -> do not write ckpt during validation
#
# Preserved (identical to 16n production):
#   - Pretrained ckpt: date0612-...-from-epoch2-iter20480-release-load (TP=8 PP=1)
#   - Tokenizer:       /ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
#   - Optimizer:       muon (momentum 0.95, ns_steps 5, matched_adamw_rms 0.10)
#   - Loss:            --cross-entropy-loss-fusion
#   - Precision:       --bf16
#   - Sequence parallel: DISABLED (NoSP -- proven necessary to avoid iter-1 NaN)
#   - Recompute:       full + uniform + 1 layer (required without SP at TP=8)
#   - Trainable:       language_model + adapter + vision_model
#   - Init:            --finetune --no-load-optim --no-load-rng (release ckpt)
#   - Packed env:      OFFLINE_PACKING_BMR=1 + OFFLINE_PACKED_DATA=1
#
# Data: /ov21/47m_sft_packed_8k/dataset.yaml
#   1) 47m_part1   161303 bins
#   2) 47m_part2   161106 bins
#   3) 47m_part3   161351 bins
#   4) 47m_part4   161323 bins
#   5) 47m_part5   161348 bins
#   6) 47m_part6   161175 bins
#   7) 47m_part7   161207 bins
#   TOTAL          1128813 bins, 1 epoch @ GBS=8 = 141102 iter (full epoch not the goal here)
# =============================================================================

TP="${1:-8}"
PP="${2:-1}"
SEQ_LEN="${3:-8192}"
MBS="${4:-1}"
GBS="${5:-8}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1128813}"
EPOCHS="${EPOCHS:-1}"
# Default: 20 iter for startup validation. Override $6 to set arbitrary NSTEP,
# or set FULL_EPOCH=1 to compute ceil(TOTAL_SAMPLES * EPOCHS / GBS).
if [[ "${FULL_EPOCH:-0}" == "1" ]]; then
    NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"
else
    NSTEP="${6:-20}"
fi
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"  # 0 = do not save during validation
CUSTOM_PIPELINE_LAYERS="${CUSTOM_PIPELINE_LAYERS:-0,12,12,12}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/vlm/yinxie/code/OV21/baseline/checkpoints}"
DATA_PATH=${DATA_PATH:-"/ov21/47m_sft_packed_8k/dataset.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/vlm/yinxie/code/OV2/OV2_public_main/checkpoints/date0612-muon-mixed-6sets-packed-64k-16n-tp8-gbs16-NoSP-from-epoch2-iter20480-release-load"}

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1

# Memory: prevent fragmentation OOM at long-seq cross-entropy (~5GB per CE call).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Kept for parity with 16n production (originally for SP TP-comm overlap):
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=(
        "localhost"
    )
fi

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
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
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
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
fi

SAVE_CKPT_PATH=$OUTPUT_DIR/$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
cp "$0" "${SAVE_CKPT_PATH}/$(basename "$0")"
GPUS_PER_NODE=8

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DP_SIZE=$((WORLD_SIZE / TP / PP))
if (( WORLD_SIZE % (TP * PP) != 0 )); then
    echo "Error: WORLD_SIZE=${WORLD_SIZE} must be divisible by TP*PP=$((TP * PP))."
    exit 1
fi
if (( GBS % DP_SIZE != 0 )); then
    echo "Error: GBS=${GBS} must be divisible by DP_SIZE=${DP_SIZE}."
    exit 1
fi
if (( MBS != 1 )); then
    echo "Error: packed training requires MBS=1."
    exit 1
fi
echo "World Size: ${WORLD_SIZE}, DP Size: ${DP_SIZE}, GBS: ${GBS}, MBS: ${MBS}, NSTEP: ${NSTEP}"

MASTER_ADDR=${MASTER_ADDR:-"${list_ip[0]}"}
MASTER_PORT=${MASTER_PORT:-"26020"}

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
    --model-name llava-onevision2-4b-p16m3
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 2
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 102400
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer muon
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --muon-momentum 0.95
    --muon-ns-steps 5
    --muon-matched-adamw-rms 0.10
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --initial-loss-scale 65536
    --bf16
    --cross-entropy-loss-fusion
    --load "$CHECKPOINT_PATH"
    --finetune
    --no-load-optim
    --no-load-rng
    --save "$SAVE_CKPT_PATH"
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
)

# Only emit --save-interval when SAVE_INTERVAL > 0; passing 0 makes Megatron
# save every step which we don't want during a startup validation run.
if (( SAVE_INTERVAL > 0 )); then
    TRAINING_ARGS+=(--save-interval "$SAVE_INTERVAL")
fi

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
    # --sequence-parallel REMOVED. Bisect (Track B) proved SP causes iter-1 NaN
    # at this seq-length / TP / model. Memory fits with full+uniform+1 recompute.
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

if [[ $PP -gt 1 && -n "$CUSTOM_PIPELINE_LAYERS" ]]; then
    MODEL_PARALLEL_ARGS+=(--custom-pipeline-layers "${CUSTOM_PIPELINE_LAYERS}")
fi

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
