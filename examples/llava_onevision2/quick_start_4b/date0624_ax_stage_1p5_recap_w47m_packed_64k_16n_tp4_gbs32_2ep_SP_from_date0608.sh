#!/usr/bin/env bash
# =============================================================================
# BEST-SPEED config (2026-06-24): LLaVA-OneVision-2-4B SFT, 64k packed.
#   Sequence-Parallel (SP) + TP=4   ->   ~1.6x vs the old NoSP/TP=8 baseline.
#
# Measured on 2 nodes (TP=8 vs 4), GBS=4, 64k, IDENTICAL bins, wall-clock ms/iter:
#   NoSP TP=8 : 25,209 ms   (1.00x, old prod)
#   SP   TP=8 : 20,554 ms   (1.23x)
#   SP   TP=4 : 15,862 ms   (1.59x)   <-- THIS CONFIG
#
# Why each choice (all empirically verified on nodes 22/26):
#   * --sequence-parallel = ON.  The earlier "SP causes iter-1 NaN" is FIXED in
#     this baseline: the offline packer left cu_seqlens[-1] < tokens width (a few
#     trailing pad tokens in no attention segment); harmless without SP, but under
#     SP the scatter/gather length mismatch produced NaN dgrad in linear_fc2.
#     Fix lives in aiak_training_llm/train/pretrain/pretrain_llava_onevision2.py
#     ("Cover the packing-pad tail").  SP loss == NoSP loss bit-for-bit.
#   * TP=4.  The 4B model was over-sharded at TP=8 (comm/efficiency-bound).  SP
#     frees enough memory that TP=4 now fits at 64k (~50/80 GB) and runs 1.3x
#     faster than TP=8.  >>> REQUIRES the load ckpt resharded TP=8 -> TP=4 <<<
#     (see README_BEST_SPEED.md).
#   * recompute full/uniform/1 = KEEP.  selective/looser OOMs at 64k; the ~30 GB
#     headroom at TP=4 is not enough to relax it.
#   * NO --tp-comm-overlap.  Tested: ~16% SLOWER here (no CUDA Multicast -> CUDA
#     IPC fallback overhead exceeds the intra-node NVLink TP-comm savings).
#   * --num-workers 2.  Raising to 4/8 made no measurable difference (compute-bound).
#   * Do NOT use TP=2 (OOMs on the fp32 cross-entropy logits) or TP=2+CP=2
#     (CP+packed RoPE is broken in this codebase: fused needs TE>=1.12; native
#     path off-by-one).
#
# Code: runs the FIXED baseline at AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5
#       (contains both the SP tail-fix and adapter tp_comm_overlap fix).
# Data: recap pandas family, weights = real bin counts (350,100 total bins).
# Parallelism @ 16 nodes: TP=4 PP=1 -> DP=32  =>  GBS must be divisible by 32.
# NSTEP default = ceil(TOTAL_SAMPLES * EPOCHS / GBS); set TOTAL_SAMPLES to the
#       yaml's real total bins (currently 350,100) and EPOCHS as desired.
# =============================================================================
TP="${1:-4}"
PP="${2:-1}"
SEQ_LEN="${3:-102400}"
MBS="${4:-1}"
GBS="${5:-32}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-350100}"
EPOCHS="${EPOCHS:-2}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"
CUSTOM_PIPELINE_LAYERS="${CUSTOM_PIPELINE_LAYERS:-0,12,12,12}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/LLaVA-OneVision-2.5/checkpoints/date0624-recap-w47m-packed-64k-16n-tp4-gbs32-2ep-SP-from-date0608}"
DATA_PATH=${DATA_PATH:-"/data/yamls/mixed_6sets_packed_64k/mixed_6sets_packed_64k/dataset_fix_recap_w47m_20260624.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/LLaVA-OneVision-2.5/checkpoints/date0608-release-load-TP4"}

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1

# Memory: prevent fragmentation OOM at long-seq cross-entropy (~5GB per CE call).
# expandable_segments allows PyTorch to grow allocations rather than failing on fragmented pools.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Kept (was originally for sequence-parallel TP-comm overlap; harmless without SP):
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=(
        "192.0.2.19"
        "192.0.2.20"
        "192.0.2.21"
        "192.0.2.22"
        "192.0.2.23"
        "192.0.2.25"
        "192.0.2.26"
        "192.0.2.27"
        "192.0.2.28"
        "192.0.2.29"
        "192.0.2.30"
        "192.0.2.31"
        "192.0.2.32"
        "192.0.2.33"
        "192.0.2.34"
        "192.0.2.35"
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
MASTER_PORT=${MASTER_PORT:-"26042"}

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
    --lr-warmup-iters 200
    --initial-loss-scale 65536
    --bf16
    --cross-entropy-loss-fusion
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 500
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
    --sequence-parallel
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
