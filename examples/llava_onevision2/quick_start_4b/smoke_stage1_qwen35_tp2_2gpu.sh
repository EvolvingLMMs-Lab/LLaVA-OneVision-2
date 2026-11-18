#!/usr/bin/env bash
# =============================================================================
# SELF-CONTAINED stage-1 alignment SMOKE (TP=2 PARITY CHECK)
#   LLaVA-OneVision2-4B / Qwen3.5 LLM | single node, 2 GPUs, TP=2 PP=1 + SP.
#
# Purpose: prove the TP=2 converted checkpoint LOADS into a TP=2-parallel model
#   with 0 missing / 0 unexpected keys and produces a FINITE loss through the
#   full vision->adapter->Qwen3.5-hybrid-LLM forward/backward. A wrong sharding
#   axis would surface here as a shape-mismatch load error or a NaN/garbage loss.
#   Sequence-parallel is ENABLED: the GatedDeltaNet (linear-attn) params are
#   REPLICATED across TP ranks by the converter and SP's reduce_scatter/tp_size
#   is what corrects that replication -> SP is required for TP>1 correctness.
#
#   All args inlined (no script-calls-script), mirroring the TP1 smoke.
# Usage (inside container):
#   CUDA_VISIBLE_DEVICES=<g0>,<g1> \
#     bash examples/llava_onevision2/quick_start_4b/smoke_stage1_qwen35_tp2_2gpu.sh
# =============================================================================
set -u

TP="${1:-2}"
PP="${2:-1}"
SEQ_LEN="${3:-4096}"
MBS="${4:-1}"
GBS="${5:-1}"
NSTEP="${6:-3}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/LLaVA-OneVision-2.5/tmp_smoke_qwen35_tp2}"
DATA_PATH="${DATA_PATH:-/data/47m_sft_packed_8k/dataset.yaml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33}"
# Converted Qwen3.5-backbone mcore checkpoint (TP2/PP1).
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/models/llava_onevision2_4b_qwen35_p16m3_mcore_tp2pp1}"

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE="$TP"
NNODES=1
MASTER_ADDR=127.0.0.1
MASTER_PORT="${MASTER_PORT:-26088}"

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DP_SIZE=$((WORLD_SIZE / TP / PP))
echo "World Size: ${WORLD_SIZE}, DP Size: ${DP_SIZE}, GBS: ${GBS}, MBS: ${MBS}, NSTEP: ${NSTEP}"
echo "Visible GPU(s): ${CUDA_VISIBLE_DEVICES:-<all>}"

SAVE_CKPT_PATH="$OUTPUT_DIR/$(basename "$0" .sh)"
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"
mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --model-name llava-onevision2-4b-qwen35
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 32
    --linear-conv-kernel-dim 4
    --full-attention-interval 4
    --mrope-section 11 11 10
    --mrope-interleaved
    --partial-rotary-factor 0.25
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 1
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules adapter
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 102400
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
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
    --ckpt-format torch
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_${NSTEP}steps.log"

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
