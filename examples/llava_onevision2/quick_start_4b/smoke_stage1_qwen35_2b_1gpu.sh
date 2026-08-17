#!/usr/bin/env bash
# =============================================================================
# SELF-CONTAINED stage-1 alignment SMOKE  |  LLaVA-OneVision2-4B / Qwen3.5 LLM
# Single node, SINGLE GPU, TP=1 PP=1, adapter-only (freeze ViT + LLM), 5 iters.
#
# Purpose: prove "the new Qwen3.5-backbone VLM can be trained" end-to-end:
#   loads the converted mcore checkpoint, runs a handful of real data batches
#   through the full vision->adapter->Qwen3.5-hybrid-LLM forward/backward, and
#   prints a finite loss. NOT a real training run.
#
# This is ONE complete self-contained script (no script-calls-script): the
# torchrun launch, env, data plumbing, all Qwen3.5 hybrid-attention args, and
# the mcore-checkpoint load are inlined below.
#
# Usage (inside container llava_megatron_container):
#   CUDA_VISIBLE_DEVICES=<free_gpu> \
#     bash examples/llava_onevision2/quick_start_4b/smoke_stage1_qwen35_1gpu.sh
#
# Overridable via positional args: TP PP SEQ_LEN MBS GBS NSTEP
# =============================================================================
set -u

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-4096}"      # samples truncate to this; caps the 248320-vocab CE logits (~4GiB) to fit the shared ~24GB GPU. Must be large enough to hold a p16m3 image+QA (2048 is too small).
MBS="${4:-1}"             # packed training requires MBS=1
GBS="${5:-1}"             # single GPU, DP=1 -> GBS=1
NSTEP="${6:-5}"           # smoke: 5 iters
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}"

# ---- repo / model / data / tokenizer paths ---------------------------------
AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/LLaVA-OneVision-2.5/tmp_smoke_qwen35}"
DATA_PATH="${DATA_PATH:-/data/47m_sft_packed_8k/dataset.yaml}"
# p16m3 tokenizer (Qwen2.5-VL token space: image_token_id=151655), consistent
# with the packed p16m3 data AND the provider's image/video token ids.
TOKENIZER_PATH="${TOKENIZER_PATH:-/models/ov2_qwen35_processor}"
# Converted Qwen3.5-backbone mcore checkpoint (TP1/PP1, loads 0 missing/0 unexpected).
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/models/llava_onevision2_2b_qwen35_p16m3_mcore_tp1pp1}"

# Offline-packed data (both required for correct THD attention isolation).
export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=0  # NON-packed: packed_seq_params=None so the qwen35 gated-deltanet skips cu_seqlens_cpu (unset by get_batch) AND the real get_rope_index path runs

# Memory: avoid fragmentation OOM at cross-entropy on the padded vocab.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- single-node / single-GPU launcher -------------------------------------
GPUS_PER_NODE=1
NNODES=1
MASTER_ADDR=127.0.0.1
MASTER_PORT="${MASTER_PORT:-26077}"

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DP_SIZE=$((WORLD_SIZE / TP / PP))
echo "World Size: ${WORLD_SIZE}, DP Size: ${DP_SIZE}, GBS: ${GBS}, MBS: ${MBS}, NSTEP: ${NSTEP}"
echo "Visible GPU(s): ${CUDA_VISIBLE_DEVICES:-<all>}"

SAVE_CKPT_PATH="$OUTPUT_DIR/$(basename "$0" .sh)"
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"
mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"
cp "$0" "${SAVE_CKPT_PATH}/$(basename "$0")"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ---- model: the NEW Qwen3.5-backbone arch ----------------------------------
MODEL_ARGS=(
    --model-name llava-onevision2-2b-qwen35
    # Qwen3.5 hybrid-attention backbone args (also carried by the arch config;
    # passed explicitly per the training-recipe contract).
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 16
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
    --use-distributed-optimizer
    --distributed-backend nccl
    # No SP (single GPU). Activation recompute to fit the shared ~24GB budget.
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
