# =============================================================================
# LLaVA-OneVision-2.0 / 4B-p14m2 / Single-node QuickStart tutorial
#
# Trains LLaVA-OneVision-2.0-4B (p14m2) on ov2_quickstart (L=8192):
#   <REPO_ROOT>/ov2_quickstart/packed_mixed_sft_cap_v30s/
#   (4 nodes × ~55k bins = 219,907 packed sequences from 2.03M input samples
#    mixing SFT 1M + caption 1M + 30s-video 50k)
#
# This is the *tutorial* entry point — single node, 8 GPUs, no list_ip /
# NODE_RANK plumbing. For the production multi-node recipe see
# `ax_stage_1_alignment_p14m3_packed.sh` in this same directory.
#
# Why these gates are mandatory (verified pretrain_llava_onevision2.py +
# task_encoder.py, see skill: offline-packing-env-vars):
#   - OFFLINE_PACKING_BMR=1  -> per sub-sample encode via MultiMixQASample
#                              (multi-turn aware; correct text/labels per
#                               sub-sample inside a packed sequence)
#   - OFFLINE_PACKED_DATA=1  -> batch() reads real cu_lengths/max_lengths,
#                              LLM forward gets PackedSeqParams(qkv_format="thd",
#                              cu_seqlens_q=...) so flash-attn varlen kernel
#                              enforces causal attention WITHIN each sub-sample
#                              and zero attention across sub-sample boundaries.
#
# Both gates require MBS=1 (one packed sequence per micro-batch).
# =============================================================================

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-10192}"
MBS="${4:-1}"
GBS="${5:-16}"
EPOCHS="${6:-1}"
# Bin count of ov2_quickstart (sum of node_{a..d} .info.yaml shard_counts).
# Verified by energon load smoke test: 54480 + 54854 + 54785 + 54788 = 219907.
TOTAL_BINS="${TOTAL_BINS:-219907}"
# ceil(TOTAL_BINS * EPOCHS / GBS) so the last partial global-batch still trains.
NSTEP=$(( (TOTAL_BINS * EPOCHS + GBS - 1) / GBS ))
CUSTOM_PIPELINE_LAYERS="${CUSTOM_PIPELINE_LAYERS:-0,12,12,12}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-./output/quick_start_4b}"
DATA_PATH="${DATA_PATH:-./ov2_quickstart/packed_mixed_sft_cap_v30s/dataset.yaml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./ov2_quickstart/ov_encoder_p14m22_qwen3_hf}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./ov2_quickstart/ov_encoder_p14m22_qwen3_mcore_tp1pp1}"

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-26000}"
NNODES=1
NODE_RANK=0

echo "--- LLaVA-OneVision-2.0 4B QuickStart (single node) ---"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "TP=${TP} PP=${PP} MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN} NSTEP=${NSTEP}"
echo "DATA_PATH: ${DATA_PATH}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"

SAVE_CKPT_PATH="$OUTPUT_DIR/$(basename "$0" .sh)"
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --model-name llava-onevision2-4b-p14m2
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

TRAINING_ARGS=(
    --training-phase sft
    # Full-parameter QuickStart: train adapter + vision_model + language_model.
    # `all` is the default and triggers the full-param path in
    # llava_onevision2_provider.py (no module is frozen).
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
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
    --save-interval 2000
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
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
