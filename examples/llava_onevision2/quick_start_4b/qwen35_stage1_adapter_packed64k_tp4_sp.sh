#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2 4B *Qwen3.5* backbone — stage-1 alignment (adapter-only),
# packed 64k, TP=4 + Sequence-Parallel.  Launch via launch_date05_training.sh
# (pass --ips "<node list>"; it exports LIST_IP/MASTER_ADDR/MASTER_PORT).
#
# WHAT IS DIFFERENT vs the Qwen3 (p16m3) scripts — all REQUIRED for Qwen3.5:
#   * AIAK_TRAINING_PATH -> /vlm/yinxie/code/OV21/qwen35  (the tree with the
#     qwen35 hybrid-attention backbone; the old baseline tree has no qwen3_5).
#   * --model-name llava-onevision2-4b-qwen35  (hybrid GatedDeltaNet+full attn;
#     language_model_type=qwen3_5 is set by the arch config, NOT the CLI).
#   * TOKENIZER = the Qwen3.5 processor (vocab 248320, image_pad=248056).  The
#     old p16m33 tokenizer is vocab 151936 and is INCOMPATIBLE.
#   * --load = the TP4 mcore ckpt converted from the merged Qwen3.5 HF model.
#   * SP + M-RoPE fix (commit 21fc2db) is required and already in this tree.
#
# >>> CRITICAL DATA PREREQUISITE <<<
#   The packed dataset MUST be re-packed/re-tokenized with the Qwen3.5
#   tokenizer (248320 vocab; image/video/vision token ids 248056/57/53/54).
#   The existing p16m33-packed bins hold Qwen2/Qwen3 token ids (151655 image
#   token, 151936 vocab) and would feed WRONG ids to the Qwen3.5 model.
#   Point DATA_PATH at a Qwen3.5-tokenized packed 64k yaml before a real run.
#
# Stage: default trains the ADAPTER ONLY (the merged model ships a randomly
#   initialized merger/adapter — align it first).  Set TRAINABLE to
#   "language_model adapter vision_model" for the later full (stage-1.5) run.
#
# Parallelism: TP=4 PP=1 -> DP = WORLD/4.  GBS must be divisible by DP.
#   MBS MUST be 1 (packing).  SP requires TP>1 (satisfied).
# =============================================================================
TP="${1:-4}"
PP="${2:-1}"
SEQ_LEN="${3:-65536}"
MBS="${4:-1}"
GBS="${5:-32}"
TRAINABLE="${TRAINABLE:-adapter}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-350100}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/vlm/yinxie/code/OV21/qwen35}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/vlm/yinxie/code/OV21/qwen35/checkpoints}"
# CRITICAL: must be a Qwen3.5-tokenizer packed dataset (see header). Placeholder:
DATA_PATH=${DATA_PATH:-"/vlm/yinxie/code/yamls/qwen35_packed_64k/REPLACE_WITH_QWEN35_PACKED.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/vlm/yinxie/pretrain_models/ov2_qwen35_processor"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/vlm/yinxie/pretrain_models/llava_onevision2_4b_qwen35_p16m3_mcore_tp4pp1"}

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=( "localhost" )
fi

CURRENT_IP=$(hostname -I | awk "{print \$1}")
[ -z "$CURRENT_IP" ] && CURRENT_IP=$(hostname -i 2>/dev/null | awk "{print \$1}")

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

NNODES=${#list_ip[@]}
MASTER_ADDR=${list_ip[0]}
if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1; MASTER_ADDR=127.0.0.1; NODE_RANK=0
else
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]] && { NODE_RANK=$i; break; }
    done
    [ "$NODE_RANK" -eq -1 ] && { echo "Error: IP ($CURRENT_IP) not in list."; exit 1; }
fi
echo "--- ${NNODES} node(s) | master ${MASTER_ADDR} | rank ${NODE_RANK} | ip ${CURRENT_IP} ---"

SAVE_CKPT_PATH=$OUTPUT_DIR/$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"
mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"
cp "$0" "${SAVE_CKPT_PATH}/$(basename "$0")"
GPUS_PER_NODE=8

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DP_SIZE=$((WORLD_SIZE / TP / PP))
(( WORLD_SIZE % (TP * PP) == 0 )) || { echo "Error: WORLD_SIZE=${WORLD_SIZE} % (TP*PP)=$((TP*PP)) != 0"; exit 1; }
(( GBS % DP_SIZE == 0 )) || { echo "Error: GBS=${GBS} % DP_SIZE=${DP_SIZE} != 0"; exit 1; }
(( MBS == 1 )) || { echo "Error: packed training requires MBS=1."; exit 1; }
echo "World ${WORLD_SIZE} | DP ${DP_SIZE} | GBS ${GBS} | MBS ${MBS} | SEQ ${SEQ_LEN} | NSTEP ${NSTEP} | trainable=${TRAINABLE}"

MASTER_PORT=${MASTER_PORT:-"26055"}
if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" )
else
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" )
fi

MODEL_ARGS=( --model-name llava-onevision2-4b-qwen35 )

# Qwen3.5 hybrid-attention hyperparams. These also have defaults in the arch
# config (copied onto args), but we pass them explicitly for clarity/override.
# NOTE: language_model_type is set by the arch config (not a valid CLI choice).
QWEN35_ARGS=(
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
    --num-workers 2
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules ${TRAINABLE}
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
    --adam-beta2 0.999
    --adam-eps 1e-08
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

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seq${SEQ_LEN}_gbs${GBS}_${NSTEP}steps.log"

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${QWEN35_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
