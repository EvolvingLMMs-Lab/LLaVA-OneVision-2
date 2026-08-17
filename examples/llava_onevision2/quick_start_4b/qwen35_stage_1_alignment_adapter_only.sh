#!/bin/bash
# =============================================================================
# Stage-1 Alignment / 4B-*Qwen3.5* / adapter-only / NON-packed
#
# Hyperparameters aligned to the official p16m3 stage-1 alignment recipe
# (ax_stage_1_alignment_p16m3_adapter_only.sh): non-packed 558k LLaVA
# alignment data, TP1/PP1, seq 32k, GBS 256, lr 2e-5, adapter-only.
#
# Qwen3.5-specific (vs the p16m3 recipe):
#   * AIAK_TRAINING_PATH -> /workspace/LLaVA-OneVision-2.5 (has the qwen3_5 backbone)
#   * --model-name llava-onevision2-4b-qwen35 (language_model_type=qwen3_5 from arch config)
#   * TOKENIZER = Qwen3.5 processor (vocab 248320); alignment wds stores raw
#     text+images and is tokenized live, so it is reused as-is.
#   * --load = the qwen35 mcore ckpt at the SAME TP as this run.
#   * QWEN35_ARGS (linear-attn / M-RoPE / partial-rotary) + ViT per-frame attn.
#   * --cross-entropy-loss-fusion ADDED: Qwen3.5 vocab (248320) is 63% larger
#     than p16m3 (151936); at TP=1 the seq-32000 CE would otherwise be heavy.
#
# >>> MEMORY NOTE <<< If TP=1 OOMs on the 248320-vocab cross-entropy at seq
#   32000 (even with fusion), rerun with TP=2 (arg1=2) and the matching ckpt
#   .../llava_onevision2_4b_qwen35_p16m3_mcore_tp2pp1 (add --sequence-parallel
#   for TP>1 if desired).
# =============================================================================
TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-32000}"
MBS="${4:-1}"
GBS="${5:-256}"
TRAINABLE="${TRAINABLE:-adapter}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-558128}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/LLaVA-OneVision-2.5/checkpoints}"
DATA_PATH=${DATA_PATH:-"/vlm/data/blip_laion_cc_sbu_558k_wds"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/models/ov2_qwen35_processor"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/models/llava_onevision2_4b_qwen35_p16m3_mcore_tp${TP}pp${PP}"}

export OFFLINE_PACKING_BMR=0
export OFFLINE_PACKED_DATA=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=( "localhost" )
fi

CURRENT_IP=$(hostname -I | awk '{print $1}')
[ -z "$CURRENT_IP" ] && CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')

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
echo "World ${WORLD_SIZE} | DP ${DP_SIZE} | GBS ${GBS} | MBS ${MBS} | SEQ ${SEQ_LEN} | NSTEP ${NSTEP} | trainable=${TRAINABLE}"

MASTER_PORT=${MASTER_PORT:-"26000"}
if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" )
else
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" )
fi

MODEL_ARGS=( --model-name llava-onevision2-4b-qwen35 )

# Qwen3.5 hybrid-attention hyperparams (also defaulted by the arch config).
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
    --vision-frame-windows-size 1
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 2
    --chat-template qwen2-vl
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules ${TRAINABLE}
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 2.0e-5
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
    --cross-entropy-loss-fusion
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 1000
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
# Sequence-parallel only helps (and is only valid) for TP>1.
if (( TP > 1 )); then
    MODEL_PARALLEL_ARGS+=( --sequence-parallel )
fi

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
