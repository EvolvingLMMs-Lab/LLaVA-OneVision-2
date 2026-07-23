#!/bin/bash
# =============================================================================
# Continue-training (VIT+ADAPTER) / 4B-Qwen3.5 / PACKED / Muon
#
# Resumes from the COMPLETED Qwen3.5 stage-1 adapter-only alignment checkpoint
# (iter_0002181) and continues training with adapter + vision_model trainable
# (LLM frozen), on the packed SFT mix, using the muon optimizer.
#
# Modeled on the official p16m3 packed-muon recipe
#   date0511_ax_stage_1_alignment_p16m3_packed_new16_muon.sh
# with the Qwen3.5-specific plumbing carried over from the stage-1 script:
#   * AIAK_TRAINING_PATH -> /workspace/LLaVA-OneVision-2.5 (qwen3_5 backbone)
#   * --model-name llava-onevision2-4b-qwen35 + QWEN35_ARGS (linear-attn / M-RoPE)
#   * Qwen3.5 processor tokenizer (vocab 248320); packed wds stores raw
#     text+images tokenized live, so it is reused as-is with this tokenizer.
#   * --cross-entropy-loss-fusion (large 248320 vocab CE at TP1).
#
# NEW-STAGE semantics: the stage-1 checkpoint provides ONLY the pretrained
# weights (initialization). --finetune loads those weights, resets iteration to
# 0 and skips optimizer/rng/dataloader state, so this stage trains from step 0
# with a fresh muon optimizer and a fresh cosine LR schedule (warmup from 0).
#
# Nodes: 192.0.2.12 (cluster2) + 192.0.2.14 (cluster1). Cross-cluster is one
# RoCE fabric and works with UNIFORM auto-detect containers (both on the same
# llava_megatron:26.07 image, no forced-IB NCCL env) -> keep NCCL auto-detect.
# =============================================================================
TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-32000}"
MBS="${4:-1}"
GBS="${5:-48}"   # 6 nodes x8 = DP48 @ TP1 (1 microbatch/DP)
TRAINABLE="${TRAINABLE:-adapter vision_model}"
# Activation recompute: none | full | block:<N>. Measured on 6xA800 GBS48:
# none OOMs (GatedDeltaNet qkvz/chunk intermediates too big even with frozen
# LLM); block:16 (recompute 16, store 16) = 4.6 s/step vs full 5.2 (=12%% faster)
# at 73GB peak. Default block:16.
RECOMPUTE="${RECOMPUTE:-block:16}"
# RESUME=1: continue THIS run from its own latest checkpoint (loads weights +
# muon optimizer + iteration; lr schedule continues). Default 0 = fresh start
# from the stage-1 checkpoint via --finetune (step 0, fresh optimizer).
RESUME="${RESUME:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8000000}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/LLaVA-OneVision-2.5/checkpoints}"
DATA_PATH=${DATA_PATH:-"/ov2/dataset_sft/seed85m_video20m_47m5p_packed/dataset.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/models/ov2_qwen35_processor"}
# Resume weights from the completed stage-1 adapter-only alignment run.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/LLaVA-OneVision-2.5/checkpoints/qwen35_stage_1_alignment_adapter_only"}

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=( "192.0.2.12" "192.0.2.14" "192.0.2.17" "192.0.2.27" "192.0.2.40" "192.0.2.43" )
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
(( MBS == 1 )) || { echo "Error: packed training requires MBS=1."; exit 1; }
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
    --num-workers 12
    --chat-template qwen2-vl
)

RECOMPUTE_ARGS=()
case "$RECOMPUTE" in
  full)    RECOMPUTE_ARGS=( --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 );;
  block:*) RECOMPUTE_ARGS=( --recompute-granularity full --recompute-method block --recompute-num-layers "${RECOMPUTE#block:}" );;
esac

if [[ "$RESUME" == "1" ]]; then
    LOAD_ARGS=( --load "$SAVE_CKPT_PATH" )
else
    LOAD_ARGS=( --finetune --load "$CHECKPOINT_PATH" --no-load-optim --no-load-rng )
fi

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
    --optimizer muon
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --muon-momentum 0.95
    --muon-ns-steps 5
    --muon-matched-adamw-rms 0.2
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --cross-entropy-loss-fusion
    "${LOAD_ARGS[@]}"
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
    ${RECOMPUTE_ARGS[@]:+"${RECOMPUTE_ARGS[@]}"} \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
