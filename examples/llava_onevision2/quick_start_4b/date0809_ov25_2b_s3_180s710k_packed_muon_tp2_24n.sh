#!/bin/bash
# =============================================================================
# STAGE-3 (s3) FULL-MODEL continue-training / 2B-OV2.5 (Qwen3.5 hybrid LLM)
# LONG-CONTEXT 64k packed data / Muon / TP2+SP
# -----------------------------------------------------------------------------
# Continues from the COMPLETED s2 full-model run (date0805 ...tp2_24n) at
# iter_0081000. TP4 was ARCHITECTURALLY IMPOSSIBLE (Qwen3.5-2B full-attn layers
# have only 2 KV heads / num_query_groups=2, not divisible by TP4), so this stage
# stays TP2 and loads the s2 TP2 checkpoint DIRECTLY (bit-exact, no conversion).
# --finetune = weights only, fresh muon, step 0, fresh cosine.
#
# WHY 64k long-context: stage-3 data is packed to 64k-token bins and a single
# 180s video sample can reach ~50k tokens (>32768). So seq-length AND
# max-position-embeddings are raised to 65536. This stage is effectively a
# long-context extension (positions 32768..65536 were unseen by s2 weights).
# TP2 (not TP4) gives NO extra activation relief vs s2's 32k, and seq doubles to
# 64k, so RECOMPUTE=full is the safe default (relax to block:N/none if headroom).
#
# Stage naming (user): s0=adapter-only, s1=vit+adapter, s2=full model,
#                      s3=full model on long-context 64k data (this script).
#
# DATA: /vlm/yinxie/code/yamls/qwen35_2b/stage3_ov1547m_180s710k.yaml
#   47m single-image 45% (147,480 bins) + 180s video 55% (180,204 bins)
#   = 327,684 bins total. GBS48 -> 1 epoch = 6827 iters (default EPOCHS=1).
#
# PARALLELISM: 24 nodes x 8 = 192 GPUs; TP2 PP1 -> DP96; GBS96 = 1 mb/DP.
#
# ---- DECISIONS BAKED IN ------------------------------------------------------
#   SEQ_LEN=65536, MAX_POS=65536   (required by 64k bins + long video samples)
#   GBS=96   (TP2->DP96; GBS48 infeasible at TP2/24nodes, min=96; effective batch
#            2x vs the requested 48, so NSTEP halves; per-GPU microbatch still 1)
#   TP=2     (TP4 impossible: 2 KV heads); PP=1 (no pipeline bubble)
#   LR=2.0e-5   (same as s2; adjust if you want lower for long-context)
#   EPOCHS=1 -> NSTEP=3414         (edit EPOCHS if needed)
#   RECOMPUTE=full   (64k seq at TP2 = 2x s2's activation, no TP relief -> full
#                     recompute safe default; try block:N/none once headroom seen)
# =============================================================================
TP="${1:-2}"
PP="${2:-1}"
SEQ_LEN="${3:-102400}"
MBS="${4:-1}"
GBS="${5:-96}"                                   # TP2 -> DP96; GBS96 = 1 mb/DP
MAX_POS="${MAX_POS:-102400}"                      # long-context extension
TRAINABLE="${TRAINABLE:-language_model adapter vision_model}"  # FULL MODEL
RECOMPUTE="${RECOMPUTE:-full}"                   # 64k seq at TP2 (no TP relief vs s2 32k) -> full recompute safe; set block:N/none if headroom
RESUME="${RESUME:-0}"                            # 0=--finetune from TP4 base; 1=resume THIS run
TOTAL_BINS="${TOTAL_BINS:-327684}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_BINS * EPOCHS + GBS - 1) / GBS ))}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/vlm/yinxie/code/OV21/qwen35}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/vlm/yinxie/code/OV21/qwen35/checkpoints}"
DATA_PATH=${DATA_PATH:-"/vlm/yinxie/code/yamls/qwen35_2b/stage3_ov1547m_180s710k.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/vlm/yinxie/pretrain_models/ov2_qwen35_processor"}
# s3 initializes from the s2-final TP2 checkpoint DIRECTLY (bit-exact, no conversion).
# --finetune loads iter_0081000 weights (latest in this dir), resets iter/optimizer.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/vlm/yinxie/code/OV21/qwen35/checkpoints/date0805_qwen35_2b_fullmodel_packed_muon_tp2_24n"}

export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Sequence-chunked cross-entropy: 64k x 248k-vocab logits OOM the fp32 CE buffer (~30GB)
# at TP2. Chunk=8 shrinks the transient fp32 buffer ~8x (numerically identical CE).
export CE_SEQ_CHUNKS="${CE_SEQ_CHUNKS:-8}"

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=( "172.16.5.32" "172.16.5.33" "172.16.5.34" "172.16.5.35" "172.16.5.36" "172.16.5.38" "172.16.5.39" "172.16.5.18" "172.16.5.41" "172.16.5.42" "172.16.5.43" "172.16.5.44" "172.16.5.45" "172.16.5.46" "172.16.5.47" "172.16.5.49" "172.16.5.81" "172.16.5.82" "172.16.5.85" "172.16.5.87" "172.16.5.88" "172.16.5.90" "172.16.5.91" "172.16.5.92" )
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
echo "World ${WORLD_SIZE} | DP ${DP_SIZE} | GBS ${GBS} | MBS ${MBS} | SEQ ${SEQ_LEN} | MAXPOS ${MAX_POS} | NSTEP ${NSTEP} | trainable=${TRAINABLE}"

MASTER_PORT=${MASTER_PORT:-"26000"}
if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" )
else
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" )
fi

MODEL_ARGS=( --model-name llava-onevision2-2b-qwen35 )

QWEN35_ARGS=(
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 16
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
    --num-workers 16
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
    --max-position-embeddings "${MAX_POS}"
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
    --distributed-timeout-minutes 30   # 64k monster bins can stall a rank; 30min tolerates NFS/preproc spikes
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
