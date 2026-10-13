#!/bin/bash
# =============================================================================
# Stage-1 Alignment (s0) / 35B-A3B *Qwen3.6* MoE / adapter-only / NON-packed
# (35B-A3B dims: 40 hybrid layers / hidden 2048 / 16h-2kv / linear value heads 32
#  / MoE every layer: 256 routed experts top-8 (moe_ffn 512) + gated shared expert 512)
# 1 node x 8 A800, TP1 EP8 PP1 -> DP8 (each EP rank holds 32 routed experts).
# Recipe identical to the 4B/9B/27B s0 (558k LLaVA alignment data, seq 32k,
# GBS 256, lr 2e-5 -> 1e-6 cosine, adapter-only, 1 epoch = 2181 steps).
#
# MoE-specific:
#   * --load = mcore ckpt split into EP8 shards (release/mp_rank_00_{000..007}).
#   * Router math == HF qwen3_5_moe: pre-softmax OFF (topk on logits, softmax
#     over the k) == HF softmax(all)->topk->renormalize. Router in fp32.
#   * Shared-expert sigmoid gate enabled by the qwen35 layer spec
#     (QWEN35_SHARED_EXPERT_GATE, default on).
#   * Load-balancing aux loss OFF (coeff 0, type none): the LLM is frozen in
#     s0, and a nonzero aux loss would back-propagate into the adapter and
#     make it optimize expert balance instead of alignment.
# =============================================================================
TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-32000}"
MBS="${4:-1}"
GBS="${5:-256}"
EP="${EP:-8}"
TRAINABLE="${TRAINABLE:-adapter}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-558128}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/vlm/yinxie/code/OV21/qwen35}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-/vlm/yinxie/ablation/splice_s0}"
DATA_PATH=${DATA_PATH:-"/vlm/data/blip_laion_cc_sbu_558k_wds"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/vlm/yinxie/pretrain_models/ov2_qwen35_processor"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/vlm/yinxie/pretrain_models/llava_onevision2_35b_a3b_qwen36_p16m3_v1010_mcore_tp${TP}pp${PP}ep${EP}"}

export OFFLINE_PACKING_BMR=0
export OFFLINE_PACKED_DATA=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export QWEN35_SHARED_EXPERT_GATE=1

if [[ -n "${LIST_IP:-}" ]]; then
    read -ra list_ip <<< "$LIST_IP"
else
    declare -a list_ip=( "172.16.5.91" )
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
# Auto-resume: once this run has saved, load its own latest checkpoint (model + optimizer + rng
# + dataloader state) instead of the init checkpoint. Used by the watchdog relaunch.
if [[ -f "$SAVE_CKPT_PATH/latest_checkpointed_iteration.txt" ]]; then
    CHECKPOINT_PATH="$SAVE_CKPT_PATH"
    echo "[resume] loading own checkpoint: $SAVE_CKPT_PATH (iter $(cat "$SAVE_CKPT_PATH/latest_checkpointed_iteration.txt"))"
fi
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"
mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"
cp "$0" "${SAVE_CKPT_PATH}/$(basename "$0")"
GPUS_PER_NODE=8

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DP_SIZE=$((WORLD_SIZE / TP / PP))
(( WORLD_SIZE % (TP * PP) == 0 )) || { echo "Error: WORLD_SIZE=${WORLD_SIZE} % (TP*PP)=$((TP*PP)) != 0"; exit 1; }
(( GBS % DP_SIZE == 0 )) || { echo "Error: GBS=${GBS} % DP_SIZE=${DP_SIZE} != 0"; exit 1; }
(( DP_SIZE % EP == 0 )) || { echo "Error: DP_SIZE=${DP_SIZE} % EP=${EP} != 0"; exit 1; }
echo "World ${WORLD_SIZE} | DP ${DP_SIZE} | EP ${EP} | GBS ${GBS} | MBS ${MBS} | SEQ ${SEQ_LEN} | NSTEP ${NSTEP} | trainable=${TRAINABLE}"

MASTER_PORT=${MASTER_PORT:-"26000"}
if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" )
else
    DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" )
fi

MODEL_ARGS=( --model-name llava-onevision2-35b-a3b-qwen35 )

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

# Qwen3.5-MoE (num_experts=256 and moe_ffn_hidden_size=512 come from the arch config).
MOE_ARGS=(
    --expert-model-parallel-size "${EP}"
    --moe-router-topk 8
    --moe-shared-expert-intermediate-size 512
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --moe-router-dtype fp32
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
# Activation recompute (infrastructure only, does not change training math).
# Default OFF: with full recompute this run used ~19 GB/GPU of 80, so recompute only cost time.
# RECOMPUTE=full restores full/uniform/1 (the vision tower follows the LLM setting).
RECOMPUTE="${RECOMPUTE:-none}"
if [[ "$RECOMPUTE" == "full" ]]; then
    DATA_ARGS+=( --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 )
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
    --save-interval "${SAVE_INTERVAL:-250}"
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
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_ep${EP}_pp${PP}_seq${SEQ_LEN}_gbs${GBS}_${NSTEP}steps_rc${RECOMPUTE}.log"

PYTHONPATH="/vlm/yinxie/pyshim:$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${QWEN35_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
