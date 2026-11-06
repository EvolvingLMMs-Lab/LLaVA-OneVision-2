#!/bin/bash
# =============================================================================
# LLaVA-OneVision-2.1 / 4B-qwen35 / single-node QuickStart
#
# This is the tutorial entry point for OV2.1 and the reference target for
# performance work. It is a stripped-down but otherwise faithful copy of the
# production SFT recipe: same model, same parallelism per GPU, same packed data
# path, same optimizer, same environment gates. Only the fleet is smaller
# (1 node instead of 24) and the data is a small bundle.
#
# Run it bare, no arguments needed:
#     bash examples/llava_onevision2/quick_start_4b/quick_start.sh
#
# Positional args (same order as the OV2.0 quick_start.sh):
#     TP PP SEQ_LEN MBS GBS EPOCHS       defaults: 4 1 131072 1 4 1
# Paths come from the environment (see BUNDLE below).
#
# -----------------------------------------------------------------------------
# WHAT IS DIFFERENT FROM OV2.0, AND WHY IT MATTERS FOR OPTIMISATION
# -----------------------------------------------------------------------------
# 1. HYBRID LM. The language model is Qwen3.5-4B: GatedDeltaNet linear-attention
#    layers with full attention only every --full-attention-interval (4). Of the
#    32 LM layers, 8 are attention and 24 are GDN. Any attention-kernel work
#    (cuDNN fused attention, FlashAttention-2/3/4, fused q/k-norm+RoPE) therefore
#    touches 1/4 of the LM. The GDN path is the larger share of LM time and has
#    had no kernel work at all.
#
# 2. LONG PACKED SEQUENCES. Production packs to 110k tokens inside a 131072
#    seq-length. Attention cost is dominated by the sum of squared sub-sequence
#    lengths within a packed row, and the per-sequence host launch overhead is
#    negligible (see the measurements below). This is a different regime from a
#    ~10k-token recipe, where host-side scheduling dominates.
#
# 3. TP4 + SEQUENCE PARALLEL + GDN HEAD PARALLEL. Production runs TP=4 with
#    --sequence-parallel and --gdn-head-parallel (GDN sharded by head across TP).
#    This quick start keeps TP=4, so 8 GPUs give DP=2 with exactly the production
#    per-GPU memory profile.
#
# 4. MUON optimizer, not Adam.
#
# 5. Several environment gates below are load-bearing, not tuning knobs.
#    Removing one does not just slow training down, it OOMs or hangs. Each is
#    commented with what happened when it was absent.
#
# -----------------------------------------------------------------------------
# WHERE THE TIME GOES (measured on the 24-node production run, 2026-11-06)
# -----------------------------------------------------------------------------
# One full per-rank timer block at a 45.6 s step, DP=48, GBS=48, 110k packed:
#
#     forward-compute       mean 14.7 s    max/mean 1.09x
#     backward-compute      mean 10.5 s    min 1.9 s   max 30.0 s   max/mean 2.87x
#     optimizer 0.55 s | all-grads-sync 0.21 s | batch-generator 0.12 s
#
#     step ~= forward 14.7 + backward(slowest rank) 30.0 + ~1  = 45.7   (observed 45.6)
#
# The step waits for the slowest data-parallel rank's backward. Backward spans
# 15.5x across ranks in a single step (1.9 s to 30.0 s) because the packed bins
# differ in how many sub-sequences they hold: attention cost is ~ L^2/n at a
# fixed 110k total, and n ranges from 1 to 146 across data families. Balancing
# backward to its own mean would take the step from 45.6 s to ~26 s.
#
# Two notes for anyone porting cross-rank load balancing here:
#   - Production runs GBS == DP (one micro-batch per rank). A balancer that
#     reassigns whole sequences between ranks cannot change the maximum in that
#     configuration - it is a permutation. It needs GBS >= 2 * DP to have
#     anything to trade. This quick start defaults to GBS = 2 * DP so the
#     non-degenerate case is the one you see first.
#   - What a reassignment moves is the whole packed-sample dict including
#     pixel_values (decoded and patchified, not the compressed images). A
#     vision-heavy 110k bin is ~52 MB on disk as JPEG and an order of magnitude
#     more in tensor form, and production DP spans 24 nodes.
#
# -----------------------------------------------------------------------------
# BUNDLE
# -----------------------------------------------------------------------------
# The packed dataset is the one from the OV2.0 quick start - no new data needed:
#     https://huggingface.co/datasets/lmms-lab-encoder/ov2_quickstart
# It loads unchanged under OV2.1: both packs use `PackedCaptioningSample`, and the
# only difference between the two `sample_loader.py` files is the `data_type`
# marker OV2.1 adds for streaming data, which is `Optional[...] = None` on the
# dataclass. OV2.0 bins simply load with `data_type = None`, i.e. the offline
# path, which is what they are.
#
# The checkpoint and tokenizer cannot be reused: OV2.0 ships a Qwen3 dense model
# (`ov_encoder_p14m22_qwen3_*`), OV2.1-4B is the Qwen3.5 hybrid with a 248320
# vocabulary. Both come from the OV2.1 side.
#
# Expected layout under $BUNDLE_DIR (override any path individually):
#     ov21_quickstart/
#       packed_mixed_sft_cap_v30s/dataset.yaml      from lmms-lab-encoder/ov2_quickstart
#       ov25_4b_qwen35_streaming_tokenizer/         OV2.1 HF tokenizer + config (vocab 248320)
#       ov25_4b_qwen35_mcore_tp4pp1/                OV2.1 Megatron checkpoint, TP4 PP1
#
# The checkpoint is TP-specific: changing TP needs a converted checkpoint
# (examples/llava_onevision2/convert/).
#
# Two things to know about running OV2.0 data through OV2.1:
#   - The pack was budgeted with the Qwen3 tokenizer and a p14/merge2 vision
#     tower. OV2.1 tokenizes the same text with a different vocabulary and
#     patchifies the same JPEGs at p16/merge3, so the realised row lengths differ
#     from the original 8192 target. SEQ_LEN defaults to 16384 here to leave
#     room; rows that still exceed it are dropped by the packing-skip path.
#   - That is also why this default is not the production sequence length. To
#     exercise the long-sequence regime the production recipe lives in, point
#     DATA_PATH at a 110k pack and run
#         bash quick_start.sh 4 1 131072 1 4
# =============================================================================

# =============================================================================

TP="${1:-4}"
PP="${2:-1}"
SEQ_LEN="${3:-16384}"
MBS="${4:-1}"
GBS="${5:-4}"
EPOCHS="${6:-1}"

BUNDLE_DIR="${BUNDLE_DIR:-./ov21_quickstart}"
DATA_PATH="${DATA_PATH:-${BUNDLE_DIR}/packed_mixed_sft_cap_v30s/dataset.yaml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BUNDLE_DIR}/ov25_4b_qwen35_streaming_tokenizer}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BUNDLE_DIR}/ov25_4b_qwen35_mcore_tp4pp1}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/quick_start_4b_qwen35}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/OV2.1}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
# pyshim: fla autotune cap + energon leak fix. REQUIRED for long sequences -
# without it a long row triggers an fla autotune sweep that has hard-hung nodes
# with Xid 31.
PYSHIM_PATH="${PYSHIM_PATH:-${AIAK_TRAINING_PATH%/}/pyshim}"

TRAINABLE="${TRAINABLE:-language_model adapter vision_model}"   # full model
RECOMPUTE="${RECOMPUTE:-full}"

# One epoch over the blend, as min(weight, available bins) per dataset:
#   - OV2.1 production blends carry bin counts as weights, some sub-sampled
#     below what the dataset holds, so the weight is the per-epoch size.
#   - The OV2.0 bundle carries the input-sample count as the weight (2,033,704)
#     while holding 218,907 bins, so there the available count is the size.
# Taking the minimum is right for both.
TOTAL_BINS="${TOTAL_BINS:-$(python3 -c "
import os, sys, yaml
mp = sys.argv[1]
root = os.path.dirname(os.path.abspath(mp))
total = 0
for ds in yaml.safe_load(open(mp))['splits']['train']['datasets']:
    info = os.path.join(root, ds['path'], '.nv-meta', '.info.yaml')
    if not os.path.exists(info):
        info = os.path.join(ds['path'], '.nv-meta', '.info.yaml')
    have = sum(yaml.safe_load(open(info))['shard_counts'].values())
    total += min(int(ds['weight']), have)
print(total)
" "$DATA_PATH" 2>/dev/null)}"
[ -n "$TOTAL_BINS" ] || { echo "Error: cannot count bins from $DATA_PATH (set TOTAL_BINS to override)."; exit 1; }
NSTEP=$(( (TOTAL_BINS * EPOCHS + GBS - 1) / GBS ))

# --- environment gates -------------------------------------------------------
# Correctness gates. OFFLINE_PACKING_BMR makes each sub-sample encode through
# MultiMixQASample (multi-turn aware, correct labels per sub-sample);
# OFFLINE_PACKED_DATA makes batch() emit real cu_lengths so the LLM forward gets
# PackedSeqParams(qkv_format="thd") and attention stays causal WITHIN each
# sub-sample and zero ACROSS sub-sample boundaries. Both require MBS=1.
export OFFLINE_PACKING_BMR=1
export OFFLINE_PACKED_DATA=1
export MROPE_PACKED_FIX=1                  # per-token M-RoPE over the packed row (packed M-RoPE bugfix)

# Memory gates. These are not tuning knobs; each one is here because its absence
# broke a real run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85
export CUDA_MEMORY_FRACTION=0.94           # cap the caching allocator: NCCL allocates its buffers lazily at the
                                           # first grad sync, which died on "Failed to CUDA calloc" with a full cache
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CE_SEQ_CHUNKS=32                    # 128k x 248k-vocab cross-entropy in 4k-token chunks; identical CE
export VISION_CHUNK_PATCHES=2500           # ViT+adapter in image-aligned chunks; the chunk size sets the stage-0
                                           # high water mark (5000 left NCCL unable to allocate at grad sync)
export VISION_RECOMPUTE=0                  # off at PP1: recomputing the tower saved 13 GB but cost 3%
export MUON_DIST_GRAM=1                    # Muon distributed-Gram Newton-Schulz (math-identical, faster)
export ENERGON_LEAKFIX=1                   # energon per-dataset sample-retention fix (worker RSS 17 -> 95 GB without it)
export ENERGON_PREFETCH_FACTOR=1           # halve dataloader shared memory
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=134217728

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-26000}"
NNODES=1
NODE_RANK=0

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DP_SIZE=$((WORLD_SIZE / TP / PP))
(( WORLD_SIZE % (TP * PP) == 0 )) || { echo "Error: WORLD_SIZE=${WORLD_SIZE} % (TP*PP)=$((TP * PP)) != 0"; exit 1; }
(( GBS % DP_SIZE == 0 )) || { echo "Error: GBS=${GBS} % DP_SIZE=${DP_SIZE} != 0"; exit 1; }
(( MBS == 1 )) || { echo "Error: packed training requires MBS=1 (OFFLINE_PACKED_DATA)."; exit 1; }
[ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ] \
    || { echo "Error: init checkpoint $CHECKPOINT_PATH not found."; exit 1; }
[ -d "$TOKENIZER_PATH" ] || { echo "Error: tokenizer $TOKENIZER_PATH not found."; exit 1; }

echo "--- LLaVA-OneVision-2.1 4B-qwen35 QuickStart (single node) ---"
echo "World ${WORLD_SIZE} | TP ${TP} | PP ${PP} | DP ${DP_SIZE} | GBS ${GBS} | MBS ${MBS}"
echo "micro-batches per rank: $((GBS / DP_SIZE))   (1 means every rank trains exactly one packed sequence per step)"
echo "SEQ ${SEQ_LEN} | bins ${TOTAL_BINS} | epochs ${EPOCHS} | NSTEP ${NSTEP} | trainable=${TRAINABLE}"
echo "DATA       : ${DATA_PATH}"
echo "TOKENIZER  : ${TOKENIZER_PATH}"
echo "CHECKPOINT : ${CHECKPOINT_PATH}"

SAVE_CKPT_PATH="$OUTPUT_DIR/$(basename "$0" .sh)"
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"
mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"
cp "$0" "${SAVE_CKPT_PATH}/$(basename "$0")"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=( --model-name llava-onevision2-4b-qwen35 )

# Qwen3.5-4B hybrid attention. full-attention-interval 4 over 32 layers means
# 8 attention layers and 24 GatedDeltaNet layers.
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
    --num-workers 4                       # 2 starved on 110k bins (CPU-side image decode): steps stalled up to 34 min
    --chat-template qwen2-vl
    # Streaming SFT recipe. No effect on offline samples; kept so the quick start
    # exercises the same loss path as production.
    --streaming-loss-scope last-segment
    --state-token-focal-loss
    --state-token-focal-gamma 2.0
    --state-token-focal-captioning
    --state-token-response-alpha-max 5
)

RECOMPUTE_ARGS=()
case "$RECOMPUTE" in
  full)    RECOMPUTE_ARGS=( --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 );;
  block:*) RECOMPUTE_ARGS=( --recompute-granularity full --recompute-method block --recompute-num-layers "${RECOMPUTE#block:}" );;
  none)    RECOMPUTE_ARGS=();;
esac

# Auto-resume from this run's own save dir; otherwise start from the bundle.
if [[ -f "$SAVE_CKPT_PATH/latest_checkpointed_iteration.txt" ]]; then
    LOAD_ARGS=( --load "$SAVE_CKPT_PATH" )
else
    LOAD_ARGS=( --finetune --load "$CHECKPOINT_PATH" --no-load-optim --no-load-rng )
fi

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules ${TRAINABLE}
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
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
    --save-interval 250
    --async-save
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --use-tp-pp-dp-mapping
    --tensor-model-parallel-size "${TP}"
    --sequence-parallel
    --gdn-head-parallel                   # GDN sharded by head across TP (the checkpoint is stored head-parallel)
    --use-distributed-optimizer
    --distributed-backend nccl
    --empty-unused-memory-level 1         # empty_cache() before optimizer.step(): the model-parallel all-reduce
                                          # after the step could not cudaMalloc 6 MB with a full cache
    --distributed-timeout-minutes 120     # a 110k bin can stall a rank; tolerates I/O and preprocessing spikes
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
    --timing-log-option all               # per-rank timers in the log: this is how the backward straggler is found.
                                          # NOTE forward-compute is barrier-synchronised (--barrier-with-L1-time
                                          # defaults on), so its per-rank values are all the global max and carry no
                                          # spread. backward-compute is not barriered - use it to find the pacer.
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=( --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seq${SEQ_LEN}_gbs${GBS}_${NSTEP}steps.log"

PYTHONPATH="$PYSHIM_PATH:$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
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
