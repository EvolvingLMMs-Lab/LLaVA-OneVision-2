#!/usr/bin/env bash
# Terminal-friendly Stage 1.5 chain runner for MobileLLM-R1 + FastViT.
#
# Usage:
#   bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_english_branch_chain.sh imagenet
#   bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_english_branch_chain.sh imagenet datacomp1b
#
# Defaults prepare the full English branch into local Energon/WebDataset shards,
# then run 1000 iterations. Set LOCAL_HF_DATA_ROOT to read an already-downloaded
# HF-style dataset directory instead of streaming from Hugging Face.

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <hf_branch> [<hf_branch> ...]"
    echo "Example: $0 imagenet datacomp1b"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5}"
cd "$REPO_ROOT"

HF_REPO_ID="${HF_REPO_ID:-mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M}"
LANG_CODE="${LANG_CODE:-EN}"
LANG_LOWER="$(echo "$LANG_CODE" | tr '[:upper:]' '[:lower:]')"

PYTHON_BIN="${PYTHON_BIN:-/home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/python}"
TORCHRUN="${TORCHRUN:-/home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/torchrun}"
export PYTHON_BIN TORCHRUN

START_CKPT="${START_CKPT:-$REPO_ROOT/stage_1_alignment_mobilellm_140m_fastvlm_faithful}"
HF_STAGE1_REPO_ID="${HF_STAGE1_REPO_ID:-ranazayed19/mobilellm-fastvit-stage1-2500}"
HF_STAGE1_LOCAL_DIR="${HF_STAGE1_LOCAL_DIR:-$START_CKPT}"
export HF_STAGE1_REPO_ID HF_STAGE1_LOCAL_DIR

ensure_start_checkpoint() {
    if [[ -f "$START_CKPT/latest_checkpointed_iteration.txt" ]]; then
        echo "[ckpt] Stage 1 alignment checkpoint: $START_CKPT"
        return
    fi

    if [[ -z "$HF_STAGE1_REPO_ID" ]]; then
        echo "[Error] Missing START_CKPT/latest_checkpointed_iteration.txt: $START_CKPT"
        echo "[Error] Set START_CKPT to a local checkpoint or HF_STAGE1_REPO_ID to a Hugging Face model repo."
        exit 1
    fi

    echo "[ckpt] Local Stage 1 checkpoint missing: $START_CKPT"
    echo "[ckpt] Downloading Stage 1 checkpoint from Hugging Face: $HF_STAGE1_REPO_ID"
    echo "[ckpt] Destination: $HF_STAGE1_LOCAL_DIR"
    "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_STAGE1_REPO_ID"],
    repo_type="model",
    local_dir=os.environ["HF_STAGE1_LOCAL_DIR"],
)
PY
    START_CKPT="$HF_STAGE1_LOCAL_DIR"

    if [[ ! -f "$START_CKPT/latest_checkpointed_iteration.txt" ]]; then
        echo "[Error] Hugging Face checkpoint download finished, but tracker is still missing: $START_CKPT/latest_checkpointed_iteration.txt"
        exit 1
    fi
}

ensure_start_checkpoint
CHECKPOINT_PATH="$START_CKPT"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/midtraining_full_${LANG_LOWER}}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/data/hf_cache_midtraining_stream}"
mkdir -p "$DATA_ROOT" "$CACHE_DIR"
LOCAL_HF_DATA_ROOT="${LOCAL_HF_DATA_ROOT:-}"
VERIFY_LOCAL_HF_DATA="${VERIFY_LOCAL_HF_DATA:-1}"

TP="${TP:-1}"
PP="${PP:-1}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GBS="${GBS:-4}"
NSTEP="${NSTEP:-1000}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-26035}"
export GPUS_PER_NODE CUDA_VISIBLE_DEVICES

MAX_SAMPLES="${MAX_SAMPLES:-0}"
SHARD_MAXCOUNT="${SHARD_MAXCOUNT:-1000}"
SHARD_MAXSIZE="${SHARD_MAXSIZE:-2000000000}"
INDEX_WORKERS="${INDEX_WORKERS:-8}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
KEEP_PREPARED_DATA="${KEEP_PREPARED_DATA:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

export MIDTRAIN_TRAINABLE_MODULES="${MIDTRAIN_TRAINABLE_MODULES:-language_model adapter vision_model}"
export NO_SAVE_OPTIM_RNG="${NO_SAVE_OPTIM_RNG:-1}"
export PRINT_DATA_SAMPLE="${PRINT_DATA_SAMPLE:-0}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export VERBOSE_MODEL_DEBUG="${VERBOSE_MODEL_DEBUG:-0}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-rana-zayed-mbzuai}"
export WANDB_PROJECT="${WANDB_PROJECT:-llava-ov-1_5}"

sanitize_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g'
}

prepare_branch() {
    local branch="$1"
    local branch_safe="$2"
    local data_path="$3"

    if [[ -f "$data_path/.nv-meta/dataset.yaml" ]]; then
        echo "[$branch] prepared dataset already exists: $data_path"
        return
    fi

    local data_glob="${DATA_FILES_GLOB:-$branch/$LANG_CODE/*/*.parquet}"
    local prepare_args=(
        --repo-id "$HF_REPO_ID" \
        --data-files "$data_glob" \
        --output-dir "$data_path" \
        --max-samples "$MAX_SAMPLES" \
        --maxcount "$SHARD_MAXCOUNT" \
        --maxsize "$SHARD_MAXSIZE" \
        --shard-prefix "${branch_safe}-${LANG_LOWER}" \
        --cache-dir "$CACHE_DIR" \
        --index-workers "$INDEX_WORKERS" \
        --min-free-gb "$MIN_FREE_GB"
    )

    if [[ -n "$LOCAL_HF_DATA_ROOT" ]]; then
        echo "[$branch] verifying local HF files: $LOCAL_HF_DATA_ROOT / $data_glob"
        if [[ "$VERIFY_LOCAL_HF_DATA" == "1" ]]; then
            "$PYTHON_BIN" tools/check_hf_local_parquet_complete.py \
                --repo-id "$HF_REPO_ID" \
                --data-files "$data_glob" \
                --local-data-root "$LOCAL_HF_DATA_ROOT"
        fi
        echo "[$branch] preparing local HF files: $LOCAL_HF_DATA_ROOT / $data_glob"
        prepare_args+=(--local-data-root "$LOCAL_HF_DATA_ROOT")
    else
        echo "[$branch] preparing HF files: $data_glob"
    fi

    "$PYTHON_BIN" tools/prepare_hf_caption_parquet_to_energon_wds.py "${prepare_args[@]}"
}

run_branch() {
    local branch="$1"
    local index="$2"
    local branch_safe
    branch_safe="$(sanitize_name "$branch")"

    local data_path="$DATA_ROOT/${branch_safe}_${LANG_LOWER}_webdataset"
    local save_path="$REPO_ROOT/stage_1_5_midtraining_mobilellm_fastvit_${branch_safe}_${LANG_LOWER}_full_wandb"

    echo "===================================================================="
    echo "Stage 1.5 branch: $branch/$LANG_CODE"
    echo "Load checkpoint : $CHECKPOINT_PATH"
    echo "Data path       : $data_path"
    echo "Save checkpoint : $save_path"
    echo "Steps           : $NSTEP"
    echo "Global batch    : $GBS"
    echo "Log interval    : $LOG_INTERVAL"
    echo "W&B             : ${WANDB_MODE:-online}/${WANDB_PROJECT}/${WANDB_NAME_PREFIX:-stage1_5_mobilellm_fastvit}_${branch_safe}_${LANG_LOWER}_full_${NSTEP}steps_${GPUS_PER_NODE}gpu"
    echo "Max samples prep: $MAX_SAMPLES (0 means full branch)"
    echo "===================================================================="

    prepare_branch "$branch" "$branch_safe" "$data_path"

    if [[ "$PREPARE_ONLY" == "1" ]]; then
        echo "[$branch] PREPARE_ONLY=1, skipping training."
        return
    fi

    export DATA_PATH="$data_path"
    export CHECKPOINT_PATH
    export SAVE_CKPT_PATH="$save_path"
    export TENSORBOARD_PATH="$save_path/tensorboard"
    export MASTER_PORT="$((MASTER_PORT_BASE + index))"
    export WANDB_NAME="${WANDB_NAME_PREFIX:-stage1_5_mobilellm_fastvit}_${branch_safe}_${LANG_LOWER}_full_${NSTEP}steps_${GPUS_PER_NODE}gpu"

    bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_imagenet_en.sh \
        "$TP" "$PP" "$SEQ_LEN" "$MBS" "$GBS" "$NSTEP"

    if [[ ! -f "$save_path/latest_checkpointed_iteration.txt" ]]; then
        echo "[$branch] missing checkpoint tracker after training: $save_path"
        exit 1
    fi

    CHECKPOINT_PATH="$save_path"
    echo "[$branch] completed. Latest checkpoint root for next branch: $CHECKPOINT_PATH"

    if [[ "$KEEP_PREPARED_DATA" == "0" ]]; then
        echo "[$branch] removing prepared data: $data_path"
        rm -rf "$data_path"
    fi
}

index=0
for branch in "$@"; do
    run_branch "$branch" "$index"
    index=$((index + 1))
done

echo "All requested English branches completed."
echo "Final checkpoint root: $CHECKPOINT_PATH"
echo "$CHECKPOINT_PATH" > "$REPO_ROOT/stage_1_5_latest_checkpoint_path.txt"
