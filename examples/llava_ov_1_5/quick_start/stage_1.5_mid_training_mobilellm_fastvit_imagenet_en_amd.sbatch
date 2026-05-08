#!/bin/bash

#SBATCH --job-name=stage15_imagenet_amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=stage15_imagenet_amd_%j.out
#SBATCH --error=stage15_imagenet_amd_%j.err

set -euo pipefail

# ---- ENV SETUP (AMD) ----
source ~/.bashrc
conda activate "${CONDA_ENV:-mobile_vlm}"

export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0

export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}"

# RCCL/NCCL runtime hints.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export NCCL_P2P_ENABLE="${NCCL_P2P_ENABLE:-1}"

export REPO_ROOT="${REPO_ROOT:-/vast/users/salman.khan/mobile_vlm/llava_ov1.5/LLaVA-OneVision-1.5}"
cd "$REPO_ROOT" || { echo "[Error] Repo root not found: $REPO_ROOT"; exit 1; }

export AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_ROOT}"
export AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-$REPO_ROOT/aiak_megatron}"
export PYTHONPATH="${AIAK_MAGATRON_PATH}:${AIAK_TRAINING_PATH}:${PYTHONPATH:-}"

export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"

export START_CKPT="${START_CKPT:-$REPO_ROOT/stage_1_alignment_mobilellm_140m_fastvlm_faithful}"
export HF_STAGE1_REPO_ID="${HF_STAGE1_REPO_ID:-ranazayed19/mobilellm-fastvit-stage1-2500}"
export HF_STAGE1_LOCAL_DIR="${HF_STAGE1_LOCAL_DIR:-$START_CKPT}"
export LOCAL_HF_DATA_ROOT="${LOCAL_HF_DATA_ROOT:-$REPO_ROOT/data/LLaVA-OneVision-Mid-Training-EN}"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/midtraining_full_en}"
export CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/data/hf_cache_midtraining_stream}"

if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    DETECTED_GPUS="$(awk -F, '{print NF}' <<< "$HIP_VISIBLE_DEVICES")"
else
    DETECTED_GPUS="${SLURM_GPUS_ON_NODE:-1}"
fi
export GPUS_PER_NODE="${GPUS_PER_NODE:-$DETECTED_GPUS}"
export TP="${TP:-1}"
export PP="${PP:-1}"
export SEQ_LEN="${SEQ_LEN:-4096}"
export MBS="${MBS:-1}"
export GBS="${GBS:-8}"
export NSTEP="${NSTEP:-1000}"
export MIN_FREE_GB="${MIN_FREE_GB:-100}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export PRINT_DATA_SAMPLE="${PRINT_DATA_SAMPLE:-0}"
export VERBOSE_MODEL_DEBUG="${VERBOSE_MODEL_DEBUG:-0}"

# The local ImageNet files were already verified on the login node. Keep this
# disabled by default so compute nodes do not need Hugging Face API access.
export VERIFY_LOCAL_HF_DATA="${VERIFY_LOCAL_HF_DATA:-0}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-rana-zayed-mbzuai}"
export WANDB_PROJECT="${WANDB_PROJECT:-llava-ov-1_5}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-stage1_5_mobilellm_fastvit_amd}"

if [[ "$WANDB_ENABLE" == "1" && -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
    echo "[WARN] W&B is enabled, but WANDB_API_KEY is not set and ~/.netrc was not found."
    echo "[WARN] Run 'wandb login' or submit with WANDB_API_KEY set if online W&B logging fails."
fi

echo "=== ENV CHECK ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TORCHRUN=$TORCHRUN"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "START_CKPT=$START_CKPT"
echo "HF_STAGE1_REPO_ID=$HF_STAGE1_REPO_ID"
echo "LOCAL_HF_DATA_ROOT=$LOCAL_HF_DATA_ROOT"
echo "WANDB_ENABLE=$WANDB_ENABLE"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME_PREFIX=$WANDB_NAME_PREFIX"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY=set"
else
    echo "WANDB_API_KEY=not_set"
fi
"$PYTHON_BIN" - <<'PY'
import sys
print("sys.executable=", sys.executable)
import torch
print("torch=", torch.__version__)
print("hip=", torch.version.hip)
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
import megatron.energon
import datasets
import webdataset
import yaml
import tqdm
import PIL
print("midtraining deps ok")
PY
echo "=== END ENV CHECK ==="

if [[ ! -f "$START_CKPT/latest_checkpointed_iteration.txt" ]]; then
    echo "[ckpt] Stage 1 checkpoint not found locally yet. The branch runner will download it from Hugging Face if needed."
fi

if [[ ! -d "$LOCAL_HF_DATA_ROOT/imagenet" ]]; then
    echo "[Error] Missing local ImageNet data under: $LOCAL_HF_DATA_ROOT/imagenet"
    exit 1
fi

df -h "$REPO_ROOT" "$LOCAL_HF_DATA_ROOT" || true

bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_english_branch_chain.sh imagenet
