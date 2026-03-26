#!/bin/bash

#SBATCH --job-name=llava_stage1_4b_amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --qos=skqos
#SBATCH --partition=faculty
#SBATCH --output=/vast/users/salman.khan/mobile_vlm/llava_ov1.5/LLaVA-OneVision-1.5/Stage1/logs/%x-%j.out

# ---- ENV SETUP (AMD) ----
source ~/.bashrc
conda activate mobile_vlm

export MIOPEN_DISABLE_CACHE=1
export PYTORCH_TUNABLEOP_ENABLED=0

export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH}"

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Force HuggingFace offline mode to use local files only
#export HF_HUB_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1


# RCCL/NCCL runtime hints (tune as needed)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export NCCL_P2P_ENABLE=${NCCL_P2P_ENABLE:-1}
# export NCCL_SOCKET_IFNAME=eno1   # uncomment and set to your NIC if needed
 
# Resolve repo root relative to this script (Stage1/..)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/vast/users/salman.khan/mobile_vlm/llava_ov1.5/LLaVA-OneVision-1.5
cd "$REPO_ROOT" || exit 1


# Go to repo root
cd "$REPO_ROOT" || { echo "[Error] Repo root not found: $REPO_ROOT"; exit 1; }

echo "=== ENV CHECK ==="
which conda
which python
python -V
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
python -c "import sys; print('sys.executable=', sys.executable)"
python -c "import torch; print('torch=', torch.__version__)" || echo "TORCH NOT FOUND"
pip -V
pip list | grep -E "torch|pytorch" || true
echo "=== END ENV CHECK ==="

# Required environment variables
export AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_ROOT}"
export AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-$REPO_ROOT/aiak_megatron}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/LLaVA-558K-Webdataset}"
# Use MobileLLM tokenizer and checkpoint (let stage_1_alignment_mobilellm_140m.sh set defaults)
export TOKENIZER_PATH="${TOKENIZER_PATH:-facebook/MobileLLM-R1-140M}"
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-$REPO_ROOT/checkpoints/mobilellm-fastvit-merged-tp1-pp1}"
# Add megatron to PYTHONPATH so imports work
export PYTHONPATH="${AIAK_MAGATRON_PATH}:${AIAK_TRAINING_PATH}:${PYTHONPATH}"

echo "AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH}"
echo "AIAK_MAGATRON_PATH=${AIAK_MAGATRON_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "PYTHONPATH=${PYTHONPATH}"

# Weights & Biases configuration
export WANDB_API_KEY="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"
export WANDB_PROJECT="llava-ov-1_5"
export WANDB_NAME="fastvit_integration"
bash examples/llava_ov_1_5/quick_start/stage_1_alignment_mobilellm_140m.sh