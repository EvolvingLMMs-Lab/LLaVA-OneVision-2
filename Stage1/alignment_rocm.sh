#!/bin/bash
# AMD/ROCm alignment launcher (separate from alignment.sh)
# Adjust SBATCH partition/queue to your cluster settings.

#SBATCH --job-name=llava_stage1_4b_rocm
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH -p amd-gpu            # TODO: set your AMD GPU partition/queue
#SBATCH --gres=gpu:2          # match quick_start GPUS_PER_NODE=2
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=2   # one task per GPU
#SBATCH --cpus-per-task=16
#SBATCH --output=/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/Stage1/logs/%x-%j.out

#!/bin/bash

# ---- ENV SETUP (ROCm) ----
source ~/.bashrc
# Optional: override with `CONDA_ENV=myenv` before running
CONDA_ENV=${CONDA_ENV:-llava-ov-4b-clean}
if command -v conda >/dev/null 2>&1; then
	if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
		conda activate "$CONDA_ENV"
	else
		echo "[Warn] Conda env '$CONDA_ENV' not found; continuing without activation"
	fi
fi

export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}

# CUDA-only Apex extensions are disabled on ROCm
export APEX_CUDA_EXT=0

# RCCL/NCCL runtime hints (tune as needed)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export NCCL_P2P_ENABLE=${NCCL_P2P_ENABLE:-1}
# export NCCL_SOCKET_IFNAME=eno1   # uncomment and set to your NIC if needed

# Resolve repo root relative to this script (Stage1/..)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Go to repo root
cd "$REPO_ROOT" || { echo "[Error] Repo root not found: $REPO_ROOT"; exit 1; }

# Required environment variables
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-$REPO_ROOT} \
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data/LLaVA-558K-Webdataset} \
TOKENIZER_PATH=${TOKENIZER_PATH:-$REPO_ROOT/checkpoints/LLaVA-OneVision-1.5-4B-stage0} \
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1} \

echo "AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

# Launch quick-start script (uses torchrun and nccl backend which maps to RCCL on ROCm)
QS="$REPO_ROOT/examples/llava_ov_1_5/quick_start/stage_1_alignment_llava_ov_4b.sh"
if [[ ! -f "$QS" ]]; then
	echo "[Error] Quick-start script not found: $QS"
	exit 1
fi
bash "$QS"
