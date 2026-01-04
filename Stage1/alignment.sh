#!/bin/bash
#SBATCH --job-name=llava_stage1_4b
#SBATCH --time=72:00:00             
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4                # 4 A100 GPUs on a single node
#SBATCH --mem=230G                  # node RAM, not GPU RAM
#SBATCH --ntasks-per-node=4         # one task per GPU
#SBATCH --cpus-per-task=16          # adjust if you want fewer CPU cores
#SBATCH --output=/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/Stage1/logs/%x-%j.out     # logs/llava_stage1_4b-<jobid>.out

# ---- ENV SETUP ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-ov-4b-clean
# conda activate apex_cuda120
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export APEX_CUDA_EXT=1

# Go to repo root on CIAI cluster
# cd /l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5
# # Go to repo root on 156 machine 
cd /share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5

# ============================================================
# Required environment variables:
#   AIAK_TRAINING_PATH  Root directory of the AIAK-Training-LLM project
#   DATA_PATH           Directory with WebDataset shards (.tar) for pretraining
#   TOKENIZER_PATH      Hugging Face tokenizer directory
#   CHECKPOINT_PATH     Megatron-formatted checkpoint directory (e.g., mcore TP1/PP1)
#   SAVE_CKPT_PATH      Output directory for saving training checkpoints
# export CUDA_HOME=/apps/local/nvidia/cuda-12.0
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Use CUDA 12.1 libraries but system nvcc (CUDA 10.1) since 12.1 doesn't have nvcc
export CUDA_HOME=/usr
export PATH="/usr/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"


# AIAK_TRAINING_PATH=/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5 \
# DATA_PATH=/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/data/LLaVA-558K-Webdataset \
# TOKENIZER_PATH=/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/checkpoints/LLaVA-OneVision-1.5-4B-stage0 \
# CHECKPOINT_PATH=/l/users/rana.zayed/new_fastvlm/LLaVA-OneVision-1.5/checkpoints/LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \

# echo "AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH}"
# echo "DATA_PATH=${DATA_PATH}"
# echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
# echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
# echo "SLURM_NODELIST=${SLURM_NODELIST}"

export CUDA_VISIBLE_DEVICES=7
export GPUS_PER_NODE=1

bash examples/llava_ov_1_5/quick_start/stage_1_alignment_llava_ov_4b.sh 1 1
