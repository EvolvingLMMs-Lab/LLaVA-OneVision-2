#!/usr/bin/env bash
set -euo pipefail

cd /share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5

export PATH="/home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=4
export GPUS_PER_NODE=1
export MASTER_PORT=26050
export SAVE_INTERVAL=50
export NO_LOAD_OPTIM_RNG=1

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(
    sed -n 's/^export WANDB_API_KEY="\([^"]*\)".*/\1/p' Stage1/alignment.sh | head -n 1
  )"
  export WANDB_API_KEY
fi

export WANDB_PROJECT="${WANDB_PROJECT:-llava-ov-1_5}"
export WANDB_NAME="${WANDB_NAME:-mobilellm_integration}"

exec bash examples/llava_ov_1_5/quick_start/stage_1_alignment_mobilellm_140m.sh \
  1 1 32768 1 4 500
