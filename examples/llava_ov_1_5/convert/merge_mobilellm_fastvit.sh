#!/bin/bash
# Merge MobileLLM-R1-140M language model with FastViT vision encoder
# This creates a complete multimodal checkpoint ready for training

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-1.5}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LANGUAGE_MODEL_PATH=$1  # MobileLLM language model (e.g., checkpoints/MobileLLM-R1-140M-mcore-tp1-pp1)
VISION_MODEL_PATH=$2     # FastViT vision encoder (e.g., checkpoints/llava-fastvithd_1.5b_stage3_mcore_tp1_pp1)
SAVE_PATH=$3            # Output merged checkpoint path
TP=${4:-1}              # Tensor parallel size
PP=${5:-1}              # Pipeline parallel size

if [ -z "$LANGUAGE_MODEL_PATH" ] || [ -z "$VISION_MODEL_PATH" ] || [ -z "$SAVE_PATH" ]; then
    echo "Usage: $0 <LANGUAGE_MODEL_PATH> <VISION_MODEL_PATH> <SAVE_PATH> [TP] [PP]"
    echo "Example: $0 checkpoints/MobileLLM-R1-140M-mcore-tp1-pp1 checkpoints/llava-fastvithd_1.5b_stage3_mcore_tp1_pp1 checkpoints/mobilellm-fastvit-merged-tp1-pp1 1 1"
    exit 1
fi

echo "=========================================="
echo "Merging MobileLLM + FastViT"
echo "=========================================="
echo "Language Model: $LANGUAGE_MODEL_PATH"
echo "Vision Model: $VISION_MODEL_PATH"
echo "Output: $SAVE_PATH"
echo "Tensor Parallel: $TP"
echo "Pipeline Parallel: $PP"
echo "=========================================="

# Merge the checkpoints
# Note: We're only merging language model + vision model
# The adapter will be randomly initialized during training
echo "Merging language model and vision model..."
echo "Note: Adapter will be randomly initialized during training"

# Create a simple Python script to merge just language and vision models
python -c "
import torch
import sys
import os

# Load language model checkpoint
lang_ckpt = torch.load('$LANGUAGE_MODEL_PATH/release/mp_rank_00/model_optim_rng.pt', map_location='cpu', weights_only=False)
print(f'Loaded language model: {len(lang_ckpt[\"model\"])} keys')

# Load vision model checkpoint
vis_ckpt = torch.load('$VISION_MODEL_PATH/release/mp_rank_00/model_optim_rng.pt', map_location='cpu', weights_only=False)
print(f'Loaded vision model: {len(vis_ckpt[\"model\"])} keys')

# Merge vision model into language model
merged_model = lang_ckpt['model'].copy()
vision_key_count = 0
for k, v in vis_ckpt['model'].items():
    merged_model[k] = v
    vision_key_count += 1
    if vision_key_count <= 5:
        print(f'  Added: {k}')

print(f'Merged model: {len(merged_model)} keys ({vision_key_count} vision keys)')

# Save merged checkpoint
os.makedirs('$SAVE_PATH/release/mp_rank_00', exist_ok=True)
merged_ckpt = {
    'model': merged_model,
    'checkpoint_version': 3.0,
    'args': lang_ckpt.get('args', {}),
    'iteration': 0
}
torch.save(merged_ckpt, '$SAVE_PATH/release/mp_rank_00/model_optim_rng.pt')
print(f'Saved merged checkpoint to $SAVE_PATH/release/mp_rank_00/model_optim_rng.pt')
"

# Create iteration marker
echo "release" > $SAVE_PATH/latest_checkpointed_iteration.txt

echo "=========================================="
echo "✓ Merge complete!"
echo "Output: $SAVE_PATH"
echo "=========================================="
echo ""
echo "To use this checkpoint in training:"
echo "  Set: --pretrained-checkpoint \"$SAVE_PATH\""
echo ""
