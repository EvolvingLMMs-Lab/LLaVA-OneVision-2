#!/bin/bash
# Convert FastViT vision tower from HuggingFace checkpoint to Megatron format
# This script extracts ONLY the vision tower weights, ignoring the language model

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-1.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1  # HF checkpoint path (e.g., checkpoints/llava-fastvithd_1.5b_stage3)
SAVE=$2  # Megatron output path (e.g., checkpoints/fastvit_vision_mcore_tp1)
TP=${3:-1}  # Tensor parallel size (default: 1)
PP=${4:-1}  # Pipeline parallel size (default: 1)

if [ -z "$LOAD" ] || [ -z "$SAVE" ]; then
    echo "Usage: $0 <HF_CHECKPOINT_PATH> <MCORE_OUTPUT_PATH> [TP] [PP]"
    echo "Example: $0 checkpoints/llava-fastvithd_1.5b_stage3 checkpoints/fastvit_vision_mcore_tp1 1 1"
    exit 1
fi

# Convert to absolute paths if relative
if [[ "$LOAD" != /* ]]; then
    LOAD="$AIAK_TRAINING_PATH/checkpoints/$LOAD"
fi
if [[ "$SAVE" != /* ]]; then
    SAVE="$AIAK_TRAINING_PATH/checkpoints/$SAVE"
fi

SAVE_VISION_MODEL=./tmp/vision-model-mcore-fastvit

echo "=========================================="
echo "Converting FastViT vision tower"
echo "=========================================="
echo "Source (HF): $LOAD"
echo "Target (Megatron): $SAVE"
echo "Tensor Parallel: $TP"
echo "Pipeline Parallel: $PP"
echo "=========================================="

# Extract vision tower weights from HuggingFace checkpoint
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/fastvit.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/fastvit.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# Copy the converted weights to final destination
mkdir -p $SAVE
cp -r $SAVE_VISION_MODEL/release $SAVE/

# Create iteration marker
echo "release" > $SAVE/latest_checkpointed_iteration.txt

# Cleanup
rm -rf $SAVE_VISION_MODEL

echo "=========================================="
echo "✓ FastViT vision tower conversion complete!"
echo "Output: $SAVE/release"
echo "=========================================="
echo ""
echo "To use this checkpoint with MobileLLM-140M training:"
echo "  1. Set: PRETRAINED_VISION_CHECKPOINT=\"$SAVE\""
echo "  2. Add flag: --load \$PRETRAINED_VISION_CHECKPOINT"
echo "  3. Add flag: --no-load-strict (to skip language model weights)"
echo ""
