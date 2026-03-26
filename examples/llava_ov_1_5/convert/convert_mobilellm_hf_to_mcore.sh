#!/bin/bash
# Convert MobileLLM-R1-140M from HuggingFace checkpoint to Megatron format

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-1.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1  # HF checkpoint path (e.g., checkpoints/MobileLLM-R1-140M)
SAVE=$2  # Megatron output path (e.g., checkpoints/MobileLLM-R1-140M-mcore-tp1-pp1)
TP=${3:-1}  # Tensor parallel size (default: 1)
PP=${4:-1}  # Pipeline parallel size (default: 1)

if [ -z "$LOAD" ] || [ -z "$SAVE" ]; then
    echo "Usage: $0 <HF_CHECKPOINT_PATH> <MCORE_OUTPUT_PATH> [TP] [PP]"
    echo "Example: $0 checkpoints/MobileLLM-R1-140M checkpoints/MobileLLM-R1-140M-mcore-tp1-pp1 1 1"
    exit 1
fi

SAVE_LANGUAGE_MODEL=./tmp/mobilellm-language-mcore

echo "=========================================="
echo "Converting MobileLLM-R1-140M"
echo "=========================================="
echo "Source (HF): $LOAD"
echo "Target (Megatron): $SAVE"
echo "Tensor Parallel: $TP"
echo "Pipeline Parallel: $PP"
echo "=========================================="

# Convert language model
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/mobilellm-140m.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# Copy the converted weights to final destination
mkdir -p $SAVE
cp -r $SAVE_LANGUAGE_MODEL/release $SAVE/

# Create iteration marker
echo "release" > $SAVE/latest_checkpointed_iteration.txt

# Cleanup
rm -rf $SAVE_LANGUAGE_MODEL

echo "=========================================="
echo "✓ MobileLLM-R1-140M conversion complete!"
echo "Output: $SAVE/release"
echo "=========================================="
