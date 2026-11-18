#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision2 4B p16m2 - Convert HuggingFace checkpoint to Megatron-Core
# =============================================================================

set -euo pipefail

LOAD=${1:?LOAD HuggingFace checkpoint path is required}
SAVE=${2:?SAVE Megatron-Core checkpoint path is required}
TP=${3:?TP is required}
PP=${4:?PP is required}
CUSTOM_PIPELINE_LAYERS=${5:-}

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"
CONFIG_DIR="$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p16m2"

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore
SAVE_VISION_MODEL=./tmp/vision-model-mcore
SAVE_ADAPTER=./tmp/adapter-mcore
SAVE_PATCH=./tmp/patch-mcore

python "$CONVERT_CHECKPOINT_PATH/model.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path="$CONFIG_DIR/qwen3.json" \
    --tensor_model_parallel_size="$TP" \
    --pipeline_model_parallel_size="$PP" \
    ${CUSTOM_PIPELINE_LAYERS:+--custom_pipeline_layers="$CUSTOM_PIPELINE_LAYERS"} \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE_LANGUAGE_MODEL" \
    --safetensors \
    --no_save_optim \
    --no_load_optim

python "$CONVERT_CHECKPOINT_PATH/model.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path="$CONFIG_DIR/vision-model.json" \
    --tensor_model_parallel_size="$TP" \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE_VISION_MODEL" \
    --safetensors \
    --no_save_optim \
    --no_load_optim

python "$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/adapter.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path="$CONFIG_DIR/adapter.json" \
    --tensor_model_parallel_size="$TP" \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE_ADAPTER"

python "$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/vision_patch.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size="$TP" \
    --common_config_path="$CONFIG_DIR/vision-patch.json" \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE_PATCH"

python "$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/merge_megatron.py" \
    --megatron_path "$AIAK_MAGATRON_PATH" \
    --language_model_path "$SAVE_LANGUAGE_MODEL/release" \
    --vision_model_path "$SAVE_VISION_MODEL/release" \
    --vision_patch "$SAVE_PATCH/release" \
    --adapter_path "$SAVE_ADAPTER/release" \
    --save_ckpt_path "$SAVE/release" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP"

echo release > "$SAVE/latest_checkpointed_iteration.txt"
rm -rf "$SAVE_LANGUAGE_MODEL" "$SAVE_VISION_MODEL" "$SAVE_ADAPTER" "$SAVE_PATCH"
