# =============================================================================
# LLaVA-OneVision2 4B (Qwen3.5 hybrid LLM) - Convert HuggingFace -> Megatron-Core
# =============================================================================
#
# Qwen3.5 counterpart of convert_4b_hf_to_mcore.sh. The LLM backbone is a hybrid
# stack (linear gated-delta-net + periodic full-attention, interleaved 3D M-RoPE,
# partial rotary 0.25), so the *language* step uses the dedicated converter
# custom/llava_onevision2/qwen35_hf_to_mcore.py (built on the validated
# aiak_training_llm.models.qwen35.hf_convert layer math) INSTEAD of the generic
# model.py + qwen3.json path. Vision tower / adapter / vision-patch reuse the same
# validated converters as the Qwen3 4B model (identical p16/merge3 ViT + adapter).
#
# Usage:
#   bash convert_4b_qwen35_hf_to_mcore.sh <LOAD> <SAVE> [TP] [PP]
#
#   LOAD  merged HF checkpoint (LlavaOnevision2ForConditionalGeneration)
#   SAVE  output Megatron-Core checkpoint dir
#   TP    tensor-parallel size  (default 1)   -- TP in {1,2,4}, PP=1
#   PP    pipeline-parallel size (default 1)
#
# Example:
#   bash convert_4b_qwen35_hf_to_mcore.sh \
#     /models/llava_onevision2_4b_qwen35_p16m3 \
#     /models/llava_onevision2_4b_qwen35_p16m3_mcore_tp1pp1 1 1
# =============================================================================
set -e

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"
CUSTOM="$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2"
CFG="$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p16m3"

LOAD=$1
SAVE=$2
TP=${3:-1}
PP=${4:-1}

# Qwen3.5-4B hybrid stack: 32 decoder layers; full-attention every 4th layer.
NUM_LAYERS=${NUM_LAYERS:-32}
FULL_ATTENTION_INTERVAL=${FULL_ATTENTION_INTERVAL:-4}

export PYTHONPATH="$AIAK_MAGATRON_PATH:$CONVERT_CHECKPOINT_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH"

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore-qwen35
SAVE_VISION_MODEL=./tmp/vision-model-mcore
SAVE_ADAPTER=./tmp/adapter-mcore
SAVE_PATCH=./tmp/patch-mcore

# ---- language: Qwen3.5 hybrid LLM (dedicated converter) ---------------------
python $CUSTOM/qwen35_hf_to_mcore.py \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --num_layers=$NUM_LAYERS \
    --full_attention_interval=$FULL_ATTENTION_INTERVAL \
    --tensor_model_parallel_size=$TP

# ---- vit (reuse validated converter) ----------------------------------------
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CFG/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# ---- adapter (reuse validated converter) ------------------------------------
python $CUSTOM/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CFG/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# ---- vision patch in vit (reuse validated converter) ------------------------
python $CUSTOM/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CFG/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# ---- merge language + vision + adapter + patch ------------------------------
python $CUSTOM/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL $SAVE_VISION_MODEL $SAVE_ADAPTER $SAVE_PATCH
echo "Done. Megatron-Core checkpoint saved to: $SAVE"
