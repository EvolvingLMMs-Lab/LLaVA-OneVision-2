# =============================================================================
# LLaVA-OneVision2 4B-p14m3 – Convert HuggingFace checkpoint to Megatron-Core
# =============================================================================
#
# Usage:
#   bash convert_4b_hf_to_mcore.sh <LOAD> <SAVE> <TP> <PP> [CUSTOM_PIPELINE_LAYERS]
#
# Arguments:
#   LOAD                    Path to the source HuggingFace checkpoint
#   SAVE                    Path to save the Megatron-Core checkpoint
#   TP                      Tensor parallel size
#   PP                      Pipeline parallel size
#   CUSTOM_PIPELINE_LAYERS  (optional) Comma-separated layer counts per PP stage.
#                           Use this when the ViT encoder occupies stage-0 and
#                           you want an uneven LLM-layer split across stages.
#
# Recommended splits for the 4B model (36 LLM layers, 300M ViT on stage-0):
#   PP=4 : 0,12,12,12  ← stage-0 holds ViT only; stages 1-3 each get 12 layers
#   PP=3 : 0,18,18     ← stage-0 holds ViT only; stages 1-2 each get 18 layers
#
# Examples:
#   bash convert_4b_hf_to_mcore.sh /src /dst 2 4 0,12,12,12
#   bash convert_4b_hf_to_mcore.sh /src /dst 1 1
# =============================================================================

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1
SAVE=$2
TP=$3
PP=$4
CUSTOM_PIPELINE_LAYERS=$5

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore
SAVE_VISION_MODEL=./tmp/vision-model-mcore
SAVE_ADAPTER=./tmp/adapter-mcore
SAVE_PATCH=./tmp/patch-mcore


# llama: language expert
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p14m3/qwen3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    ${CUSTOM_PIPELINE_LAYERS:+--custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS} \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p14m3/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p14m3/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch in vit
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p14m3/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
