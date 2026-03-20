# =============================================================================
# LLaVA-OneVision2 30B-A3B – Convert HuggingFace checkpoint to Megatron-Core
# =============================================================================
#
# Usage:
#   bash convert_30b_a3b_hf_to_mcore.sh <LOAD> <SAVE> <PP> <EP>
#   bash convert_30b_a3b_hf_to_mcore.sh <LOAD> <SAVE> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#   bash convert_30b_a3b_hf_to_mcore.sh <LOAD> <SAVE> <TP> <PP> <EP>
#   bash convert_30b_a3b_hf_to_mcore.sh <LOAD> <SAVE> <TP> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#
# Arguments:
#   LOAD  Path to the source HuggingFace checkpoint
#   SAVE  Path to save the Megatron-Core checkpoint
#   TP    Tensor parallel size (optional, defaults to 1)
#   PP    Pipeline parallel size (recommended 1 for this script)
#   EP    Expert parallel size
#   CUSTOM_PIPELINE_LAYERS  (optional) Comma-separated layer counts per PP stage,
#                           used for custom PP layer layouts.
# =============================================================================

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1
SAVE=$2
CUSTOM_PIPELINE_LAYERS=

if [[ $# -eq 4 ]]; then
    TP=1
    PP=$3
    EP=$4
elif [[ $# -eq 5 ]]; then
    if [[ "$5" == *","* ]]; then
        TP=1
        PP=$3
        EP=$4
        CUSTOM_PIPELINE_LAYERS=$5
    else
        TP=${3:-1}
        PP=${4:-1}
        EP=${5:-8}
    fi
elif [[ $# -ge 6 ]]; then
    TP=${3:-1}
    PP=${4:-1}
    EP=${5:-8}
    CUSTOM_PIPELINE_LAYERS=$6
else
    TP=${3:-1}
    PP=${4:-1}
    EP=${5:-8}
fi

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore
SAVE_VISION_MODEL=./tmp/vision-model-mcore
SAVE_ADAPTER=./tmp/adapter-mcore
SAVE_PATCH=./tmp/patch-mcore


# llm (moe)
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/qwen3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    ${CUSTOM_PIPELINE_LAYERS:+--custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS} \
    --num_experts=128 \
    --expert_parallel_size=$EP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/vision-model.json \
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
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch in vit
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge (tp x ep aware)
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/merge_megatron_qwen3_30b_a3b.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
