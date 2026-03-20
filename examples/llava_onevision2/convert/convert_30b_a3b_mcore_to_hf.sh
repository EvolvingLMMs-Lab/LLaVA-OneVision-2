# =============================================================================
# LLaVA-OneVision2 30B-A3B – Convert Megatron-Core checkpoint to HuggingFace
# =============================================================================
#
# Usage:
#   bash convert_30b_a3b_mcore_to_hf.sh <LOAD> <SAVE> <PP> <EP>
#   bash convert_30b_a3b_mcore_to_hf.sh <LOAD> <SAVE> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#   bash convert_30b_a3b_mcore_to_hf.sh <LOAD> <SAVE> <TP> <PP> <EP>
#   bash convert_30b_a3b_mcore_to_hf.sh <LOAD> <SAVE> <TP> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#
# Arguments:
#   LOAD  Path to the source Megatron-Core checkpoint
#   SAVE  Path to save the HuggingFace checkpoint
#   TP    Tensor parallel size (optional, defaults to 1)
#   PP    Pipeline parallel size
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
    --load_platform=mcore \
    --megatron_path $AIAK_MAGATRON_PATH \
    --save_platform=huggingface \
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
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$TP;i++)); do
        from=$(printf "mp_rank_%02d_000_000" $i)
        to=$(printf "mp_rank_%02d" $i)
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-30b-a3b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/merge_huggingface.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
