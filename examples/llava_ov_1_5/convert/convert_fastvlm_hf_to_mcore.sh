#!/bin/bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-ov-4b-clean

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-1.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1
SAVE=$2
TP=$3
PP=$4

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore
SAVE_VISION_MODEL=./tmp/vision-model-mcore
SAVE_ADAPTER=./tmp/adapter-mcore

# llm (Now compatible: FastVLM Qwen2-1.5B matches our model config)
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/qwen3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [ $? -ne 0 ]; then
    echo "Language model conversion failed!"
    exit 1
fi

# fastvit
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

if [ $? -ne 0 ]; then
    echo "FastViT conversion failed!"
    exit 1
fi

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

if [ $? -ne 0 ]; then
    echo "Adapter conversion failed!"
    exit 1
fi

# merge (all 3 components now compatible!)
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP

if [ $? -ne 0 ]; then
    echo "Merge failed!"
    exit 1
fi

echo "Conversion completed successfully!"
echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER

