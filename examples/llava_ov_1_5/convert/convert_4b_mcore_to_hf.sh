# Setup paths 
# set base training path
AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-1.5}"
#megatron path
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
# conversion script path
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1 # source checkpoint directory (Megatron format)
SAVE=$2 # Destination checkpoint directory (Hugging Face format)
TP=$3 # tensor parallel size
PP=$4 # pipeline parallel size
# Usage: bash script.sh <megatron_checkpoint> <hf_output> <tp> <pp>

# create temp directories
mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore # temp dir for Qwen language model
SAVE_VISION_MODEL=./tmp/vision-model-mcore # temp dir for Rice vision encoder
SAVE_ADAPTER=./tmp/adapter-mcore # temp dir for projection adapter
SAVE_PATCH=./tmp/patch-mcore # temp dir for vision patch embedding

#These temporary directories hold intermediate conversions before final merge.

# llama: language expert
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \  # load from Megatron MCore format
    --megatron_path $AIAK_MAGATRON_PATH \
    --save_platform=huggingface \  #save to Hugging Face format
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/qwen3.json \  #config file for Qwen3
    --tensor_model_parallel_size=$TP \ 
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \  #source megatron checkpoint
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \  #output temp directory
    --safetensors \  #save as .safetensors format
    --no_save_optim \  #do not save optimizer states
    --no_load_optim ##do not load optimizer states

# vit
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD #if no pipeline parallelism, load directly from source
else
    # If pipeline parallel, vision is only on rank 0, need to reorganize
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$TP;i++)); do
        from=`printf "mp_rank_%02d_000" $i` # mp_rank_00_000, mp_rank_01_000, etc.
        to=`printf "mp_rank_%02d" $i` # mp_rank_00, mp_rank_01, etc.
        cp -r $LOAD/$from $LOAD_PATH/$to # Copy first pipeline stage to new location
    done
fi

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/vision-model.json \  # Config for Rice vision model
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \  # Vision is NOT pipeline parallel (always 1)
    --load_ckpt_path=$LOAD_PATH \   # Use reorganized path if PP>1
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH # Remove temporary reorganized directory if created
fi

# adapter
#Extracts adapter/projection layer weights (vision features → language embedding space
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch
# Convert Vision Patch Embedding
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/merge_huggingface.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

# Delete temporary intermediate files, keep only final merged checkpoint.
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
