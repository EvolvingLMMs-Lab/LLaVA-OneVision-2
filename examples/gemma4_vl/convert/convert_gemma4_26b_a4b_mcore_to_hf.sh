# =============================================================================
# Gemma4-VL 26B-A4B – Convert Megatron-Core checkpoint to HuggingFace
# =============================================================================
#
# Usage:
#   HF_REF=/path/to/source/hf bash convert_gemma4_26b_a4b_mcore_to_hf.sh <LOAD> <SAVE> <PP> <EP>
#   HF_REF=/path/to/source/hf bash convert_gemma4_26b_a4b_mcore_to_hf.sh <LOAD> <SAVE> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#   HF_REF=/path/to/source/hf bash convert_gemma4_26b_a4b_mcore_to_hf.sh <LOAD> <SAVE> <TP> <PP> <EP>
#   HF_REF=/path/to/source/hf bash convert_gemma4_26b_a4b_mcore_to_hf.sh <LOAD> <SAVE> <TP> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#
# Arguments:
#   LOAD  Path to the source Megatron-Core checkpoint (parent of `release/`,
#         the path produced by convert_gemma4_26b_a4b_hf_to_mcore.sh).
#   SAVE  Path to save the HuggingFace checkpoint.
#   TP    Tensor parallel size (optional, defaults to 1).
#   PP    Pipeline parallel size.
#   EP    Expert parallel size.
#   CUSTOM_PIPELINE_LAYERS  (optional) Comma-separated layer counts per PP stage.
#
# Required environment:
#   HF_REF  Path to the original HuggingFace checkpoint directory. Needed
#           because the standalone llm.py / vit.py read num_layers,
#           num_experts, layer_types and other architecture metadata directly
#           from the source `config.json` (data-driven, no JSON template).
#           The shell wrapper seeds this `config.json` into the per-component
#           save dirs before running each converter.
#
# Notes:
#   * Gemma4-VL LLM round-trip is bit-equivalent (657/657 tensors verified at
#     P2.5). ViT/adapter/patch MG->HF conversion paths are wired through the
#     same standalone scripts.
#   * The final HF dict is assembled by reusing the OV2 generic
#     custom/llava_onevision2/merge_huggingface.py which simply merges 4
#     state-dicts by key (model-agnostic).
#   * P2.8: when PP>1, the LLM ckpt uses 3D mp_rank_TT_PPP_EEE naming; vit /
#     adapter / vision_patch live solely on PP stage 0, so we materialise a
#     single 1D-named mp_rank_TT view of stage-0 EP-rank-0 shards and feed it
#     to all three of those standalone converters.
# =============================================================================

if [[ -z "$HF_REF" ]]; then
    echo "ERROR: HF_REF environment variable must be set to the source HuggingFace dir" >&2
    echo "  Example: HF_REF=/ov2/pretrain_models/google/gemma-4-26B-A4B-it bash $0 ..." >&2
    exit 1
fi
if [[ ! -f "$HF_REF/config.json" ]]; then
    echo "ERROR: HF_REF=$HF_REF does not contain config.json" >&2
    exit 1
fi

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
        EP=${5:-1}
    fi
elif [[ $# -ge 6 ]]; then
    TP=${3:-1}
    PP=${4:-1}
    EP=${5:-1}
    CUSTOM_PIPELINE_LAYERS=$6
else
    TP=${3:-1}
    PP=${4:-1}
    EP=${5:-1}
fi

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-hf
SAVE_VISION_MODEL=./tmp/vision-model-hf
SAVE_ADAPTER=./tmp/adapter-hf
SAVE_PATCH=./tmp/patch-hf

mkdir -p $SAVE_LANGUAGE_MODEL $SAVE_VISION_MODEL
cp $HF_REF/config.json $SAVE_LANGUAGE_MODEL/config.json
cp $HF_REF/config.json $SAVE_VISION_MODEL/config.json

if [[ -d "$LOAD/release" ]]; then
    MCORE_LOAD=$LOAD/release
else
    MCORE_LOAD=$LOAD
fi


if [[ $PP -eq 1 ]]; then
    NONLLM_LOAD=$MCORE_LOAD
    NONLLM_EP=$EP
else
    NONLLM_LOAD=$MCORE_LOAD/tmp_nonllm_view/
    NONLLM_EP=1
    rm -rf $NONLLM_LOAD
    mkdir -p $NONLLM_LOAD
    for ((i=0;i<$TP;i++)); do
        from=$(printf "mp_rank_%02d_000_000" $i)
        to=$(printf "mp_rank_%02d" $i)
        cp -r $MCORE_LOAD/$from $NONLLM_LOAD/$to
    done
fi


python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/llm.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    ${CUSTOM_PIPELINE_LAYERS:+--custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS} \
    --load_ckpt_path=$MCORE_LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/vit.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=$NONLLM_EP \
    --load_ckpt_path=$NONLLM_LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim


python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/gemma4-26b-a4b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=$NONLLM_EP \
    --load_ckpt_path=$NONLLM_LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=$NONLLM_EP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/gemma4-26b-a4b/vision-patch.json \
    --load_ckpt_path=$NONLLM_LOAD \
    --save_ckpt_path=$SAVE_PATCH

if [[ $MCORE_LOAD != $NONLLM_LOAD ]]; then
    rm -rf $NONLLM_LOAD
fi

# merge – reuse OV2 generic 4-dict merger (model-agnostic)
python $CONVERT_CHECKPOINT_PATH/custom/llava_onevision2/merge_huggingface.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

cp $HF_REF/config.json $SAVE/config.json

rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
