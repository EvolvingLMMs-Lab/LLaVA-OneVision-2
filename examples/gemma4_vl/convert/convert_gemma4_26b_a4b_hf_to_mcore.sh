# =============================================================================
# Gemma4-VL 26B-A4B – Convert HuggingFace checkpoint to Megatron-Core
# =============================================================================
#
# Usage:
#   bash convert_gemma4_26b_a4b_hf_to_mcore.sh <LOAD> <SAVE> <PP> <EP>
#   bash convert_gemma4_26b_a4b_hf_to_mcore.sh <LOAD> <SAVE> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#   bash convert_gemma4_26b_a4b_hf_to_mcore.sh <LOAD> <SAVE> <TP> <PP> <EP>
#   bash convert_gemma4_26b_a4b_hf_to_mcore.sh <LOAD> <SAVE> <TP> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#
# Arguments:
#   LOAD  Path to the source HuggingFace checkpoint
#   SAVE  Path to save the Megatron-Core checkpoint
#   TP    Tensor parallel size (optional, defaults to 1)
#   PP    Pipeline parallel size (recommended 1 for this script)
#   EP    Expert parallel size
#   CUSTOM_PIPELINE_LAYERS  (optional) Comma-separated layer counts per PP stage,
#                           used for custom PP layer layouts.
#
# Notes:
#   * Gemma4-VL uses standalone Python converters under
#     tools/convert_checkpoint/custom/gemma4_vl/ for both LLM and ViT
#     (no JSON config for those two; ViT/LLM read num_layers/num_experts
#     directly from the HuggingFace config.json).
#   * patch + adapter still use small JSON configs under
#     tools/convert_checkpoint/config/gemma4-26b-a4b/.
#   * P2.5 ships with TP=PP=EP=1 verified end to end (16680 keys).
#   * P2.7 adds TP/EP sharding support: LLM emits 2D mp_rank_TT_EEE when
#     EP>1 (Qwen3 convention), and ViT/adapter/patch always emit 1D
#     mp_rank_TT (broadcast across the EP axis at merge time).
#   * P2.8 adds PP sharding support: LLM emits 3D mp_rank_TT_PPP_EEE
#     when PP>1 (any EP, including EP=1, Qwen3 convention), with layers
#     locally renumbered (0..N_local-1) per PP rank. ViT/adapter/patch
#     still emit 1D mp_rank_TT and the merger broadcasts them across
#     EP and into PP stage 0 only.
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
SAVE_LANGUAGE_MODEL=./tmp/language-mcore
SAVE_VISION_MODEL=./tmp/vision-model-mcore
SAVE_ADAPTER=./tmp/adapter-mcore
SAVE_PATCH=./tmp/patch-mcore


# llm (moe) – standalone converter, no JSON config
python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/llm.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    ${CUSTOM_PIPELINE_LAYERS:+--custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS} \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit – standalone converter, no JSON config
python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/vit.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/gemma4-26b-a4b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch in vit
python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/gemma4-26b-a4b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge (1D mp_rank, no TP/EP sharding at P2.5)
python $CONVERT_CHECKPOINT_PATH/custom/gemma4_vl/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
