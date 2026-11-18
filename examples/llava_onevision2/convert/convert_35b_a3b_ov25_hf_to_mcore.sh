#!/bin/bash
# =============================================================================
# Qwen3.5-MoE (35B-A3B) LLaVA-OneVision2: merged HuggingFace -> Megatron-Core.
#
# Separate from convert_4b_ov25_hf_to_mcore.sh: that one is for the dense
# Qwen3.5 backbones, while this one fixes TP=1 (expert weights have no TP rule)
# and adds the expert-parallel split that a MoE training run loads.
#
# Two steps:
#   1. HF -> mcore TP1/PP1, all 256 experts in one shard (language + ViT +
#      adapter + patch embed, merged);
#   2. qwen35_split_ep.py -> release/mp_rank_00_{e:03d}, i.e. the layout
#      megatron writes when expert_model_parallel_size > 1.
#
# Usage:
#   bash convert_35b_a3b_ov25_hf_to_mcore.sh <LOAD_HF> <SAVE> [EP]
#   env: NUM_LAYERS (40), FULL_ATTENTION_INTERVAL (4), NUM_EXPERTS (256),
#        KEEP_TP1 (0 -> delete the intermediate TP1 checkpoint), AIAK_TRAINING_PATH
# =============================================================================
set -e

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"
CUSTOM="$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2"
CFG="$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p16m3"   # ViT/adapter are shared p16m3

LOAD=$1
SAVE=$2
EP=${3:-8}
TP=1
PP=1
NUM_LAYERS=${NUM_LAYERS:-40}
FULL_ATTENTION_INTERVAL=${FULL_ATTENTION_INTERVAL:-4}
NUM_EXPERTS=${NUM_EXPERTS:-256}

[[ -n "$LOAD" && -n "$SAVE" ]] || { echo "usage: $0 <LOAD_HF> <SAVE> [EP]"; exit 1; }
export PYTHONPATH="$AIAK_MAGATRON_PATH:$CONVERT_CHECKPOINT_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH"
echo "[wrapper] 35B-A3B MoE: TP=$TP PP=$PP EP=$EP NUM_LAYERS=$NUM_LAYERS NUM_EXPERTS=$NUM_EXPERTS"

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-mcore-qwen35-moe
SAVE_VISION_MODEL=./tmp/vision-model-mcore-moe
SAVE_ADAPTER=./tmp/adapter-mcore-moe
SAVE_PATCH=./tmp/patch-mcore-moe
if [[ "$EP" == "1" ]]; then SAVE_TP1="$SAVE"; else SAVE_TP1="${SAVE}_tp1_premerge"; fi

# ---- language: Qwen3.5-MoE hybrid LLM (router + experts + shared expert) ------
python $CUSTOM/qwen35_hf_to_mcore.py \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --num_layers=$NUM_LAYERS \
    --full_attention_interval=$FULL_ATTENTION_INTERVAL \
    --tensor_model_parallel_size=$TP

# ---- vit ---------------------------------------------------------------------
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

# ---- adapter -----------------------------------------------------------------
python $CUSTOM/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CFG/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# ---- vision patch embed ------------------------------------------------------
python $CUSTOM/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CFG/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# ---- merge into one mcore checkpoint ----------------------------------------
python $CUSTOM/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE_TP1/release \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP
echo release > $SAVE_TP1/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL $SAVE_VISION_MODEL $SAVE_ADAPTER $SAVE_PATCH

# ---- expert-parallel split ---------------------------------------------------
if [[ "$EP" != "1" ]]; then
    python $CUSTOM/qwen35_split_ep.py \
        --src $SAVE_TP1 \
        --dst $SAVE \
        --ep $EP \
        --num_experts $NUM_EXPERTS
    if [[ "${KEEP_TP1:-0}" == "1" ]]; then
        echo "[wrapper] intermediate TP1 checkpoint kept at: $SAVE_TP1"
    else
        rm -rf "$SAVE_TP1"
    fi
fi

echo "Done. Megatron-Core checkpoint (EP=$EP) saved to: $SAVE"
