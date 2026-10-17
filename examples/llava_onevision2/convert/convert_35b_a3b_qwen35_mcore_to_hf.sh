#!/bin/bash
# =============================================================================
# Qwen3.5-MoE (35B-A3B) LLaVA-OneVision2: Megatron-Core -> merged HuggingFace.
#
# Separate from convert_4b_qwen35_mcore_to_hf.sh because a MoE checkpoint differs
# in three ways that do not belong in the dense wrapper:
#   * expert parallelism: shards are release/mp_rank_{t:02d}_{e:03d}, and the
#     routed experts have to be gathered and renumbered (the language converter
#     does this via --expert_model_parallel_size);
#   * the vision / adapter / patch converters only understand mp_rank_{t:02d},
#     so they read a rank-0 view (those weights are replicated across EP ranks);
#   * no hidden-size auto-detect: 35B-A3B and the dense 2B share hidden_size
#     2048, so size probing is the wrong way to pick the reference here.
#
# MoE is TP=1 only (qwen35_hf_to_mcore.py does not TP-shard expert weights, and
# qwen35_mcore_to_hf.py refuses to un-shard such a checkpoint).
#
# Usage:
#   bash convert_35b_a3b_qwen35_mcore_to_hf.sh <LOAD> <SAVE> [EP] [REFERENCE]
#     LOAD       mcore checkpoint dir (with or without the release/ level)
#     SAVE       output merged-HF dir
#     EP         expert-parallel size of the checkpoint (default: auto-count)
#     REFERENCE  merged-HF model supplying config.json / modeling / tokenizer
#   env: NUM_LAYERS (40), FULL_ATTENTION_INTERVAL (4), FUSED_EXPERTS (0),
#        SKIP_TP_CHECK (0), AIAK_TRAINING_PATH
# =============================================================================
set -e

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/vlm/yinxie/code/OV21/qwen35}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"
CUSTOM="$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2"
CFG="$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p16m3"   # ViT/adapter are shared p16m3

LOAD=$1
SAVE=$2
EP=${3:-auto}
REFERENCE=${4:-/vlm/yinxie/pretrain_models/llava_onevision2_35b_a3b_qwen36_p16m3_v1010}
TP=1
PP=1
NUM_LAYERS=${NUM_LAYERS:-40}
FULL_ATTENTION_INTERVAL=${FULL_ATTENTION_INTERVAL:-4}

[[ -n "$LOAD" && -n "$SAVE" ]] || { echo "usage: $0 <LOAD> <SAVE> [EP] [REFERENCE]"; exit 1; }
export PYTHONPATH="$AIAK_MAGATRON_PATH:$CONVERT_CHECKPOINT_PATH:$AIAK_TRAINING_PATH:$CONVERT_CHECKPOINT_PATH/..:$PYTHONPATH"

# All mcore loaders expect the dir that directly contains mp_rank_* .
if [[ -d "$LOAD/release" ]]; then LOAD_REL="$LOAD/release"; else LOAD_REL="$LOAD"; fi

# ---- expert parallelism ------------------------------------------------------
if [[ "$EP" == "auto" ]]; then
    EP=$(ls -d $LOAD_REL/mp_rank_00_[0-9][0-9][0-9] 2>/dev/null | wc -l | tr -d ' ')
    [[ "$EP" == "0" ]] && EP=1
fi
echo "[wrapper] 35B-A3B MoE: EP=$EP TP=$TP PP=$PP NUM_LAYERS=$NUM_LAYERS REFERENCE=$REFERENCE"

mkdir -p ./tmp/
# The vision / adapter / patch converters only know mp_rank_{t:02d}; give them a
# rank-0 view of the EP shards (vision, adapter and patch are EP-replicated).
LOAD_REL_NONLANG="$LOAD_REL"
if [[ "$EP" != "1" ]]; then
    rm -rf ./tmp/ep-rank0; mkdir -p ./tmp/ep-rank0
    _src="$(cd "$LOAD_REL" && pwd)/mp_rank_00_000"
    [[ -d "$_src" ]] || { echo "[wrapper] missing EP shard $_src"; exit 1; }
    ln -s "$_src" ./tmp/ep-rank0/mp_rank_00
    LOAD_REL_NONLANG=./tmp/ep-rank0
    echo "[wrapper] rank-0 view for vision/adapter/patch -> $LOAD_REL_NONLANG"
fi

# ---- refuse checkpoints whose replicated GDN params drifted across TP ranks ----
# (vacuous at TP=1, kept so the MoE path cannot silently skip a check the dense
#  path enforces; SKIP_TP_CHECK=1 bypasses.)
if [[ "${SKIP_TP_CHECK:-0}" != "1" ]]; then
    python $CUSTOM/qwen35_tp_consistency.py check "$LOAD_REL_NONLANG" || exit 1
fi

SAVE_LANGUAGE_MODEL=./tmp/language-hf-qwen35-moe
SAVE_VISION_MODEL=./tmp/vision-model-hf-moe
SAVE_ADAPTER=./tmp/adapter-hf-moe
SAVE_PATCH=./tmp/patch-hf-moe

# ---- the vision reverse converter needs megatron args; a checkpoint produced by
#      the forward converter has none, so synthesize them into the rank-0 view ----
HAS_ARGS=$(python -c "import torch;d=torch.load('$LOAD_REL_NONLANG/mp_rank_00/model_optim_rng.pt',map_location='cpu',weights_only=False);print('args' in d)")
if [ "$HAS_ARGS" != "True" ]; then
    echo "[wrapper] checkpoint has no megatron args -> synthesizing for vision reverse"
    TMP_VIS_ARGS=./tmp/vis-args-mcore-moe
    WORK_REL=./tmp/work-release-moe
    rm -rf $TMP_VIS_ARGS $WORK_REL
    python $CONVERT_CHECKPOINT_PATH/model.py \
        --load_platform=huggingface --save_platform=mcore \
        --megatron_path $AIAK_MAGATRON_PATH \
        --common_config_path=$CFG/vision-model.json \
        --tensor_model_parallel_size=$TP --pipeline_model_parallel_size=1 \
        --load_ckpt_path=$REFERENCE --save_ckpt_path=$TMP_VIS_ARGS \
        --safetensors --no_save_optim --no_load_optim
    python $CUSTOM/qwen35_inject_args.py \
        --args_from $TMP_VIS_ARGS/release --src $LOAD_REL_NONLANG --dst $WORK_REL --tp $TP
    rm -rf $TMP_VIS_ARGS
    # Only the non-language view is rewritten: the language converter reads the real
    # EP shards (it needs every expert) and does not need megatron args.
    LOAD_REL_NONLANG="$WORK_REL"
fi

# ---- language: Qwen3.5-MoE hybrid LLM ---------------------------------------
FUSED_FLAG=""
[[ "${FUSED_EXPERTS:-0}" == "1" ]] && FUSED_FLAG="--fused_experts"
python $CUSTOM/qwen35_mcore_to_hf.py \
    --load_ckpt_path=$LOAD_REL \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --num_layers=$NUM_LAYERS \
    --full_attention_interval=$FULL_ATTENTION_INTERVAL \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_model_parallel_size=$EP \
    --hf_config_path=$REFERENCE \
    $FUSED_FLAG

# ---- vit ---------------------------------------------------------------------
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CFG/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_REL_NONLANG \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# ---- adapter -----------------------------------------------------------------
python $CUSTOM/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CFG/adapter.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD_REL_NONLANG \
    --save_ckpt_path=$SAVE_ADAPTER

# ---- vision patch embed ------------------------------------------------------
python $CUSTOM/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --common_config_path=$CFG/vision-patch.json \
    --load_ckpt_path=$LOAD_REL_NONLANG \
    --save_ckpt_path=$SAVE_PATCH

# ---- merge into one HF model -------------------------------------------------
python $CUSTOM/merge_huggingface.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

mkdir -p "$SAVE"
for f in config.json configuration_llava_onevision2.py modeling_llava_onevision2.py \
         processing_llava_onevision2.py video_processing_llava_onevision2.py \
         codec_video_processing_llava_onevision2.py video_preprocessor_config.json \
         generation_config.json processor_config.json preprocessor_config.json \
         tokenizer_config.json tokenizer.json tokenizer.model vocab.json merges.txt \
         chat_template.jinja special_tokens_map.json added_tokens.json; do
    [[ -f "$REFERENCE/$f" ]] && cp "$REFERENCE/$f" "$SAVE/$f"
done

rm -rf $SAVE_LANGUAGE_MODEL $SAVE_VISION_MODEL $SAVE_ADAPTER $SAVE_PATCH \
       ./tmp/work-release-moe ./tmp/ep-rank0
echo "Done. Merged HuggingFace model saved to: $SAVE"
