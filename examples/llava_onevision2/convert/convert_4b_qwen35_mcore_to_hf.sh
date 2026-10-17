# =============================================================================
# LLaVA-OneVision2 4B (Qwen3.5 hybrid LLM) - Convert Megatron-Core -> HuggingFace
# =============================================================================
#
# REVERSE of convert_4b_qwen35_hf_to_mcore.sh. The LLM backbone is a hybrid stack
# (linear gated-delta-net + periodic full-attention, interleaved 3D M-RoPE,
# partial rotary 0.25), so the *language* step uses the dedicated reverse converter
# custom/llava_onevision2/qwen35_mcore_to_hf.py (built on the validated
# aiak_training_llm.models.qwen35.hf_convert mcore_to_hf_* layer math) INSTEAD of
# the generic model.py + qwen3.json path. Vision tower / adapter / vision-patch
# REUSE the exact same validated mcore->hf converters as the Qwen3 4B p16m3 model.
#
# The resulting HF dir is a full merged LlavaOnevision2ForConditionalGeneration:
# language + visual weights, plus config/modeling/processor/tokenizer copied from
# the reference merged HF model (only the weights are regenerated).
#
# Usage:
#   bash convert_4b_qwen35_mcore_to_hf.sh <LOAD> <SAVE> [TP] [PP] [REFERENCE]
#
#   LOAD       Megatron-Core checkpoint root (contains release/mp_rank_*)  OR a
#              dir that directly contains mp_rank_* .
#   SAVE       output merged HuggingFace dir.
#   TP         tensor-parallel size (default 1)   -- TP in {1,2,4}
#   PP         pipeline-parallel size (default 1) -- PP=1/PP=2 supported (2D mp_rank_{t}_{p} merged in the LLM loader)
#   REFERENCE  merged HF model to copy config/modeling/tokenizer from
#              (default /vlm/yinxie/pretrain_models/llava_onevision2_4b_qwen35_p16m3)
#
# Example:
#   bash convert_4b_qwen35_mcore_to_hf.sh \
#     /vlm/yinxie/pretrain_models/llava_onevision2_4b_qwen35_p16m3_mcore_tp4pp1 \
#     /vlm/yinxie/pretrain_models/llava_onevision2_4b_qwen35_p16m3_from_mcore 4 1
# =============================================================================
set -e

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/vlm/yinxie/code/OV21/qwen35}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"
CUSTOM="$CONVERT_CHECKPOINT_PATH/custom/llava_onevision2"
CFG="$CONVERT_CHECKPOINT_PATH/config/llava-onevision2-4b-p16m3"

LOAD=$1
SAVE=$2
TP=${3:-1}
PP=${4:-1}

# ---- auto-detect model size from the ckpt (hidden_size of the embedding) so the
#      right REFERENCE (config/modeling source) and NUM_LAYERS are chosen even when
#      arg5/NUM_LAYERS are omitted. 2048 -> Qwen3.5-2B, 2560 -> Qwen3.5-4B. ---------
_probe_rel="$LOAD"; [[ -d "$LOAD/release" ]] && _probe_rel="$LOAD/release"
_probe_shard="$(ls -d $_probe_rel/mp_rank_00 $_probe_rel/mp_rank_00_000 2>/dev/null | head -1)/model_optim_rng.pt"
# hidden size + MoE flag + decoder layer count in one load (the ckpt is large).
read -r _HID _IS_MOE _NLAYER < <(PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH" python - "$_probe_shard" <<PYEOF
import re, sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = d.get("model", d)
hid, moe, layers = 0, 0, set()
for k, v in m.items():
    if k.endswith("embedding.word_embeddings.weight"):
        hid = v.shape[1]
    if ".mlp.router.weight" in k:
        moe = 1
    lm = re.search(r"decoder\.layers\.(\d+)\.", k)
    if lm:
        layers.add(int(lm.group(1)))
print(hid, moe, (max(layers) + 1) if layers else 0)
PYEOF
)
# MoE (Qwen3.5-MoE, e.g. 35B-A3B) shares hidden_size 2048 with the dense 2B, so the
# router probe has to win over the hidden-size table.
if [[ "$_IS_MOE" == "1" ]]; then
    _DEF_REF=/vlm/yinxie/pretrain_models/llava_onevision2_35b_a3b_qwen36_p16m3_v1010
    _DEF_LAYERS=${_NLAYER:-40}
elif [[ "$_HID" == "2048" ]]; then
    _DEF_REF=/vlm/yinxie/pretrain_models/llava_onevision2_2b_qwen35_p16m3; _DEF_LAYERS=24
elif [[ "$_HID" == "2560" ]]; then
    _DEF_REF=/vlm/yinxie/pretrain_models/llava_onevision2_4b_qwen35_p16m3; _DEF_LAYERS=32
elif [[ "$_HID" == "4096" ]]; then
    _DEF_REF=/vlm/yinxie/pretrain_models/llava_onevision2_9b_qwen35_p16m3_v0904; _DEF_LAYERS=32
elif [[ "$_HID" == "5120" ]]; then
    _DEF_REF=/vlm/yinxie/pretrain_models/llava_onevision2_27b_qwen38_p16m3_v0923; _DEF_LAYERS=64
else
    echo "[wrapper] WARNING: unknown hidden_size=$_HID; falling back to 4B defaults"
    _DEF_REF=/vlm/yinxie/pretrain_models/llava_onevision2_4b_qwen35_p16m3; _DEF_LAYERS=32
fi
REFERENCE=${5:-$_DEF_REF}
NUM_LAYERS=${NUM_LAYERS:-$_DEF_LAYERS}
FULL_ATTENTION_INTERVAL=${FULL_ATTENTION_INTERVAL:-4}
echo "[wrapper] detected hidden=$_HID -> REFERENCE=$REFERENCE NUM_LAYERS=$NUM_LAYERS"

export PYTHONPATH="$AIAK_MAGATRON_PATH:$CONVERT_CHECKPOINT_PATH:$AIAK_TRAINING_PATH:$CONVERT_CHECKPOINT_PATH/..:$PYTHONPATH"

# All mcore loaders expect the dir that directly contains mp_rank_* .
if [[ -d "$LOAD/release" ]]; then
    LOAD_REL="$LOAD/release"
else
    LOAD_REL="$LOAD"
fi

# ---- refuse checkpoints whose replicated GDN params drifted across TP ranks (pre SP-FIX
#      training): the converter reads TP rank 0 only and would export a rank-0-only model.
#      Repair with `qwen35_tp_consistency.py resync <src> <dst>` and convert <dst>.
#      Set SKIP_TP_CHECK=1 to bypass. ----------------------------------------------------
# ---- a --gdn-head-parallel checkpoint holds per-rank GDN head shards: rebuild the replicated
#      layout (weights only) first; everything below reads replicated GDN tensors. ------------
# ---- expert parallelism (MoE only). PP=1 EP>1 stores mp_rank_{t:02d}_{e:03d}; the
#      language converter gathers the experts itself, but the vision / adapter / patch
#      converters only know mp_rank_{t:02d}, so they get a rank-0 view (those weights are
#      replicated across EP ranks). EP=<n> overrides the auto-count. -------------------
EP=${EP:-auto}
if [[ "$EP" == "auto" ]]; then
    if [[ "$_IS_MOE" == "1" && "$PP" == "1" ]]; then
        EP=$(ls -d $LOAD_REL/mp_rank_00_[0-9][0-9][0-9] 2>/dev/null | wc -l | tr -d ' ')
        [[ "$EP" == "0" ]] && EP=1
    else
        EP=1
    fi
fi
LOAD_REL_NONLANG="$LOAD_REL"
if [[ "$EP" != "1" ]]; then
    echo "[wrapper] MoE checkpoint with EP=$EP -> rank-0 view for vision/adapter/patch"
    rm -rf ./tmp/ep-rank0; mkdir -p ./tmp/ep-rank0
    for ((_t = 0; _t < TP; _t++)); do
        _src="$(cd "$LOAD_REL" && pwd)/$(printf 'mp_rank_%02d_000' $_t)"
        [[ -d "$_src" ]] || { echo "[wrapper] missing EP shard $_src"; exit 1; }
        ln -s "$_src" "./tmp/ep-rank0/$(printf 'mp_rank_%02d' $_t)"
    done
    LOAD_REL_NONLANG=./tmp/ep-rank0
    # TP/GDN checks below assume the mp_rank_{t:02d} layout; run them on the same view.
    LOAD_REL_CHECK="$LOAD_REL_NONLANG"
fi

if [[ "$(python $CUSTOM/qwen35_gdn_head_tp.py detect "${LOAD_REL_CHECK:-$LOAD_REL}")" == "head_parallel" ]]; then
    echo "[wrapper] GDN head-parallel checkpoint -> unsharding to replicated layout"
    rm -rf ./tmp/gdn-unsharded
    python $CUSTOM/qwen35_gdn_head_tp.py unshard "$LOAD_REL" ./tmp/gdn-unsharded || exit 1
    LOAD_REL=./tmp/gdn-unsharded
fi

if [[ "${SKIP_TP_CHECK:-0}" != "1" ]]; then
    python $CUSTOM/qwen35_tp_consistency.py check "${LOAD_REL_CHECK:-$LOAD_REL}" || exit 1
fi

mkdir -p ./tmp/
SAVE_LANGUAGE_MODEL=./tmp/language-hf-qwen35
SAVE_VISION_MODEL=./tmp/vision-model-hf
SAVE_ADAPTER=./tmp/adapter-hf
SAVE_PATCH=./tmp/patch-hf

# ---- ensure the checkpoint carries megatron `args` (needed only by the vision
#      tower reverse via model.py). Real training checkpoints always have it; the
#      qwen35 forward language converter writes an args-less {"model":...}. If it
#      is missing, synthesize args from a throwaway forward vision conversion of
#      the reference model (identical arch) and copy the checkpoint into a working
#      dir with args injected (originals are never mutated). --------------------
HAS_ARGS=$(python -c "import torch;d=torch.load('$(ls -d $LOAD_REL/mp_rank_00 $LOAD_REL/mp_rank_00_000 2>/dev/null | head -1)/model_optim_rng.pt',map_location='cpu',weights_only=False);print('args' in d)")
if [ "$HAS_ARGS" != "True" ]; then
    echo "[wrapper] checkpoint has no megatron args -> synthesizing for vision reverse"
    TMP_VIS_ARGS=./tmp/vis-args-mcore
    WORK_REL=./tmp/work-release
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
    LOAD_REL_NONLANG="$WORK_REL"
    # EP>1: the language converter keeps reading the real EP shards (it needs all experts
    # and does not need megatron args); only the vision/adapter/patch view is rewritten.
    [[ "$EP" == "1" ]] && LOAD_REL="$WORK_REL"
fi

# ---- language: Qwen3.5 hybrid LLM (dedicated reverse converter) -------------
python $CUSTOM/qwen35_mcore_to_hf.py \
    --load_ckpt_path=$LOAD_REL \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --num_layers=$NUM_LAYERS \
    --full_attention_interval=$FULL_ATTENTION_INTERVAL \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_model_parallel_size=$EP \
    --hf_config_path=$REFERENCE

# ---- vit (reuse validated mcore->hf converter) ------------------------------
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

# ---- adapter (reuse validated mcore->hf converter) --------------------------
python $CUSTOM/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CFG/adapter.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD_REL_NONLANG \
    --save_ckpt_path=$SAVE_ADAPTER

# ---- vision patch (reuse validated mcore->hf converter) ---------------------
python $CUSTOM/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --common_config_path=$CFG/vision-patch.json \
    --load_ckpt_path=$LOAD_REL_NONLANG \
    --save_ckpt_path=$SAVE_PATCH

# ---- merge language + vision + adapter + patch into one HF state dict --------
python $CUSTOM/merge_huggingface.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

# ---- copy config / modeling / processor / tokenizer (weights already written)
mkdir -p "$SAVE"
for f in config.json configuration_llava_onevision2.py modeling_llava_onevision2.py \
         processing_llava_onevision2.py video_processing_llava_onevision2.py \
         codec_video_processing_llava_onevision2.py video_preprocessor_config.json \
         generation_config.json processor_config.json preprocessor_config.json \
         tokenizer_config.json tokenizer.json tokenizer.model vocab.json merges.txt \
         chat_template.jinja special_tokens_map.json added_tokens.json; do
    if [[ -f "$REFERENCE/$f" ]]; then
        cp "$REFERENCE/$f" "$SAVE/$f"
    fi
done

rm -rf $SAVE_LANGUAGE_MODEL $SAVE_VISION_MODEL $SAVE_ADAPTER $SAVE_PATCH ./tmp/work-release ./tmp/ep-rank0
echo "Done. Merged HuggingFace model saved to: $SAVE"
