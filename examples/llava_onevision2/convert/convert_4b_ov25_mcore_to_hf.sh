# =============================================================================
# LLaVA-OneVision2 4B (Qwen3.5 hybrid LLM) - Convert Megatron-Core -> HuggingFace
# =============================================================================
#
# REVERSE of convert_4b_ov25_hf_to_mcore.sh. The LLM backbone is a hybrid stack
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
#   bash convert_4b_ov25_mcore_to_hf.sh <LOAD> <SAVE> [TP] [PP] [REFERENCE]
#
#   LOAD       Megatron-Core checkpoint root (contains release/mp_rank_*)  OR a
#              dir that directly contains mp_rank_* .
#   SAVE       output merged HuggingFace dir.
#   TP         tensor-parallel size (default 1)   -- TP in {1,2,4}
#   PP         pipeline-parallel size (default 1) -- PP=1/PP=2 supported (2D mp_rank_{t}_{p} merged in the LLM loader)
#   REFERENCE  merged HF model to copy config/modeling/tokenizer from
#              (default /models/llava_onevision2_4b_qwen35_p16m3)
#
# Streaming models (STREAMING_EXPORT=auto|1|0, default auto): a checkpoint trained with the streaming
# loss (--streaming-loss-scope=last-segment / --state-token-focal-loss in its args, or a <silence>
# token in its --hf-tokenizer-path) is exported as a streaming model: the tokenizer/processor are
# copied from the tokenizer it was TRAINED with (all trained ids kept, e.g. <captioning>, nothing
# re-registered), and chat_template.jinja becomes the dual-mode template -- offline by default (the
# trainer's qwen2-vl default system), streaming with apply_chat_template(..., streaming=True) (the
# trainer's STREAMING_SYSTEM_PROMPT) -- with "<T.T seconds>" before timestamped images in both modes.
# Both modes are checked against the training format. Override the tokenizer with TOKENIZER_SRC=<dir>.
# Non-streaming exports are unaffected.
#
# Example:
#   bash convert_4b_ov25_mcore_to_hf.sh \
#     /models/llava_onevision2_4b_qwen35_p16m3_mcore_tp4pp1 \
#     /models/llava_onevision2_4b_qwen35_p16m3_from_mcore 4 1
# =============================================================================
set -e

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2.5}"
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
_HID=$(PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH" python - "$(ls -d $_probe_rel/mp_rank_00 $_probe_rel/mp_rank_00_000 2>/dev/null | head -1)/model_optim_rng.pt" <<PYEOF
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = d.get("model", d)
for k, v in m.items():
    if k.endswith("embedding.word_embeddings.weight"):
        print(v.shape[1]); break
PYEOF
)
if [[ "$_HID" == "2048" ]]; then
    _DEF_REF=/models/llava_onevision2_2b_qwen35_p16m3; _DEF_LAYERS=24
elif [[ "$_HID" == "2560" ]]; then
    _DEF_REF=/models/llava_onevision2_4b_qwen35_p16m3; _DEF_LAYERS=32
elif [[ "$_HID" == "4096" ]]; then
    _DEF_REF=/models/llava_onevision2_9b_qwen35_p16m3_v0904; _DEF_LAYERS=32
elif [[ "$_HID" == "5120" ]]; then
    _DEF_REF=/models/llava_onevision2_27b_qwen38_p16m3_v0923; _DEF_LAYERS=64
else
    echo "[wrapper] WARNING: unknown hidden_size=$_HID; falling back to 4B defaults"
    _DEF_REF=/models/llava_onevision2_4b_qwen35_p16m3; _DEF_LAYERS=32
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

# ---- what was this checkpoint trained as? (read before any weights-only rewrite below) -------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)   # tools shipped with this script, independent of AIAK_TRAINING_PATH
read -r TRAIN_TOKENIZER TRAIN_STREAMING < <(python - "$(ls -d $LOAD_REL/mp_rank_00 $LOAD_REL/mp_rank_00_000 2>/dev/null | head -1)/model_optim_rng.pt" <<'PYEOF'
import json, os, sys, torch
args = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("args")
tok = getattr(args, "hf_tokenizer_path", None) or "-"
# --streaming-loss-scope carries default="all", so argparse stores it in the args of EVERY run and
# the checkpoint keeps it. Only a value the user actually chose is evidence of streaming training;
# taking the default as evidence would mark every checkpoint streaming and fail the finalize check
# below on ordinary exports. The tokenizer probe underneath is the real fallback, so let it run.
scope = getattr(args, "streaming_loss_scope", None)
streaming = scope not in (None, "all") or bool(getattr(args, "state_token_focal_loss", False))
tj = os.path.join(tok, "tokenizer.json")
if not streaming and os.path.isfile(tj):
    streaming = any(a["content"] == "<silence>" for a in json.load(open(tj)).get("added_tokens", []))
print(tok, int(streaming))
PYEOF
)
STREAMING_EXPORT=${STREAMING_EXPORT:-auto}
[[ $STREAMING_EXPORT == auto ]] && STREAMING_EXPORT=${TRAIN_STREAMING:-0}
echo "[wrapper] trained tokenizer=$TRAIN_TOKENIZER streaming_checkpoint=${TRAIN_STREAMING:-?} -> STREAMING_EXPORT=$STREAMING_EXPORT"
if [[ $STREAMING_EXPORT == 1 ]]; then
    STREAMING_TOKENIZER=${TOKENIZER_SRC:-$TRAIN_TOKENIZER}
    [[ -f $STREAMING_TOKENIZER/tokenizer.json ]] || {
        echo "[wrapper] streaming export needs the training tokenizer, not found: '$STREAMING_TOKENIZER'" >&2
        echo "[wrapper] pass TOKENIZER_SRC=<the --hf-tokenizer-path used in training>" >&2; exit 1; }
fi

# ---- refuse checkpoints whose replicated GDN params drifted across TP ranks (pre SP-FIX
#      training): the converter reads TP rank 0 only and would export a rank-0-only model.
#      Repair with `qwen35_tp_consistency.py resync <src> <dst>` and convert <dst>.
#      Set SKIP_TP_CHECK=1 to bypass. ----------------------------------------------------
# ---- a --gdn-head-parallel checkpoint holds per-rank GDN head shards: rebuild the replicated
#      layout (weights only) first; everything below reads replicated GDN tensors. ------------
if [[ "$(python $CUSTOM/qwen35_gdn_head_tp.py detect "$LOAD_REL")" == "head_parallel" ]]; then
    echo "[wrapper] GDN head-parallel checkpoint -> unsharding to replicated layout"
    rm -rf ./tmp/gdn-unsharded
    python $CUSTOM/qwen35_gdn_head_tp.py unshard "$LOAD_REL" ./tmp/gdn-unsharded || exit 1
    LOAD_REL=./tmp/gdn-unsharded
fi

if [[ "${SKIP_TP_CHECK:-0}" != "1" ]]; then
    python $CUSTOM/qwen35_tp_consistency.py check "$LOAD_REL" || exit 1
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
        --args_from $TMP_VIS_ARGS/release --src $LOAD_REL --dst $WORK_REL --tp $TP
    rm -rf $TMP_VIS_ARGS
    LOAD_REL="$WORK_REL"
fi

# ---- language: Qwen3.5 hybrid LLM (dedicated reverse converter) -------------
python $CUSTOM/qwen35_mcore_to_hf.py \
    --load_ckpt_path=$LOAD_REL \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --num_layers=$NUM_LAYERS \
    --full_attention_interval=$FULL_ATTENTION_INTERVAL \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --hf_config_path=$REFERENCE

# ---- vit (reuse validated mcore->hf converter) ------------------------------
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --common_config_path=$CFG/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_REL \
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
    --load_ckpt_path=$LOAD_REL \
    --save_ckpt_path=$SAVE_ADAPTER

# ---- vision patch (reuse validated mcore->hf converter) ---------------------
python $CUSTOM/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MAGATRON_PATH \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --common_config_path=$CFG/vision-patch.json \
    --load_ckpt_path=$LOAD_REL \
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

rm -rf $SAVE_LANGUAGE_MODEL $SAVE_VISION_MODEL $SAVE_ADAPTER $SAVE_PATCH ./tmp/work-release
echo "Done. Merged HuggingFace model saved to: $SAVE"

# =============================================================================
# Streaming checkpoints only: make the export loadable AND aligned with training.
#
# Everything above copies config/modeling/processor/tokenizer verbatim from
# REFERENCE, which for a streaming model is the wrong tokenizer (no <silence> ..
# <caption_begin>: they would tokenize as sub-words and the streaming signal
# would be silently lost) and a chat template that cannot render streaming turns.
# Copy the tokenizer the model was actually TRAINED with, untouched, then install
# the dual-mode template and check both modes against the training format.
#
# A non-streaming export leaves this block alone and ends above, unchanged.
# =============================================================================
if [[ $STREAMING_EXPORT == 1 ]]; then
    python $REPO_ROOT/tools/data_preprocess/register_streaming_tokens.py \
        --src "$STREAMING_TOKENIZER" --dst "$SAVE" --no-streaming-tokens || exit 1
    python $REPO_ROOT/tools/data_preprocess/finalize_streaming_export.py \
        --dst "$SAVE" --train-tokenizer "$STREAMING_TOKENIZER" --repo "$REPO_ROOT" \
        --template "$SCRIPT_DIR/chat_template_ov_dual.jinja" || {
        echo "[finalize] streaming export check FAILED; $SAVE is not aligned with training" >&2; exit 1; }
    echo "Done. Streaming HuggingFace model: $SAVE"
fi
