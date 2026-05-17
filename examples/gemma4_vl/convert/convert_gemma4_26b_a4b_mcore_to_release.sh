# =============================================================================
# Gemma4-VL 26B-A4B – Re-shard Megatron-Core checkpoint (mcore -> mcore)
# =============================================================================
#
# Usage (target layout via positional args; source layout via env vars):
#   HF_REF=/path/to/source/hf bash convert_gemma4_26b_a4b_mcore_to_release.sh <LOAD> <SAVE> <PP> <EP>
#   HF_REF=... SRC_TP=1 SRC_PP=1 SRC_EP=1 bash ... <LOAD> <SAVE> <TP> <PP> <EP>
#   HF_REF=... SRC_TP=1 SRC_PP=1 SRC_EP=1 bash ... <LOAD> <SAVE> <TP> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#
# Required environment:
#   HF_REF   Path to the original HuggingFace checkpoint directory (forwarded
#            to mcore_to_hf.sh; see that script for why it is required).
#
# Optional environment (source mcore layout, defaults TP=PP=EP=1):
#   SRC_TP   Source tensor parallel size      (default 1)
#   SRC_PP   Source pipeline parallel size    (default 1)
#   SRC_EP   Source expert parallel size      (default 1)
#
# Implementation: round-trips through HF as an intermediate format, reusing the
# two sibling scripts (mcore_to_hf -> hf_to_mcore). The intermediate HF
# directory tmp_hf is removed at the end. The mcore_to_hf step is invoked
# with the SOURCE layout (so the standalone converters select the correct
# loader); the hf_to_mcore step is invoked with the TARGET layout.
# =============================================================================

if [[ -z "$HF_REF" ]]; then
    echo "ERROR: HF_REF environment variable must be set to the source HuggingFace dir" >&2
    echo "  Example: HF_REF=/ov2/pretrain_models/google/gemma-4-26B-A4B-it bash $0 ..." >&2
    exit 1
fi

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"

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

SRC_TP="${SRC_TP:-1}"
SRC_PP="${SRC_PP:-1}"
SRC_EP="${SRC_EP:-1}"

bash $AIAK_TRAINING_PATH/examples/gemma4_vl/convert/convert_gemma4_26b_a4b_mcore_to_hf.sh \
    $LOAD tmp_hf $SRC_TP $SRC_PP $SRC_EP $CUSTOM_PIPELINE_LAYERS

bash $AIAK_TRAINING_PATH/examples/gemma4_vl/convert/convert_gemma4_26b_a4b_hf_to_mcore.sh \
    tmp_hf $SAVE $TP $PP $EP $CUSTOM_PIPELINE_LAYERS

rm -rf tmp_hf
