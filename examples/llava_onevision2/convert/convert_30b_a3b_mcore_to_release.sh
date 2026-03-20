# =============================================================================
# LLaVA-OneVision2 30B-A3B – Re-shard Megatron-Core checkpoint (mcore -> mcore)
# =============================================================================
#
# Usage:
#   bash convert_30b_a3b_mcore_to_release.sh <LOAD> <SAVE> <PP> <EP>
#   bash convert_30b_a3b_mcore_to_release.sh <LOAD> <SAVE> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
#   bash convert_30b_a3b_mcore_to_release.sh <LOAD> <SAVE> <TP> <PP> <EP>
#   bash convert_30b_a3b_mcore_to_release.sh <LOAD> <SAVE> <TP> <PP> <EP> <CUSTOM_PIPELINE_LAYERS>
# =============================================================================

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

bash $AIAK_TRAINING_PATH/examples/llava_onevision2/convert/convert_30b_a3b_mcore_to_hf.sh \
    $LOAD tmp_hf $TP $PP $EP $CUSTOM_PIPELINE_LAYERS

bash $AIAK_TRAINING_PATH/examples/llava_onevision2/convert/convert_30b_a3b_hf_to_mcore.sh \
    tmp_hf $SAVE $TP $PP $EP $CUSTOM_PIPELINE_LAYERS

rm -rf tmp_hf
