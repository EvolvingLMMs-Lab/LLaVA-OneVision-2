# =============================================================================
# LLaVA-OneVision2 4B-p14m3 – Re-shard Megatron-Core checkpoint (mcore → mcore)
# =============================================================================
#
# Converts an existing mcore checkpoint to a new TP/PP layout by round-tripping
# through HuggingFace format.  Pass the same CUSTOM_PIPELINE_LAYERS that the
# source checkpoint was trained with so that it is correctly read, and the same
# value for the target layout as well.
#
# Usage:
#   bash convert_4b_p14m3_mcore_to_release.sh <LOAD> <SAVE> <TP> <PP> [CUSTOM_PIPELINE_LAYERS]
#
# Arguments:
#   LOAD                    Path to the source Megatron-Core checkpoint
#   SAVE                    Path to save the re-sharded Megatron-Core checkpoint
#   TP                      Tensor parallel size (for both source and target)
#   PP                      Pipeline parallel size (for both source and target)
#   CUSTOM_PIPELINE_LAYERS  (optional) Comma-separated layer counts per PP stage,
#                           must match the layout the checkpoint was saved with.
#
# Recommended splits for the 4B model (36 LLM layers, 300M ViT on stage-0):
#   PP=4 : 0,12,12,12  ← stage-0 holds ViT only; stages 1-3 each get 12 layers
#   PP=3 : 0,18,18     ← stage-0 holds ViT only; stages 1-2 each get 18 layers
#
# Examples:
#   bash convert_4b_p14m3_mcore_to_release.sh /src /dst 2 4 0,12,12,12
#   bash convert_4b_p14m3_mcore_to_release.sh /src /dst 1 1
# =============================================================================

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1
SAVE=$2
TP=$3
PP=$4
CUSTOM_PIPELINE_LAYERS=$5


bash $AIAK_TRAINING_PATH/examples/llava_onevision2/convert/convert_4b_p14m3_mcore_to_hf.sh \
    $LOAD tmp_hf $TP $PP $CUSTOM_PIPELINE_LAYERS

bash $AIAK_TRAINING_PATH/examples/llava_onevision2/convert/convert_4b_p14m3_hf_to_mcore.sh \
    tmp_hf $SAVE $TP $PP $CUSTOM_PIPELINE_LAYERS

rm -rf tmp_hf
