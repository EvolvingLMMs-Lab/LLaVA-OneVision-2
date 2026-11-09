#!/bin/bash
# Build FlashAttention-3 (Dao-AILab/flash-attention v2.8.3, hopper/, package flash_attn_3) for the experimental switch
# OV2_OPT_FA3_ATTN, against the training image's torch, and install it with --no-deps into $OV2_EXTRAS_DIR/fa3.
# No GPU needed. Only what this model needs is built: bf16, head dims 64 (vision) and 128 (LM), sm90; FP16/FP8,
# head dims 96/192/256, split-KV, paged/append KV, softcap and the sm80 path are disabled (FLASH_ATTENTION_DISABLE_*).
# setup.py fetches nvcc 12.6 / ptxas 12.8 from NVIDIA's redist (FA3's recommended toolchain).
#
#   git clone --branch v2.8.3 --recursive https://github.com/Dao-AILab/flash-attention $WORK/flash-attention
#   docker run --rm --net host -e WORK=$WORK -e OV2_EXTRAS_DIR=$OV2_EXTRAS_DIR -v $WORK:$WORK \
#       -v $OV2_EXTRAS_DIR:$OV2_EXTRAS_DIR -v $PWD:/src llava_megatron:26.05 bash /src/tools/kernels/build_fa3.sh
set -euxo pipefail
WORK=${WORK:?set WORK to a directory containing the flash-attention v2.8.3 clone (with csrc/cutlass)}
OV2_EXTRAS_DIR=${OV2_EXTRAS_DIR:?set OV2_EXTRAS_DIR (fa3/ is created there)}
export HOME=$WORK/home TMPDIR=$WORK/fa3_build_tmp
mkdir -p "$HOME" "$TMPDIR"
cd "$WORK/flash-attention/hopper"
export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_SM80=TRUE
export MAX_JOBS=${MAX_JOBS:-24} NVCC_THREADS=${NVCC_THREADS:-4} FLASH_ATTN_LOCAL_VERSION=ov2
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python setup.py bdist_wheel 2>&1 | tail -40
rm -rf "$OV2_EXTRAS_DIR/fa3"
pip install --no-deps --no-build-isolation --target "$OV2_EXTRAS_DIR/fa3" dist/flash_attn_3-*.whl
ls -la "$OV2_EXTRAS_DIR/fa3"
