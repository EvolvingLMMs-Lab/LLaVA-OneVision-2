#!/bin/bash
# Smoke test: pandas70m_gt10min (multi-frame video data with shared patch_positions.npy)
#
# Layout per sample:
#   images: 384 video frame .jpg paths
#   patch_positions: a single .npy file (concat of all frames' patch positions)
#
# Pipeline expects manifest mode (s1) -> manifest-driven s2 -> bin packing -> WDS.
# s4 is run inside docker for energon/.idx support.
#
# Usage: bash run_pandas70m_smoke.sh
set -euo pipefail
export PYTHONUNBUFFERED=1

SRC=/ov21/smoke_test/pandas70m/pandas70m_smoke.jsonl
OUT=/ov21/smoke_test/pandas70m/output
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
IMG=llava_megatron:25.12

mkdir -p "$OUT"
LOG=${OUT}/smoke.log
echo "[smoke] host=$(hostname) src=$SRC out=$OUT lines=$(wc -l <"$SRC")" | tee "$LOG"

# ---------- Step 1-3 on bare metal (s1/s2 use conda env, s3 is pure python) ----------
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llava_main_xy

# Stage-2 optimal env (auto_pipe.sh defaults — same as 47M run)
export MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2 PYTHONMALLOC=malloc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export S2_USE_SPAWN=1 FAST_TOKENIZER=1 FAST_IMG_META=1
# Conservative caps for a 500-sample smoke (avoid spawn overhead dominating).
export S2_MAX_PROCESSES=8 S2_MAX_THREADS_PER_PROC=4
export S2_MEM_PER_PROC_GB=2 S2_MAXTASKS_PER_CHILD=10

# Run steps 1-3 via auto_pipe with --no-preprocess for s4 disabled here.
# We skip s4 inside auto_pipe by stopping it after step 3 -- easiest is to set
# -S 1 fresh (force_run not needed for empty OUT) and rerun s4 explicitly in
# docker afterwards. But auto_pipe doesn't support a "stop after step N" flag,
# so we run s1/s2/s3 manually below to keep control.

# ---- Step 1: build manifest ----
MANIFEST=${OUT}/manifest.jsonl
if [[ ! -f "${OUT}/.status/step1.done" ]]; then
    mkdir -p "${OUT}/.status"
    echo "[smoke] STEP 1: build manifest" | tee -a "$LOG"
    python ${PIPE}/s1_build_manifest.py \
        -i "$SRC" --image-root / -o "$MANIFEST" --overwrite 2>&1 | tee -a "$LOG"
    touch "${OUT}/.status/step1.done"
fi

# ---- Step 2: token lengths ----
TOK=${OUT}/token_info.txt
MAXLEN=65536
if [[ ! -f "${OUT}/.status/step2.done" ]]; then
    echo "[smoke] STEP 2: compute token lengths (max-len=${MAXLEN})" | tee -a "$LOG"
    python ${PIPE}/s2_compute_token_lengths.py \
        --manifest "$MANIFEST" --output "$TOK" \
        --model-path "$MODEL" --max-len ${MAXLEN} --task-type sft \
        --chunk-size 100 --min-workers 4 --max-workers 32 --timeout 300 2>&1 | tee -a "$LOG"
    touch "${OUT}/.status/step2.done"
fi

# ---- Step 3: bin packing ----
BINS=${OUT}/bins_${MAXLEN}.pkl
if [[ ! -f "${OUT}/.status/step3.done" ]]; then
    echo "[smoke] STEP 3: bin packing" | tee -a "$LOG"
    python ${PIPE}/s3_bin_packing.py \
        --input "$TOK" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 4 \
        --algorithm bfd --patch-slack 0.30 2>&1 | tee -a "$LOG"
    touch "${OUT}/.status/step3.done"
fi

# ---- Step 4: bins -> webdataset (inside docker for energon/.idx) ----
WDS=${OUT}/webdataset
if [[ -d "$WDS" ]] && ls "$WDS"/*.tar 1>/dev/null 2>&1; then
    rm -rf "$WDS"
fi
mkdir -p "$WDS"
echo "[smoke] STEP 4: bins -> webdataset (docker)" | tee -a "$LOG"
docker run --rm \
    --name "smoke_pandas70m_$$" \
    -v /vlm:/vlm -v /ov2:/ov2 -v /ov21:/ov21 -v /rice_vl:/rice_vl \
    -w "${PIPE}" --entrypoint bash \
    "${IMG}" -c "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=${PIPE}:\${PYTHONPATH:-}
        python s4_bins_to_webdataset.py \
            --bins-file '${BINS}' \
            --manifest  '${MANIFEST}' \
            --output-dir '${WDS}' \
            --workers 16 \
            --max-samples-per-shard 200 \
            --shard-prefix pandas70m_smoke \
            --sample-class-name PackedCaptioningSample \
            --longest-edge 3072 --jpeg-quality 92
    " 2>&1 | tee -a "$LOG"

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
echo "[smoke] DONE  tars=${NTAR}  idx=${NIDX}  size=$(du -sh $WDS | awk '{print $1}')" | tee -a "$LOG"
