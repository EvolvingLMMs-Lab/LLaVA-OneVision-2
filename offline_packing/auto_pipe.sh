#!/bin/bash
# ============================================================================
# Offline Packing Pipeline for SFT Training Data
# ============================================================================
#
# Stages:
#   1. Build lightweight manifest.jsonl     (s1_build_manifest.py)
#   2. Compute token lengths                (s2_compute_token_lengths.py)
#   3. Bin packing                          (s3_bin_packing.py)
#   4. Bins -> WebDataset                   (s4_bins_to_webdataset.py)
#
# The new step-1 no longer splits the input JSONL into per-sample files and
# no longer copies images / .npy files. Instead, it produces a single
# manifest.jsonl whose entries carry absolute source paths and the raw
# sample payload. Steps 2 and 4 read directly from those absolute paths,
# halving disk usage and removing the redundant copy/read cycle.
#
# Resume is supported via status files in ${OUTPUT_BASE}/.status/.
#
# ============================================================================
# Usage Examples
# ============================================================================
#
# 1) Minimal required invocation:
#    bash auto_pipe.sh \
#        -i /path/to/input.jsonl \
#        -r /path/to/images \
#        -o /path/to/output_base \
#        -L 64000
#
# 2) Full invocation with common overrides:
#    bash auto_pipe.sh \
#        -i /path/to/input.jsonl \
#        -r /path/to/image_root \
#        -m /path/to/tokenizer_or_model \
#        -o /path/to/output_dir \
#        -L 64000 \
#        -M 200 \
#        -w 32 \
#        --max-pixels 12845056 \
#        --direct-workers 32 \
#        --shard-prefix pretrain \
#        --sample-class PackedCaptioningSample \
#        --max-samples-per-shard 10000 \
#        --max-shard-size 3000000000
#
# 3) Resume from a specific step (re-run step 3 and step 4):
#    bash auto_pipe.sh -i ... -r ... -o ... -L 64000 -S 3
#
# 4) Resume step 4 without restarting finished shards:
#    bash auto_pipe.sh -i ... -r ... -o ... -L 64000 -S 4 --resume-direct
#
# 5) Force re-run everything:
#    bash auto_pipe.sh -i ... -r ... -o ... -L 64000 -f
#
# 6) Show pipeline status and exit:
#    bash auto_pipe.sh -o /path/to/output_dir -L 64000 -s
#
# 7) Disable NPY mode (let s2 do image processing instead):
#    bash auto_pipe.sh -i ... -r ... -o ... -L 64000 --no-npy
#
# ============================================================================
#
# Author: LLaVA-OneVision Team
# ============================================================================

set -e
set -o pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Stage-2 optimal defaults (validated on 47M SFT run, May 2026)
# ----------------------------------------------------------------------------
# These env vars are read by s2_compute_token_lengths.py. They produce ~24x
# speedup and 44x lower per-worker RSS vs the legacy fork+AutoProcessor path:
#
#   Worker RSS:     12 GB  -> 0.27 GB
#   Real Mem Used:  79-99% -> ~10%
#   s2 throughput:  3 chunks/min -> 40-90 chunks/min
#
# Layer 1: glibc / BLAS memory hygiene
#   - MALLOC_ARENA_MAX=2 caps per-thread arenas (avoids RSS growth from many
#     small mallocs landing in distinct arenas).
#   - MALLOC_TRIM_THRESHOLD_ lowers the threshold above which free() returns
#     pages to the kernel; combined with explicit malloc_trim() in the worker
#     loop, keeps RSS flat across long runs.
#   - OMP/OPENBLAS/MKL=1: numerical libs default to oversubscribe; capping
#     them prevents 64-thread BLAS blowups under our own multiprocessing.
#
# Layer 2: spawn context (S2_USE_SPAWN=1)
#   - Default fork inherits parent's ~10 GB RSS via copy-on-write; on the
#     first write each worker's pages become private, ballooning effective
#     usage to 5-6 GB/worker. Spawn workers start fresh (~700 MB) and stay
#     small. Required when running >16 workers on a ~256 GB box.
#
# Layer 3: FAST_TOKENIZER=1
#   - Skip AutoProcessor (which materializes input_ids tensor + attention
#     mask per call). Use tokenizer.encode() directly. Byte-equal output
#     verified on 100/100 A/B samples.
#
# Layer 4: FAST_IMG_META=1
#   - Read W/H from JPEG/PNG header instead of decoding pixels. With
#     Qwen2-VL smart_resize replica we compute exact patch counts without
#     ever materializing an RGB array. ~5-50x faster per image.
#
# Override any of these by exporting before running this script.
# ============================================================================
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-131072}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONMALLOC=${PYTHONMALLOC:-malloc}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

export S2_USE_SPAWN=${S2_USE_SPAWN:-1}
export FAST_TOKENIZER=${FAST_TOKENIZER:-1}
export FAST_IMG_META=${FAST_IMG_META:-1}

# Concurrency caps. Tuned for ~128-core / 256 GB nodes. The script also
# auto-scales down based on system load. Override per-host as needed.
#
# Sizing rationale (post FAST_TOKENIZER + spawn):
#   Worker RSS settles around 270 MB → memory is NOT the bottleneck.
#   CPU is. Each worker spends ~95% of its time in tokenizer.encode() +
#   Jinja2 render, both of which HOLD the GIL. Threads inside a process
#   only help during file-header syscalls (smart_resize is pure math).
#   So real parallelism ≈ # processes, not # processes × # threads.
#
#   At 24 procs we observed ~25% CPU on 128 cores; bumping to 64 procs
#   targets 60-70% CPU and 2.5-3x more throughput. Memory budget:
#     64 × 0.3 GB ≈ 20 GB / 256 GB ≈ 8%   (plenty of headroom)
#
#   THREADS_PER_PROC dropped from 8→4: more threads just fight for the
#   GIL once we already have enough procs to saturate cores.
#
#   MAXTASKS_PER_CHILD raised from 3→10: with FAST_TOKENIZER, worker RSS
#   no longer grows monotonically, so the spawn-restart cost (~3-5s per
#   worker for transformers import) no longer pays for itself every 3
#   chunks. 10 chunks/child = ~1 restart every 5000 samples per worker.
export S2_MAX_PROCESSES=${S2_MAX_PROCESSES:-64}
export S2_MAX_THREADS_PER_PROC=${S2_MAX_THREADS_PER_PROC:-4}
export S2_MEM_PER_PROC_GB=${S2_MEM_PER_PROC_GB:-2}
export S2_MAXTASKS_PER_CHILD=${S2_MAXTASKS_PER_CHILD:-10}

usage() {
    echo "Usage: $0 -i INPUT_JSONL -r IMAGE_ROOT -o OUTPUT_BASE -L MAX_TOKEN_LEN [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -i, --input             Path to input JSONL file"
    echo "  -r, --image-root        Path to image root directory"
    echo "  -o, --output            Output base directory"
    echo "  -L, --max-token-len     Max token length (e.g. 64000)"
    echo ""
    echo "Optional:"
    echo "  -m, --model-path        Path to tokenizer (default: Qwen2.5-VL-3B-Instruct)"
    echo "      --max-pixels        Max image pixels for Step 2 processor resize"
    echo "  -S, --from-step         Start from step (1-4), reset subsequent steps"
    echo "  -M, --max-samples       Max samples per bin (default: 200)"
    echo "  -w, --max-workers       Max workers for step 2 tokenization (default: 32)"
    echo "      --direct-workers    Workers for step 4 conversion (default: 32)"
    echo "      --shard-prefix      WebDataset shard prefix (default: pretrain)"
    echo "      --sample-class      Sample class name (default: PackedCaptioningSample)"
    echo "      --max-samples-per-shard  Max samples per shard (default: 10000)"
    echo "      --max-shard-size    Max shard size in bytes (default: 3000000000)"
    echo "      --no-npy            Disable NPY mode (s2 will use image processing)"
    echo "      --s1-num-processes  Cap step 1 worker processes (default: 0 = s1 default)"
    echo "      --s2-num-processes  Cap step 2 worker processes (default: 0 = auto)"
    echo "      --s2-min-workers    Min workers kept alive in step 2 (default: 4)"
    echo "      --chunk-size        Items per step 1/2 chunk (default: 500)"
    echo "      --resume-direct     Pass --resume to step 4 conversion"
    echo "  -f, --force             Force re-run all steps"
    echo "  -s, --status            Show current pipeline status and exit"
    echo "  -h, --help              Show this help message"
    exit 1
}

INPUT_JSONL=""
IMAGE_ROOT=""
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"   # default model dir: set via -m flag or MODEL/MODEL_PATH env
OUTPUT_BASE=""
FORCE_RUN=false
SHOW_STATUS=false
FROM_STEP=0
MAX_TOKEN_LEN=""
OVERRIDE_MAX_SAMPLES=""
OVERRIDE_MAX_WORKERS=""
NO_NPY_MODE=false
DIRECT_WORKERS=32
SHARD_PREFIX="pretrain"
SAMPLE_CLASS_NAME="PackedCaptioningSample"
MAX_SAMPLES_PER_SHARD=10000
MAX_SHARD_SIZE=3000000000
RESUME_DIRECT=false
OVERRIDE_MAX_PIXELS=""
S1_NUM_PROCESSES=0
S2_NUM_PROCESSES=0
S2_MIN_WORKERS=4
OVERRIDE_CHUNK_SIZE=""
LONGEST_EDGE=3072
JPEG_QUALITY=92
NO_PREPROCESS=false
MAX_PATCHES_PER_BIN=0
MIN_PATCHES_PER_BIN=0
PATCH_SLACK=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)           INPUT_JSONL="$2"; shift 2 ;;
        -r|--image-root)      IMAGE_ROOT="$2"; shift 2 ;;
        -m|--model-path)      MODEL_PATH="$2"; shift 2 ;;
        --max-pixels)         OVERRIDE_MAX_PIXELS="$2"; shift 2 ;;
        -o|--output)          OUTPUT_BASE="$2"; shift 2 ;;
        -S|--from-step)
            FROM_STEP="$2"
            if [[ ! "$FROM_STEP" =~ ^[1-4]$ ]]; then
                echo "Error: --from-step must be 1-4"
                exit 1
            fi
            shift 2 ;;
        -L|--max-token-len)   MAX_TOKEN_LEN="$2"; shift 2 ;;
        -M|--max-samples)     OVERRIDE_MAX_SAMPLES="$2"; shift 2 ;;
        -w|--max-workers)     OVERRIDE_MAX_WORKERS="$2"; shift 2 ;;
        --direct-workers)     DIRECT_WORKERS="$2"; shift 2 ;;
        --shard-prefix)       SHARD_PREFIX="$2"; shift 2 ;;
        --sample-class)       SAMPLE_CLASS_NAME="$2"; shift 2 ;;
        --max-samples-per-shard) MAX_SAMPLES_PER_SHARD="$2"; shift 2 ;;
        --max-shard-size)     MAX_SHARD_SIZE="$2"; shift 2 ;;
        --no-npy)             NO_NPY_MODE=true; shift ;;
        --s1-num-processes)   S1_NUM_PROCESSES="$2"; shift 2 ;;
        --s2-num-processes)   S2_NUM_PROCESSES="$2"; shift 2 ;;
        --s2-min-workers)     S2_MIN_WORKERS="$2"; shift 2 ;;
        --chunk-size)         OVERRIDE_CHUNK_SIZE="$2"; shift 2 ;;
        --resume-direct)      RESUME_DIRECT=true; shift ;;
        --longest-edge)       LONGEST_EDGE="$2"; shift 2 ;;
        --jpeg-quality)       JPEG_QUALITY="$2"; shift 2 ;;
        --no-preprocess)      NO_PREPROCESS=true; shift ;;
        --max-patches-per-bin) MAX_PATCHES_PER_BIN="$2"; shift 2 ;;
        --min-patches-per-bin) MIN_PATCHES_PER_BIN="$2"; shift 2 ;;
        --patch-slack)        PATCH_SLACK="$2"; shift 2 ;;
        -f|--force)           FORCE_RUN=true; shift ;;
        -s|--status)          SHOW_STATUS=true; shift ;;
        -h|--help)            usage ;;
        *)                    echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "${MAX_TOKEN_LEN}" ]]; then
    echo "Error: --max-token-len is required"; usage
fi
if [[ ! "${MAX_TOKEN_LEN}" =~ ^[0-9]+$ ]] || [[ "${MAX_TOKEN_LEN}" -le 0 ]]; then
    echo "Error: --max-token-len must be a positive integer"; exit 1
fi

if [[ "${SHOW_STATUS}" == "true" ]]; then
    [[ -z "${OUTPUT_BASE}" ]] && { echo "Error: --output required for status"; usage; }
else
    [[ -z "${INPUT_JSONL}" ]] && { echo "Error: --input is required"; usage; }
    [[ -z "${IMAGE_ROOT}" ]]  && { echo "Error: --image-root is required"; usage; }
    [[ -z "${OUTPUT_BASE}" ]] && { echo "Error: --output is required"; usage; }
fi

# ============================================================================
# Configuration
# ============================================================================
MAX_SAMPLES_PER_BIN=${OVERRIDE_MAX_SAMPLES:-200}
PACKING_ALGORITHM="bfd"
CHUNK_SIZE=${OVERRIDE_CHUNK_SIZE:-500}
MAX_WORKERS=${OVERRIDE_MAX_WORKERS:-32}
TIMEOUT=180
TASK_TYPE="sft"

# Derived paths
MANIFEST_FILE="${OUTPUT_BASE}/manifest.jsonl"
TOKEN_INFO="${OUTPUT_BASE}/token_info.txt"
BINS_FILE="${OUTPUT_BASE}/bins_${MAX_TOKEN_LEN//000/k}len.pkl"
BINS_LOG_FILE="${OUTPUT_BASE}/bins_${MAX_TOKEN_LEN//000/k}len_packing.log"
WEBDATASET_OUTPUT="${OUTPUT_BASE}/webdataset"
STATUS_DIR="${OUTPUT_BASE}/.status"

get_param_suffix() { echo "${MAX_TOKEN_LEN}_${MAX_SAMPLES_PER_BIN}"; }
get_step_status_file() {
    local step=$1
    if [[ "${step}" -eq 1 ]]; then
        echo "${STATUS_DIR}/step1.done"
    else
        echo "${STATUS_DIR}/direct_step${step}_$(get_param_suffix).done"
    fi
}

# ============================================================================
# Helper functions
# ============================================================================
log_step() {
    echo ""
    echo "============================================================================"
    echo "  Step $1: $2"
    echo "============================================================================"
    echo ""
}
log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"; }
log_warn()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2; }

check_file_exists() {
    [[ -f "$1" ]] || { log_error "Required file not found: $1"; exit 1; }
}
check_dir_exists() {
    [[ -d "$1" ]] || { log_error "Required directory not found: $1"; exit 1; }
}

mark_step_done() {
    local step=$1
    local status_file
    status_file=$(get_step_status_file "${step}")
    mkdir -p "${STATUS_DIR}"
    echo "completed at $(date '+%Y-%m-%d %H:%M:%S')" > "${status_file}"
    if [[ "${step}" -gt 1 ]]; then
        echo "MAX_TOKEN_LEN=${MAX_TOKEN_LEN}" >> "${status_file}"
        echo "MAX_SAMPLES_PER_BIN=${MAX_SAMPLES_PER_BIN}" >> "${status_file}"
    fi
    log_info "Step ${step} marked as complete (params: ${MAX_TOKEN_LEN}/${MAX_SAMPLES_PER_BIN})"
}

is_step_done() {
    local step=$1
    [[ -f "$(get_step_status_file "${step}")" ]]
}

reset_from_step() {
    local from_step=$1
    log_info "Resetting status from step ${from_step} onwards..."
    for i in $(seq "${from_step}" 4); do
        local sf
        sf=$(get_step_status_file "${i}")
        if [[ -f "${sf}" ]]; then
            rm -f "${sf}"
            log_info "  Reset step ${i} status"
        fi
    done
}

check_prerequisites() {
    local target_step=$1
    for i in $(seq 1 $((target_step - 1))); do
        if ! is_step_done "${i}"; then
            log_error "Cannot start from step ${target_step}: step ${i} is not completed"
            exit 1
        fi
    done
}

show_status() {
    echo ""
    echo "============================================================================"
    echo "  Pipeline Status: ${OUTPUT_BASE}"
    echo "============================================================================"
    echo ""
    echo "  Configuration:"
    echo "    - Max token length:    ${MAX_TOKEN_LEN}"
    echo "    - Max samples per bin: ${MAX_SAMPLES_PER_BIN}"
    echo "    - Packing algorithm:   ${PACKING_ALGORITHM}"
    echo "    - Direct workers:      ${DIRECT_WORKERS}"
    echo "    - Shard prefix:        ${SHARD_PREFIX}"
    echo ""

    local steps=("Build manifest.jsonl" "Compute token lengths" "Bin packing" "Bins -> WebDataset")
    local outputs=("${MANIFEST_FILE}" "${TOKEN_INFO}" "${BINS_FILE}" "${WEBDATASET_OUTPUT}")

    for i in 1 2 3 4; do
        local sf
        sf=$(get_step_status_file "${i}")
        local step_name="${steps[$((i-1))]}"
        local output_path="${outputs[$((i-1))]}"
        if [[ -f "${sf}" ]]; then
            local done_time
            done_time=$(head -1 "${sf}")
            echo "  [DONE] Step ${i}: ${step_name}"
            echo "      ${done_time}"
        else
            echo "  [TODO] Step ${i}: ${step_name}"
            echo "      Not completed"
        fi
        echo "      Output: ${output_path}"
        echo ""
    done
    echo "============================================================================"
}

# ============================================================================
# Status Check Mode
# ============================================================================
if [[ "${SHOW_STATUS}" == "true" ]]; then
    show_status
    exit 0
fi

# ============================================================================
# Main Pipeline
# ============================================================================
log_info "Starting offline packing pipeline..."
log_info "Input: ${INPUT_JSONL}"
log_info "Output base: ${OUTPUT_BASE}"
log_info "Max token length: ${MAX_TOKEN_LEN}"
log_info "Max samples per bin: ${MAX_SAMPLES_PER_BIN}"
if [[ -n "${OVERRIDE_MAX_PIXELS}" ]]; then
    log_info "Max image pixels: ${OVERRIDE_MAX_PIXELS}"
fi

if [[ "${FORCE_RUN}" == "true" ]]; then
    log_warn "Force mode enabled - will re-run all steps"
    rm -rf "${STATUS_DIR}"
fi

if [[ "${FROM_STEP}" -gt 0 ]]; then
    log_info "Starting from step ${FROM_STEP} (as requested)"
    check_prerequisites "${FROM_STEP}"
    reset_from_step "${FROM_STEP}"
fi

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${STATUS_DIR}"

show_status

# ----------------------------------------------------------------------------
# Step 1: Build manifest.jsonl from raw JSONL/JSON input
# ----------------------------------------------------------------------------
if is_step_done 1 && [[ "${FORCE_RUN}" != "true" ]]; then
    log_info "Skipping Step 1 (already completed)"
else
    log_step 1 "Build lightweight manifest.jsonl"

    check_file_exists "${INPUT_JSONL}"
    check_dir_exists "${IMAGE_ROOT}"

    S1_CMD=(python "${SCRIPT_DIR}/s1_build_manifest.py"
        -i "${INPUT_JSONL}"
        --image-root "${IMAGE_ROOT}"
        -o "${MANIFEST_FILE}"
        --overwrite)

    if [[ -n "${OVERRIDE_CHUNK_SIZE}" ]]; then
        S1_CMD+=(--chunk-size "${CHUNK_SIZE}")
    fi

    if [[ "${S1_NUM_PROCESSES}" -gt 0 ]]; then
        S1_CMD+=(--processes "${S1_NUM_PROCESSES}")
    fi

    "${S1_CMD[@]}"

    mark_step_done 1
    log_info "Step 1 complete. Output: ${MANIFEST_FILE}"
fi

# ----------------------------------------------------------------------------
# Step 2: Compute token lengths (manifest-driven)
# ----------------------------------------------------------------------------
if is_step_done 2 && [[ "${FORCE_RUN}" != "true" ]]; then
    log_info "Skipping Step 2 (already completed)"
else
    log_step 2 "Compute token lengths for each sample"

    check_file_exists "${MANIFEST_FILE}"
    check_dir_exists "${MODEL_PATH}"

    S2_CMD=(python "${SCRIPT_DIR}/s2_compute_token_lengths.py"
        --manifest "${MANIFEST_FILE}"
        --output "${TOKEN_INFO}"
        --model-path "${MODEL_PATH}"
        --max-len "${MAX_TOKEN_LEN}"
        --task-type "${TASK_TYPE}"
        --chunk-size "${CHUNK_SIZE}"
        --min-workers "${S2_MIN_WORKERS}"
        --num-processes "${S2_NUM_PROCESSES}"
        --max-workers "${MAX_WORKERS}"
        --timeout "${TIMEOUT}")

    if [[ -n "${OVERRIDE_MAX_PIXELS}" ]]; then
        S2_CMD+=(--max-pixels "${OVERRIDE_MAX_PIXELS}")
    fi

    if [[ "${NO_NPY_MODE}" == "true" ]]; then
        S2_CMD+=(--no-npy)
        log_info "NPY mode disabled, will use image processing"
    fi

    "${S2_CMD[@]}"

    mark_step_done 2
    log_info "Step 2 complete. Output: ${TOKEN_INFO}"
fi

# ----------------------------------------------------------------------------
# Step 3: Bin packing
# ----------------------------------------------------------------------------
if is_step_done 3 && [[ "${FORCE_RUN}" != "true" ]]; then
    log_info "Skipping Step 3 (already completed)"
else
    log_step 3 "Bin packing (group samples by token capacity)"

    check_file_exists "${TOKEN_INFO}"

    log_info "Bin packing log will be saved to: ${BINS_LOG_FILE}"

    python "${SCRIPT_DIR}/s3_bin_packing.py" \
        --input "${TOKEN_INFO}" \
        --output "${BINS_FILE}" \
        --capacity "${MAX_TOKEN_LEN}" \
        --max-samples "${MAX_SAMPLES_PER_BIN}" \
        --algorithm "${PACKING_ALGORITHM}" \
        --max-patches-per-bin "${MAX_PATCHES_PER_BIN}" \
        --min-patches-per-bin "${MIN_PATCHES_PER_BIN}" \
        --patch-slack "${PATCH_SLACK}" \
        2>&1 | tee "${BINS_LOG_FILE}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "Step 3 failed. Check log: ${BINS_LOG_FILE}"
        exit 1
    fi

    mark_step_done 3
    log_info "Step 3 complete. Output: ${BINS_FILE}"
fi

# ----------------------------------------------------------------------------
# Step 4: Bins -> WebDataset
# ----------------------------------------------------------------------------
if is_step_done 4 && [[ "${FORCE_RUN}" != "true" ]]; then
    log_info "Skipping Step 4 (already completed)"
else
    log_step 4 "Bins -> WebDataset"

    check_file_exists "${BINS_FILE}"
    check_file_exists "${MANIFEST_FILE}"

    log_info "timestamp_decimal policy:"
    log_info "  - Read from manifest 'raw' field; missing/None falls back to 1."
    log_info "  - Every bin payload in the tar will contain 'timestamp_decimal'."

    DIRECT_CMD=(python "${SCRIPT_DIR}/s4_bins_to_webdataset.py"
        --bins-file "${BINS_FILE}"
        --manifest "${MANIFEST_FILE}"
        --output-dir "${WEBDATASET_OUTPUT}"
        --workers "${DIRECT_WORKERS}"
        --max-samples-per-shard "${MAX_SAMPLES_PER_SHARD}"
        --max-shard-size "${MAX_SHARD_SIZE}"
        --shard-prefix "${SHARD_PREFIX}"
        --sample-class-name "${SAMPLE_CLASS_NAME}")

    if [[ "${RESUME_DIRECT}" == "true" ]]; then
        DIRECT_CMD+=(--resume)
    fi

    DIRECT_CMD+=(--longest-edge "${LONGEST_EDGE}" --jpeg-quality "${JPEG_QUALITY}")
    if [[ "${NO_PREPROCESS}" == "true" ]]; then
        DIRECT_CMD+=(--no-preprocess)
    fi

    "${DIRECT_CMD[@]}"

    mark_step_done 4
    log_info "Step 4 complete. Output: ${WEBDATASET_OUTPUT}"
fi

# ============================================================================
# Pipeline Complete
# ============================================================================
echo ""
echo "============================================================================"
echo "  Pipeline Complete!"
echo "============================================================================"
echo ""
echo "  Outputs:"
echo "    - Manifest:       ${MANIFEST_FILE}"
echo "    - Token info:     ${TOKEN_INFO}"
echo "    - Bins file:      ${BINS_FILE}"
echo "    - Packing log:    ${BINS_LOG_FILE}"
echo "    - WebDataset:     ${WEBDATASET_OUTPUT}"
echo ""
echo "============================================================================"
