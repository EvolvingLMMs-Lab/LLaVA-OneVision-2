#!/bin/bash
# ============================================================================
# Offline Pipeline (No-Packing) -- raw JSONL -> WebDataset
# ============================================================================
#
# Stages (per part):
#   1. Build manifest.jsonl                  (s1_build_manifest.py)
#   2. Build trivial bins.pkl + inject       (s_make_trivial_bins.py)
#      timestamp_decimal=2 into raw payload
#   3. Bins -> WebDataset                    (s4_bins_to_webdataset.py)
#
# Each output WebDataset sample contains EXACTLY ONE original sample
# (sample_count = 1). We re-use s4_bins_to_webdataset because the training
# code already treats sample_count==1 the same as sample_count>1 -- it just
# iterates len(sample.images) once. The default sample-class
# (PackedCaptioningSample) is therefore correct.
#
# Status / resume is supported via per-part status files in
# ${OUTPUT_BASE}/${part_dir}/.status/.
#
# ============================================================================
# Usage
# ============================================================================
#
#   bash auto_pipe_no_packing.sh
#
# Optional overrides (env or CLI flags) -- see usage() below.
#
# Designed to run inside the llava_megatron:25.12 image. No extra installs
# required (webdataset / megatron.energon / PIL / numpy / tqdm are present).
#
# ============================================================================

set -e
set -o pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Defaults (matches the answers in the chat)
# ----------------------------------------------------------------------------
INPUT_GLOB_DEFAULT="/data/streaming/180s/all_180s_96w_done_part*.jsonl"
OUTPUT_BASE_DEFAULT="/data/streaming/180s_wds"
IMAGE_ROOT_DEFAULT="/ov21"   # source jsonl uses absolute paths, but s1 still
                             # requires a directory to exist for sanity check.

INPUT_GLOB="${INPUT_GLOB:-${INPUT_GLOB_DEFAULT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"
IMAGE_ROOT="${IMAGE_ROOT:-${IMAGE_ROOT_DEFAULT}}"

# Defaults from auto_pipe.sh's "Optional" section
SHARD_PREFIX="${SHARD_PREFIX:-pretrain}"
SAMPLE_CLASS_NAME="${SAMPLE_CLASS_NAME:-PackedCaptioningSample}"
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-10000}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-3000000000}"
DIRECT_WORKERS="${DIRECT_WORKERS:-32}"
LONGEST_EDGE="${LONGEST_EDGE:-3072}"
JPEG_QUALITY="${JPEG_QUALITY:-92}"
TIMESTAMP_DECIMAL="${TIMESTAMP_DECIMAL:-2}"

# Behavioural switches
FORCE_RUN=false
SHOW_STATUS=false
RESUME_DIRECT=false
SELECTED_PARTS=""   # comma-separated part stems, e.g. all_180s_96w_done_part00

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

  Convert /data/streaming/180s/all_180s_96w_done_part*.jsonl into WebDataset
  shards (no token-packing). Each WebDataset sample wraps exactly one
  original sample. timestamp_decimal=${TIMESTAMP_DECIMAL} is injected into
  every sample's raw payload.

Required env-overridable defaults:
  INPUT_GLOB        glob for input JSONL parts
                    (default: ${INPUT_GLOB_DEFAULT})
  OUTPUT_BASE       output base directory
                    (default: ${OUTPUT_BASE_DEFAULT})
  IMAGE_ROOT        image root (passed to s1_build_manifest)
                    (default: ${IMAGE_ROOT_DEFAULT})

Optional overrides (also env-overridable):
  --shard-prefix             default: ${SHARD_PREFIX}
  --sample-class             default: ${SAMPLE_CLASS_NAME}
  --max-samples-per-shard    default: ${MAX_SAMPLES_PER_SHARD}
  --max-shard-size           default: ${MAX_SHARD_SIZE}
  --direct-workers           default: ${DIRECT_WORKERS}
  --longest-edge             default: ${LONGEST_EDGE}
  --jpeg-quality             default: ${JPEG_QUALITY}
  --timestamp-decimal        default: ${TIMESTAMP_DECIMAL}
  --parts STEM[,STEM,...]    only process selected part stems (e.g.
                             all_180s_96w_done_part00,all_180s_96w_done_part01)
  --resume-direct            pass --resume to s4 (skip already-done shards)
  -f, --force                force re-run all parts and steps
  -s, --status               show per-part status and exit
  -h, --help                 show this help
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-glob)            INPUT_GLOB="$2"; shift 2 ;;
        --output)                OUTPUT_BASE="$2"; shift 2 ;;
        --image-root)            IMAGE_ROOT="$2"; shift 2 ;;
        --shard-prefix)          SHARD_PREFIX="$2"; shift 2 ;;
        --sample-class)          SAMPLE_CLASS_NAME="$2"; shift 2 ;;
        --max-samples-per-shard) MAX_SAMPLES_PER_SHARD="$2"; shift 2 ;;
        --max-shard-size)        MAX_SHARD_SIZE="$2"; shift 2 ;;
        --direct-workers)        DIRECT_WORKERS="$2"; shift 2 ;;
        --longest-edge)          LONGEST_EDGE="$2"; shift 2 ;;
        --jpeg-quality)          JPEG_QUALITY="$2"; shift 2 ;;
        --timestamp-decimal)     TIMESTAMP_DECIMAL="$2"; shift 2 ;;
        --parts)                 SELECTED_PARTS="$2"; shift 2 ;;
        --resume-direct)         RESUME_DIRECT=true; shift ;;
        -f|--force)              FORCE_RUN=true; shift ;;
        -s|--status)             SHOW_STATUS=true; shift ;;
        -h|--help)               usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"; }
log_warn()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2; }

log_step() {
    echo ""
    echo "----------------------------------------------------------------------------"
    echo "  [${CURRENT_PART}] Step $1: $2"
    echo "----------------------------------------------------------------------------"
}

is_in_selected() {
    local stem="$1"
    if [[ -z "${SELECTED_PARTS}" ]]; then
        return 0
    fi
    local item
    IFS=',' read -ra _arr <<< "${SELECTED_PARTS}"
    for item in "${_arr[@]}"; do
        if [[ "${item}" == "${stem}" ]]; then
            return 0
        fi
    done
    return 1
}

mark_done() { mkdir -p "$(dirname "$1")"; date '+%Y-%m-%d %H:%M:%S' > "$1"; }
is_done()   { [[ -f "$1" ]]; }

# ----------------------------------------------------------------------------
# Discover parts
# ----------------------------------------------------------------------------
shopt -s nullglob
INPUT_FILES=( ${INPUT_GLOB} )
shopt -u nullglob
if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    log_error "No input files matched glob: ${INPUT_GLOB}"
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

# ----------------------------------------------------------------------------
# Status mode
# ----------------------------------------------------------------------------
show_status_for_part() {
    local stem="$1"
    local part_base="${OUTPUT_BASE}/${stem}"
    local status_dir="${part_base}/.status"
    echo "  Part: ${stem}"
    for step in 1 2 3; do
        local sf="${status_dir}/step${step}.done"
        if [[ -f "${sf}" ]]; then
            echo "    [DONE] step${step}: $(head -1 "${sf}")"
        else
            echo "    [TODO] step${step}"
        fi
    done
    if [[ -d "${part_base}/webdataset" ]]; then
        local n
        n=$(ls "${part_base}/webdataset"/*.tar 2>/dev/null | wc -l)
        echo "    Tars produced: ${n}"
    fi
}

if [[ "${SHOW_STATUS}" == "true" ]]; then
    echo "============================================================================"
    echo "  No-Packing Pipeline Status"
    echo "============================================================================"
    echo "  Output base: ${OUTPUT_BASE}"
    echo "  Input glob:  ${INPUT_GLOB}"
    echo ""
    for f in "${INPUT_FILES[@]}"; do
        stem="$(basename "${f}" .jsonl)"
        show_status_for_part "${stem}"
        echo ""
    done
    echo "============================================================================"
    exit 0
fi

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
echo "============================================================================"
echo "  Offline No-Packing Pipeline (raw JSONL -> WebDataset)"
echo "============================================================================"
log_info "Input glob:        ${INPUT_GLOB}"
log_info "Discovered parts:  ${#INPUT_FILES[@]}"
log_info "Output base:       ${OUTPUT_BASE}"
log_info "Image root:        ${IMAGE_ROOT}"
log_info "shard-prefix:      ${SHARD_PREFIX}"
log_info "sample-class:      ${SAMPLE_CLASS_NAME}"
log_info "timestamp_decimal: ${TIMESTAMP_DECIMAL}"
log_info "longest-edge:      ${LONGEST_EDGE}"
log_info "jpeg-quality:      ${JPEG_QUALITY}"
log_info "direct-workers:    ${DIRECT_WORKERS}"
[[ -n "${SELECTED_PARTS}" ]] && log_info "selected parts:    ${SELECTED_PARTS}"
echo ""

# ----------------------------------------------------------------------------
# Per-part loop
# ----------------------------------------------------------------------------
for INPUT_JSONL in "${INPUT_FILES[@]}"; do
    PART_STEM="$(basename "${INPUT_JSONL}" .jsonl)"

    if ! is_in_selected "${PART_STEM}"; then
        log_info "Skipping (not in --parts): ${PART_STEM}"
        continue
    fi

    PART_BASE="${OUTPUT_BASE}/${PART_STEM}"
    STATUS_DIR="${PART_BASE}/.status"
    MANIFEST_FILE="${PART_BASE}/manifest.jsonl"
    BINS_FILE="${PART_BASE}/bins.pkl"
    WDS_DIR="${PART_BASE}/webdataset"
    CURRENT_PART="${PART_STEM}"

    mkdir -p "${PART_BASE}"
    mkdir -p "${STATUS_DIR}"

    if [[ "${FORCE_RUN}" == "true" ]]; then
        log_warn "[${PART_STEM}] Force mode: clearing status."
        rm -f "${STATUS_DIR}"/step*.done
    fi

    echo ""
    echo "============================================================================"
    echo "  Part: ${PART_STEM}"
    echo "  Input:  ${INPUT_JSONL}"
    echo "  Output: ${PART_BASE}"
    echo "============================================================================"

    # ----- Step 1: build manifest --------------------------------------------
    SF1="${STATUS_DIR}/step1.done"
    if is_done "${SF1}" && [[ "${FORCE_RUN}" != "true" ]]; then
        log_info "[${PART_STEM}] Skip step 1 (manifest already built)"
    else
        log_step 1 "Build manifest.jsonl"
        python "${SCRIPT_DIR}/s1_build_manifest.py" \
            -i "${INPUT_JSONL}" \
            --image-root "${IMAGE_ROOT}" \
            -o "${MANIFEST_FILE}" \
            --no-shuffle \
            --overwrite
        mark_done "${SF1}"
        log_info "[${PART_STEM}] Step 1 done -> ${MANIFEST_FILE}"
    fi

    # ----- Step 2: trivial bins (+ inject timestamp_decimal) -----------------
    SF2="${STATUS_DIR}/step2.done"
    if is_done "${SF2}" && [[ "${FORCE_RUN}" != "true" ]]; then
        log_info "[${PART_STEM}] Skip step 2 (bins already built)"
    else
        log_step 2 "Build trivial bins.pkl (1 sample per bin) and inject timestamp_decimal=${TIMESTAMP_DECIMAL}"
        python "${SCRIPT_DIR}/s_make_trivial_bins.py" \
            --manifest "${MANIFEST_FILE}" \
            --bins-out "${BINS_FILE}" \
            --inject-timestamp-decimal "${TIMESTAMP_DECIMAL}" \
            --rewrite-manifest
        mark_done "${SF2}"
        log_info "[${PART_STEM}] Step 2 done -> ${BINS_FILE}"
    fi

    # ----- Step 3: bins -> WebDataset ----------------------------------------
    SF3="${STATUS_DIR}/step3.done"
    if is_done "${SF3}" && [[ "${FORCE_RUN}" != "true" ]]; then
        log_info "[${PART_STEM}] Skip step 3 (webdataset already built)"
    else
        log_step 3 "Bins -> WebDataset"
        DIRECT_CMD=(python "${SCRIPT_DIR}/s4_bins_to_webdataset.py"
            --bins-file "${BINS_FILE}"
            --manifest "${MANIFEST_FILE}"
            --output-dir "${WDS_DIR}"
            --workers "${DIRECT_WORKERS}"
            --max-samples-per-shard "${MAX_SAMPLES_PER_SHARD}"
            --max-shard-size "${MAX_SHARD_SIZE}"
            --shard-prefix "${SHARD_PREFIX}"
            --sample-class-name "${SAMPLE_CLASS_NAME}"
            --longest-edge "${LONGEST_EDGE}"
            --jpeg-quality "${JPEG_QUALITY}")

        if [[ "${RESUME_DIRECT}" == "true" ]]; then
            DIRECT_CMD+=(--resume)
        fi

        "${DIRECT_CMD[@]}"
        mark_done "${SF3}"
        log_info "[${PART_STEM}] Step 3 done -> ${WDS_DIR}"
    fi

    log_info "[${PART_STEM}] All steps complete."
done

echo ""
echo "============================================================================"
echo "  Pipeline complete."
echo "============================================================================"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Run with -s to inspect per-part status."
echo "============================================================================"
