#!/usr/bin/env bash
# Single-node cut_frames launcher for the ov2_quickstart subset:
#   - 50,000 videos sliced from the 30s_v0 SFT JSONL
#   - 8 frames per video (1 fps -> capped at 8 by --max-frames)
#   - max 350x350 pixels (122_500), aligned to factor = patch_size * factor_multiplier = 14 * 2 = 28
#     -> actual max resolution becomes 336x336
#   - 32 parallel workers
#
# All paths are env-var overridable. The defaults assume the standard
# ov2_quickstart layout under ${QS_ROOT} and a system python3 with cv2/ffmpeg
# available. Adjust per your environment:
#
#   QS_ROOT=/data/ov2_quickstart \
#   INPUT_JSONL=$QS_ROOT/30s_v0_head50k.jsonl \
#   OUTPUT_JSONL=$QS_ROOT/30s_v0_8frames_350_5w.jsonl \
#   OUTPUT_DIR=$QS_ROOT/frames \
#   bash cut_frames_quickstart_8f_350.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

QS_ROOT="${QS_ROOT:-./ov2_quickstart}"
INPUT_JSONL="${INPUT_JSONL:-${QS_ROOT}/30s_v0_head50k.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${QS_ROOT}/30s_v0_8frames_350_5w.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${QS_ROOT}/frames}"
SCRIPT="${SCRIPT:-${HERE}/../core/run_cut_frames.py}"
PYTHON="${PYTHON:-python3}"
LOG="${LOG:-${QS_ROOT}/logs/cut_frames_quickstart_8f_350.log}"
TMP_FFMPEG_DIR="${TMP_FFMPEG_DIR:-/tmp/ov2_ffmpeg_frames}"
STRIP_PREFIX="${STRIP_PREFIX:-}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG}")" "${TMP_FFMPEG_DIR}"

if [[ ! -r "${INPUT_JSONL}" ]]; then
  echo "[FATAL] Missing input JSONL: ${INPUT_JSONL}" >&2
  exit 1
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "[FATAL] Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -r "${SCRIPT}" ]]; then
  echo "[FATAL] Missing script: ${SCRIPT}" >&2
  exit 1
fi

echo "[INFO] Input:       ${INPUT_JSONL} ($(wc -l < "${INPUT_JSONL}") lines)"
echo "[INFO] Output JSONL: ${OUTPUT_JSONL}"
echo "[INFO] Frames dir:  ${OUTPUT_DIR}"
echo "[INFO] Log:         ${LOG}"
echo "[INFO] Launching $(date)"

STRIP_ARGS=()
if [[ -n "${STRIP_PREFIX}" ]]; then
  STRIP_ARGS=(--strip-prefix "${STRIP_PREFIX}")
fi

exec "${PYTHON}" "${SCRIPT}" \
  --input-jsonl       "${INPUT_JSONL}" \
  --output-jsonl      "${OUTPUT_JSONL}" \
  --output-dir        "${OUTPUT_DIR}" \
  --max-frames        8 \
  --sample-fps        1 \
  --max-pixels        122500 \
  --patch-size        14 \
  --factor-multiplier 2 \
  --num-workers       32 \
  "${STRIP_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"
