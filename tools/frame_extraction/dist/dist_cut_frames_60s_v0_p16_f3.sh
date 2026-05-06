#!/usr/bin/env bash
# Distributed cut_frames launcher: 60s_v0 SFT, patch_size=16, factor_multiplier=3, max_pixels=432*432.
# Static mapping: part00..part19 -> hosts in /tmp/wpcplc10_hosts (line order).
# Stagger 5s per node to avoid NFS thundering herd.
# State persisted to _dispatch_state/ for monitor/stop/scheduler to consume.
set -euo pipefail

JSONL_ROOT="/ov2/dataset_jsonl/2000w_frames_v2/60s/parts_p16_f3"
OUTPUT_FRAMES_ROOT="/ov2/dataset_sft_video_frames/60s_v0_max_60_max_432x432_p16_f3"
OUTPUT_JSONL_ROOT="/ov2/dataset_jsonl/2000w_frames_v2/60s"
STATE_DIR="${OUTPUT_JSONL_ROOT}/_dispatch_state"
SCRIPT="/ov2/xiangan/LLaVA-OneVision-2/tools/frame_extraction/core/run_cut_frames.py"
PYTHON="/ov2/miniconda3/envs/ov2_dataset/bin/python3.12"
HOSTS_FILE="/tmp/wpcplc10_hosts"
JOB_TAG="cut_frames_60s_v0_p16_f3"
MIN_FREE_GB=200
STAGGER_SECONDS=5

mapfile -t HOSTS < "${HOSTS_FILE}"
if [[ "${#HOSTS[@]}" -ne 20 ]]; then
  echo "[FATAL] Expected 20 hosts in ${HOSTS_FILE}, got ${#HOSTS[@]}" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"
: > "${STATE_DIR}/parts.assigned"
: > "${STATE_DIR}/parts.failed"
touch "${STATE_DIR}/parts.done"

# Persist config so other tools can read it.
cat > "${STATE_DIR}/config.env" <<EOF
JSONL_ROOT="${JSONL_ROOT}"
OUTPUT_FRAMES_ROOT="${OUTPUT_FRAMES_ROOT}"
OUTPUT_JSONL_ROOT="${OUTPUT_JSONL_ROOT}"
STATE_DIR="${STATE_DIR}"
SCRIPT="${SCRIPT}"
PYTHON="${PYTHON}"
HOSTS_FILE="${HOSTS_FILE}"
JOB_TAG="${JOB_TAG}"
EOF

# Generate the actual python command for a part. Used by both launcher and scheduler.
cat > "${STATE_DIR}/run_part.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.env"
part="$1"
input_jsonl="${JSONL_ROOT}/60s_v0_slim_${part}.jsonl"
output_jsonl="${OUTPUT_JSONL_ROOT}/60s_v0_slim_done_${part}.jsonl"
log="/tmp/${JOB_TAG}_${part}.log"
mkdir -p "${OUTPUT_JSONL_ROOT}" "${OUTPUT_FRAMES_ROOT}" /ov2/tmp_ffmpeg_frames
exec setsid "${PYTHON}" "${SCRIPT}" \
  --input-jsonl       "${input_jsonl}" \
  --output-jsonl      "${output_jsonl}" \
  --output-dir        "${OUTPUT_FRAMES_ROOT}" \
  --sample-fps        1 \
  --max-frames        60 \
  --max-pixels        186624 \
  --patch-size        16 \
  --factor-multiplier 3 \
  --num-workers       32 \
  --strip-prefix      /ov2/dataset_mid_source_split/ \
  > "${log}" 2>&1 < /dev/null &
INNER
chmod +x "${STATE_DIR}/run_part.sh"

echo "[INFO] Preflight checks on all 20 nodes..."
preflight_failed=0
for i in "${!HOSTS[@]}"; do
  ip="${HOSTS[$i]}"
  part=$(printf "part%02d" "$i")
  input_jsonl="${JSONL_ROOT}/60s_v0_slim_${part}.jsonl"

  result=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR "${ip}" "
    set -e
    [[ -r '${input_jsonl}' ]] || { echo MISSING_INPUT; exit 0; }
    [[ \$(wc -l < '${input_jsonl}') -gt 0 ]] || { echo EMPTY_INPUT; exit 0; }
    command -v ffmpeg >/dev/null || { echo NO_FFMPEG; exit 0; }
    [[ -x '${PYTHON}' ]] || { echo NO_PYTHON; exit 0; }
    '${PYTHON}' -c 'import cv2' 2>/dev/null || { echo NO_CV2; exit 0; }
    [[ -w /ov2/tmp_ffmpeg_frames ]] || mkdir -p /ov2/tmp_ffmpeg_frames || { echo NO_TMPDIR; exit 0; }
    pgrep -f '[r]un_cut_frames.py.*60s_v0_slim_${part}\.jsonl' >/dev/null && { echo ALREADY_RUNNING; exit 0; }
    free_gb=\$(df -BG --output=avail '${OUTPUT_FRAMES_ROOT%/*}' 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ \${free_gb:-0} -ge ${MIN_FREE_GB} ]] || { echo \"LOW_DISK_\${free_gb}G\"; exit 0; }
    echo OK
  " 2>&1 | tail -1)

  if [[ "${result}" != "OK" ]]; then
    echo "  [FAIL] ${ip} (${part}): ${result}"
    preflight_failed=1
  else
    echo "  [OK]   ${ip} (${part})"
  fi
done

if [[ ${preflight_failed} -ne 0 ]]; then
  echo "[FATAL] Preflight failed. Fix issues above and re-run." >&2
  exit 1
fi

echo "[INFO] Preflight passed. Launching with ${STAGGER_SECONDS}s stagger..."
for i in "${!HOSTS[@]}"; do
  ip="${HOSTS[$i]}"
  part=$(printf "part%02d" "$i")
  log="/tmp/${JOB_TAG}_${part}.log"

  scp -q -o StrictHostKeyChecking=no -o LogLevel=ERROR "${STATE_DIR}/run_part.sh" "${STATE_DIR}/config.env" "${ip}:/tmp/"
  pid=$(ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR -n "${ip}" "
    bash /tmp/run_part.sh '${part}' >/dev/null 2>&1
    sleep 1
    pgrep -f '[r]un_cut_frames.py.*60s_v0_slim_${part}\.jsonl' | head -1
  ")

  if [[ -z "${pid}" ]]; then
    echo "  [FAIL] ${ip} ${part} did not start"
    echo "${part} ${ip} - $(date +%s) FAILED_TO_START 0" >> "${STATE_DIR}/parts.assigned"
  else
    echo "  [OK]   ${ip} ${part} pid=${pid} log=${log}"
    echo "${part} ${ip} ${pid} $(date +%s) RUNNING 0" >> "${STATE_DIR}/parts.assigned"
  fi

  if (( i < ${#HOSTS[@]} - 1 )); then
    sleep ${STAGGER_SECONDS}
  fi
done

echo ""
echo "[DONE] All 20 jobs launched. State dir: ${STATE_DIR}"
echo "  Monitor:   bash $(dirname "$0")/dist_cut_frames_60s_v0_p16_f3_monitor.sh"
echo "  Scheduler: setsid bash $(dirname "$0")/dist_cut_frames_60s_v0_p16_f3_scheduler.sh > /tmp/${JOB_TAG}_scheduler.log 2>&1 < /dev/null & disown"
echo "  Stop:      bash $(dirname "$0")/dist_cut_frames_60s_v0_p16_f3_stop.sh"
