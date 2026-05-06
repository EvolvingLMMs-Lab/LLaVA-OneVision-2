#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="/ov2/dataset_jsonl/2000w_frames_v2/30s/_dispatch_state"
source "${STATE_DIR}/config.env"

POLL_INTERVAL=60
MAX_RETRIES=3
RESTART_GRACE=10

echo "[scheduler] Started pid=$$ at $(date)"
echo "[scheduler] Polling every ${POLL_INTERVAL}s, max ${MAX_RETRIES} retries per part"

while true; do
  tmp=$(mktemp -p "${STATE_DIR}" parts.assigned.XXXXXX)
  any_active=0

  while read -r part ip pid ts state retry; do
    [[ -z "${part:-}" ]] && continue

    if [[ "${state}" == "DONE" || "${state}" == "FAILED" ]]; then
      echo "${part} ${ip} ${pid} ${ts} ${state} ${retry}" >> "${tmp}"
      continue
    fi
    any_active=1

    input_jsonl="${JSONL_ROOT}/30s_v0_slim_${part}.jsonl"
    output_jsonl="${OUTPUT_JSONL_ROOT}/30s_v0_slim_done_${part}.jsonl"
    total=$(wc -l < "${input_jsonl}" 2>/dev/null || echo 0)

    probe=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o LogLevel=ERROR -n "${ip}" \
      "wc -l < '${output_jsonl}' 2>/dev/null || echo 0; pgrep -f '[r]un_cut_frames.py.*${part}\\.jsonl' >/dev/null && echo ALIVE || echo DEAD" 2>/dev/null \
      || printf '0\nUNREACHABLE\n')
    done_n=$(echo "$probe" | sed -n 1p)
    alive=$(echo "$probe" | sed -n 2p)
    done_n=${done_n:-0}

    if [[ ${total} -gt 0 && ${done_n} -ge ${total} ]]; then
      echo "[scheduler] $(date '+%H:%M:%S') ${part}@${ip} DONE (${done_n}/${total})"
      echo "${part} ${ip} ${pid} ${ts} DONE ${retry}" >> "${tmp}"
      grep -qxF "${part}" "${STATE_DIR}/parts.done" 2>/dev/null || echo "${part}" >> "${STATE_DIR}/parts.done"
      continue
    fi

    if [[ "${alive}" == "ALIVE" ]]; then
      echo "${part} ${ip} ${pid} ${ts} RUNNING ${retry}" >> "${tmp}"
      continue
    fi

    if [[ "${alive}" == "UNREACHABLE" ]]; then
      echo "[scheduler] $(date '+%H:%M:%S') ${part}@${ip} UNREACHABLE, will retry next poll"
      echo "${part} ${ip} ${pid} ${ts} ${state} ${retry}" >> "${tmp}"
      continue
    fi

    if [[ ${retry} -ge ${MAX_RETRIES} ]]; then
      echo "[scheduler] $(date '+%H:%M:%S') ${part}@${ip} FAILED after ${retry} retries (${done_n}/${total})"
      echo "${part} ${ip} ${pid} ${ts} FAILED ${retry}" >> "${tmp}"
      grep -qxF "${part}" "${STATE_DIR}/parts.failed" 2>/dev/null || echo "${part} ${ip} ${done_n}/${total}" >> "${STATE_DIR}/parts.failed"
      continue
    fi

    new_retry=$((retry + 1))
    echo "[scheduler] $(date '+%H:%M:%S') ${part}@${ip} DEAD (${done_n}/${total}), restart ${new_retry}/${MAX_RETRIES}"
    new_pid=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR -n "${ip}" "
      [[ -f /tmp/run_part.sh ]] || exit 1
      bash /tmp/run_part.sh '${part}' >/dev/null 2>&1
      sleep ${RESTART_GRACE}
      pgrep -f '[r]un_cut_frames.py.*${part}\.jsonl' | head -1
    " 2>/dev/null || echo "")

    if [[ -z "${new_pid}" ]]; then
      echo "[scheduler]   restart FAILED to spawn PID"
      echo "${part} ${ip} ${pid} ${ts} DEAD ${new_retry}" >> "${tmp}"
    else
      echo "[scheduler]   restarted with pid=${new_pid}"
      echo "${part} ${ip} ${new_pid} $(date +%s) RUNNING ${new_retry}" >> "${tmp}"
    fi
  done < "${STATE_DIR}/parts.assigned"

  mv "${tmp}" "${STATE_DIR}/parts.assigned"

  if [[ ${any_active} -eq 0 ]]; then
    echo "[scheduler] $(date) all parts terminal (DONE/FAILED), exiting"
    break
  fi

  sleep "${POLL_INTERVAL}"
done
