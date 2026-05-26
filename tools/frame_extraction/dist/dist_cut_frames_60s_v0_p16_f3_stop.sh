#!/usr/bin/env bash
# Stop: SIGTERM then SIGKILL all per-part workers across nodes; first kill local scheduler.
#
# Required env vars:
#   STATE_DIR  - dispatch state dir produced by the launcher (must contain config.env)
set -euo pipefail

STATE_DIR="${STATE_DIR:?set STATE_DIR to the dispatch state dir from the launcher}"
source "${STATE_DIR}/config.env"

GRACE="${GRACE:-30}"

if pgrep -f "[d]ist_cut_frames_60s_v0_p16_f3_scheduler.sh" >/dev/null; then
  echo "[INFO] Killing local scheduler first (so it can't respawn workers)..."
  pkill -f "[d]ist_cut_frames_60s_v0_p16_f3_scheduler.sh" || true
  sleep 2
fi

echo "[INFO] Sending SIGTERM to all workers..."
while read -r part ip pid ts state retry; do
  [[ -z "${part:-}" ]] && continue
  [[ "${state}" == "DONE" || "${state}" == "FAILED" ]] && continue
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o LogLevel=ERROR -n "${ip}" \
    "pkill -TERM -f '[r]un_cut_frames.py.*60s_v0_slim_${part}\\.jsonl' && echo '  [TERM] ${ip} ${part}' || echo '  [NONE] ${ip} ${part} (no proc)'" \
    || echo "  [UNREACHABLE] ${ip} ${part}"
done < "${STATE_DIR}/parts.assigned"

echo "[INFO] Waiting ${GRACE}s for graceful exit..."
sleep "${GRACE}"

echo "[INFO] Force killing any survivors with SIGKILL..."
while read -r part ip pid ts state retry; do
  [[ -z "${part:-}" ]] && continue
  [[ "${state}" == "DONE" || "${state}" == "FAILED" ]] && continue
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o LogLevel=ERROR -n "${ip}" \
    "pgrep -f '[r]un_cut_frames.py.*60s_v0_slim_${part}\\.jsonl' >/dev/null && pkill -KILL -f '[r]un_cut_frames.py.*60s_v0_slim_${part}\\.jsonl' && echo '  [KILL] ${ip} ${part}' || echo '  [GONE] ${ip} ${part}'" \
    || echo "  [UNREACHABLE] ${ip} ${part}"
done < "${STATE_DIR}/parts.assigned"

echo "[DONE] All workers stopped."
