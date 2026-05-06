#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="/ov2/dataset_jsonl/2000w_frames_v2/60s/_dispatch_state"
source "${STATE_DIR}/config.env"

GRACE=30

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
