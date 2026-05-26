#!/usr/bin/env bash
# Stop: pkill -9 the per-part workers on each remote node.
#
# Required env vars:
#   STATE_DIR  - dispatch state dir produced by the launcher (must contain config.env)
#   SSHPASS    - SSH password for sshpass (do not commit)
set -euo pipefail

STATE_DIR="${STATE_DIR:?set STATE_DIR to the dispatch state dir from the launcher}"
source "${STATE_DIR}/config.env"
PASS="${SSHPASS:?set SSHPASS env (do not commit a literal password)}"
SSH="sshpass -p ${PASS} ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o LogLevel=ERROR"

mapfile -t HOSTS < "${HOSTS_FILE}"
for i in "${!HOSTS[@]}"; do
  ip="${HOSTS[$i]}"
  part=$(printf "part%02d" "$i")
  echo "[stop] killing ${part}@${ip}..."
  ${SSH} -n root@"${ip}" "pkill -9 -f '[r]un_cut_frames.py.*${part}\\.jsonl' || true"
done
echo "[stop] done"
