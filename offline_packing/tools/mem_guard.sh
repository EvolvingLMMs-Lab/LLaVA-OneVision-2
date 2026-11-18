#!/bin/bash
# Memory guard: kill watched process group if MEM% > THRESHOLD for 3 consecutive checks
#
# Usage: bash mem_guard.sh <part_idx> [threshold_pct] [pattern] [done_file] [logdir]
#
# Defaults preserve the original 47M behaviour:
#   pattern   = "run_47m_part.sh <idx>"
#   done_file = /ov21/47m_sft_packed/47m_part<idx>/token_info.txt
#   logdir    = /ov21/47m_sft_packed/47m_part<idx>/
#
# Caller may override pattern/done_file/logdir for other datasets.
set -uo pipefail
IDX=${1:?"usage: $0 <part_idx> [threshold] [pattern] [done_file] [logdir]"}
TH=${2:-92}
PATTERN=${3:-"run_47m_part.sh ${IDX}"}
DONE_FILE=${4:-/ov21/47m_sft_packed/47m_part${IDX}/token_info.txt}
LOGDIR=${5:-/ov21/47m_sft_packed/47m_part${IDX}}
mkdir -p "$LOGDIR" 2>/dev/null || true
LOG=${LOGDIR}/mem_guard.log
log(){ echo "[memguard $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
log "started part=$IDX threshold=${TH}% pattern='${PATTERN}' done_file=${DONE_FILE}"
hi=0
while true; do
  if [[ -f "${DONE_FILE}" ]]; then
    log "${DONE_FILE} exists, watched job finished. exiting."
    exit 0
  fi
  PGID=$(ps -eo pgid,cmd | grep -F -- "${PATTERN}" | grep -v grep | awk '{print $1}' | head -1)
  if [[ -z "$PGID" ]]; then
    log "no '${PATTERN}' process found; exiting"
    exit 0
  fi
  # Use MemAvailable (kernel's honest "free + reclaimable") instead of MemUsed.
  # Why: workers fork-share large parent pages → RSS double-counts shared
  # memory. `free` can show used=98% when real pressure is moderate.
  # MEM_USED here means (1 - MemAvailable/MemTotal), the true scarcity metric.
  MEM_AVAIL_PCT=$(awk '/MemAvailable:/{a=$2} /MemTotal:/{t=$2} END{printf("%d", a*100/t)}' /proc/meminfo)
  MEM=$((100 - MEM_AVAIL_PCT))
  if (( MEM > TH )); then
    hi=$((hi+1))
    log "WARN MEM_real_used=${MEM}% (avail=${MEM_AVAIL_PCT}%) >${TH}% (consec=${hi}) PGID=${PGID}"
    if (( hi >= 3 )); then
      log "EMERGENCY MEM_real_used=${MEM}%; killing PGID=${PGID}"
      kill -KILL -- -${PGID} 2>/dev/null || true
      sleep 5
      log "killed; exiting (please relaunch with lower S2_MAX_PROCESSES)"
      exit 1
    fi
  else
    hi=0
  fi
  sleep 15
done
