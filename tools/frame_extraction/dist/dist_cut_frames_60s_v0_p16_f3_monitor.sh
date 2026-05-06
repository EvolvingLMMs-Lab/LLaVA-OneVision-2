#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="/ov2/dataset_jsonl/2000w_frames_v2/60s/_dispatch_state"
source "${STATE_DIR}/config.env"
REFRESH=10

trap 'tput cnorm; echo; exit 0' INT TERM
tput civis

declare -A prev_done prev_ts

fmt_eta() {
  local secs=$1
  if [[ ${secs} -le 0 ]]; then printf "--"; return; fi
  local d=$((secs / 86400))
  local h=$(( (secs % 86400) / 3600 ))
  local m=$(( (secs % 3600) / 60 ))
  if [[ ${d} -gt 0 ]]; then printf "%dd%02dh" "$d" "$h"
  elif [[ ${h} -gt 0 ]]; then printf "%dh%02dm" "$h" "$m"
  else printf "%dm" "$m"
  fi
}

while true; do
  clear
  now=$(date +%s)
  printf "=== %s monitor === %s (refresh %ds, ctrl+c to exit)\n\n" \
    "${JOB_TAG}" "$(date '+%Y-%m-%d %H:%M:%S')" "${REFRESH}"
  printf "%-7s %-15s %-9s %-13s %-6s %-9s %-8s %-5s %s\n" \
    PART IP STATE DONE/TOTAL PROG% RATE/s ETA RETRY LOG_TAIL
  printf -- '----------------------------------------------------------------------------------------------------------------------------\n'

  declare -A row_pid row_state row_retry
  while read -r part ip pid ts state retry; do
    [[ -z "${part:-}" ]] && continue
    row_pid["$part"]="$pid"
    row_state["$part"]="$state"
    row_retry["$part"]="$retry"
  done < "${STATE_DIR}/parts.assigned"

  total_done=0
  total_total=0
  total_rate_milli=0
  alive=0
  for i in $(seq 0 19); do
    part=$(printf "part%02d" "$i")
    state="${row_state[$part]:-?}"
    retry="${row_retry[$part]:-0}"
    ip=$(awk -v p="$part" '$1==p{print $2; exit}' "${STATE_DIR}/parts.assigned")

    input_jsonl="${JSONL_ROOT}/60s_v0_slim_${part}.jsonl"
    log="/tmp/${JOB_TAG}_${part}.log"

    total=$(wc -l < "${input_jsonl}" 2>/dev/null || echo 0)
    probe=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o LogLevel=ERROR -n "${ip}" \
      "grep -oE '抽帧进度: [0-9]+/[0-9]+' '${log}' 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+' | cut -d/ -f1 || echo 0; pgrep -f '[r]un_cut_frames.py.*60s_v0_slim_${part}\\.jsonl' >/dev/null && echo ALIVE || echo DEAD; tail -1 '${log}' 2>/dev/null | tr -d '\r' | cut -c1-80" 2>/dev/null || printf '%s\n' 0 UNREACHABLE -)
    done_n_val=$(echo "$probe" | sed -n 1p)
    [[ -z "${done_n_val}" ]] && done_n_val=0
    alive_flag=$(echo "$probe" | sed -n 2p)
    log_tail=$(echo "$probe" | sed -n 3p)

    [[ "$alive_flag" == "ALIVE" ]] && alive=$((alive+1))
    total=${total:-0}; done_n_val=${done_n_val:-0}
    total_done=$((total_done + done_n_val))
    total_total=$((total_total + total))
    pct=0
    [[ ${total} -gt 0 ]] && pct=$((100 * done_n_val / total))

    rate_str="--"
    eta_str="--"
    if [[ -n "${prev_done[$part]:-}" && -n "${prev_ts[$part]:-}" ]]; then
      dt=$((now - prev_ts[$part]))
      dn=$((done_n_val - prev_done[$part]))
      if [[ ${dt} -gt 0 && ${dn} -gt 0 ]]; then
        rate_milli=$((1000 * dn / dt))
        total_rate_milli=$((total_rate_milli + rate_milli))
        rate_str=$(printf "%d.%01d" $((rate_milli / 1000)) $(((rate_milli % 1000) / 100)))
        remaining=$((total - done_n_val))
        if [[ ${remaining} -gt 0 ]]; then
          eta_secs=$((remaining * 1000 / rate_milli))
          eta_str=$(fmt_eta "$eta_secs")
        else
          eta_str="done"
        fi
      fi
    fi
    prev_done[$part]=$done_n_val
    prev_ts[$part]=$now

    printf "%-7s %-15s %-9s %6d/%-6d %4d%% %-9s %-8s %-5s %s\n" \
      "$part" "${ip:-?}" "$alive_flag" "$done_n_val" "$total" "$pct" "$rate_str" "$eta_str" "$retry" "${log_tail:-}"
  done

  printf -- '----------------------------------------------------------------------------------------------------------------------------\n'
  pct_total=0
  [[ ${total_total} -gt 0 ]] && pct_total=$((100 * total_done / total_total))
  total_eta_str="--"
  total_rate_str="--"
  if [[ ${total_rate_milli} -gt 0 ]]; then
    total_rate_str=$(printf "%d.%01d" $((total_rate_milli / 1000)) $(((total_rate_milli % 1000) / 100)))
    remaining=$((total_total - total_done))
    if [[ ${remaining} -gt 0 ]]; then
      total_eta_secs=$((remaining * 1000 / total_rate_milli))
      total_eta_str=$(fmt_eta "$total_eta_secs")
    else
      total_eta_str="done"
    fi
  fi
  printf "TOTAL: %d/%d (%d%%)  RATE: %s vid/s  ETA: %s  ALIVE: %d/20  FAILED: %d\n" \
    "$total_done" "$total_total" "$pct_total" "$total_rate_str" "$total_eta_str" \
    "$alive" "$(wc -l < "${STATE_DIR}/parts.failed" 2>/dev/null || echo 0)"

  sleep "${REFRESH}"
done
