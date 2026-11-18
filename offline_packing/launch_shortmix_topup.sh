#!/bin/bash
# Launch gt20min-768 shortmix topup across idle hosts. 10 parts.
set -uo pipefail
SDIR=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
OUT_BASE=/ov21/pandas70m_gt20min_768frames_packed_64k_shortmixtopup
LOG_DIR=${OUT_BASE}/logs; mkdir -p "${LOG_DIR}"
declare -A HOST_PARTS=( ["172.16.5.19"]="0 7" ["172.16.5.27"]="1 8" ["172.16.5.40"]="2 9" ["172.16.5.93"]="3" ["172.16.5.94"]="4" ["172.16.5.95"]="5" ["172.16.5.96"]="6" )
ACTION="${1:-launch}"
case "${ACTION}" in
  launch)
    for ip in "${!HOST_PARTS[@]}"; do for idx in ${HOST_PARTS[$ip]}; do
      pidx=$(printf '%02d' $idx)
      log="${LOG_DIR}/part${pidx}_${ip}.log"
      echo "[${ip}] launching topup idx=${idx}"
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${ip}" "nohup bash ${SDIR}/tools/run_shortmix_topup_part.sh ${idx} > '${log}' 2>&1 &" 
    done; done; echo; echo 'All launched.' ;;
  status)
    for ip in $(echo "${!HOST_PARTS[@]}" | tr ' ' '\n' | sort); do for idx in ${HOST_PARTS[$ip]}; do
      pidx=$(printf '%02d' $idx); sd="${OUT_BASE}/part${pidx}/.status"
      a=$([[ -f "${sd}/stepA.done" ]] && echo A || echo -); b=$([[ -f "${sd}/stepB.done" ]] && echo B || echo -); c=$([[ -f "${sd}/stepC.done" ]] && echo C || echo -)
      dn=$([[ -f "${OUT_BASE}/part${pidx}/DONE.txt" ]] && echo DONE || echo ...)
      nt=$(ls "${OUT_BASE}/part${pidx}/webdataset"/*.tar 2>/dev/null | wc -l)
      printf '  %-13s part%s [%s%s%s] %s tars=%s\n' "${ip}" "${pidx}" "$a" "$b" "$c" "$dn" "$nt"
    done; done ;;
esac
