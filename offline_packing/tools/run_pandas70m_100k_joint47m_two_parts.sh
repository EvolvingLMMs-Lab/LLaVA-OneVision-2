#!/bin/bash
set -uo pipefail
A=${1:?need part_a}
B=${2:?need part_b}
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
LOG=/ov21/pandas70m_100k_joint47m_packed_64k/host_$(hostname).log
mkdir -p /ov21/pandas70m_100k_joint47m_packed_64k
echo "[$(date)] starting joint part_a=${A} on $(hostname)" | tee -a "$LOG"
bash ${PIPE}/tools/run_pandas70m_100k_joint47m_part.sh ${A} 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[$(date)] joint part_a=${A} exited rc=${RC}" | tee -a "$LOG"
echo "[$(date)] starting joint part_b=${B} on $(hostname)" | tee -a "$LOG"
bash ${PIPE}/tools/run_pandas70m_100k_joint47m_part.sh ${B} 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[$(date)] joint part_b=${B} exited rc=${RC}" | tee -a "$LOG"
echo "[$(date)] BOTH JOINT DONE on $(hostname)" | tee -a "$LOG"
