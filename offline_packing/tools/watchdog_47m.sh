#!/bin/bash
# Watchdog: when s2 (token_info.txt) completes but s3 hasn't started with the
# desired --patch-slack value, kill the running auto_pipe and re-launch s3+s4
# with --patch-slack=$DESIRED_SLACK.
#
# Idempotent: writes a marker to avoid re-triggering.
#
# Usage:  bash watchdog_47m.sh <part_idx> <desired_slack>
#   e.g. bash watchdog_47m.sh 1 0.30

set -uo pipefail

# Memory hygiene (mostly affects s4; s2 is already done by the time we run).
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

IDX=${1:?"usage: $0 <part_idx> <desired_slack>"}
SLACK=${2:-0.30}
OUT=/ov21/47m_sft_packed/47m_part${IDX}
SRC=/ov2/chengzheng/ov2_wds/47m_sft_cleaned_64k_20w/jsonls/47m_${IDX}.jsonl
MODEL=/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
WD_LOG=${OUT}/watchdog.log
WD_MARK=${OUT}/watchdog.triggered

log() { echo "[watchdog $(date '+%H:%M:%S')] $*" | tee -a "$WD_LOG"; }

log "started for part${IDX}, target slack=${SLACK}"

while true; do
  # Already done? Exit.
  if [[ -f "${OUT}/DONE.txt" ]]; then
    log "DONE.txt present, exiting"
    exit 0
  fi

  # Already intervened? Just wait for the re-run to finish.
  if [[ -f "$WD_MARK" ]]; then
    sleep 60
    continue
  fi

  # token_info.txt exists AND has non-zero size AND s3 not yet started?
  if [[ -s "${OUT}/token_info.txt" ]]; then
    # Check whether s3 already wrote bins; if it has, we're too late on this run
    # (but the new launch will overwrite, so still proceed).
    s2_done_line=$(grep 'Step 2 marked as complete' "${OUT}/run.log" 2>/dev/null | head -1)
    if [[ -z "$s2_done_line" ]]; then
      # token_info.txt exists but s2-mark not yet logged; wait a moment.
      sleep 5
      continue
    fi

    log "s2 complete (token_info.txt ready). Intercepting..."

    # 1) Find auto_pipe.sh pid for this run and kill the whole group.
    PIDS=$(pgrep -f "auto_pipe.sh.*47m_part${IDX}" || true)
    log "auto_pipe pids: ${PIDS:-none}"
    for pid in $PIDS; do
      log "kill TERM $pid (and children)"
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    # Hard kill anything still around.
    for pid in $PIDS; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    pkill -KILL -f "s3_bin_packing.py.*47m_part${IDX}" 2>/dev/null || true
    pkill -KILL -f "s4_bins_to_webdataset.py.*47m_part${IDX}" 2>/dev/null || true

    # 2) Wipe step3/step4 status markers so auto_pipe re-runs them.
    rm -f "${OUT}/.status/step3_"* "${OUT}/.status/step4_"* "${OUT}/bins_32klen.pkl"
    log "cleared step3/step4 markers; bins_32klen.pkl removed"

    touch "$WD_MARK"

    # 3) Re-launch full pipeline; s1/s2 will be skipped via existing markers.
    log "re-launching auto_pipe.sh with --patch-slack ${SLACK}"
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate llava_main_xy
    nohup bash "${PIPE}/auto_pipe.sh" \
        -i "${SRC}" -r / -m "${MODEL}" -o "${OUT}" -L 32000 \
        -w 32 --direct-workers 32 \
        --shard-prefix "47m_part${IDX}" \
        --sample-class PackedCaptioningSample \
        --max-samples-per-shard 5000 \
        --longest-edge 3072 --jpeg-quality 92 \
        --no-npy \
        --patch-slack "${SLACK}" \
        >> "${OUT}/run.log" 2>&1 &
    RELAUNCH_PID=$!
    log "re-launched pid=${RELAUNCH_PID}; waiting for completion..."

    wait "${RELAUNCH_PID}" 2>/dev/null
    if [[ -f "${OUT}/DONE.txt" ]]; then
      log "DONE marker present after relaunch (created by run_47m_part.sh). Done."
    else
      # run_47m_part.sh wraps auto_pipe.sh and touches DONE.txt. When we
      # re-launch auto_pipe.sh directly we have to touch it ourselves.
      if grep -q 'Pipeline Complete' "${OUT}/run.log" 2>/dev/null; then
        touch "${OUT}/DONE.txt"
        log "Pipeline Complete detected; DONE.txt touched"
      else
        log "auto_pipe exited but no Pipeline Complete marker. Inspect run.log."
      fi
    fi
    exit 0
  fi

  sleep 60
done
