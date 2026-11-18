#!/usr/bin/env bash
# Generic launcher for date05* OV2 training scripts.
#
# Example:
#   bash examples/llava_onevision2/quick_start_4b/launch_date05_training.sh \
#     --script examples/llava_onevision2/quick_start_4b/date0520_ax_stage_2_alignment_p16m3_4n_muon_from_adapter_lr2e5.sh \
#     --ips "172.16.5.20 172.16.5.22 172.16.5.26 172.16.5.31" \
#     --gbs 128 --master-port 26012 --prewarm

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vlm/yinxie/code/OV2/OV2_public_main}"
CONTAINER="${CONTAINER:-llava_megatron_container_ax}"
SCRIPT=""
IPS=""
TP=1
PP=1
SEQ_LEN=32000
MBS=1
GBS=""
NSTEP=""
MASTER_PORT="26020"
LAUNCH_LOG_NAME=""
DRY_RUN=0
PREWARM=0
CHECK_IDLE=1
TOKENIZER_PATH="/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"
EXTRA_ENVS=()
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

usage() {
  cat <<'USAGE'
Usage:
  launch_date05_training.sh --script <path> --ips "ip1 ip2 ..." --gbs <int> [options]

Required:
  --script PATH          Training script to run. Relative paths are resolved from repo root.
  --ips "IP IP ..."      Space/comma separated target node list, in rank order.
  --gbs INT              Global batch size passed as positional arg 5.

Common options:
  --tp INT               Tensor parallel size. Default: 1
  --pp INT               Pipeline parallel size. Default: 1
  --seq-len INT          Sequence length. Default: 32000
  --mbs INT              Micro batch size. Default: 1
  --nstep INT            Optional train iters passed as positional arg 6.
  --master-port INT      MASTER_PORT override. Default: 26020
  --container NAME       Docker container name. Default: llava_megatron_container_ax
  --repo-root PATH       Repo root visible on all nodes. Default: /vlm/yinxie/code/OV2/OV2_public_main
  --env KEY=VALUE        Extra env override. Repeatable, e.g. --env OUTPUT_DIR=/path
  --prewarm              Prewarm HF AutoConfig/AutoTokenizer once per node before launch.
  --tokenizer-path PATH  Tokenizer/model path for --prewarm.
  --skip-idle-check      Do not check for existing torchrun/train.py processes.
  --launch-log-name STR  Remote /tmp launch log suffix. Default: script basename.
  --dry-run              Print commands without executing.
  -h, --help             Show this help.

Notes:
  - The launcher sets LIST_IP to the provided IP list, so date05 scripts do not need editing.
  - It passes positional args as: TP PP SEQ_LEN MBS GBS [NSTEP].
  - It sets MASTER_PORT, REPO_ROOT, and any --env KEY=VALUE overrides inside docker exec.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

normalize_ips() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $raw | awk 'NF && !seen[$0]++ {print}' | paste -sd' ' -
}

quote_for_bash() {
  printf '%q' "$1"
}

run_or_print() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script) SCRIPT="${2:-}"; shift 2 ;;
    --ips) IPS="${2:-}"; shift 2 ;;
    --gbs) GBS="${2:-}"; shift 2 ;;
    --tp) TP="${2:-}"; shift 2 ;;
    --pp) PP="${2:-}"; shift 2 ;;
    --seq-len) SEQ_LEN="${2:-}"; shift 2 ;;
    --mbs) MBS="${2:-}"; shift 2 ;;
    --nstep) NSTEP="${2:-}"; shift 2 ;;
    --master-port) MASTER_PORT="${2:-}"; shift 2 ;;
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:-}"; shift 2 ;;
    --env) EXTRA_ENVS+=("${2:-}"); shift 2 ;;
    --prewarm) PREWARM=1; shift ;;
    --tokenizer-path) TOKENIZER_PATH="${2:-}"; shift 2 ;;
    --skip-idle-check) CHECK_IDLE=0; shift ;;
    --launch-log-name) LAUNCH_LOG_NAME="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$SCRIPT" ]] || die "--script is required"
[[ -n "$IPS" ]] || die "--ips is required"
[[ -n "$GBS" ]] || die "--gbs is required"

IPS="$(normalize_ips "$IPS")"
[[ -n "$IPS" ]] || die "empty --ips after normalization"

if [[ "$SCRIPT" != /* ]]; then
  SCRIPT="$REPO_ROOT/$SCRIPT"
fi
[[ -f "$SCRIPT" ]] || die "script not found: $SCRIPT"

SCRIPT_BASENAME="$(basename "$SCRIPT" .sh)"
LAUNCH_LOG_NAME="${LAUNCH_LOG_NAME:-$SCRIPT_BASENAME}"
REMOTE_LOG="/tmp/${LAUNCH_LOG_NAME}_launch.log"

for value in "$TP" "$PP" "$SEQ_LEN" "$MBS" "$GBS" "$MASTER_PORT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "numeric argument expected, got: $value"
done
if [[ -n "$NSTEP" ]]; then
  [[ "$NSTEP" =~ ^[0-9]+$ ]] || die "--nstep must be numeric"
fi

for env_kv in "${EXTRA_ENVS[@]}"; do
  [[ "$env_kv" == *=* ]] || die "--env must be KEY=VALUE, got: $env_kv"
done

read -r -a IP_ARRAY <<< "$IPS"
MASTER_ADDR="${IP_ARRAY[0]}"

echo "== OV2 date05 launcher =="
echo "script:       $SCRIPT"
echo "ips:          $IPS"
echo "master:       $MASTER_ADDR:$MASTER_PORT"
echo "container:    $CONTAINER"
echo "args:         TP=$TP PP=$PP SEQ_LEN=$SEQ_LEN MBS=$MBS GBS=$GBS NSTEP=${NSTEP:-auto}"
echo "launch log:   $REMOTE_LOG"
if [[ ${#EXTRA_ENVS[@]} -gt 0 ]]; then
  echo "extra envs:    ${EXTRA_ENVS[*]}"
fi

if [[ "$CHECK_IDLE" -eq 1 ]]; then
  echo "== Checking idle state =="
  for ip in "${IP_ARRAY[@]}"; do
    echo "-- $ip"
    idle_cmd='ps -ef | grep -v grep | grep -E "torchrun|aiak_training_llm/train.py" || true; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " "; echo'
    run_or_print ssh "${SSH_OPTS[@]}" "$ip" "docker exec $CONTAINER bash -lc $(quote_for_bash "$idle_cmd")"
  done
fi

if [[ "$PREWARM" -eq 1 ]]; then
  echo "== Prewarming HF tokenizer/config cache =="
  prewarm_cmd=$(cat <<PY
python - <<'INNERPY'
from transformers import AutoConfig, AutoTokenizer
p = "$TOKENIZER_PATH"
c = AutoConfig.from_pretrained(p, trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=False)
print(type(c).__name__, len(t))
INNERPY
PY
)
  for ip in "${IP_ARRAY[@]}"; do
    echo "-- $ip"
    run_or_print ssh "${SSH_OPTS[@]}" "$ip" "docker exec $CONTAINER bash -lc $(quote_for_bash "$prewarm_cmd")"
  done
fi

echo "== Launching =="
for ip in "${IP_ARRAY[@]}"; do
  env_prefix="LIST_IP=$(quote_for_bash "$IPS") MASTER_ADDR=$(quote_for_bash "$MASTER_ADDR") MASTER_PORT=$(quote_for_bash "$MASTER_PORT")"
  for env_kv in "${EXTRA_ENVS[@]}"; do
    key="${env_kv%%=*}"
    val="${env_kv#*=}"
    env_prefix+=" $key=$(quote_for_bash "$val")"
  done
  args="$TP $PP $SEQ_LEN $MBS $GBS"
  if [[ -n "$NSTEP" ]]; then
    args+=" $NSTEP"
  fi
  inner="cd $(quote_for_bash "$REPO_ROOT") && $env_prefix nohup bash $(quote_for_bash "$SCRIPT") $args > $(quote_for_bash "$REMOTE_LOG") 2>&1 &"
  echo "-- $ip"
  run_or_print ssh "${SSH_OPTS[@]}" "$ip" "docker exec $CONTAINER bash -lc $(quote_for_bash "$inner")"
done

echo "== Done issuing launch commands =="
echo "Monitor example:"
echo "  ssh ${IP_ARRAY[0]} \"docker exec $CONTAINER bash -lc 'tail -n 40 $REMOTE_LOG'\""
