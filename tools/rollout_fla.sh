#!/usr/bin/env bash
# Roll out flash-linear-attention (required by the Qwen3.5 GatedDeltaNet
# linear-attention layers) into the training container on every node.
# Idempotent: skips nodes where fla already imports.
#
# Usage:
#   bash tools/rollout_fla.sh --ips "192.0.2.11 192.0.2.12 ..." \
#        [--container llava_megatron_container] \
#        [--proxy http://proxy.example.com:8888] [--version 0.5.1] [--dry-run]
set -uo pipefail

CONTAINER="${CONTAINER:-llava_megatron_container}"
IPS=""
PROXY="http://proxy.example.com:8888"
VERSION="0.5.1"
DRY=0

die(){ echo "ERROR: $*" >&2; exit 1; }
normalize_ips(){ echo "$1" | tr ,   | tr -s  ; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ips) IPS="${2:-}"; shift 2 ;;
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --proxy) PROXY="${2:-}"; shift 2 ;;
    --version) VERSION="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) grep "^#" "$0" | sed "s/^# \{0,1\}//"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done
[[ -n "$IPS" ]] || die "--ips is required (space/comma separated node list)"
IPS="$(normalize_ips "$IPS")"
read -r -a IP_ARRAY <<< "$IPS"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
PKG="flash-linear-attention==${VERSION}"

echo "container: $CONTAINER | proxy: $PROXY | pkg: $PKG"
echo "nodes:     ${IP_ARRAY[*]}"
echo "----------------------------------------"

FAIL=()
for ip in "${IP_ARRAY[@]}"; do
  printf "[%s] " "$ip"
  # idempotent install + verify inside the container
  inner="python -c \"import fla\" 2>/dev/null && echo ALREADY || { \
      http_proxy=${PROXY} https_proxy=${PROXY} pip install --no-cache-dir ${PKG} >/tmp/fla_install.log 2>&1 && \
      python -c \"import fla; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; from fla.modules.convolution import causal_conv1d; print('INSTALLED', fla.__version__)\" ; }"
  cmd=(ssh "${SSH_OPTS[@]}" "$ip" "docker exec $CONTAINER bash -lc $(printf %q "$inner")")
  if [[ "$DRY" == "1" ]]; then echo "DRY: ${cmd[*]}"; continue; fi
  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  # print last meaningful line
  echo "$out" | grep -aE "ALREADY|INSTALLED|Error|error|No module|Could not|Failed" | tail -1 || echo "(rc=$rc)"
  if [[ $rc -ne 0 ]] || ! echo "$out" | grep -qaE "ALREADY|INSTALLED"; then
    FAIL+=("$ip")
  fi
done

echo "----------------------------------------"
if [[ ${#FAIL[@]} -eq 0 ]]; then
  echo "OK: fla present on all ${#IP_ARRAY[@]} node(s)."
else
  echo "FAILED on: ${FAIL[*]}"; exit 1
fi
