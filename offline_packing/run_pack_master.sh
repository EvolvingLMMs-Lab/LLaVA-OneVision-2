#!/bin/bash
# Master-based dynamic-budget packing driver.
#   usage: run_pack_master.sh <input_unified_jsonl> <output_base> [config_yaml]
# Steps: copy config -> s0 (dynamic resample from master) -> auto_pipe s1-s4 (B downscale)
#        -> energon prepare (container: .idx + .info.yaml). NO transient frames (B).
# Env: PY, PIPE (default: this script's dir), MODEL (REQUIRED), IMG, EXTRA_MOUNTS.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="${PIPE:-$SCRIPT_DIR}"
PY="${PY:-python}"
IN=${1:?"input unified jsonl"}
BASE=${2:?"output base dir"}
CFG=${3:-$PIPE/dynamic_budget_default.yaml}
MODEL="${MODEL:?set MODEL to the tokenizer/model dir (must match the training model)}"
IMG=${IMG:-llava_megatron:26.05}
_root() { printf ' -v /%s:/%s' "$(echo "$1"|cut -d/ -f2)" "$(echo "$1"|cut -d/ -f2)"; }
MOUNTS="$(_root "$BASE")${EXTRA_MOUNTS:-}"
mkdir -p "$BASE"
LOG=$BASE/_pack.log
exec > >(tee -a "$LOG") 2>&1

echo "==== copy config -> target (provenance) $(date) ===="
cp "$CFG" "$BASE/dynamic_budget.yaml"
echo "config: $CFG -> $BASE/dynamic_budget.yaml"

EXP=$BASE/s0_expanded.jsonl
echo "==== [s0] dynamic resample from master $(date) ===="
$PY $PIPE/s0_resample_from_master.py --input-jsonl "$IN" --output-jsonl "$EXP" --config "$BASE/dynamic_budget.yaml"

echo "==== [s1-s4] auto_pipe L=64000 $(date) ===="
export MODEL PYTHONUNBUFFERED=1
bash $PIPE/auto_pipe.sh -i "$EXP" -r / -o "$BASE" -L 64000 --shard-prefix pack

echo "==== [energon prepare] container -> .idx + .info.yaml $(date) ===="
docker run --rm -i $MOUNTS --entrypoint bash "$IMG" -c "cd '$BASE/webdataset' && printf 'y\n' | energon prepare --num-workers 12 --split-parts 'train:pack-{000000..999999}.tar' . >/dev/null 2>&1"

W=$BASE/webdataset
echo "==== VERIFY $(date) ===="
echo "tar=$(ls $W/*.tar 2>/dev/null|wc -l) idx=$(ls $W/*.tar.idx 2>/dev/null|wc -l) info.yaml=$([ -f $W/.nv-meta/.info.yaml ] && echo YES || echo NO) config=$([ -f $BASE/dynamic_budget.yaml ] && echo YES)"
echo "PACK_MASTER_DONE"
