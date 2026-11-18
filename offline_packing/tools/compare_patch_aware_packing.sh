#!/bin/bash
# Compare token-only BFD vs patch-aware BFD on the same input.
# Runs auto_pipe.sh twice:
#   A. baseline       — no patch constraint
#   B. patch-aware    — --patch-slack 0.15  (cap ≈ avg_patches × 1.15)
# Then prints a side-by-side ViT-load comparison from the produced bins_*.pkl.
#
# Usage:
#   bash compare_patch_aware_packing.sh \
#       -i /path/input.jsonl -r /image/root -m /model/path \
#       -o /out/base -L 32000
#
# The Python environment must contain transformers / webdataset / sortedcontainers.

set -euo pipefail
export PYTHONUNBUFFERED=1

INPUT=""; IMG_ROOT="/"; MODEL=""; OUT=""; LEN=32000
SLACK=0.15
PIPE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) INPUT="$2"; shift 2;;
    -r) IMG_ROOT="$2"; shift 2;;
    -m) MODEL="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    -L) LEN="$2"; shift 2;;
    --slack) SLACK="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done
[[ -z "$INPUT" || -z "$MODEL" || -z "$OUT" ]] && { echo "need -i -m -o"; exit 2; }

OUT_A="${OUT}_baseline"
OUT_B="${OUT}_patch${SLACK//./}"

echo "[cmp] A. baseline -> ${OUT_A}"
bash "${PIPE_DIR}/auto_pipe.sh" \
  -i "${INPUT}" -r "${IMG_ROOT}" -m "${MODEL}" -o "${OUT_A}" -L "${LEN}" \
  --shard-prefix cmpA --no-npy

echo "[cmp] B. patch-aware (slack=${SLACK}) -> ${OUT_B}"
bash "${PIPE_DIR}/auto_pipe.sh" \
  -i "${INPUT}" -r "${IMG_ROOT}" -m "${MODEL}" -o "${OUT_B}" -L "${LEN}" \
  --shard-prefix cmpB --no-npy --patch-slack "${SLACK}"

python - <<PY
import pickle, numpy as np, glob, os
def stat(pkl):
    bins = pickle.load(open(pkl,"rb"))
    toks = np.array([sum(int(it[1]) for it in b) for b in bins])
    pats = np.array([sum(int(it[3]) if len(it)>3 else 0 for it in b) for b in bins])
    print(f"  bins={len(bins)}")
    print(f"  token mean={toks.mean():.0f}  p10={np.percentile(toks,10):.0f}  max={toks.max()}")
    print(f"  patches mean={pats.mean():.0f}  std={pats.std():.0f}  CV={pats.std()/max(pats.mean(),1):.3f}")
    print(f"          min={pats.min()}  max={pats.max()}  p50={np.percentile(pats,50):.0f}  "
          f"p99={np.percentile(pats,99):.0f}  p99/p50={np.percentile(pats,99)/max(np.percentile(pats,50),1):.2f}x")
for tag, base in [("A baseline","${OUT_A}"), ("B patch${SLACK}","${OUT_B}")]:
    pkl = sorted(glob.glob(os.path.join(base,"bins_*.pkl")))[-1]
    print(f"[{tag}] {pkl}")
    stat(pkl)
PY
