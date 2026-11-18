#!/bin/bash
# 100k-sample comparative smoke test: token-only BFD vs patch-aware BFD.
# Runs auto_pipe.sh twice on the same 100k input and prints ViT-load comparison.
set -euo pipefail
export PYTHONUNBUFFERED=1
# Optional: CONDA_ENV=<name> activates that conda env first (needs transformers / webdataset).
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

SRC=/data/jsonls/sft_part1.jsonl
BASE=/data/smoke_test
INPUT=${BASE}/input_100k.jsonl
PIPE=/workspace/LLaVA-OneVision-2.5/offline_packing
MODEL=/models/lmms-lab/LLaVA-OneVision-2-4B-p16m33

[[ -f $INPUT ]] || head -n 100000 "$SRC" > "$INPUT"
echo "[100k] input lines: $(wc -l <"$INPUT")"

run_pipe () {
  local tag=$1; shift
  local out=${BASE}/packed_100k_L32k_${tag}
  echo "[100k] === ${tag} === -> ${out}"
  bash "${PIPE}/auto_pipe.sh" \
      -i "${INPUT}" -r / -m "${MODEL}" -o "${out}" -L 32000 \
      -w 16 --direct-workers 16 \
      --shard-prefix "smoke100k_${tag}" \
      --sample-class PackedCaptioningSample \
      --max-samples-per-shard 5000 \
      --longest-edge 3072 --jpeg-quality 92 \
      --no-npy "$@" 2>&1 | tail -200
}

run_pipe baseline
run_pipe patch15 --patch-slack 0.15

python - <<PY
import pickle, numpy as np, glob, os
def stat(tag, pkl):
    bins = pickle.load(open(pkl, "rb"))
    toks = np.array([sum(int(it[1]) for it in b) for b in bins])
    pats = np.array([sum(int(it[3]) if len(it) > 3 else 0 for it in b) for b in bins])
    cv = pats.std() / max(pats.mean(), 1)
    print(f"\n[{tag}] {pkl}")
    print(f"  bins={len(bins)}  samples={sum(len(b) for b in bins)}")
    print(f"  token util mean={toks.mean()/32000*100:.2f}%  p10={np.percentile(toks,10)/32000*100:.2f}%")
    print(f"  patches mean={pats.mean():.0f}  std={pats.std():.0f}  CV={cv:.3f}")
    print(f"          min={pats.min():,}  max={pats.max():,}  "
          f"p50={np.percentile(pats,50):.0f}  p99={np.percentile(pats,99):.0f}  "
          f"p99/p50={np.percentile(pats,99)/max(np.percentile(pats,50),1):.2f}x")
for tag in ("baseline","patch15"):
    pkl = sorted(glob.glob(f"${BASE}/packed_100k_L32k_{tag}/bins_*.pkl"))[-1]
    stat(tag, pkl)
PY

echo "[100k] DONE"
