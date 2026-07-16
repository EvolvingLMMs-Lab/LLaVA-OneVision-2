#!/bin/bash
# Joint packing: 1 pandas70m_100k part (heavy video) + 1/6 chunk of 1 47m part
# (light SFT) — co-packed into 64k bins to maximize fill from 62.8% → ~98%.
#
# Plan:
#  * IDX = 0..11
#  * p70m source: /ov21/pandas70m_gt10min_100k_packed_64k/p70m100k_part${IDX}/
#                 (already has manifest.jsonl + token_info.txt from earlier run)
#  * 47m source : 1/6 chunk of /ov21/47m_sft_packed_64k/47m_part${M47_PART}/
#                 where M47_PART = (IDX // 6) + 1   → 47m_part1 (IDX 0-5) or 47m_part2 (6-11)
#                 chunk index    = IDX % 6  (every 6th line, offset by chunk index)
#  * Each part gets ~335k 47m samples, fits perfectly into the 11k p70m bins
#    without spawning extra 47m-only bins. Efficiency: ~98% (vs 62.8% p70m-alone).
#  * Total 47m consumed across 12 parts: 12 × 335k = 4.02M (≈28% of 14M pool;
#    47m_part3..7 are untouched and continue to be served by /ov21/47m_sft_packed_64k/)
#
# We:
#  1. Write joint token_info (namespaced: p70m_... / m47_...)
#  2. Write joint manifest.jsonl (renames "name" field to namespaced version too)
#  3. Run s3 BFD with capacity 64000 over the union (patch-slack=0; 47m has 0 patches)
#  4. Run s4 (in docker) → webdataset
#
# Output: /ov21/pandas70m_100k_joint47m_packed_64k/joint_part${IDX}/

set -euo pipefail
export PYTHONUNBUFFERED=1

IDX=${1:?"usage: $0 <part_idx 0..11>"}
PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
IMG=${IMG:-llava_megatron:25.12}
MAXLEN=64000

M47_PART=$(( IDX / 6 + 1 ))     # 1 for IDX 0-5, 2 for IDX 6-11
M47_CHUNK=$(( IDX % 6 ))        # 0..5

P70M_DIR=/ov21/pandas70m_gt10min_100k_packed_64k/p70m100k_part${IDX}
M47_DIR=/ov21/47m_sft_packed_64k/47m_part${M47_PART}
OUT=/ov21/pandas70m_100k_joint47m_packed_64k/joint_part${IDX}

[[ -f "${P70M_DIR}/token_info.txt" ]] || { echo "MISSING p70m token_info: ${P70M_DIR}"; exit 2; }
[[ -f "${M47_DIR}/token_info.txt"  ]] || { echo "MISSING 47m token_info:  ${M47_DIR}";  exit 2; }
mkdir -p "$OUT"
LOG=${OUT}/run.log
echo "[joint-${IDX}] host=$(hostname) p70m=part${IDX} m47=part${M47_PART}/chunk${M47_CHUNK}/6 out=${OUT}" | tee "$LOG"

# Memory guard (joint pack is light: just s3 + s4; no s2 here, threshold is high)
nohup bash ${PIPE}/tools/mem_guard.sh ${IDX} 92 \
    "run_pandas70m_100k_joint47m_part.sh ${IDX}" \
    "${OUT}/bins_64klen.pkl" \
    "${OUT}" \
    > ${OUT}/mem_guard.log 2>&1 &
GUARD_PID=$!
trap "kill ${GUARD_PID} 2>/dev/null || true" EXIT

# --- glibc / BLAS hygiene ---
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llava_main_xy

STATUS=${OUT}/.status; mkdir -p "$STATUS"

JOINT_TOK=${OUT}/token_info.txt
JOINT_MAN=${OUT}/manifest.jsonl
BINS=${OUT}/bins_64klen.pkl
WDS=${OUT}/webdataset

# ---------- Step J1: build joint token_info ----------
if [[ ! -f ${STATUS}/joint1.done ]]; then
    echo "[joint-${IDX}] J1: write joint token_info" | tee -a "$LOG"
    python - <<PY 2>&1 | tee -a "$LOG"
import hashlib
out = "${JOINT_TOK}"
n_p70m = 0
with open(out, "w") as fout:
    with open("${P70M_DIR}/token_info.txt") as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            parts = line.split(":")
            parts[0] = "p70m_" + parts[0]
            fout.write(":".join(parts) + "\n")
            n_p70m += 1
    n_m47 = 0
    # Chunk 47m by stable hash of name (NOT line index!) so that token_info and
    # manifest yield the same set of samples regardless of file ordering.
    with open("${M47_DIR}/token_info.txt") as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            parts = line.split(":")
            name = parts[0]
            h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            if h % 6 != ${M47_CHUNK}: continue
            parts[0] = "m47_" + name
            fout.write(":".join(parts) + "\n")
            n_m47 += 1
print(f"p70m samples: {n_p70m}")
print(f"47m  samples: {n_m47}")
print(f"joint total : {n_p70m + n_m47}")
PY
    touch ${STATUS}/joint1.done
fi

# ---------- Step J2: build joint manifest (with namespaced name) ----------
if [[ ! -f ${STATUS}/joint2.done ]]; then
    echo "[joint-${IDX}] J2: write joint manifest" | tee -a "$LOG"
    python - <<PY 2>&1 | tee -a "$LOG"
import json, hashlib
out = "${JOINT_MAN}"
n_p70m = 0
n_m47 = 0
with open(out, "w") as fout:
    with open("${P70M_DIR}/manifest.jsonl") as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            obj["name"] = "p70m_" + obj["name"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_p70m += 1
    # Same hash-based chunking as J1 so the two files stay aligned by name set.
    with open("${M47_DIR}/manifest.jsonl") as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            name = obj["name"]
            h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            if h % 6 != ${M47_CHUNK}: continue
            obj["name"] = "m47_" + name
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_m47 += 1
print(f"p70m  manifest lines: {n_p70m}")
print(f"47m   manifest lines: {n_m47}")
print(f"joint manifest lines: {n_p70m + n_m47}")
PY
    touch ${STATUS}/joint2.done
fi

# ---------- Step J3: bin packing (no patch-slack; 47m has 0 patches → would skew avg) ----------
if [[ ! -f ${STATUS}/joint3.done ]]; then
    echo "[joint-${IDX}] J3: bin packing capacity=${MAXLEN}" | tee -a "$LOG"
    python ${PIPE}/s3_bin_packing.py \
        --input "$JOINT_TOK" --output "$BINS" \
        --capacity ${MAXLEN} --max-samples 200 \
        --algorithm bfd --patch-slack 0 2>&1 | tee -a "$LOG"
    touch ${STATUS}/joint3.done
fi

# ---------- Step J4: webdataset (docker for energon/.idx) ----------
mkdir -p "$WDS"
[[ ! -f ${STATUS}/joint4.done ]] && {
    echo "[joint-${IDX}] J4: bins -> webdataset (docker)" | tee -a "$LOG"
    docker run --rm \
        --name "ov2_joint_part${IDX}_$$" \
        -v /vlm:/vlm -v /ov2:/ov2 -v /ov21:/ov21 -v /rice_vl:/rice_vl \
        -w "${PIPE}" --entrypoint bash \
        "${IMG}" -c "
            set -euo pipefail
            export PYTHONUNBUFFERED=1
            export PYTHONPATH=${PIPE}:\${PYTHONPATH:-}
            python s4_bins_to_webdataset.py \
                --bins-file '${BINS}' \
                --manifest  '${JOINT_MAN}' \
                --output-dir '${WDS}' \
                --workers 16 \
                --max-samples-per-shard 500 \
                --shard-prefix 'p70m100k_joint47m_part${IDX}_64k' \
                --sample-class-name PackedCaptioningSample \
                --longest-edge 3072 --jpeg-quality 92
        " 2>&1 | tee -a "$LOG"
    touch ${STATUS}/joint4.done
}

NTAR=$(ls "$WDS"/*.tar 2>/dev/null | wc -l)
NIDX=$(ls "$WDS"/*.tar.idx 2>/dev/null | wc -l)
SZ=$(du -sh "$WDS" 2>/dev/null | awk '{print $1}')
echo "[joint-${IDX}] tars=${NTAR} idx=${NIDX} size=${SZ}" | tee -a "$LOG"

if (( NTAR > 0 )) && (( NIDX == NTAR )); then
    touch "${OUT}/DONE.txt"
    echo "[joint-${IDX}] DONE at $(date)" | tee -a "$LOG"
else
    echo "[joint-${IDX}] INCOMPLETE" | tee -a "$LOG"
    exit 4
fi
