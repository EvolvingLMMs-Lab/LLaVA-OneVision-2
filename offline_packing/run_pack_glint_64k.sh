#!/bin/bash
# Glint video-MC-QA pack -> 64k webdataset.  (variant of run_pack_master_47m.sh)
# ONLY difference vs the pandas driver: s3 --max-samples is configurable (MAXSAMP, default 4)
# because glint videos are SHORT (~12-16k tok each) -> pack MULTIPLE videos/bin, then light 47m topup fills the small remaining slack.
#   usage: run_pack_glint_64k.sh <input_master_jsonl> <output_base> [config_yaml] [shard_idx shard_cnt]
# Chain: s0(dynamic) -> s1 -> s2 -> s3(MAXSAMP videos/bin) -> s3b(47m random topup, light) -> combine -> s4(container,B) -> .info.yaml
# Env: PY, PIPE (default: this script's dir), MODEL (REQUIRED), IMG, POOL_TOKENINFO_GLOB (REQUIRED),
#      POOL_MANIFEST_GLOB (REQUIRED), EXTRA_MOUNTS, BLOCKLIST, MAXSAMP, TOPUP_PERBIN, S2_WORKERS/S4_WORKERS/S4_CPUS.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="${PIPE:-$SCRIPT_DIR}"
PY="${PY:-python}"
IN=${1:?input jsonl}; BASE=${2:?output base}; CFG=${3:-$PIPE/dynamic_budget_default.yaml}
SIDX=${4:-0}; SCNT=${5:-1}   # 47m pool disjoint shard (for distributed dedup); default no shard
MODEL="${MODEL:?set MODEL to the tokenizer/model dir (must match the training model)}"
POOL_TOKENINFO_GLOB="${POOL_TOKENINFO_GLOB:?set to the 47m pool token_info glob, e.g. /path/to/47m_packed/*/token_info.txt}"
POOL_MANIFEST_GLOB="${POOL_MANIFEST_GLOB:?set to the 47m pool manifest glob, e.g. /path/to/47m_packed/*/manifest.jsonl}"
IMG=${IMG:-llava_megatron:26.05}
_root() { printf ' -v /%s:/%s' "$(echo "$1"|cut -d/ -f2)" "$(echo "$1"|cut -d/ -f2)"; }
MOUNTS="$(_root "$BASE")${EXTRA_MOUNTS:-}"
export PYTHONUNBUFFERED=1
S2W=${S2_WORKERS:-12}; S4W=${S4_WORKERS:-8}; S4CPUS=${S4_CPUS:-8}   # conservative on borrowed training hosts
CAP=${CAP:-64000}; MAXP=$(( 608822 * CAP / 64000 ))   # pack bin length (64k default)
MAXSAMP=${MAXSAMP:-4}          # glint: MULTIPLE short videos per bin (was 1 for long pandas)
TOPUP_PERBIN=${TOPUP_PERBIN:-200}   # 47m per-bin cap; naturally light since bins already near-full
mkdir -p "$BASE"; cp "$CFG" "$BASE/dynamic_budget.yaml"
LOG=$BASE/_pack.log; exec > >(tee -a "$LOG") 2>&1
echo "==== glint pack: MAXSAMP=$MAXSAMP CAP=$CAP  config -> $BASE/dynamic_budget.yaml  $(date) ===="

echo "==== [s0] dynamic resample $(date) ===="
$PY $PIPE/s0_resample_from_master.py --input-jsonl "$IN" --output-jsonl "$BASE/s0_expanded.jsonl" --config "$BASE/dynamic_budget.yaml"

echo "==== [s1] manifest $(date) ===="
$PY $PIPE/s1_build_manifest.py -i "$BASE/s0_expanded.jsonl" --image-root / -o "$BASE/manifest.jsonl" --overwrite

echo "==== [s2] token lengths $(date) ===="
$PY $PIPE/s2_compute_token_lengths.py --manifest "$BASE/manifest.jsonl" --output "$BASE/token_info.txt" \
    --model-path "$MODEL" --max-len $CAP --task-type sft --chunk-size 500 --min-workers 2 --max-workers $S2W --timeout 180

echo "==== [s3] bin packing ($MAXSAMP videos/bin, BFD, leave slack for light 47m) $(date) ===="
$PY $PIPE/s3_bin_packing.py --input "$BASE/token_info.txt" --output "$BASE/bins_64klen.pkl" \
    --capacity $CAP --max-samples $MAXSAMP --algorithm bfd --max-patches-per-bin 0 --min-patches-per-bin 0 --patch-slack 0

echo "==== [s3b] 47m random topup (shard $SIDX/$SCNT) $(date) ===="
if [ -n "${BLOCKLIST:-}" ] && [ -f "${BLOCKLIST}" ]; then cp "$BLOCKLIST" "$BASE/blocklist.txt"; echo "blocklist: $(wc -l < "$BASE/blocklist.txt") names from $BLOCKLIST"; else : > "$BASE/blocklist.txt"; fi
$PY $PIPE/s3b_topup_with_47m.py --bins-in "$BASE/bins_64klen.pkl" --bins-out "$BASE/bins_topup.pkl" \
    --pool-tokeninfo-glob "$POOL_TOKENINFO_GLOB" --pool-blocklist "$BASE/blocklist.txt" \
    --selected-out "$BASE/selected_47m.txt" --image-only --capacity $CAP --max-samples $TOPUP_PERBIN --max-patches $MAXP \
    --topup-budget-samples 2000000 --selection-mode random --pool-shard-index "$SIDX" --pool-shard-count "$SCNT" --seed 12345

echo "==== [combine] video manifest + selected 47m $(date) ===="
$PY $PIPE/tools/build_combined_manifest.py --fix-manifest "$BASE/manifest.jsonl" --selected "$BASE/selected_47m.txt" \
    --pool-manifest-glob "$POOL_MANIFEST_GLOB" --output "$BASE/manifest_combined.jsonl"

echo "==== [s4] bins -> webdataset (container, B downscale + energon .info.yaml) $(date) ===="
docker run --rm --cpus="$S4CPUS" $MOUNTS \
    -w "$PIPE" --entrypoint bash "$IMG" -c "
        set -euo pipefail; export PYTHONUNBUFFERED=1 PYTHONPATH=$PIPE:\${PYTHONPATH:-}
        nice -n 19 python s4_bins_to_webdataset.py --bins-file '$BASE/bins_topup.pkl' --manifest '$BASE/manifest_combined.jsonl' \
            --output-dir '$BASE/webdataset' --workers $S4W --max-samples-per-shard 500 --shard-prefix pack \
            --sample-class-name PackedCaptioningSample --longest-edge 3072 --jpeg-quality 92
    "

W=$BASE/webdataset
echo "==== VERIFY $(date) ===="
echo "tar=$(ls $W/*.tar 2>/dev/null|wc -l) idx=$(ls $W/*.tar.idx 2>/dev/null|wc -l) info.yaml=$([ -f $W/.nv-meta/.info.yaml ] && echo YES || echo NO)"
echo "PACKGLINT_DONE"
