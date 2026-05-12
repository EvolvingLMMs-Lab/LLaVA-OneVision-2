# Make adjustments according to the actual data.
OUT_WDS_DIR='/vlm/data/offline_paclking_datasets/bmr_sft_780k-8k'
IN_SAMPLE_DIR='/workspace/data4packing/RiceVL/data_procs/raw_packing_data_mr_sft_780k-8k-fast'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_EXE="$(cd "${SCRIPT_DIR}/../.." && pwd)/s5_convert_to_webdataset.py"

mkdir -p ./logs

python -u "${PY_EXE}" \
  --input-dir "${IN_SAMPLE_DIR}" \
  --output-dir "${OUT_WDS_DIR}" \
  --mode bmr_pack \
  --max-samples-per-shard 5000 \
  2>&1 | tee ./logs/s3_proc_mr_sft_780k-8k.log
