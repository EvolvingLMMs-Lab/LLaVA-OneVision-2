# Make adjustments according to the actual data.
OUT_WDS_DIR='/vlm/data/offline_paclking_datasets/bmr_sft_780k-8k'
IN_SAMPLE_DIR='/workspace/data4packing/RiceVL/data_procs/raw_packing_data_mr_sft_780k-8k-fast'
PY_EXE="/workspace/AIAK-Training-LLM/tools/data_preprocess/convert_packedsample_to_wds.py"

python -u ${PY_EXE}  --output_dir ${OUT_WDS_DIR} --json_file ${IN_SAMPLE_DIR} --video_dir ${IN_SAMPLE_DIR}   --image_dir ${IN_SAMPLE_DIR}  --mode bmr_pack   --maxcount 5000 2>&1 | tee ./logs/s3_proc_mr_sft_780k-8k.log