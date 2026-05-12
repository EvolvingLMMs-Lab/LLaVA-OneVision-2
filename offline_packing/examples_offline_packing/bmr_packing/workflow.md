### Multi-Turn Conversations + Mixed Workflow
- Step-1: 
    - `python s1_bmr_sft_data_proc_indcoding.py 2>&1 tee ./logs/output.log`
    - `python s1_get_tokenlens_v4-sft.py --config ./configs/s1_config_BMR_sft_780k.yaml`
- Step-2: `s2_bmr_sft_packing.ipynb`
    - section `Step2-1`: bin-packing processing (You can set different bin-packing strategies and their parameters based on your understanding of the element distribution in the hash_buckets.).
    - section `hash_buckets‘s element distribution`: Visualize the element distribution in hash_buckets to help design bin-packing strategies.
    - section `Step2-2`: Convert pre-packing format to post-packing format.
- Step-3: `bash s3_test_bmr_sft_780k-8k.sh`
- Step-4: `export OFFLINE_PACKED_DATA='1' & export OFFLINE_PACKING_BMR='1'`(Settings in the training script remain unchanged).
