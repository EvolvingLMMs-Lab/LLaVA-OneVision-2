# AMD Terminal Midtraining Handoff

This branch contains the MobileLLM-R1 + FastViT/FastVLM integration and the
scripts needed to start Stage 1.5. Keep model checkpoints outside Git and copy
them between machines directly.

## 1. Get the Code

```bash
git clone git@github.com:RanaZay/LLaVA-OneVision-1.5.git
cd LLaVA-OneVision-1.5
git checkout mobile-llm-integration
git pull
git lfs pull
```

Copy the Stage 1 alignment checkpoint from the GPU machine to the AMD machine.
From the GPU machine, run:

```bash
rsync -avP \
  stage_1_alignment_mobilellm_140m_fastvlm_faithful/latest_checkpointed_iteration.txt \
  stage_1_alignment_mobilellm_140m_fastvlm_faithful/iter_0002500 \
  user@AMD_HOST:/path/to/LLaVA-OneVision-1.5/stage_1_alignment_mobilellm_140m_fastvlm_faithful/
```

On the AMD machine, verify the Stage 1 checkpoint is present:

```bash
cat stage_1_alignment_mobilellm_140m_fastvlm_faithful/latest_checkpointed_iteration.txt
ls -lh stage_1_alignment_mobilellm_140m_fastvlm_faithful/iter_0002500/mp_rank_00/model_optim_rng.pt
```

Expected iteration: `2500`.

## 2. Start a Terminal Session

Use `tmux` so training survives SSH disconnects:

```bash
tmux new -s mobile_vlm_midtrain
```

Detach without stopping training:

```text
Ctrl-b then d
```

Reattach:

```bash
tmux attach -t mobile_vlm_midtrain
```

## 3. Set Environment

Adjust paths for the AMD machine:

```bash
export REPO_ROOT="$PWD"
export PYTHON_BIN=/path/to/your/env/bin/python
export TORCHRUN=/path/to/your/env/bin/torchrun

export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="llava-ov-1_5"
```

If the AMD machine has a different Conda env, replace `PYTHON_BIN` and
`TORCHRUN` accordingly.

For the MBZUAI AMD cluster used here, the Stage 1.5 ImageNet job can also be
submitted with the ROCm Slurm launcher:

```bash
cd ~/mobile_vlm/llava_ov1.5/LLaVA-OneVision-1.5

# Optional: export WANDB_API_KEY before submitting if the env is not already logged in.
sbatch examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_imagenet_en_amd.sbatch
```

This launcher mirrors the ROCm setup from `Stage1/alignment_rocm.sh`, uses
`conda activate mobile_vlm`, requests 8 AMD GPUs, and reads local ImageNet data
from `data/LLaVA-OneVision-Mid-Training-EN`.

## 4. Run Full English Branch Midtraining

This prepares the full HF English branch into local Energon/WebDataset shards,
then runs 1000 Stage 1.5 iterations. `MAX_SAMPLES=0` means no artificial cap.

ImageNet EN:

```bash
LOCAL_HF_DATA_ROOT="$PWD/data/LLaVA-OneVision-Mid-Training-EN" \
MAX_SAMPLES=0 \
NSTEP=1000 \
GBS=4 \
MIN_FREE_GB=100 \
bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_english_branch_chain.sh imagenet
```

Multiple English branches, chained one after another:

```bash
LOCAL_HF_DATA_ROOT="$PWD/data/LLaVA-OneVision-Mid-Training-EN" \
MAX_SAMPLES=0 \
NSTEP=1000 \
GBS=4 \
MIN_FREE_GB=100 \
bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_english_branch_chain.sh imagenet datacomp1b
```

The second branch automatically loads from the checkpoint produced by the first.

To verify local ImageNet files before training:

```bash
python tools/check_hf_local_parquet_complete.py \
  --local-data-root data/LLaVA-OneVision-Mid-Training-EN \
  --data-files 'imagenet/EN/*/*.parquet'
```

## 5. Useful Options

```bash
START_CKPT=/path/to/checkpoint_dir
DATA_ROOT=/large/disk/midtraining_full_en
CACHE_DIR=/large/disk/hf_cache_midtraining_stream
LOCAL_HF_DATA_ROOT=/large/disk/LLaVA-OneVision-Mid-Training-EN
KEEP_PREPARED_DATA=1
PREPARE_ONLY=1
MIDTRAIN_TRAINABLE_MODULES="language_model adapter vision_model"
```

For a quick smoke test only:

```bash
MAX_SAMPLES=4000 NSTEP=1000 bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_mobilellm_fastvit_english_branch_chain.sh imagenet
```

For real full-data preparation, keep `MAX_SAMPLES=0`.
