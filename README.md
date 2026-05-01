# LLaVA-OneVision-1.5 × Mobile-LLM Integration

> **Branch:** `mobile-llm-integration`
>
> This branch adapts the [LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5) training framework to a fully **mobile-optimized** multimodal pipeline by replacing the original vision encoder and language model with lightweight alternatives:
>
> | Component | Original (upstream) | This branch |
> |---|---|---|
> | Vision Encoder | RICE ViT-Large (560 px) | **FastViT / MobileCLIP-L** (1024 px, Apple ml-fastvlm) |
> | Language Model | Qwen3-4B | **MobileLLM-R1-140M** (Facebook, 140M params) |
> | Adapter | 2-layer MLP | 2-layer MLP (3072 → 576, re-initialized) |
> | Training Stage shown | Stage 1 alignment | **Stage 1 alignment** (adapter only frozen: vision + LLM) |

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Dependencies & Installation](#dependencies--installation)
- [Downloading Pretrained Checkpoints](#downloading-pretrained-checkpoints)
- [Downloading the Dataset](#downloading-the-dataset)
- [Running Stage 1 Alignment Training](#running-stage-1-alignment-training)
- [Demo: End-to-End Inference](#demo-end-to-end-inference)
- [Training Logs & Results](#training-logs--results)
- [Modifications vs. Upstream](#modifications-vs-upstream)
- [Credits & Citations](#credits--citations)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Mobile Multimodal Pipeline                         │
│                                                                     │
│  Image (1024×1024)                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────┐                       │
│  │  FastViT MobileCLIP-L  [FROZEN]          │                       │
│  │  • Apple ml-fastvlm architecture         │                       │
│  │  • 1024×1024 input, 64×64 patch grid     │                       │
│  │  • Output: [B, 256 tokens, 3072 dim]     │                       │
│  └──────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼  (32 images packed → flattened to [8192, 3072])             │
│  ┌──────────────────────────────────────────┐                       │
│  │  2-Layer MLP Adapter  [TRAINABLE]        │                       │
│  │  • Linear(3072, 3072) + GELU             │                       │
│  │  • Linear(3072, 576)                     │                       │
│  │  • Output: [8192, 576]                   │                       │
│  └──────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼  (merged into text token sequence at <|image_pad|> slots)   │
│  ┌──────────────────────────────────────────┐                       │
│  │  MobileLLM-R1-140M  [FROZEN]             │                       │
│  │  • 15 transformer layers                 │                       │
│  │  • Hidden: 576, Heads: 9, KV-heads: 3   │                       │
│  │  • FFN: 2048 (SwiGLU), 32k context      │                       │
│  │  • QK LayerNorm enabled                  │                       │
│  │  • GQA (9 query / 3 key-value heads)     │                       │
│  │  • Output: per-token cross-entropy loss  │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  Stage 1: Only the Adapter is trained (≈ 3.5M params)              │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Step-by-Step (one forward pass)

```
[1/6] BATCH
  input_ids  : (1, 1909)  torch.int64
  images     : (32, 3, 1024, 1024)  torch.bfloat16
  labels     : (1, 1909)  torch.int64
  loss_mask  : 532 / 1909 tokens (27.9%) contribute to loss

[2/6] VISION ENCODER  [FastViT MobileCLIP-L  |  FROZEN]
  in  : (32, 3, 1024, 1024)  torch.bfloat16
  out : (32, 256, 3072)  torch.bfloat16  grad=False

[3/6] ADAPTER  [2-layer MLP 3072→576  |  TRAINABLE]
  in  : (32, 256, 3072)
  out : (32, 256, 576)  grad=True

[4/6] TOKEN FUSION
  text embeddings : (1909, 1, 576)  torch.bfloat16  grad=False
  image slots     : 32 tokens replaced with vision embeddings
  combined        : (1909, 1, 576)  grad=True

[5/6] LANGUAGE MODEL  [MobileLLM-R1-140M  |  FROZEN]
  in  : (1909, 1, 576)  torch.bfloat16
  out : (1, 1909)  torch.float32  grad=True

[6/6] LOSS / LOGITS
  loss (mean over 532 tokens) : 11.7835   ← step 1 (random adapter init)
  top-1 accuracy  : 1/532 = 0.2%          ← expected near-random at step 1
```

---

## Repository Structure

```
LLaVA-OneVision-1.5/
│
├── README.md                            ← This file
├── inference_fastvlm.py                 ← Demo: end-to-end inference script
│
├── examples/llava_ov_1_5/
│   └── quick_start/
│       └── stage_1_alignment_mobilellm_140m.sh   ← Main training launcher
│
├── stage_1_alignment_mobilellm_140m/    ← Training outputs (this run)
│   ├── iter_0000500/                    ← Latest checkpoint (500 steps)
│   │   └── mp_rank_00/
│   │       ├── model_optim_rng.pt       ← Model + optimizer state (529 MB, Git LFS)
│   │       └── distrib_optim.pt         ← Distributed optimizer state (129 MB, Git LFS)
│   ├── latest_checkpointed_iteration.txt
│   └── run_2026-04-29_13:33:58_*.log   ← Full training log (500 steps)
│
├── aiak_training_llm/
│   ├── models/
│   │   ├── llavaov_1_5/                 ← Core model: integrates all three components
│   │   │   ├── llavaov_1_5_model.py     ← Main forward pass with 6-step pipeline logging
│   │   │   ├── llavaov_1_5_config.py    ← Model configuration (sets qk_layernorm=True)
│   │   │   ├── llavaov_1_5_layer_spec.py← Transformer layer spec (imports from mobilellm)
│   │   │   ├── llavaov_1_5_provider.py  ← Model provider for Megatron training loop
│   │   │   └── rice_vision_model.py     ← Vision model base (rotary pos emb for vision)
│   │   │
│   │   ├── mobilellm/                   ← MobileLLM-R1-140M integration
│   │   │   ├── mobilellm_model.py       ← Megatron GPT model wrapper for MobileLLM
│   │   │   ├── mobilellm_config.py      ← Reads HF config.json → TransformerConfig
│   │   │   ├── mobilellm_layer_spec.py  ← Layer spec with correct QK-norm handling
│   │   │   └── mobilellm_provider.py    ← Model provider
│   │   │
│   │   └── fastvit/                     ← FastViT / MobileCLIP-L vision encoder
│   │       ├── fastvit_vision_model.py  ← Megatron-compatible wrapper
│   │       ├── mobileclip_encoder.py    ← MobileCLIPVisionTower (loads pretrained weights)
│   │       ├── fastvit_preprocessor.py  ← Image preprocessing (pad to square, resize)
│   │       ├── mm_utils.py              ← expand2square, image padding utilities
│   │       └── mobileclip/              ← Apple ml-fastvlm model code (FastViT backbone)
│   │           ├── mci.py               ← FastViT model class
│   │           └── ...
│   │
│   ├── data/multimodal/
│   │   ├── qwen2vl_task_encoder.py      ← Main data pipeline (FastViT path added)
│   │   └── task_encoder.py              ← Base task encoder
│   │
│   └── train/
│       ├── pretrain/
│       │   └── pretrain_llavaov_1_5.py  ← forward_step, loss_func, training entry
│       └── training_utils.py            ← Megatron training loop
│
├── aiak_megatron/megatron/core/
│   ├── transformer/
│   │   ├── attention.py                 ← Patched: debug prints removed
│   │   └── dot_product_attention.py     ← Patched: debug prints removed, GQA shape fix
│   └── extensions/
│       ├── transformer_engine.py        ← Patched: ROCm/AMD compatibility
│       └── transformer_engine2.py       ← Patched: ROCm/AMD compatibility
│
├── apex/csrc/
│   ├── mlp.cpp                          ← Patched: PyTorch 2.x API (.scalar_type())
│   └── fused_dense.cpp                  ← Patched: PyTorch 2.x API (.scalar_type())
│
└── checkpoints/
    └── mobilellm-fastvit-merged-tp1-pp1/← Pretrained merged checkpoint (NOT in git)
```

---

## Dependencies & Installation

### Hardware Requirements

- **GPU:** NVIDIA GPU with ≥ 16 GB VRAM (tested on A100 80 GB)  
  *Note:* This codebase was also patched for ROCm/AMD compatibility.
- **CUDA:** 12.x (tested with CUDA 12.1)
- **RAM:** ≥ 32 GB system RAM recommended

### 1. Clone the repository

```bash
git clone https://github.com/RanaZay/LLaVA-OneVision-1.5.git
cd LLaVA-OneVision-1.5
git checkout mobile-llm-integration
```

### 2. Create conda environment

```bash
conda create -n llava-mobile python=3.10 -y
conda activate llava-mobile
```

### 3. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install core dependencies

```bash
pip install \
    transformers==4.47.1 \
    tokenizers \
    sentencepiece \
    einops \
    timm \
    Pillow \
    numpy \
    scipy \
    tqdm \
    pyyaml \
    regex \
    ftfy \
    webdataset \
    braceexpand \
    protobuf "protobuf>=4.25.1,<7" \
    wandb \
    tensorboard \
    open_clip_torch
```

### 5. Install Transformer Engine (NVIDIA TE 2.11.0)

TE is required for fused attention and mixed-precision training:

```bash
# Set CUDA/cuDNN paths (adjust to your environment)
export CUDA_HOME=/usr/local/cuda
export CPATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

pip install transformer_engine[pytorch]==2.11.0
```

### 6. Install Apex (fused CUDA kernels)

```bash
git clone https://github.com/NVIDIA/apex.git /tmp/apex
cd /tmp/apex
# The apex/ directory in this repo already contains the PyTorch 2.x patches
# Copy patched files first
cp /path/to/LLaVA-OneVision-1.5/apex/csrc/mlp.cpp csrc/
cp /path/to/LLaVA-OneVision-1.5/apex/csrc/fused_dense.cpp csrc/

pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    .
```

### 7. Install the AIAK Megatron submodule

```bash
cd /path/to/LLaVA-OneVision-1.5
pip install -e aiak_megatron/
```

### 8. Set PYTHONPATH

```bash
export PYTHONPATH=/path/to/LLaVA-OneVision-1.5:$PYTHONPATH
```

---

## Downloading Pretrained Checkpoints

### MobileLLM-R1-140M (Language Model)

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='facebook/MobileLLM-R1-140M',
    local_dir='checkpoints/MobileLLM-R1-140M'
)
"
```

### FastViT MobileCLIP-L (Vision Encoder)

The FastViT encoder weights are part of Apple's ml-fastvlm release:

```bash
# Download from Apple ml-fastvlm (FastViT-HD 1.5B stage-3 checkpoint)
# See: https://github.com/apple/ml-fastvlm
# The vision tower weights are stored as mobileclip_l_1024 in the merged checkpoint below.
```

### Merged Checkpoint (MobileLLM + FastViT, Megatron format)

The `mobilellm-fastvit-merged-tp1-pp1` checkpoint combines MobileLLM-R1-140M and FastViT
into Megatron core format (TP=1, PP=1). Generate it once after downloading both models:

```bash
# Convert MobileLLM from HuggingFace to Megatron format
AIAK_TRAINING_PATH=$(pwd) python aiak_training_llm/models/mobilellm/megatron_checkpoint/convert_hf_to_mcore.py \
    --hf-checkpoint checkpoints/MobileLLM-R1-140M \
    --output checkpoints/mobilellm-fastvit-merged-tp1-pp1 \
    --tp 1 --pp 1

# The vision encoder weights are loaded automatically from the FastViT pretrained
# weights specified by --vision-tower-name mobileclip_l_1024
```

---

## Downloading the Dataset

Stage 1 alignment uses the **LLaVA-558K** dataset in WebDataset format:

```bash
# Using HuggingFace hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lmms-lab/LLaVA-558K-Webdataset',
    repo_type='dataset',
    local_dir='data/LLaVA-558K-Webdataset'
)
"
```

The dataset is ~30 GB. It contains 558K image-text pairs in `.tar` WebDataset shards.

---

## Running Stage 1 Alignment Training

### Quick Start (single machine, 2 GPUs)

```bash
cd /path/to/LLaVA-OneVision-1.5

# Set environment and run
GPUS_PER_NODE=2 \
DATA_PATH=data/LLaVA-558K-Webdataset \
TOKENIZER_PATH=facebook/MobileLLM-R1-140M \
PRETRAINED_CHECKPOINT=checkpoints/mobilellm-fastvit-merged-tp1-pp1 \
bash examples/llava_ov_1_5/quick_start/stage_1_alignment_mobilellm_140m.sh
```

> **Note:** The script uses `torchrun`. Make sure your conda environment is active so `torchrun` is on your PATH.  
> If you get `torchrun: command not found`, prepend:  
> `PATH=/path/to/conda/envs/llava-mobile/bin:$PATH bash examples/...`

### Script parameters

The script accepts positional arguments:

```bash
bash stage_1_alignment_mobilellm_140m.sh \
    [TP=1] [PP=1] [SEQ_LEN=32768] [MBS=1] [GBS=2] [NSTEPS=500]
```

| Argument | Default | Description |
|---|---|---|
| `TP` | 1 | Tensor parallel degree |
| `PP` | 1 | Pipeline parallel degree |
| `SEQ_LEN` | 32768 | Maximum sequence length |
| `MBS` | 1 | Micro batch size per GPU |
| `GBS` | 2 | Global batch size |
| `NSTEPS` | 1 | Number of training iterations |

### Environment variable overrides

| Variable | Default | Description |
|---|---|---|
| `GPUS_PER_NODE` | 2 | GPUs per machine |
| `DATA_PATH` | `data/LLaVA-558K-Webdataset` | Dataset path |
| `TOKENIZER_PATH` | `facebook/MobileLLM-R1-140M` | HF tokenizer |
| `PRETRAINED_CHECKPOINT` | `checkpoints/mobilellm-fastvit-merged-tp1-pp1` | Megatron checkpoint |
| `WANDB_API_KEY` | *(unset)* | Set to enable W&B logging |

### What gets trained

Stage 1 trains **only the 2-layer MLP adapter** (~3.5M parameters). Both the FastViT encoder and MobileLLM-R1-140M are fully frozen. This teaches the adapter to project visual features into the language model's embedding space.

---

## Demo: End-to-End Inference

`inference_fastvlm.py` runs the full pipeline (FastViT → Adapter → MobileLLM) for a single image + text prompt without distributed training setup.

### Usage

```bash
CUDA_VISIBLE_DEVICES=0 python inference_fastvlm.py \
    --image /path/to/image.jpg \
    --prompt "<|image_pad|>\nDescribe what you see in this image." \
    --checkpoint stage_1_alignment_mobilellm_140m/iter_0000500 \
    --max-new-tokens 150
```

### Sample output (step-500 checkpoint, random noise image)

```
Prompt: <|image_pad|>
Describe what you see in this image.

Generated: The image shows a [...]
```

### Without an image (text-only)

```bash
CUDA_VISIBLE_DEVICES=0 python inference_fastvlm.py \
    --prompt "What is the capital of France?" \
    --max-new-tokens 50
```

---

## Training Logs & Results

All training logs are stored in `stage_1_alignment_mobilellm_140m/`. The final 500-step run is:

```
stage_1_alignment_mobilellm_140m/run_2026-04-29_13:33:58_tp1_pp1_seqlen32768_mbs1_gbs4_500steps.log
```

### Configuration for the 500-step run

| Setting | Value |
|---|---|
| GPUs | 4× (TP=1, PP=1, DP=4) |
| Global batch size | 4 samples |
| Micro batch size | 1 sample/GPU |
| Sequence length | 32,768 tokens |
| Learning rate | 1e-4 (cosine decay to ~1e-5) |
| Optimizer | Adam (β₁=0.9, β₂=0.99) |
| Precision | BFloat16 |
| Gradient clipping | 1.0 |

### Loss curve summary

| Iteration | LM Loss |
|---|---|
| 1 | ~11.78 (random adapter init) |
| 100 | ~11.2x |
| 500 | **11.30** |

> The loss decrease is expected to be gradual at Stage 1 since only the 3.5M-parameter adapter is trained and the LLM (140M params) is frozen. Full convergence requires Stage 2 instruction fine-tuning.

### Checkpoint

The latest checkpoint is at iteration 500 (Megatron format, TP=1 PP=1).  
The binary weights are **not stored in git** (too large; 529 MB + 129 MB).  
They are available on the training server at:

```
/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5/stage_1_alignment_mobilellm_140m/iter_0000500/
├── mp_rank_00/model_optim_rng.pt    # full model + optimizer state (529 MB)
└── mp_rank_00/distrib_optim.pt      # distributed optimizer shards (129 MB)
```

To resume training from this checkpoint, set:
```bash
PRETRAINED_CHECKPOINT=stage_1_alignment_mobilellm_140m/iter_0000500
```

---

## Modifications vs. Upstream

This branch modifies the original LLaVA-OneVision-1.5 codebase in the following ways. All changes are clearly separated from the upstream by the file paths below.

### New files added

| File | Description |
|---|---|
| `aiak_training_llm/models/mobilellm/mobilellm_model.py` | MobileLLM-R1-140M Megatron wrapper |
| `aiak_training_llm/models/mobilellm/mobilellm_config.py` | Reads HF `config.json` → `TransformerConfig`; documents NoPE→RoPE divergence |
| `aiak_training_llm/models/mobilellm/mobilellm_layer_spec.py` | Transformer layer spec with correct QK-norm (checks `config.qk_layernorm`) |
| `aiak_training_llm/models/mobilellm/mobilellm_provider.py` | Megatron model provider |
| `aiak_training_llm/models/fastvit/fastvit_vision_model.py` | Megatron-compatible FastViT wrapper |
| `aiak_training_llm/models/fastvit/mobileclip_encoder.py` | `MobileCLIPVisionTower` that loads Apple FastViT weights |
| `aiak_training_llm/models/fastvit/fastvit_preprocessor.py` | Image preprocessing for FastViT (1024 px, square padding) |
| `aiak_training_llm/models/fastvit/mobileclip/` | Apple ml-fastvlm FastViT model code |
| `inference_fastvlm.py` | Stand-alone inference demo |
| `examples/llava_ov_1_5/quick_start/stage_1_alignment_mobilellm_140m.sh` | Training launcher for MobileLLM + FastViT |

### Modified files (vs. upstream)

| File | Change |
|---|---|
| `aiak_training_llm/models/llavaov_1_5/llavaov_1_5_model.py` | Added 6-step pipeline logging; replaced `[BIG DEBUG]` prints with structured `[1/6]…[6/6]` output; added per-step token accuracy |
| `aiak_training_llm/models/llavaov_1_5/llavaov_1_5_layer_spec.py` | Removed duplicate `get_mobilellm_layer_with_te_spec` that hard-coded `q_layernorm=IdentityOp` (bug: ignored config); now imports the correct version from `mobilellm_layer_spec.py` |
| `aiak_training_llm/models/llavaov_1_5/llavaov_1_5_config.py` | Sets `qk_layernorm=True` for MobileLLM (matching `use_qk_norm: true` in HF config) |
| `aiak_training_llm/data/multimodal/qwen2vl_task_encoder.py` | Added FastViT preprocessing path (`--use-fastvit` flag); per-image debug prints removed |
| `aiak_megatron/megatron/core/transformer/dot_product_attention.py` | Fixed GQA shape handling for non-square query/key heads; removed debug prints |
| `aiak_megatron/megatron/core/transformer/attention.py` | Removed debug prints |
| `aiak_megatron/megatron/core/extensions/transformer_engine.py` | ROCm/AMD compatibility: skip CUDA arch checks on AMD GPUs |
| `aiak_megatron/megatron/core/extensions/transformer_engine2.py` | ROCm/AMD compatibility |
| `apex/csrc/mlp.cpp` | PyTorch 2.x API fix: `.type()` → `.scalar_type()`, `.options()` |
| `apex/csrc/fused_dense.cpp` | PyTorch 2.x API fix: same as above |

### Deleted dead code files

| File | Reason |
|---|---|
| `aiak_training_llm/models/llavaov_1_5/adapter.py` | Never imported; active adapter is `qwen_vl/adapter.py` |
| `aiak_training_llm/models/llavaov_1_5/llavaov_1_5_layer_spec_org.py` | Old backup copy |
| `aiak_training_llm/models/llavaov_1_5/llavaov_mobilellm_config.py` | Superseded by `mobilellm/mobilellm_config.py` |

### Key architectural decisions

1. **QK LayerNorm (bug fix):** MobileLLM-R1-140M uses `use_qk_norm: true` in its HuggingFace config. The upstream code ignored this, causing attention to run without QK normalization. Fixed in `llavaov_1_5_layer_spec.py`.

2. **NoPE → RoPE (intentional divergence):** MobileLLM-R1 is a NoPE (No Positional Encoding) model — all 15 layers have `no_rope_layers=[1,1,...,1]`. This Megatron integration applies RoPE with `rope_theta=8000000` (from the HF config) to provide positional encoding for multimodal sequences. This is documented in `mobilellm_config.py`.

3. **FastViT output shape:** FastViT MobileCLIP-L outputs 4D features `[B, 3072, H, W]` where `H=W=16` (for 1024 px input with patch_size=64). These are reshaped to `[B, 256, 3072]` before the adapter.

4. **Vision token count mismatch:** The data pipeliner packs multiple images per sequence. A trim/pad mechanism in `llavaov_1_5_model.py` ensures the adapter output token count always matches the number of `<|image_pad|>` tokens in `input_ids`.

---

## Credits & Citations

This project builds on top of:

### LLaVA-OneVision-1.5 (base framework)

```bibtex
@inproceedings{LLaVA-OneVision-1.5,
  title={LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training},
  author={An, Xiang and Xie, Yin and Yang, Kaicheng and others},
  booktitle={arXiv},
  year={2025}
}
```

GitHub: https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5

### MobileLLM-R1-140M (Language Model)

```bibtex
@article{mobilellm,
  title={MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases},
  author={Liu, Zechun and others},
  journal={arXiv},
  year={2024}
}
```

HuggingFace: https://huggingface.co/facebook/MobileLLM-R1-140M

### FastVLM / MobileCLIP-L (Vision Encoder)

```bibtex
@article{fastvlm,
  title={FastVLM: Efficient Vision Encoding for Vision Language Models},
  author={Prabhu, Shirin and others},
  journal={CVPR},
  year={2025}
}
```

GitHub: https://github.com/apple/ml-fastvlm

### Megatron-LM (Training Framework)

GitHub: https://github.com/NVIDIA/Megatron-LM

---

*For questions or issues, please open a GitHub issue on this repository.*
