# MobileLLM Integration for LLaVA-OneVision-1.5

This document describes the integration of Facebook's MobileLLM-R1-140M as the language backbone for LLaVA-OneVision-1.5, replacing the original Qwen2.5 model.

## Overview

**MobileLLM-R1-140M** is a compact 140M parameter language model from Facebook Research optimized for on-device deployment with strong reasoning capabilities.

### Architecture Comparison

| Component | Original (Qwen2.5-4B) | MobileLLM-R1-140M |
|-----------|----------------------|-------------------|
| Parameters | 4B | 140M (35x smaller) |
| Layers | 32 | 15 |
| Hidden Size | 3584 | 576 |
| Attention Heads | 28 | 9 |
| KV Heads (GQA) | 4 | 3 |
| FFN Size | 18944 | 2048 |
| Vocab Size | 151,936 | 128,256 |
| Context Length | 32,768 | 32,768 |
| RoPE Base | 1,000,000 | 8,000,000 |

### Key Features
- **Grouped Query Attention (GQA)**: 9 query heads, 3 KV heads
- **SwiGLU Activation**: Gated linear unit for better performance
- **RMSNorm**: Fast layer normalization
- **Shared Embeddings**: Input/output embeddings are tied
- **RoPE**: Rotary position embeddings with base 8M

## Files Modified/Created

### Core Integration Files

1. **aiak_training_llm/models/llavaov_1_5/llavaov_mobilellm_config.py** (NEW)
   - MobileLLM configuration for LLaVA-OneVision
   - Registered model architectures:
     - `llava-ov-mobilellm-140m` (with SigLIP)
     - `llava-ov-mobilellm-140m-fastvit` (with FastViT)

2. **aiak_training_llm/models/llavaov_1_5/llavaov_1_5_layer_spec.py** (MODIFIED)
   - Added `get_mobilellm_layer_with_te_spec()` function
   - Configures MobileLLM-style transformer layers with:
     - GQA attention
     - SwiGLU MLP
     - RMSNorm
     - No QK LayerNorm

3. **aiak_training_llm/models/llavaov_1_5/llavaov_1_5_provider.py** (MODIFIED)
   - Auto-detects MobileLLM from model name
   - Routes to appropriate language config and layer spec
   - Supports both Qwen and MobileLLM backbones

4. **aiak_training_llm/models/llavaov_1_5/__init__.py** (MODIFIED)
   - Exports MobileLLM configuration functions

### Training Script

5. **examples/llava_ov_1_5/quick_start/stage_1_alignment_mobilellm_140m.sh** (NEW)
   - Stage 1 alignment training script
   - Configured for MobileLLM-R1-140M + FastViT
   - Trains only the adapter (vision + language frozen)

### Existing MobileLLM Files (from your previous work)

6. **aiak_training_llm/models/mobilellm/** (EXISTING)
   - `mobilellm_config.py`: Standalone MobileLLM config
   - `mobilellm_model.py`: MobileLLM model class
   - `mobilellm_layer_spec.py`: Layer specifications
   - `mobilellm_provider.py`: Model provider registration
   
   **Note**: These files are for standalone MobileLLM LM training. The LLaVA integration uses the `llavaov_1_5/llavaov_mobilellm_config.py` instead.

## Configuration Details

### Model Architecture (llava-ov-mobilellm-140m-fastvit)

```python
Language Model (MobileLLM-R1-140M):
- num_layers: 15
- hidden_size: 576
- num_attention_heads: 9
- num_query_groups: 3 (GQA)
- ffn_hidden_size: 2048
- vocab_size: 128,256
- max_position_embeddings: 32,768
- rotary_base: 8,000,000
- normalization: RMSNorm (epsilon=1e-05)
- activation: swiglu
- tied_embeddings: True

Vision Encoder (FastViT/MobileCLIP-S-384):
- image_resolution: 384x384
- patch_size: 16
- hidden_size: 768
- num_layers: 12
- num_attention_heads: 12

Adapter (Projection):
- adapter_dim: 2048
- activation: gelu
```

## Training Setup

### Prerequisites

1. **Tokenizer**: Use HuggingFace tokenizer from `facebook/MobileLLM-R1-140M`
2. **Data**: LLaVA-558K dataset in WebDataset format
3. **Checkpoints** (Optional): 
   - MobileLLM pretrained weights converted to Megatron format
   - FastViT pretrained weights

### Stage 1: Alignment Training

Train only the projection adapter while keeping vision encoder and language model frozen.

```bash
cd /path/to/LLaVA-OneVision-1.5

# Set environment variables
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="llava-ov-mobilellm"
export WANDB_NAME="stage1-alignment"

# Run training
bash examples/llava_ov_1_5/quick_start/stage_1_alignment_mobilellm_140m.sh
```

#### Training Parameters

- **Model**: `llava-ov-mobilellm-140m-fastvit`
- **Trainable**: Adapter only
- **Frozen**: Vision encoder + Language model
- **Sequence Length**: 512 tokens
- **Batch Size**: Global=1, Micro=1
- **Learning Rate**: 1e-4 → 1e-6 (cosine)
- **Training Steps**: 2,500 (adjust as needed)
- **Precision**: BF16

### Running on Your Cluster

Update these paths in the script:

```bash
# Data directory with WebDataset shards
DATA_PATH="/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5/data/LLaVA-558K-Webdataset"

# MobileLLM tokenizer (HuggingFace)
TOKENIZER_PATH="facebook/MobileLLM-R1-140M"

# (Optional) Checkpoint path if loading pretrained weights
# CHECKPOINT_PATH="/path/to/mobilellm_mcore_checkpoint"
```

## Model Registration

The MobileLLM integration uses the factory pattern for model registration:

```python
# In llavaov_mobilellm_config.py
@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5,
    model_arch="llava-ov-mobilellm-140m-fastvit"
)
```

When you specify `--model-name llava-ov-mobilellm-140m-fastvit`, the system automatically:
1. Routes to LLaVA-OneVision-1.5 model family
2. Loads MobileLLM language config
3. Configures FastViT vision encoder
4. Sets up appropriate layer specs

## Key Differences from Qwen Integration

### 1. Configuration
- **Qwen**: Uses `llavaov_1_5_config.py` → Qwen-specific configs
- **MobileLLM**: Uses `llavaov_mobilellm_config.py` → MobileLLM configs

### 2. Layer Specification
- **Qwen**: `get_qwen_layer_with_te_spec()` → QK LayerNorm enabled
- **MobileLLM**: `get_mobilellm_layer_with_te_spec()` → No QK LayerNorm

### 3. Tokenizer
- **Qwen**: `Qwen/Qwen2.5-4B-Instruct` (151,936 vocab)
- **MobileLLM**: `facebook/MobileLLM-R1-140M` (128,256 vocab)

### 4. RoPE Base
- **Qwen**: 1,000,000
- **MobileLLM**: 8,000,000 (8x larger for better long-context)

### 5. Chat Template
- **Qwen**: Uses Qwen2-VL chat template
- **MobileLLM**: TODO - needs custom chat template for vision tokens

## Verification Checklist

After integration, verify:

- [ ] Model name detection: Check logs for "Using MobileLLM-R1-140M as language backbone"
- [ ] Config loading: Verify 15 layers, 576 hidden size in logs
- [ ] Layer spec: Confirm "Using MobileLLM layer specification"
- [ ] Tokenizer: Vocab size should be 128,256
- [ ] Training: Adapter parameters update, language/vision frozen
- [ ] Memory: 140M model should use significantly less VRAM than 4B

## Expected Benefits

1. **Efficiency**: 35x fewer parameters → faster inference, lower memory
2. **On-device Deployment**: Fits on mobile/edge devices
3. **Long Context**: 32k context with efficient RoPE
4. **Reasoning**: R1 series has strong math/reasoning capabilities

## Troubleshooting

### Error: "Unknown model family or arch: llava-ov-mobilellm-140m-fastvit"

**Solution**: Ensure `llavaov_mobilellm_config.py` is imported. Check that decorators are registered before model initialization.

### Error: "Cannot load checkpoint"

**Solution**: If starting from scratch, comment out `--load` in the training script. For loading MobileLLM weights, they must be converted to Megatron format first.

### Memory Issues

**Solution**: 
- Reduce sequence length: `SEQ_LEN=256`
- Enable gradient checkpointing: Already enabled in script
- Reduce recompute layers: `--recompute-num-layers 2`

### Tokenizer Mismatch

**Solution**: Ensure using `facebook/MobileLLM-R1-140M` tokenizer, not Qwen tokenizer.

## Next Steps

1. **Stage 1 Completion**: Train adapter on full LLaVA-558K dataset
2. **Stage 2 (Optional)**: Fine-tune full model on instruction data
3. **Evaluation**: Test on VQA benchmarks
4. **Export**: Convert to ONNX/TensorRT for deployment

## References

- [MobileLLM-R1 GitHub](https://github.com/facebookresearch/MobileLLM-R1)
- [MobileLLM-R1-140M on HuggingFace](https://huggingface.co/facebook/MobileLLM-R1-140M)
- [LLaVA-OneVision Paper](https://arxiv.org/abs/2408.03326)
- [FastViT/MobileCLIP Paper](https://arxiv.org/abs/2303.15378)

## Contact

For issues specific to this integration, check:
1. Model logs in `stage_1_alignment_mobilellm_140m/`
2. WandB dashboard if configured
3. TensorBoard logs in `stage_1_alignment_mobilellm_140m/tensorboard/`
