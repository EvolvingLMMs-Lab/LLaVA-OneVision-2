# Faster training for LLaVA-OneVision-2

We provide opt-in speedups for the 4B quick-start SFT recipe ([`quick_start.sh`](../examples/llava_onevision2/quick_start_4b/quick_start.sh)). Every speedup is an environment variable switch that is off by default; with none set, training runs exactly as before. A preset contains a set of speedup switches that `quick_start.sh` loads when `OV2_PRESET` points to it.

Both presets need the `flash-attn-4` package, which is not in the training image: Blackwell runs its FA4 kernels, Hopper its sm90 kernels (the FA3 algorithm in CuTe DSL). Install it inside the container, next to the image's torch 2.7 and flash-attn 2.7.3 (`--no-deps` keeps them), then pick a preset:

```bash
export OV2_EXTRAS_DIR=/workspace/ov2_extras
pip install --no-deps --target $OV2_EXTRAS_DIR/fa4 \
    flash-attn-4==4.0.0b36 nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 \
    nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 \
    apache-tvm-ffi==0.1.15.post1 quack-kernels==0.6.5 torch-c-dlpack-ext==0.1.5

# 8-GPU Blackwell node
OV2_PRESET=blackwell bash examples/llava_onevision2/quick_start_4b/quick_start.sh
# 8 Hopper GPUs with 80 GB
OV2_PRESET=hopper_80g bash examples/llava_onevision2/quick_start_4b/quick_start.sh
```

Both presets keep the global batch size of 16 of the quick-start recipe. 

| Preset | GPU Architecture | Peak memory per GPU (8 GPUs) | Speed-up |
|---|---|---|---|
| `blackwell` | Blackwell | 157.6 GiB | 3.0× |
| `hopper_80g` | Hopper | 74.8 GiB | 2.2× |

## Switches

The following table lists the speedup switches, their effect on training, and whether they are on in each preset. The `results` column shows how a switch affects the training numbers:
- **exact**: training is bit-for-bit identical to the default code (checked in deterministic mode).
- **rounding**: the same computation at the same precision, done with other kernels or with floating-point sums in another order, so results differ only by floating-point rounding.

| switch | what it does | results | blackwell | hopper_80g |
|---|---|---|---|---|
| `OV2_OPT_NO_RECOMPUTE=1` | no full activation recomputation (on 80 GB GPUs, use it with `OV2_OPT_MEM_BUDGET_GIB`) | exact | on | on |
| `OV2_OPT_ATTN_FUSED=1` | Transformer Engine cuDNN fused attention (deterministic mode keeps flash-attn 2) | rounding | on | on |
| `OV2_OPT_FA4_ATTN=1` | `flash-attn-4` attention: FA4 kernels on Blackwell, FA3-algorithm kernels on Hopper | rounding | on | on |
| `OV2_OPT_FA4_SCOPE=all/lm/vision` | which attentions use `flash-attn-4` (default `all`) | rounding | `lm` | `lm` |
| `OV2_OPT_FA4_VISION_FROM=k` | with scope `lm`, also the vision encoder's layers from k on (1-based) | rounding | 6 | 6 |
| `OV2_OPT_FA4_EXACT_MAX=1` | FA4 forward rescales at every new row maximum; the default lazy rescaling distorts gradients of peaked attention rows (no effect on Hopper) | rounding | on | on |
| `OV2_OPT_HOST_SYNC_FREE=1` | batch to GPU without host syncs (pinned async copies, host-side metadata) | exact | on | on |
| `OV2_OPT_DEFER_LOSS_REDUCE=1` | per-micro-batch loss all-reduces issued after the last backward | exact | on | on |
| `OV2_OPT_ASYNC_TOKEN_COUNT=1` | token count filled on the device (no blocking copy) | exact | on | on |
| `OV2_OPT_INDEX_SCATTER=1` | image embeddings scattered by host-computed positions (no `nonzero` sync) | exact | on | on |
| `OV2_OPT_GC_SYNC=1` | `gc.freeze()`, then manual GC at the same iteration on every rank (`OV2_OPT_GC_INTERVAL`, default 100) | exact | on | on |
| `OV2_OPT_FUSED_VISION_ROPE=1` | vision RoPE in one Triton kernel each way | exact | on | on |
| `OV2_OPT_BITEXACT_QK_RMSNORM=1` | apex's q/k RMSNorm reproduced in Triton, without copies | exact on Blackwell; rounding on Hopper (~1e-5 of the input-gradient elements differ by one bf16 ulp) | on | on |
| `OV2_OPT_FAST_BIAS_ADD=1` | vision fc1 bias add in a vectorised Triton kernel | exact | on | on |
| `OV2_OPT_FUSED_SWIGLU=1` | LM SwiGLU in Triton | exact | on | on |
| `OV2_OPT_FUSED_QK_NORM_ROPE=1` | LM q/k RMSNorm + RoPE fused | rounding | on | on |
| `OV2_OPT_FUSED_QKV_SPLIT=1` | the qkv split fused into that kernel | rounding | on | on |
| `OV2_OPT_TRITON_CE=1` | LM cross-entropy in two Triton kernels, no fp32 logits copy | rounding | on | on |
| `OV2_OPT_CE_FUSION=1` | Megatron's fused cross-entropy | rounding | on | on |
| `OV2_OPT_DDP_AVG=1` | 1/DP gradient scaling inside the reduce-scatter | exact (power-of-two DP size) | on | on |
| `OV2_OPT_BALANCE=1`, `OV2_OPT_BALANCE_PREPULL=1` | cross-rank load balancing of each iteration's packed sequences; the next iteration's sequences are taken from the loader while the GPU finishes | rounding | on | on |
| `OV2_OPT_BALANCE_COSTS=<9 values>` | balancing cost model for another GPU (with the planned recomputation) | rounding | off | Hopper values |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | PyTorch allocator setting (allocator slack ~14 → ~1.6 GiB) | exact | off | on |
| `OV2_OPT_MEM_BUDGET_GIB=g` | per-micro-batch planner: only the recomputation needed to keep activations within g GiB | exact | off | 33 |
| `OV2_OPT_QKV_RECOMPUTE=1` | LM q/k/v freed after the attention forward, rebuilt before its backward | exact | off | on |
| `OV2_OPT_ACT_RECOMPUTE`, `OV2_OPT_FC1_RECOMPUTE_{LM,VISION}`, `OV2_OPT_RECOMPUTE_{LM,VISION}` | static recomputation choices (the planner chooses per sequence) | exact | off | off |
| `OV2_OPT_FA3_ATTN`, `OV2_OPT_FA3_VISION_FROM` | FlashAttention-3 (experimental, Hopper only; not faster than `flash-attn-4`'s sm90 kernels) | rounding | off | off |
