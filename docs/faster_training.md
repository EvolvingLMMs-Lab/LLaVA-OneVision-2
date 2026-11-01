# Faster training (opt-in)

Opt-in speedups for the 4B quick-start SFT recipe
([`quick_start.sh`](../examples/llava_onevision2/quick_start_4b/quick_start.sh)). Every switch is an environment
variable (`OV2_OPT_*`) that is off by default: with none set, training runs exactly as before. Two presets bundle them.

| preset | for | upstream → preset | speedup |
|---|---|---|---|
| `blackwell` | 8-GPU Blackwell node | 2.536 → 0.838 s/it, one epoch of the full dataset (9.6 h → 3.2 h) | **3.03×** |
| `hopper_80g` | Hopper, 80 GB | 5.770 → 2.587 s/it, full dataset, iterations 1-800 (4 GPUs) | **2.23×** |

## Usage

Both presets use FlashAttention-4, which is not in the training image. Install it inside the container, next to the
image's torch 2.7 and flash-attn 2.7.3 (`--no-deps` keeps them):

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

- `OV2_PRESET` takes a name (`examples/llava_onevision2/quick_start_4b/presets/<name>.json`) or a path; the file's `environment` is exported before `torchrun`.
  Switches can also be set one by one in the environment.
- `OV2_EXTRAS_DIR` is only searched for `fa4/` (and `fa3/`), after the image's packages. On torch 2.7 the loader
  ([`optional_kernels.py`](../aiak_training_llm/models/custom/common/optional_kernels.py)) adds two small compatibility
  shims. A switch whose library is missing fails at start-up with the install command.
- FA4 kernels are JIT-compiled during the first iterations.
- FlashAttention-3 (experimental `OV2_OPT_FA3_ATTN`, Hopper only): `tools/kernels/build_fa3.sh` builds it into
  `$OV2_EXTRAS_DIR/fa3`.

## Speed

Seconds per iteration, global batch 16 (8 GPUs: 2 micro-batches per GPU; 4 GPUs: 4).

| GPUs | data | upstream | `blackwell` | `hopper_80g` |
|---|---|---|---|---|
| 8 × Blackwell | full dataset, one epoch (13,682 it) | 2.536 | 0.838 (**3.03×**) | 0.871 (2.91×) |
| 8 × Blackwell | fixed subset | 2.684 | 0.960 (2.80×) | |
| 4 × Blackwell | fixed subset | 5.097 | 1.762 (2.89×) | |
| 4 × Hopper | full dataset, iterations 1-800 | 5.770 | | 2.587 (2.23×) |
| 4 × Hopper | fixed subset | 6.481 | 2.991 (2.17×, needs > 80 GB) | 3.142 (2.06×) |

Fixed subset: one shard of each of the four sub-datasets (6,844 packed sequences), 10 warm-up + 2 × 25 timed
iterations, upstream timed before and after the preset. The Hopper GPUs used have more memory and HBM bandwidth than
80 GB Hopper GPUs (`hopper_80g` still capped its activations as for 80 GB); memory-bound kernels will be somewhat
slower on those.

## Memory (per GPU)

| | upstream | `blackwell` | `hopper_80g` |
|---|---|---|---|
| max allocated, 4 GPUs, fixed subset (Hopper) | 50.3 GiB | 102.1 GiB | 73.3 GiB |
| nvidia-smi peak, 8 GPUs, one epoch of the full dataset (8-GPU Blackwell node) | 95.5 GiB | 157.6 GiB | **74.8 GiB** |

Both rows measure each preset on one architecture, so the columns compare directly; nvidia-smi also counts the
allocator cache, the CUDA context and NCCL buffers. At 4 GPUs each rank holds twice the optimizer shard of 8 GPUs
(+6.6 GiB). `OV2_OPT_MEM_BUDGET_GIB=33` keeps every micro-batch's activations within 33 GiB, including the
dataset's heaviest packed sequences (32,388 vision patches).

## Switches

Numerics: **bitwise** = identical results in deterministic mode; **reordered** = same math, floating-point sums in
another order; **kernel** = another attention kernel at the same precision (bf16 in/out, fp32 softmax statistics and
accumulation).

| switch | what it does | where | numerics | `blackwell` | `hopper_80g` |
|---|---|---|---|---|---|
| `OV2_OPT_NO_RECOMPUTE` | no full activation recomputation | > 80 GB, or with the planner | bitwise | 1 | 1 |
| `OV2_OPT_ATTN_FUSED` | Transformer Engine cuDNN fused attention (deterministic runs keep flash-attn 2) | portable | kernel | 1 | 1 |
| `OV2_OPT_FA4_ATTN`, `_SCOPE`, `_VISION_FROM` | FlashAttention-4 (sm100 / sm90 kernels) for the LM and vision layers ≥ k | Blackwell, Hopper | kernel | 1, `lm`, 6 | 1, `lm`, 6 |
| `OV2_OPT_FA4_EXACT_MAX` | FA4 sm100 forward rescales at every new row maximum; the default lazy rescaling distorts gradients of peaked attention rows | Blackwell (no effect on sm90) | kernel | 1 | 1 |
| `OV2_OPT_HOST_SYNC_FREE` | batch to GPU without host syncs (pinned async copies, host-side metadata) | portable | bitwise | 1 | 1 |
| `OV2_OPT_DEFER_LOSS_REDUCE` | per-micro-batch loss all-reduces issued after the last backward | portable | bitwise | 1 | 1 |
| `OV2_OPT_ASYNC_TOKEN_COUNT` | token count filled on the device (no blocking copy) | portable | bitwise | 1 | 1 |
| `OV2_OPT_INDEX_SCATTER` | image embeddings scattered by host-computed positions (no `nonzero` sync) | portable | bitwise | 1 | 1 |
| `OV2_OPT_GC_SYNC` | `gc.freeze()`, then manual GC at the same iteration on every rank (`OV2_OPT_GC_INTERVAL`, 100) | portable | bitwise | 1 | 1 |
| `OV2_OPT_FUSED_VISION_ROPE` | vision RoPE in one Triton kernel each way | portable | bitwise | 1 | 1 |
| `OV2_OPT_BITEXACT_QK_RMSNORM` | apex's q/k RMSNorm reproduced in Triton, without copies | portable | bitwise on Blackwell; on Hopper ~1e-5 of the input-gradient elements differ by one bf16 ulp | 1 | 1 |
| `OV2_OPT_FAST_BIAS_ADD` | vision fc1 bias add in a vectorised Triton kernel | portable | bitwise | 1 | 1 |
| `OV2_OPT_FUSED_SWIGLU` | LM SwiGLU in Triton | portable | bitwise | 1 | 1 |
| `OV2_OPT_FUSED_QK_NORM_ROPE`, `OV2_OPT_FUSED_QKV_SPLIT` | LM q/k RMSNorm + RoPE (+ the qkv split) fused | portable | reordered | 1 | 1 |
| `OV2_OPT_TRITON_CE` | LM cross-entropy in two Triton kernels, no fp32 logits copy | portable | reordered | 1 | 1 |
| `OV2_OPT_CE_FUSION` | Megatron's fused cross-entropy | portable | reordered | 1 | 1 |
| `OV2_OPT_DDP_AVG` | 1/DP gradient scaling inside the reduce-scatter | portable | bitwise (power-of-two DP) | 1 | 1 |
| `OV2_OPT_BALANCE`, `OV2_OPT_BALANCE_PREPULL` | cross-rank load balancing of each iteration's packed sequences | portable | reordered | 1 | 1 |
| `OV2_OPT_BALANCE_COSTS` | balancing cost model for another GPU (+ the planned recomputation) | Hopper value in preset | reordered | | 9 values |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | allocator slack ~14 → ~1.6 GiB | 80 GB | bitwise | | yes |
| `OV2_OPT_MEM_BUDGET_GIB` | per-micro-batch planner: only the recomputation needed to stay within the budget | 80 GB | bitwise | | 33 |
| `OV2_OPT_QKV_RECOMPUTE` | LM q/k/v freed after the attention forward, rebuilt before its backward | 80 GB | bitwise | | 1 |
| `OV2_OPT_ACT_RECOMPUTE`, `OV2_OPT_FC1_RECOMPUTE_{LM,VISION}`, `OV2_OPT_RECOMPUTE_{LM,VISION}` | static recomputation choices (the planner chooses per sequence) | any | bitwise | | |
| `OV2_OPT_FA3_ATTN`, `OV2_OPT_FA3_VISION_FROM` | FlashAttention-3 (experimental; not faster than FA4's sm90 kernels) | Hopper | kernel | | |

Scope: the host-side and Triton switches take effect with TP 1, CP 1 and no sequence parallelism (other layouts keep
the default path). The balancing cost model and the planner's memory model are calibrated for the 4B quick-start
model; other models need their own coefficients.

## Numerics

Unchanged by every switch: the data and its order, the model, bf16 compute, fp32 master weights, fp32 gradient
accumulation and reduction, fp32 Adam states, the loss (each packed sequence's token mean, sequences averaged with
equal weight), the optimizer and the learning-rate schedule. Checked with both presets (4 GPUs, global batch 16,
fixed subset unless stated):

1. **Deterministic records** of the first 3 iterations: losses, gradient norm and, per parameter, the reduced fp32
   gradient, the master-weight update and both Adam states (28,260 tensors). No switch, and the bitwise switches
   (alone or combined): bitwise identical to upstream (on Hopper all of them except `OV2_OPT_BITEXACT_QK_RMSNORM`, see the
   switch table). Presets: the same sequences per iteration, identical per-sequence loss weights and
   attention segments, step-1 update RMS within 1 % of upstream.
2. **Trajectories**: 100 iterations in deterministic mode, 3 data-order seeds, paired with upstream. Final loss,
   gradient norm, loss on a fixed evaluation batch and update RMS all within twice upstream's seed-to-seed standard deviation.
3. **Full data**, 8 GPUs, one epoch: `blackwell` loss on a fixed evaluation batch (16 packed sequences, eval mode) 0.5782 vs 0.5788 upstream, mean loss of the last 500
   iterations 0.7304 vs 0.7298; `hopper_80g` 0.7292 vs 0.7298.
4. cuDNN fused attention (`OV2_OPT_ATTN_FUSED`) runs only outside deterministic mode; it was checked on attention
   inputs captured from training on both architectures (error against fp64 within twice flash-attn 2's) and with
   non-deterministic trajectories of the `blackwell` preset.
