"""Opt-in performance switches (environment variables OV2_OPT_*), read by the training code. All are off by default.

With no switch set, training runs exactly as before. A switch changes how a step is executed (kernels, recomputation,
host/device scheduling, which rank trains which of an iteration's packed sequences); none changes the data, the model,
the optimizer, the loss or the fp32 master weights / gradients / Adam states. See docs/faster_training.md.
"""
import os


def flag(name, default='0'):
    """True when the environment variable OV2_OPT_<name> is set to 1."""
    return os.environ.get(f'OV2_OPT_{name}', default) == '1'


def deterministic_mode():
    """torch's deterministic algorithms or Transformer Engine's deterministic attention requested."""
    import torch
    return (torch.are_deterministic_algorithms_enabled()
            or os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1') == '0')


def apply_to_args(args):
    """Adjust performance-only Megatron options after parsing and validation."""
    if flag('NO_RECOMPUTE'):
        # Full activation recomputation only saves memory. Without it the 4B quick-start recipe keeps up to ~62 GiB of
        # activations per micro-batch; on 80 GB GPUs combine it with OV2_OPT_MEM_BUDGET_GIB.
        args.recompute_granularity = None
        args.recompute_method = None
        args.recompute_num_layers = None
    if flag('ATTN_FUSED') and not deterministic_mode():
        # Transformer Engine's cuDNN fused attention (F16_arbitrary_seqlen) instead of flash-attn 2: same bf16
        # inputs/outputs and fp32 softmax statistics, a different kernel. cuDNN 9.9 has no deterministic backward for
        # the packed (thd) layout on sm100, so deterministic runs keep the recipe's flash backend.
        from megatron.core.transformer.enums import AttnBackend
        args.attention_backend = AttnBackend.fused
    if flag('GC_SYNC'):
        # Automatic garbage collection fires at different times on different ranks (a full collection stalls a
        # rank's host thread for ~0.3 s) and every other rank then waits at the gradient reduce-scatter. Collect
        # manually instead, at the same iteration on every rank.
        args.manual_gc = True
        args.manual_gc_interval = int(os.environ.get('OV2_OPT_GC_INTERVAL', '100'))
    if flag('DDP_AVG'):
        # The 1/DP gradient scaling inside the reduce-scatter (NCCL AVG) instead of a separate pass over the fp32
        # gradient buffer; with a power-of-two data-parallel size, pre-scaling and averaging round alike.
        args.ddp_average_in_collective = True
    if flag('CE_FUSION'):
        # Megatron's fused vocab-parallel cross-entropy: the same math in fewer passes over the fp32 logits.
        args.cross_entropy_loss_fusion = True
    return args
