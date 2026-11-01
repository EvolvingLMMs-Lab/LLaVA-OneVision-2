"""Per-token cross-entropy of the language model in two Triton kernels (switch OV2_OPT_TRITON_CE, TP 1 only).

The same function and precision as Megatron's vocab_parallel_cross_entropy (and its fused variant) for one tensor-
parallel rank: logits read as bf16 and computed in fp32; per row the maximum, then sum(exp(x - max)), the loss
log(sum) - (x[target] - max) (a target outside the vocabulary, e.g. -100, contributes no predicted logit), and in the
backward the gradient (exp(x - max) / sum - [v == target]) * grad_loss, cast to the logits' dtype. Without the fp32 copy
of the [tokens, vocab] logits and the separate elementwise passes: the forward reads every logit twice (the second read
mostly from L2), the backward reads it once and writes the gradient once. Reductions in a fixed order (deterministic).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 4096


@triton.jit
def _ce_fwd_kernel(logits_ptr, stride_row, target_ptr, loss_ptr, max_ptr, sum_ptr, V,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    offs = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], float('-inf'), tl.float32)
    for start in range(0, V, BLOCK):
        cols = start + offs
        x = tl.load(base + cols, mask=cols < V, other=float('-inf')).to(tl.float32)
        m = tl.maximum(m, x)
    row_max = tl.max(m, axis=0)
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, V, BLOCK):
        cols = start + offs
        x = tl.load(base + cols, mask=cols < V, other=float('-inf')).to(tl.float32)
        acc += libdevice.exp(x - row_max)
    s = tl.sum(acc, axis=0)
    t = tl.load(target_ptr + row)
    valid = (t >= 0) & (t < V)
    xt = tl.load(base + tl.where(valid, t, 0)).to(tl.float32)
    pred = tl.where(valid, xt - row_max, 0.0)
    tl.store(loss_ptr + row, libdevice.log(s) - pred)
    tl.store(max_ptr + row, row_max)
    tl.store(sum_ptr + row, s)


@triton.jit
def _ce_bwd_kernel(logits_ptr, stride_row, grad_ptr, gstride_row, target_ptr, max_ptr, sum_ptr, gout_ptr, V,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    gbase = grad_ptr + row * gstride_row
    offs = tl.arange(0, BLOCK)
    row_max = tl.load(max_ptr + row)
    s = tl.load(sum_ptr + row)
    g = tl.load(gout_ptr + row)
    t = tl.load(target_ptr + row)
    valid = (t >= 0) & (t < V)
    for start in range(0, V, BLOCK):
        cols = start + offs
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        p = libdevice.exp(x - row_max) / s
        p = tl.where(valid & (cols == t), p - 1.0, p)
        tl.store(gbase + cols, (p * g).to(grad_ptr.dtype.element_ty), mask=mask)


class _TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        shape = target.shape
        V = logits.shape[-1]
        x = logits.reshape(-1, V)
        if x.stride(-1) != 1:
            x = x.contiguous()
        t = target.reshape(-1).contiguous()
        n = x.shape[0]
        loss = torch.empty(n, dtype=torch.float32, device=x.device)
        row_max = torch.empty(n, dtype=torch.float32, device=x.device)
        row_sum = torch.empty(n, dtype=torch.float32, device=x.device)
        _ce_fwd_kernel[(n,)](x, x.stride(0), t, loss, row_max, row_sum, V, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(x, t, row_max, row_sum)
        ctx.logits_shape = logits.shape
        return loss.view(shape)

    @staticmethod
    def backward(ctx, grad_loss):
        x, t, row_max, row_sum = ctx.saved_tensors
        V = x.shape[-1]
        n = x.shape[0]
        g = grad_loss.reshape(-1).float().contiguous()
        grad = torch.empty_like(x)
        _ce_bwd_kernel[(n,)](x, x.stride(0), grad, grad.stride(0), t, row_max, row_sum, g, V, BLOCK=_BLOCK,
                             num_warps=8)
        return grad.view(ctx.logits_shape), None


def cross_entropy(logits, target):
    """logits [s, b, V] (bf16), target [s, b] (int64) -> per-token loss [s, b] (fp32)."""
    return _TritonCrossEntropy.apply(logits, target)


_INSTALLED = False


def install():
    """Route LanguageModule.compute_language_model_loss through the Triton kernels (tensor-parallel size 1)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from megatron.core import parallel_state
    from megatron.core.models.common.language_module.language_module import LanguageModule
    original = LanguageModule.compute_language_model_loss

    def compute_language_model_loss(self, labels, logits):
        if parallel_state.get_tensor_model_parallel_world_size() != 1 or logits.dtype not in (torch.bfloat16,
                                                                                               torch.float16):
            return original(self, labels, logits)
        labels = labels.transpose(0, 1).contiguous()
        loss = cross_entropy(logits, labels)
        return loss.transpose(0, 1).contiguous()

    LanguageModule.compute_language_model_loss = compute_language_model_loss
    _INSTALLED = True
