"""The vision MLP's unfused fc1 bias add (Megatron ActivationFuncModule without bias_activation_fusion:
intermediate + bias, then F.gelu) as a vectorised Triton kernel (switch OV2_OPT_FAST_BIAS_ADD).

PyTorch runs the broadcast [tokens, 1, 4096] + [4096] bf16 add in its generic, non-vectorised elementwise kernel
(~0.24 ms at 25k tokens, ~1.7 TB/s). The kernel here computes the same thing (each sum in fp32, rounded to bf16, as
PyTorch's bf16 add) with 128-bit accesses; the backward is exactly AddBackward's: the gradient passes through to the
intermediate and is summed to the bias shape with Tensor.sum_to_size. Bitwise identical; GELU and its backward stay
PyTorch's.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, b_ptr, y_ptr, R, N, BR: tl.constexpr, BC: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < N)
    offs = rows[:, None].to(tl.int64) * N + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (x + b[None, :]).to(y_ptr.dtype.element_ty), mask=mask)


_BR, _BC, _NW = 4, 1024, 4


class _BiasAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        N = x.shape[-1]
        x2 = x.reshape(-1, N)
        R = x2.shape[0]
        y = torch.empty_like(x)
        _add_kernel[(triton.cdiv(R, _BR), triton.cdiv(N, _BC))](x2, bias, y, R, N, BR=_BR, BC=_BC, num_warps=_NW)
        ctx.bias_shape = bias.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        return dy, dy.sum_to_size(ctx.bias_shape)


def supported(x, bias):
    return (bias is not None and x.is_cuda and x.dtype == bias.dtype and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous() and bias.dim() == 1 and bias.is_contiguous() and bias.shape[0] == x.shape[-1])


def bias_add(x, bias):
    return _BiasAdd.apply(x, bias)


_INSTALLED = False


def install():
    """Megatron's MLP activation module, unfused path with a bias: the add runs in the Triton kernel."""
    global _INSTALLED
    if _INSTALLED:
        return
    from megatron.core.transformer.mlp import ActivationFuncModule as cls
    original = cls.forward

    def forward(self, intermediate, bias):
        if not self.bias_activation_fusion and bias is not None and supported(intermediate, bias):
            return self.bias_activation_func_impl(bias_add(intermediate, bias))
        return original(self, intermediate, bias)

    cls.forward = forward
    _INSTALLED = True
