"""SwiGLU of the language model's MLP in Triton kernels (switch OV2_OPT_FUSED_SWIGLU).

Megatron's bias_swiglu_impl without bias (the recipe: bias_activation_fusion, gated linear unit, no bias): forward
y = silu(a) * b with [a, b] = chunk(x, 2, -1), backward dx = cat(g * sigmoid(a) * (1 + a * (1 - sigmoid(a))) * b,
g * silu(a)); both are jit_fuser (torch.compile) functions, i.e. fp32 math on the bf16 inputs and one rounding to bf16
at the store. The same expressions here, as two kernels over [rows, F] tiles (the backward writes both halves of dx
directly instead of Inductor's concatenation kernel). Elementwise, no reduction; bitwise identical to the jit version
on this model.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(x_ptr, y_ptr, R, F, BR: tl.constexpr, BC: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < F)
    xo = rows[:, None].to(tl.int64) * (2 * F) + cols[None, :]
    a = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + xo + F, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(a)
    y = a * s * b
    tl.store(y_ptr + rows[:, None].to(tl.int64) * F + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(g_ptr, x_ptr, dx_ptr, R, F, BR: tl.constexpr, BC: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < F)
    xo = rows[:, None].to(tl.int64) * (2 * F) + cols[None, :]
    a = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + xo + F, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + rows[:, None].to(tl.int64) * F + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(a)
    da = g * s * (1.0 + a * (1.0 - s)) * b
    db = g * (a * s)
    tl.store(dx_ptr + xo, da.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dx_ptr + xo + F, db.to(dx_ptr.dtype.element_ty), mask=mask)


_BR, _BC = 4, 1024


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        F2 = x.shape[-1]
        F = F2 // 2
        x2 = x.reshape(-1, F2)
        R = x2.shape[0]
        y = torch.empty((*x.shape[:-1], F), dtype=x.dtype, device=x.device)
        _fwd_kernel[(triton.cdiv(R, _BR), triton.cdiv(F, _BC))](x2, y, R, F, BR=_BR, BC=_BC, num_warps=4)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        F2 = x.shape[-1]
        F = F2 // 2
        x2 = x.reshape(-1, F2)
        R = x2.shape[0]
        g = g.contiguous()
        dx = torch.empty_like(x)
        _bwd_kernel[(triton.cdiv(R, _BR), triton.cdiv(F, _BC))](g, x2, dx, R, F, BR=_BR, BC=_BC, num_warps=4)
        return dx


def supported(x):
    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.is_contiguous() and x.shape[-1] % 2 == 0


def fused_swiglu(x):
    return _FusedSwiGLU.apply(x)


_INSTALLED = False


def install():
    """Megatron's MLP activation module: the fused (jit) bias-free SwiGLU goes through these kernels."""
    global _INSTALLED
    if _INSTALLED:
        return
    import torch.nn.functional as Fn
    from megatron.core.transformer.mlp import ActivationFuncModule as cls
    original = cls.forward

    def forward(self, intermediate, bias):
        if (self.bias_activation_fusion and self.activation_func is Fn.silu and bias is None
                and getattr(self, 'fused_activation_type', None) == 'swiglu'
                and not self.activation_func_fp8_input_store and supported(intermediate)):
            return fused_swiglu(intermediate)
        return original(self, intermediate, bias)

    cls.forward = forward
    _INSTALLED = True
