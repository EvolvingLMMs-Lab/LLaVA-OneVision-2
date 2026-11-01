"""Bit-exact fused replacement of llava_onevision2_layer_spec.apply_rotary_pos_emb_vision (thd path).

Upstream (per layer, for q and for k, t: [T, H, D] bf16, freqs: [T, 1, 1, D/2] fp32):
    tf = t.float(); c = cos(freqs).repeat(.., 2); s = sin(freqs).repeat(.., 2)
    out = ((tf * c) + (rotate_half(tf) * s)).to(bf16),  rotate_half: r[2i] = -x[2i+1], r[2i+1] = x[2i]
That is ~10 eager kernels over fp32 copies of q and k in the forward and as many in the backward.

Here: cos/sin of the (layer-invariant) frequency table are computed once per table with the same PyTorch kernels, and
one Triton kernel per direction does the rest with the same rounding points (each product rounded to fp32, then the
sum, then round-to-nearest-even to bf16; FMA contraction disabled). The backward reproduces autograd's result exactly:
grad_tf = (R + 0.0) + g*c with R[2i] = g[2i+1]*s[2i+1], R[2i+1] = -(g[2i]*s[2i]) (the slice backwards are zero tensors
into which the rotated gradient is copied; the odd-slice buffer arrives first and the even one is added to it, which
turns -0 into +0, and the mul-by-cos gradient is added last), then rounded to bf16.
"""
import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(x_ptr, c_ptr, s_ptr, o_ptr, n_rows, H, sxt, sxh, HALF: tl.constexpr, D: tl.constexpr,
                     BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, D)
    rmask = rows < n_rows
    tok = rows // H
    head = rows % H
    m = rmask[:, None]
    xb = x_ptr + tok[:, None] * sxt + head[:, None] * sxh
    x = tl.load(xb + d[None, :], mask=m, other=0.0).to(tl.float32)
    xp = tl.load(xb + (d ^ 1)[None, :], mask=m, other=0.0).to(tl.float32)
    cs = tok[:, None] * HALF + (d % HALF)[None, :]
    c = tl.load(c_ptr + cs, mask=m, other=0.0)
    s = tl.load(s_ptr + cs, mask=m, other=0.0)
    even = (d % 2 == 0)[None, :]
    r = tl.where(even, xp * -1.0, xp)  # torch.neg (Triton's unary minus is 0 - x, which maps -0 to +0)
    a = x * c
    b = r * s
    out = a + b
    tl.store(o_ptr + rows[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=m)


@triton.jit
def _rope_bwd_kernel(g_ptr, c_ptr, s_ptr, o_ptr, n_rows, H, sgt, sgh, HALF: tl.constexpr, D: tl.constexpr,
                     BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, D)
    rmask = rows < n_rows
    tok = rows // H
    head = rows % H
    m = rmask[:, None]
    gb = g_ptr + tok[:, None] * sgt + head[:, None] * sgh
    g = tl.load(gb + d[None, :], mask=m, other=0.0).to(tl.float32)
    gp = tl.load(gb + (d ^ 1)[None, :], mask=m, other=0.0).to(tl.float32)
    cs = tok[:, None] * HALF + (d % HALF)[None, :]
    csp = tok[:, None] * HALF + ((d ^ 1) % HALF)[None, :]
    c = tl.load(c_ptr + cs, mask=m, other=0.0)
    sp = tl.load(s_ptr + csp, mask=m, other=0.0)
    even = (d % 2 == 0)[None, :]
    gr = gp * sp                       # grad of the rotated tensor at the partner position, rounded
    rr = tl.where(even, gr, gr * -1.0)  # copied into the slice-backward zeros (neg for the odd slice)
    rr = rr + 0.0                      # zeros(odd slice) + zeros(even slice): -0 -> +0
    a = g * c                          # grad through tf * cos, rounded
    out = rr + a
    tl.store(o_ptr + rows[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=m)


@triton.jit
def _rope_fwd_kernel2(x_ptr, c_ptr, s_ptr, o_ptr, n_rows, H, sxt, sxh, HALF: tl.constexpr, D: tl.constexpr,
                      BLOCK_R: tl.constexpr):
    """Same arithmetic as _rope_fwd_kernel; one coalesced load per row, even/odd lanes by tl.split / tl.join."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, D)
    m = (rows < n_rows)[:, None]
    tok = rows // H
    head = rows % H
    x = tl.load(x_ptr + tok[:, None] * sxt + head[:, None] * sxh + d[None, :], mask=m, other=0.0).to(tl.float32)
    cs = tok[:, None] * HALF + (d % HALF)[None, :]
    c = tl.load(c_ptr + cs, mask=m, other=0.0)
    s = tl.load(s_ptr + cs, mask=m, other=0.0)
    xe, xo = tl.split(tl.reshape(x, [BLOCK_R, D // 2, 2]))
    ce, co = tl.split(tl.reshape(c, [BLOCK_R, D // 2, 2]))
    se, so = tl.split(tl.reshape(s, [BLOCK_R, D // 2, 2]))
    oe = xe * ce + (xo * -1.0) * se
    oo = xo * co + xe * so
    out = tl.reshape(tl.join(oe, oo), [BLOCK_R, D])
    tl.store(o_ptr + rows[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=m)


@triton.jit
def _rope_bwd_kernel2(g_ptr, c_ptr, s_ptr, o_ptr, n_rows, H, sgt, sgh, HALF: tl.constexpr, D: tl.constexpr,
                      BLOCK_R: tl.constexpr):
    """Same arithmetic as _rope_bwd_kernel; one coalesced load per row, even/odd lanes by tl.split / tl.join."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, D)
    m = (rows < n_rows)[:, None]
    tok = rows // H
    head = rows % H
    g = tl.load(g_ptr + tok[:, None] * sgt + head[:, None] * sgh + d[None, :], mask=m, other=0.0).to(tl.float32)
    cs = tok[:, None] * HALF + (d % HALF)[None, :]
    c = tl.load(c_ptr + cs, mask=m, other=0.0)
    s = tl.load(s_ptr + cs, mask=m, other=0.0)
    ge, go = tl.split(tl.reshape(g, [BLOCK_R, D // 2, 2]))
    ce, co = tl.split(tl.reshape(c, [BLOCK_R, D // 2, 2]))
    se, so = tl.split(tl.reshape(s, [BLOCK_R, D // 2, 2]))
    re = (go * so) + 0.0
    ro = ((ge * se) * -1.0) + 0.0
    oe = re + ge * ce
    oo = ro + go * co
    out = tl.reshape(tl.join(oe, oo), [BLOCK_R, D])
    tl.store(o_ptr + rows[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=m)


_BLOCK_R = 32
_TABLE = {}


def _cos_sin(freqs):
    """cos / sin of the [T, 1, 1, D/2] frequency table, computed once per table object (PyTorch kernels)."""
    key = id(freqs)
    hit = _TABLE.get(key)
    if hit is not None and hit[0]() is freqs and hit[1] == freqs._version:
        return hit[2], hit[3]
    f = freqs.reshape(freqs.shape[0], -1)
    c = f.cos().float().contiguous()
    s = f.sin().float().contiguous()
    for k in [k for k, v in _TABLE.items() if v[0]() is None]:
        del _TABLE[k]
    _TABLE[key] = (weakref.ref(freqs), freqs._version, c, s)
    return c, s


def _launch(kernel, x, c, s, out):
    T, H, D = x.shape
    n_rows = T * H
    grid = (triton.cdiv(n_rows, _BLOCK_R),)
    kernel[grid](x, c, s, out, n_rows, H, x.stride(0), x.stride(1), HALF=D // 2, D=D, BLOCK_R=_BLOCK_R,
                 num_warps=8, enable_fp_fusion=False)
    return out


class _FusedVisionRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, c, s):
        out = torch.empty(t.shape, dtype=torch.bfloat16, device=t.device)
        _launch(_rope_fwd_kernel2, t, c, s, out)
        ctx.save_for_backward(c, s)
        return out

    @staticmethod
    def backward(ctx, g):
        c, s = ctx.saved_tensors
        if g.stride(2) != 1:
            g = g.contiguous()
        out = torch.empty(g.shape, dtype=torch.bfloat16, device=g.device)
        _launch(_rope_bwd_kernel2, g, c, s, out)
        return out, None, None


def supported(t, freqs, cu_seqlens):
    return (
        cu_seqlens is not None
        and t.dim() == 3
        and t.dtype == torch.bfloat16
        and t.is_cuda
        and t.stride(2) == 1
        and t.shape[2] % 2 == 0
        and freqs.dtype == torch.float32
        and freqs.dim() == 4
        and freqs.shape[0] == t.shape[0]
        and freqs.shape[1] == 1
        and freqs.shape[2] == 1
        and 2 * freqs.shape[3] == t.shape[2]
        and triton.next_power_of_2(t.shape[2]) == t.shape[2]
    )


def fused_apply_rotary_pos_emb_vision(t, freqs):
    c, s = _cos_sin(freqs)
    return _FusedVisionRope.apply(t, c, s)
