"""Bit-exact Triton replacement of apex's FusedRMSNorm (affine, bf16, hidden 128) used for Qwen3's q/k norms.

apex (csrc/layer_norm_cuda_kernel.cu, built with --use_fast_math) launches one 128-thread block per row with only
the first warp holding data when the row has 128 elements, and computes the weight gradient in a two-stage
reduction over 16 row partitions with very few blocks (0.45 ms per call on the q norm here). These kernels
reproduce apex's arithmetic exactly -- the same per-lane partial sums (4 consecutive elements, FMA-contracted), the
same shuffle trees, the same rounding points, rsqrt.approx -- with a different work distribution:

forward  : lane j: s_j = fma(x3,x3, fma(x2,x2, fma(x1,x1, x0*x0))), x = row[4j:4j+4]; lane 0's rotate-shuffle tree
           (adjacent pairs); sigma2 = s * 2^-7; invvar = rsqrt.approx(sigma2 + eps);
           y = bf16(gamma * bf16(invvar * x)).
grad in  : lane j: S_j = fma(((dy*g)*x), invvar, ...) over its 4 elements from 0; xor-butterfly over lanes (identical
           on every lane), + 0 + 0 from the idle warps; dx = bf16((fma(-(x*invvar), S, (128*dy)*clamp(g))) * (invvar*2^-7)).
grad w   : 16 partitions of ceil(ceil(n1/16)/16)*16 rows; per partition, row offset r (0..15) and column c, a
           sequential chain over 16-row segments: b = (dy*x)*invvar, then b = fma(dy*x, invvar, b); per partition
           A_y = (((0+b_y)+b_{y+4})+b_{y+8})+b_{y+12}, part = (A_0+A_2)+(A_1+A_3); then
           grad = ((S0+S4)+(S2+S6))+((S1+S5)+(S3+S7)), S_y = (0+part_{2y})+part_{2y+1}, rounded to bf16.
Each chain gets its own thread here (coalesced row loads, unrolled segments) instead of apex's 64 blocks.
"""
import torch
import triton
import triton.language as tl

N2 = 128
PART_SIZE = 16


@triton.jit
def _rsqrt_approx(x):
    return tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _row_addr(rows, H, sA, sB):
    return (rows // H).to(tl.int64) * sA + (rows % H).to(tl.int64) * sB


@triton.jit
def _fwd_kernel(x_ptr, w_ptr, y_ptr, inv_ptr, n1, H, sA, sB, eps, R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < n1
    base = _row_addr(rows, H, sA, sB)
    lane = tl.arange(0, 32)
    m = rmask[:, None]
    p = x_ptr + base[:, None] + 4 * lane[None, :]
    x0 = tl.load(p + 0, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(p + 1, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(p + 2, mask=m, other=0.0).to(tl.float32)
    x3 = tl.load(p + 3, mask=m, other=0.0).to(tl.float32)
    s = x0 * x0
    s = tl.fma(x1, x1, s)
    s = tl.fma(x2, x2, s)
    s = tl.fma(x3, x3, s)
    # lane 0 of the rotate-shuffle reduction: pairwise tree over adjacent lanes
    s = tl.sum(tl.reshape(s, [R, 16, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [R, 8, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [R, 4, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [R, 2, 2]), axis=2)
    s = tl.sum(s, axis=1)
    sigma2 = s * 0.0078125
    inv = _rsqrt_approx(sigma2 + eps)
    w = w_ptr + 4 * lane
    o = y_ptr + rows.to(tl.int64)[:, None] * 128 + 4 * lane[None, :]
    for k in tl.static_range(4):
        xk = tl.load(p + k, mask=m, other=0.0).to(tl.float32)
        gk = tl.load(w + k).to(tl.float32)
        yk = (inv[:, None] * xk).to(tl.bfloat16).to(tl.float32)
        tl.store(o + k, (gk[None, :] * yk).to(tl.bfloat16), mask=m)
    tl.store(inv_ptr + rows, inv, mask=rmask)


@triton.jit
def _bwd_input_kernel(dy_ptr, x_ptr, w_ptr, inv_ptr, dx_ptr, n1, H, sA, sB, kmin, R: tl.constexpr,
                      VARIANT: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < n1
    base = _row_addr(rows, H, sA, sB)
    lane = tl.arange(0, 32)
    m = rmask[:, None]
    px = x_ptr + base[:, None] + 4 * lane[None, :]
    pdy = dy_ptr + rows.to(tl.int64)[:, None] * 128 + 4 * lane[None, :]
    inv = tl.load(inv_ptr + rows, mask=rmask, other=1.0)
    s = tl.zeros([R, 32], dtype=tl.float32)
    for k in tl.static_range(4):
        xk = tl.load(px + k, mask=m, other=0.0).to(tl.float32)
        dk = tl.load(pdy + k, mask=m, other=0.0).to(tl.float32)
        gk = tl.load(w_ptr + 4 * lane + k).to(tl.float32)
        t = (dk * gk[None, :]) * xk
        s = tl.fma(t, inv[:, None], s)
    # xor butterfly (16, 8, 4, 2, 1): the same value on every lane
    s = tl.sum(tl.reshape(s, [R, 2, 16]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 2]), axis=1)
    s = tl.sum(s, axis=1)
    S = (s + 0.0) + 0.0
    term1 = inv * 0.0078125
    pdx = dx_ptr + rows.to(tl.int64)[:, None] * 128 + 4 * lane[None, :]
    for k in tl.static_range(4):
        xk = tl.load(px + k, mask=m, other=0.0).to(tl.float32)
        dk = tl.load(pdy + k, mask=m, other=0.0).to(tl.float32)
        g = tl.load(w_ptr + 4 * lane + k).to(tl.float32)
        # clamp_by_magnitude(gamma, eps) in bf16 (kmin = bf16(eps))
        g = tl.where(g >= 0, tl.where(g < kmin, kmin, g), tl.where(g > -kmin, -kmin, g))
        f0 = (128.0 * dk) * g[None, :]
        t1 = xk * inv[:, None]
        if VARIANT == 0:
            f = tl.fma(t1 * -1.0, S[:, None], f0)
        elif VARIANT == 1:
            f = tl.fma(128.0 * dk, g[None, :], (t1 * S[:, None]) * -1.0)
        else:
            f = f0 - t1 * S[:, None]
        f = f * term1[:, None]
        tl.store(pdx + k, f.to(tl.bfloat16), mask=m)


@triton.jit
def _bwd_wchain_kernel(dy_ptr, x_ptr, inv_ptr, b_ptr, n1, H, sA, sB, segs_per_part, U: tl.constexpr):
    """b[p, r, c]: apex's per-thread accumulation chain over the 16-row segments of partition p, row offset r."""
    p = tl.program_id(0)
    r = tl.program_id(1)
    c = tl.arange(0, 128)
    i1_beg = p * segs_per_part * 16
    i1_end = tl.minimum(i1_beg + segs_per_part * 16, n1)
    acc = tl.zeros([128], dtype=tl.float32)
    # first segment: plain products (or 0 when the row lies beyond the partition)
    row = i1_beg + r
    ok = row < i1_end
    base = _row_addr(row, H, sA, sB)
    xv = tl.load(x_ptr + base + c, mask=ok & (c < 128), other=0.0).to(tl.float32)
    dv = tl.load(dy_ptr + row.to(tl.int64) * 128 + c, mask=ok & (c < 128), other=0.0).to(tl.float32)
    iv = tl.load(inv_ptr + row, mask=ok, other=0.0)
    acc = tl.where(ok, (dv * xv) * iv, 0.0)
    for s0 in range(1, segs_per_part, U):
        for u in tl.static_range(U):
            seg = s0 + u
            row = i1_beg + seg * 16 + r
            ok = (row < i1_end) & (seg < segs_per_part)
            base = _row_addr(row, H, sA, sB)
            xv = tl.load(x_ptr + base + c, mask=ok & (c < 128), other=0.0).to(tl.float32)
            dv = tl.load(dy_ptr + row.to(tl.int64) * 128 + c, mask=ok & (c < 128), other=0.0).to(tl.float32)
            iv = tl.load(inv_ptr + row, mask=ok, other=0.0)
            acc = tl.where(ok, tl.fma(dv * xv, iv, acc), acc)
    tl.store(b_ptr + (p * 16 + r) * 128 + c, acc)


@triton.jit
def _chain_sum(pb, y: tl.constexpr):
    # acc2 = sum over k of warp_buf2[y + 4k] (from 0, sequential)
    a = 0.0 + tl.load(pb + y * 128)
    a = a + tl.load(pb + (y + 4) * 128)
    a = a + tl.load(pb + (y + 8) * 128)
    a = a + tl.load(pb + (y + 12) * 128)
    return a


@triton.jit
def _part(b_ptr, p: tl.constexpr, c):
    # warp rows 0,1 += rows 2,3; part = row0 + row1
    pb = b_ptr + p * 2048 + c
    return (_chain_sum(pb, 0) + _chain_sum(pb, 2)) + (_chain_sum(pb, 1) + _chain_sum(pb, 3))


@triton.jit
def _warp_sum(b_ptr, y: tl.constexpr, c):
    # cuComputeGradGammaBeta: each of 8 warps sums 2 partitions sequentially from 0
    return (0.0 + _part(b_ptr, 2 * y, c)) + _part(b_ptr, 2 * y + 1, c)


@triton.jit
def _bwd_wreduce_kernel(b_ptr, dw_ptr):
    """apex's in-block reduction of the 16 chains per partition, then cuComputeGradGammaBeta over partitions."""
    c = tl.arange(0, 128)
    g = (((_warp_sum(b_ptr, 0, c) + _warp_sum(b_ptr, 4, c)) + (_warp_sum(b_ptr, 2, c) + _warp_sum(b_ptr, 6, c)))
         + ((_warp_sum(b_ptr, 1, c) + _warp_sum(b_ptr, 5, c)) + (_warp_sum(b_ptr, 3, c) + _warp_sum(b_ptr, 7, c))))
    tl.store(dw_ptr + c, g.to(tl.bfloat16))


@triton.jit
def _split4(v, R: tl.constexpr):
    """[R, 128] -> four [R, 32] tensors x_k[:, j] = v[:, 4j + k] (pure data movement)."""
    v = tl.reshape(v, [R, 32, 2, 2])
    a0, a1 = tl.split(v)
    x0, x2 = tl.split(a0)
    x1, x3 = tl.split(a1)
    return x0, x1, x2, x3


@triton.jit
def _fwd_kernel2(x_ptr, w_ptr, y_ptr, inv_ptr, n1, H, sA, sB, eps, R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < n1
    base = _row_addr(rows, H, sA, sB)
    cols = tl.arange(0, 128)
    m = rmask[:, None]
    x = tl.load(x_ptr + base[:, None] + cols[None, :], mask=m, other=0.0).to(tl.float32)
    x0, x1, x2, x3 = _split4(x, R)
    s = x0 * x0
    s = tl.fma(x1, x1, s)
    s = tl.fma(x2, x2, s)
    s = tl.fma(x3, x3, s)
    s = tl.sum(tl.reshape(s, [R, 16, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [R, 8, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [R, 4, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [R, 2, 2]), axis=2)
    s = tl.sum(s, axis=1)
    inv = _rsqrt_approx(s * 0.0078125 + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = (inv[:, None] * x).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + rows.to(tl.int64)[:, None] * 128 + cols[None, :], (w[None, :] * y).to(tl.bfloat16), mask=m)
    tl.store(inv_ptr + rows, inv, mask=rmask)


@triton.jit
def _bwd_input_kernel2(dy_ptr, x_ptr, w_ptr, inv_ptr, dx_ptr, n1, H, sA, sB, kmin, R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < n1
    base = _row_addr(rows, H, sA, sB)
    cols = tl.arange(0, 128)
    m = rmask[:, None]
    x = tl.load(x_ptr + base[:, None] + cols[None, :], mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + rows.to(tl.int64)[:, None] * 128 + cols[None, :], mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=rmask, other=1.0)
    t = (dy * w[None, :]) * x
    t0, t1, t2, t3 = _split4(t, R)
    s = tl.fma(t0, inv[:, None], tl.zeros([R, 32], dtype=tl.float32))
    s = tl.fma(t1, inv[:, None], s)
    s = tl.fma(t2, inv[:, None], s)
    s = tl.fma(t3, inv[:, None], s)
    s = tl.sum(tl.reshape(s, [R, 2, 16]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [R, 2, 2]), axis=1)
    s = tl.sum(s, axis=1)
    S = (s + 0.0) + 0.0
    g = tl.where(w >= 0, tl.where(w < kmin, kmin, w), tl.where(w > -kmin, -kmin, w))
    f0 = (128.0 * dy) * g[None, :]
    xi = x * inv[:, None]
    f = tl.fma(xi * -1.0, S[:, None], f0) * (inv * 0.0078125)[:, None]
    tl.store(dx_ptr + rows.to(tl.int64)[:, None] * 128 + cols[None, :], f.to(tl.bfloat16), mask=m)


@triton.jit
def _bwd_wchain_tile_kernel(dy_ptr, x_ptr, inv_ptr, b_ptr, n1, H, sA, sB, segs_per_part, CB: tl.constexpr,
                            U: tl.constexpr, NS: tl.constexpr):
    """The same chains as _bwd_wchain_kernel (b[p, r, c]); one program per partition and CB columns, all 16 row
    offsets of a segment loaded as one tile, software-pipelined over segments (the chains stay sequential)."""
    p = tl.program_id(0)
    cb = tl.program_id(1)
    r = tl.arange(0, 16)
    c = cb * CB + tl.arange(0, CB)
    i1_beg = p * segs_per_part * 16
    i1_end = tl.minimum(i1_beg + segs_per_part * 16, n1)
    row = i1_beg + r
    ok = row < i1_end
    m = ok[:, None]
    base = _row_addr(row, H, sA, sB)
    xv = tl.load(x_ptr + base[:, None] + c[None, :], mask=m, other=0.0).to(tl.float32)
    dv = tl.load(dy_ptr + row.to(tl.int64)[:, None] * 128 + c[None, :], mask=m, other=0.0).to(tl.float32)
    iv = tl.load(inv_ptr + row, mask=ok, other=0.0)
    acc = tl.where(m, (dv * xv) * iv[:, None], 0.0)
    for s0 in tl.range(1, segs_per_part, U, num_stages=NS):
        for u in tl.static_range(U):
            seg = s0 + u
            row = i1_beg + seg * 16 + r
            ok = (row < i1_end) & (seg < segs_per_part)
            m = ok[:, None]
            base = _row_addr(row, H, sA, sB)
            xv = tl.load(x_ptr + base[:, None] + c[None, :], mask=m, other=0.0).to(tl.float32)
            dv = tl.load(dy_ptr + row.to(tl.int64)[:, None] * 128 + c[None, :], mask=m, other=0.0).to(tl.float32)
            iv = tl.load(inv_ptr + row, mask=ok, other=0.0)
            acc = tl.where(m, tl.fma(dv * xv, iv[:, None], acc), acc)
    tl.store(b_ptr + (p * 16 + r[:, None]) * 128 + c[None, :], acc)


_R = 16
_U = 8


def _geometry(x):
    """(n1, H, sA, sB) such that row r of the [n1, 128] view of x starts at (r // H) * sA + (r % H) * sB."""
    if x.stride(-1) != 1 or x.shape[-1] != N2:
        return None
    if x.is_contiguous():
        return x.numel() // N2, 1, N2, 0
    lead = [(s, st) for s, st in zip(x.shape[:-1], x.stride()[:-1]) if s != 1]
    if len(lead) == 1:
        return lead[0][0], 1, lead[0][1], 0
    if len(lead) == 2:
        (s0, st0), (s1, st1) = lead
        return s0 * s1, s1, st0, st1
    return None


class _BitexactRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, kmin, variant):
        n1, H, sA, sB = _geometry(x)
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        inv = torch.empty(n1, dtype=torch.float32, device=x.device)
        if variant == 0:
            _fwd_kernel2[(triton.cdiv(n1, 16),)](x, weight, y, inv, n1, H, sA, sB, eps, R=16, num_warps=2)
        else:
            _fwd_kernel[(triton.cdiv(n1, _R),)](x, weight, y, inv, n1, H, sA, sB, eps, R=_R, num_warps=4)
        ctx.save_for_backward(x, weight, inv)
        ctx.geom = (n1, H, sA, sB)
        ctx.kmin, ctx.variant = kmin, variant
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, inv = ctx.saved_tensors
        n1, H, sA, sB = ctx.geom
        dy = dy.contiguous()
        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if ctx.variant == 0:
            _bwd_input_kernel2[(triton.cdiv(n1, 16),)](dy, x, weight, inv, dx, n1, H, sA, sB, ctx.kmin, R=16,
                                                       num_warps=2)
        else:
            _bwd_input_kernel[(triton.cdiv(n1, _R),)](dy, x, weight, inv, dx, n1, H, sA, sB, ctx.kmin, R=_R,
                                                      VARIANT=ctx.variant, num_warps=4)
        numsegs = (n1 + 15) // 16
        segs_per_part = (numsegs + PART_SIZE - 1) // PART_SIZE
        b = torch.empty((PART_SIZE, 16, N2), dtype=torch.float32, device=x.device)
        _bwd_wchain_tile_kernel[(PART_SIZE, N2 // 16)](dy, x, inv, b, n1, H, sA, sB, segs_per_part, CB=16, U=_U,
                                                       NS=3, num_warps=4)
        dw = torch.empty_like(weight)
        _bwd_wreduce_kernel[(1,)](b, dw, num_warps=4)
        return dx, dw, None, None, None


def supported(x, weight):
    return (x.is_cuda and x.dtype == torch.bfloat16 and weight is not None and weight.dtype == torch.bfloat16
            and weight.dim() == 1 and weight.shape[0] == N2 and weight.is_contiguous()
            and _geometry(x) is not None and x.numel() > 0)


_KMIN = {}


def bitexact_rms_norm(x, weight, eps, variant=0):
    kmin = _KMIN.get(eps)
    if kmin is None:
        kmin = _KMIN[eps] = float(torch.tensor(eps, dtype=torch.bfloat16).float())   # bf16(eps), apex's clamp
    return _BitexactRMSNorm.apply(x, weight, float(eps), kmin, variant)
