"""Language-model q/k RMSNorm and rotary embedding in one Triton kernel each way (switch OV2_OPT_FUSED_QK_NORM_ROPE).

Upstream (Qwen3 attention with packed sequences): query/key -> q_layernorm/k_layernorm (apex FusedRMSNorm over the
head dim, fp32 statistics: y = bf16(gamma * bf16(x * rstd))) -> Transformer Engine's fused RoPE for the thd layout
(non-interleaved halves, position = token index within its sub-sample, cos/sin of the fp32 frequency table, fp32 math,
bf16 out). Here both in one pass with the same rounding points (the bf16 norm output feeds the rotation in
registers); the backward applies TE's transposed rotation (rounded to bf16 like TE's backward output), then the
RMSNorm backward (apex's formulas: dx = (N * gamma * dy - xhat * sum(gamma * dy * xhat)) * rstd / N, dgamma =
sum over rows of dy * xhat, fp32), the weight gradient summed per block in fp32 and then over blocks. Same precision;
the reductions are ordered differently from apex (not bitwise). One read and one write of q/k instead of two each,
and no separate strided copies.
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fwd_kernel(x_ptr, sx_t, sx_g, sx_h, HPG, w_ptr, cos_ptr, sin_ptr, out_ptr, rstd_ptr, R, H, eps,
                BR: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    t = rows // H
    h = rows - t * H
    hg = h // HPG
    base = t.to(tl.int64) * sx_t + hg.to(tl.int64) * sx_g + (h - hg * HPG).to(tl.int64) * sx_h
    c = tl.arange(0, HALF)
    m2 = rmask[:, None]
    x1 = tl.load(x_ptr + base[:, None] + c[None, :], mask=m2, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base[:, None] + HALF + c[None, :], mask=m2, other=0.0).to(tl.float32)
    ms = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / (2 * HALF)
    rstd = libdevice.rsqrt(ms + eps)
    w1 = tl.load(w_ptr + c).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + c).to(tl.float32)
    y1 = ((x1 * rstd[:, None]).to(tl.bfloat16).to(tl.float32) * w1[None, :]).to(tl.bfloat16).to(tl.float32)
    y2 = ((x2 * rstd[:, None]).to(tl.bfloat16).to(tl.float32) * w2[None, :]).to(tl.bfloat16).to(tl.float32)
    tb = t.to(tl.int64) * (2 * HALF)
    cs1 = tl.load(cos_ptr + tb[:, None] + c[None, :], mask=m2, other=0.0)
    cs2 = tl.load(cos_ptr + tb[:, None] + HALF + c[None, :], mask=m2, other=0.0)
    sn1 = tl.load(sin_ptr + tb[:, None] + c[None, :], mask=m2, other=0.0)
    sn2 = tl.load(sin_ptr + tb[:, None] + HALF + c[None, :], mask=m2, other=0.0)
    o1 = y1 * cs1 - y2 * sn1
    o2 = y2 * cs2 + y1 * sn2
    obase = rows.to(tl.int64) * (2 * HALF)
    tl.store(out_ptr + obase[:, None] + c[None, :], o1.to(out_ptr.dtype.element_ty), mask=m2)
    tl.store(out_ptr + obase[:, None] + HALF + c[None, :], o2.to(out_ptr.dtype.element_ty), mask=m2)
    tl.store(rstd_ptr + rows, rstd, mask=rmask)


@triton.jit
def _bwd_kernel(do_ptr, x_ptr, sx_t, sx_g, sx_h, HPG, w_ptr, cos_ptr, sin_ptr, rstd_ptr, dx_ptr, sd_t, sd_g, sd_h,
                part_ptr, R, H, BR: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    t = rows // H
    h = rows - t * H
    hg = h // HPG
    hs = h - hg * HPG
    base = t.to(tl.int64) * sx_t + hg.to(tl.int64) * sx_g + hs.to(tl.int64) * sx_h
    dbase = t.to(tl.int64) * sd_t + hg.to(tl.int64) * sd_g + hs.to(tl.int64) * sd_h
    c = tl.arange(0, HALF)
    m2 = rmask[:, None]
    obase = rows.to(tl.int64) * (2 * HALF)
    do1 = tl.load(do_ptr + obase[:, None] + c[None, :], mask=m2, other=0.0).to(tl.float32)
    do2 = tl.load(do_ptr + obase[:, None] + HALF + c[None, :], mask=m2, other=0.0).to(tl.float32)
    tb = t.to(tl.int64) * (2 * HALF)
    cs1 = tl.load(cos_ptr + tb[:, None] + c[None, :], mask=m2, other=0.0)
    cs2 = tl.load(cos_ptr + tb[:, None] + HALF + c[None, :], mask=m2, other=0.0)
    sn1 = tl.load(sin_ptr + tb[:, None] + c[None, :], mask=m2, other=0.0)
    sn2 = tl.load(sin_ptr + tb[:, None] + HALF + c[None, :], mask=m2, other=0.0)
    dy1 = (do1 * cs1 + do2 * sn2).to(tl.bfloat16).to(tl.float32)
    dy2 = (do2 * cs2 - do1 * sn1).to(tl.bfloat16).to(tl.float32)
    x1 = tl.load(x_ptr + base[:, None] + c[None, :], mask=m2, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base[:, None] + HALF + c[None, :], mask=m2, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    xh1 = x1 * rstd[:, None]
    xh2 = x2 * rstd[:, None]
    w1 = tl.load(w_ptr + c).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + c).to(tl.float32)
    c1 = dy1 * w1[None, :]
    c2 = dy2 * w2[None, :]
    s = tl.sum(c1 * xh1, axis=1) + tl.sum(c2 * xh2, axis=1)
    n = 2 * HALF
    dx1 = (n * c1 - xh1 * s[:, None]) * (rstd[:, None] / n)
    dx2 = (n * c2 - xh2 * s[:, None]) * (rstd[:, None] / n)
    tl.store(dx_ptr + dbase[:, None] + c[None, :], dx1.to(dx_ptr.dtype.element_ty), mask=m2)
    tl.store(dx_ptr + dbase[:, None] + HALF + c[None, :], dx2.to(dx_ptr.dtype.element_ty), mask=m2)
    g1 = tl.sum(tl.where(m2, dy1 * xh1, 0.0), axis=0)
    g2 = tl.sum(tl.where(m2, dy2 * xh2, 0.0), axis=0)
    tl.store(part_ptr + pid.to(tl.int64) * n + c, g1)
    tl.store(part_ptr + pid.to(tl.int64) * n + HALF + c, g2)


_BR = 32
_NW = 4


def _rows_view(x):
    """x [..., H, D] -> (R, H, D, stride over tokens, stride over heads); the leading dims must flatten to tokens."""
    *lead, H, D = x.shape
    T = 1
    for d in lead:
        T *= d
    # strides of the flattened token dim: all leading dims of size 1 except one
    st = None
    for d, s in zip(lead, x.stride()[:-2]):
        if d != 1:
            if st is not None:
                return None
            st = s
    if st is None:
        st = x.stride(-2) * H
    return T, H, D, st, x.stride(-2)


class _FusedQKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, cos, sin):
        T, H, D, st, sh = _rows_view(x)
        R = T * H
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        rstd = torch.empty(R, dtype=torch.float32, device=x.device)
        _fwd_kernel[(triton.cdiv(R, _BR),)](x, st, 0, sh, H, weight, cos, sin, out, rstd, R, H, float(eps),
                                           BR=_BR, HALF=D // 2, num_warps=_NW)
        ctx.save_for_backward(x, weight, cos, sin, rstd)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, cos, sin, rstd = ctx.saved_tensors
        T, H, D, st, sh = _rows_view(x)
        R = T * H
        dout = dout.contiguous()
        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        nb = triton.cdiv(R, _BR)
        part = torch.empty((nb, D), dtype=torch.float32, device=x.device)
        _bwd_kernel[(nb,)](dout, x, st, 0, sh, H, weight, cos, sin, rstd, dx, H * D, 0, D, part, R, H,
                           BR=_BR, HALF=D // 2, num_warps=_NW)
        dw = part.sum(dim=0).to(weight.dtype)
        return dx, dw, None, None, None


class _FusedQKVNormRope(torch.autograd.Function):
    """mixed [T, 1, ng, (np/ng + 2) * hn] (the linear_qkv output viewed per query group) -> q [T, 1, np, hn] (normed,
    rotated), k [T, 1, ng, hn] (normed, rotated), v [T, 1, ng, hn] (copied): Megatron's split + q/k layernorm + RoPE;
    the backward writes dq, dk and dv straight into one gradient of mixed (no concatenation)."""

    @staticmethod
    def forward(ctx, mixed, wq, wk, eps, cos, sin, np_, ng, hn):
        T = mixed.shape[0]
        hpg = np_ // ng
        st, sg = mixed.stride(0), mixed.stride(2)
        q = torch.empty((T, 1, np_, hn), dtype=mixed.dtype, device=mixed.device)
        k = torch.empty((T, 1, ng, hn), dtype=mixed.dtype, device=mixed.device)
        rq = torch.empty(T * np_, dtype=torch.float32, device=mixed.device)
        rk = torch.empty(T * ng, dtype=torch.float32, device=mixed.device)
        _fwd_kernel[(triton.cdiv(T * np_, _BR),)](mixed, st, sg, hn, hpg, wq, cos, sin, q, rq, T * np_, np_,
                                                  float(eps), BR=_BR, HALF=hn // 2, num_warps=_NW)
        kview = mixed[..., hpg * hn:(hpg + 1) * hn]
        _fwd_kernel[(triton.cdiv(T * ng, _BR),)](kview, st, sg, hn, 1, wk, cos, sin, k, rk, T * ng, ng,
                                                 float(eps), BR=_BR, HALF=hn // 2, num_warps=_NW)
        v = mixed[..., (hpg + 1) * hn:].contiguous()
        ctx.save_for_backward(mixed, wq, wk, cos, sin, rq, rk)
        ctx.dims = (np_, ng, hn)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        mixed, wq, wk, cos, sin, rq, rk = ctx.saved_tensors
        np_, ng, hn = ctx.dims
        T = mixed.shape[0]
        hpg = np_ // ng
        st, sg = mixed.stride(0), mixed.stride(2)
        dm = torch.empty(mixed.shape, dtype=mixed.dtype, device=mixed.device)
        dst, dsg = dm.stride(0), dm.stride(2)
        out = []
        for d, w, r, H, h0, n in ((dq, wq, rq, np_, 0, hpg), (dk, wk, rk, ng, hpg * hn, 1)):
            R = T * H
            nb = triton.cdiv(R, _BR)
            part = torch.empty((nb, hn), dtype=torch.float32, device=mixed.device)
            if d is None:
                d = torch.zeros((T, 1, H, hn), dtype=mixed.dtype, device=mixed.device)
            d = d.contiguous()
            _bwd_kernel[(nb,)](d, mixed[..., h0:], st, sg, hn, n, w, cos, sin, r, dm[..., h0:], dst, dsg, hn, part,
                               R, H, BR=_BR, HALF=hn // 2, num_warps=_NW)
            out.append(part.sum(dim=0).to(w.dtype))
        vs = dm[..., (hpg + 1) * hn:]
        if dv is None:
            vs.zero_()
        else:
            vs.copy_(dv)
        return dm, out[0], out[1], None, None, None, None, None, None


def positions(cu_seqlens, total):
    """Position of every token within its sub-sample (cached on the cu_seqlens tensor; no host sync)."""
    pos = getattr(cu_seqlens, '_ov2_positions', None)
    if pos is not None and pos.numel() == total:
        return pos
    cu = cu_seqlens.to(torch.int64)
    starts = torch.repeat_interleave(cu[:-1], cu[1:] - cu[:-1], output_size=total)
    pos = (torch.arange(total, device=cu.device, dtype=torch.int64) - starts).to(torch.int32)
    cu_seqlens._ov2_positions = pos
    return pos


def rope_tables(freqs, cu_seqlens, total):
    """cos / sin of the fp32 frequency table at every token's position, [tokens, D] fp32: computed once per packed
    sequence (cached on the cu_seqlens tensor, keyed by the frequency table) and shared by all layers and q / k."""
    key = (freqs.data_ptr(), tuple(freqs.shape))
    cache = getattr(cu_seqlens, '_ov2_rope_tables', None)
    if cache is not None and cache[0] == key and cache[1].shape[0] == total:
        return cache[1], cache[2]
    pos = positions(cu_seqlens, total).to(torch.int64)
    f = freqs.reshape(freqs.shape[0], -1).index_select(0, pos)
    cos, sin = torch.cos(f).contiguous(), torch.sin(f).contiguous()
    cu_seqlens._ov2_rope_tables = (key, cos, sin)
    return cos, sin


def supported(x, weight, freqs):
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or not x.is_cuda:
        return False
    D = x.shape[-1]
    if D % 2 or D > 256 or (D // 2) & (D // 2 - 1) or weight.numel() != D or x.stride(-1) != 1:
        return False
    if freqs is None or freqs.dtype != torch.float32 or freqs.shape[-1] != D or freqs.stride(-1) != 1:
        return False
    return _rows_view(x) is not None


def fused_qk_norm_rope(x, weight, eps, freqs, cu_seqlens):
    T = 1
    for d in x.shape[:-2]:
        T *= d
    cos, sin = rope_tables(freqs, cu_seqlens, T)
    return _FusedQKNormRope.apply(x, weight, eps, cos, sin)


def _same_tensor(a, b):
    if a is b:
        return True
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.data_ptr() == b.data_ptr()
            and a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride())


def _identity_rope(t, freqs, config=None, cu_seqlens=None, **kwargs):
    return t


_INSTALLED = False


def install():
    """SelfAttention.forward of attention layers with q/k FusedRMSNorm and Megatron's fused thd RoPE: the norms
    apply the rotation themselves (fused kernels) and the separate RoPE call becomes the identity."""
    global _INSTALLED
    if _INSTALLED:
        return
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.transformer.attention import SelfAttention
    from aiak_training_llm.models.custom.common.local_norm import FusedRMSNorm
    original = SelfAttention.forward

    def eligible(self):
        qn, kn = getattr(self, 'q_layernorm', None), getattr(self, 'k_layernorm', None)
        return (isinstance(qn, FusedRMSNorm) and isinstance(kn, FusedRMSNorm) and qn.elementwise_affine
                and kn.elementwise_affine and self.apply_rotary_fn is rope_utils.apply_rotary_pos_emb
                and self.config.apply_rope_fusion and not self.config.multi_latent_attention
                and not self.config.flash_decode)

    def forward(self, hidden_states, attention_mask, attn_mask_type=None, key_value_states=None,
                inference_params=None, rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                attention_bias=None, packed_seq_params=None, sequence_len_offset=None):
        psp = packed_seq_params
        use = (inference_params is None and rotary_pos_emb is not None and psp is not None
               and psp.qkv_format == 'thd' and psp.cu_seqlens_q is not None
               and getattr(psp, 'cu_seqlens_q_padded', None) is None
               and getattr(psp, 'cu_seqlens_kv_padded', None) is None
               and parallel_state.get_context_parallel_world_size() == 1 and eligible(self))
        if use:
            q_emb, k_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb, rotary_pos_emb)
            self.q_layernorm._ov2_rope = (q_emb, psp.cu_seqlens_q)
            self.k_layernorm._ov2_rope = (k_emb, psp.cu_seqlens_kv)
            saved = self.apply_rotary_fn
            self.apply_rotary_fn = _identity_rope
            try:
                return original(self, hidden_states, attention_mask, attn_mask_type=attn_mask_type,
                                key_value_states=key_value_states, inference_params=inference_params,
                                rotary_pos_emb=rotary_pos_emb, rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin, attention_bias=attention_bias,
                                packed_seq_params=packed_seq_params, sequence_len_offset=sequence_len_offset)
            finally:
                self.apply_rotary_fn = saved
                self.q_layernorm._ov2_rope = None
                self.k_layernorm._ov2_rope = None
        return original(self, hidden_states, attention_mask, attn_mask_type=attn_mask_type,
                        key_value_states=key_value_states, inference_params=inference_params,
                        rotary_pos_emb=rotary_pos_emb, rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias, packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset)

    SelfAttention.forward = forward

    if os.environ.get('OV2_OPT_FUSED_QKV_SPLIT', '0') == '1':
        original_qkv = SelfAttention.get_query_key_value_tensors

        def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
            qn, kn = self.q_layernorm, self.k_layernorm
            rq = qn.__dict__.get('_ov2_rope') if qn is not None else None
            rk = kn.__dict__.get('_ov2_rope') if kn is not None else None
            if rq is None or rk is None or self.config.test_mode or rq[0] is not rk[0] \
                    or not _same_tensor(rq[1], rk[1]) or qn.eps != kn.eps:
                return original_qkv(self, hidden_states, key_value_states)
            mixed, _ = self.linear_qkv(hidden_states)
            np_ = self.num_attention_heads_per_partition
            ng = self.num_query_groups_per_partition
            hn = self.hidden_size_per_attention_head
            mixed = mixed.view(*mixed.size()[:-1], ng, (np_ // ng + 2) * hn)
            if (mixed.dim() != 4 or mixed.shape[1] != 1 or mixed.stride(-1) != 1 or mixed.stride(2) != mixed.shape[-1]
                    or not supported(mixed[..., :hn], qn.weight, rq[0])):
                raise RuntimeError('fused qkv split + q/k norm + RoPE: unsupported layout')
            cos, sin = rope_tables(rq[0], rq[1], mixed.shape[0])
            q, k, v = _FusedQKVNormRope.apply(mixed, qn.weight, kn.weight, qn.eps, cos, sin, np_, ng, hn)
            return q, k, v

        SelfAttention.get_query_key_value_tensors = get_query_key_value_tensors
    _INSTALLED = True
