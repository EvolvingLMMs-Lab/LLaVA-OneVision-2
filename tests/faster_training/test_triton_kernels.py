"""Small-shape checks of the opt-in Triton kernels against their PyTorch / apex counterparts (one GPU, seconds).

    PYTHONPATH=aiak_megatron:. python -m pytest tests/faster_training -q
"""
import pytest
import torch

pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')


def _bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_fused_vision_rope_is_bitwise(monkeypatch):
    monkeypatch.delenv('OV2_OPT_FUSED_VISION_ROPE', raising=False)          # reference = the default path
    layer_spec = pytest.importorskip('aiak_training_llm.models.llava_onevision2.llava_onevision2_layer_spec')
    from aiak_training_llm.models.llava_onevision2 import fused_vision_rope
    torch.manual_seed(0)
    T, H, D = 777, 4, 64
    t = torch.randn(T, H, D, device='cuda', dtype=torch.bfloat16)
    freqs = torch.randn(T, 1, 1, D // 2, device='cuda') * 3
    cu = torch.tensor([0, 300, T], dtype=torch.int32, device='cuda')
    assert fused_vision_rope.supported(t, freqs, cu)
    g = torch.randn(T, H, D, device='cuda', dtype=torch.bfloat16)

    ref_in = t.clone().requires_grad_()
    ref = layer_spec.apply_rotary_pos_emb_vision(ref_in, freqs, None, cu_seqlens=cu)   # switch off: upstream path
    ref.backward(g)
    out_in = t.clone().requires_grad_()
    out = fused_vision_rope.fused_apply_rotary_pos_emb_vision(out_in, freqs)
    out.backward(g)
    assert _bitwise(out, ref)
    assert _bitwise(out_in.grad, ref_in.grad)


def test_bitexact_rmsnorm_matches_apex():
    apex_norm = pytest.importorskip('apex.normalization.fused_layer_norm')
    from aiak_training_llm.models.custom.common import bitexact_rmsnorm
    torch.manual_seed(0)
    T, H, N = 513, 8, bitexact_rmsnorm.N2
    base = torch.randn(T, H, 2 * N, device='cuda', dtype=torch.bfloat16)
    x = base[..., :N]                                    # strided view, as q/k sliced from the qkv projection
    norm = apex_norm.FusedRMSNorm(N, eps=1e-6).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(N, device='cuda') + 0.5)
    w = norm.weight.detach().clone().requires_grad_()
    g = torch.randn(T, H, N, device='cuda', dtype=torch.bfloat16)

    xr = x.detach().clone().requires_grad_()
    ref = norm(xr)
    ref.backward(g)
    xo = x.detach().requires_grad_()
    out = bitexact_rmsnorm.bitexact_rms_norm(xo, w, 1e-6)
    out.backward(g)
    assert _bitwise(out, ref)
    assert _bitwise(w.grad, norm.weight.grad)
    if torch.cuda.get_device_capability()[0] == 10:
        assert _bitwise(xo.grad.contiguous(), xr.grad)
    else:   # before sm100, apex and this kernel round a few input-gradient elements differently (one bf16 ulp)
        torch.testing.assert_close(xo.grad.contiguous(), xr.grad, rtol=2 ** -7, atol=2 ** -10)


def test_bias_add_is_bitwise():
    from aiak_training_llm.models.custom.common import bias_add
    torch.manual_seed(0)
    x = torch.randn(1000, 1, 4096, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
    g = torch.randn_like(x)
    xr, br = x.clone().requires_grad_(), b.clone().requires_grad_()
    ref = xr + br
    ref.backward(g)
    xo, bo = x.clone().requires_grad_(), b.clone().requires_grad_()
    out = bias_add.bias_add(xo, bo)
    out.backward(g)
    assert _bitwise(out, ref) and _bitwise(xo.grad, xr.grad) and _bitwise(bo.grad, br.grad)


def test_fused_swiglu_matches_reference():
    from aiak_training_llm.models.custom.common import fused_swiglu
    torch.manual_seed(0)
    x = torch.randn(300, 1, 2 * 1536, device='cuda', dtype=torch.bfloat16)
    g = torch.randn(300, 1, 1536, device='cuda', dtype=torch.bfloat16)

    xr = x.clone().requires_grad_()
    a, b = torch.chunk(xr.float(), 2, -1)
    ref = (torch.nn.functional.silu(a) * b).to(torch.bfloat16)
    ref.backward(g)
    xo = x.clone().requires_grad_()
    out = fused_swiglu.fused_swiglu(xo)
    out.backward(g)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(xo.grad, xr.grad)


def test_triton_cross_entropy_matches_reference():
    from aiak_training_llm.models.custom.common import triton_cross_entropy
    torch.manual_seed(0)
    S, V = 64, 151936
    logits = (torch.randn(S, 1, V, device='cuda') * 4).to(torch.bfloat16)
    target = torch.randint(0, V, (S, 1), device='cuda')
    target[::7] = -100                                   # unlabelled positions
    grad_loss = torch.rand(S, 1, device='cuda')

    lr = logits.clone().requires_grad_()
    x = lr.float()
    m = x.max(-1, keepdim=True).values.detach()           # as Megatron's cross-entropy: no gradient through the max
    valid = (target >= 0) & (target < V)
    xt = torch.gather(x, -1, target.clamp(0, V - 1).unsqueeze(-1)).squeeze(-1)
    ref = torch.log(torch.exp(x - m).sum(-1)) - torch.where(valid, xt - m.squeeze(-1), torch.zeros_like(xt))
    ref.backward(grad_loss)
    lo = logits.clone().requires_grad_()
    out = triton_cross_entropy.cross_entropy(lo, target)
    out.backward(grad_loss)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lo.grad, lr.grad)
