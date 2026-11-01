"""FlashAttention-4 (CuTe DSL kernels for sm90 / sm100, optional package: optional_kernels.load_fa4) for the packed
(thd) self-attention of the vision encoder and the language model (switch OV2_OPT_FA4_ATTN).

install() wraps Megatron's TEDotProductAttention.forward: a call on packed sequences (qkv_format 'thd', the same
cu_seqlens for q and kv, no attention bias, no context parallelism, no dropout, no sliding window, bf16/fp16, head dim
<= 128) runs flash_attn.cute.flash_attn_varlen_func instead of Transformer Engine's attention: the same function
(softmax(q k^T * softmax_scale) v inside each sub-sample, causal for the language model, bidirectional for the vision
windows), bf16 inputs and outputs, fp32 softmax statistics and accumulation. Any other call keeps TE's path.

  OV2_OPT_FA4_SCOPE=all|lm|vision   which attentions use FA4 (default all)
  OV2_OPT_FA4_VISION_FROM=<k>       with scope lm: also the vision encoder's attention of layers >= k (1-based)
  OV2_OPT_FA4_EXACT_MAX=1           sm100 forward with an exact running row maximum (see _use_exact_running_max)

Deterministic mode (torch.are_deterministic_algorithms_enabled() or NVTE_ALLOW_NONDETERMINISTIC_ALGO=0): the same
kernels with FA4's deterministic backward (deterministic=True: semaphore-ordered dQ / dK / dV accumulation), same
precision.
"""
import hashlib
import os
import sys
import tempfile

import torch

_CUTE = None
_INSTALLED = False
_QKV_RECOMPUTE = os.environ.get('OV2_OPT_QKV_RECOMPUTE', '0') == '1'
# 'all' (default), 'lm' (causal language-model attention only) or 'vision' (bidirectional only)
_SCOPE = os.environ.get('OV2_OPT_FA4_SCOPE', 'all')
_VISION_FROM = int(os.environ.get('OV2_OPT_FA4_VISION_FROM', '0') or 0)


def _cute():
    global _CUTE
    if _CUTE is None:
        from aiak_training_llm.models.custom.common.optional_kernels import load_fa4
        cute = load_fa4()
        if os.environ.get('OV2_OPT_FA4_EXACT_MAX', '0') == '1':
            _use_exact_running_max()
        _CUTE = cute
    return _CUTE


def _use_exact_running_max():
    """FA4's sm100 forward lets the running row max lag by up to 8 log2 units before rescaling (rescale_threshold 8.0
    for 16-bit inputs, flash_fwd_sm100.py): the dominant probability of a peaked row is then 2^x with x in (0, 8]
    instead of exactly 1, its bf16 rounding gives the row's output a gain error, and the backward, which recomputes P
    exactly but takes D = rowsum(dO * O) from that output, turns it into a spurious dS on rows whose true softmax
    gradient is ~0 (on this model's peaked vision-encoder attention rows: step-0 vision gradients off by 40-350%).
    FlashAttention-2 and cuDNN rescale at every new maximum.

    This loads a copy of the same kernel module with rescale_threshold 0 (rescale at every new maximum; nothing else
    changes) under its own name and makes the interface build that class. CuTe DSL compiles kernels from their Python
    source, so the copy is a real file (named by its content hash in the temp directory, written atomically). The
    sm90 forward is not affected. Written for flash-attn-4 4.0.0b36; a version without that exact line is refused."""
    import importlib.util
    import flash_attn.cute.flash_fwd_sm100 as orig
    import flash_attn.cute.interface as interface
    with open(orig.__file__) as f:
        src = f.read()
    old = 'rescale_threshold = 8.0 if const_expr(self.q_dtype.width == 16) else 0.0'
    if src.count(old) != 1:
        raise RuntimeError('OV2_OPT_FA4_EXACT_MAX: unexpected flash_fwd_sm100.py (rescale_threshold line not found '
                           'once); written for flash-attn-4 4.0.0b36')
    src = src.replace(old, 'rescale_threshold = 0.0  # exact running max')
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f'ov2_fa4_flash_fwd_sm100_exact_{digest}.py')
    if not os.path.exists(path):
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            f.write(src)
        os.replace(tmp, path)
    name = 'flash_attn.cute.flash_fwd_sm100_ov2_exact'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.DescaleTensors = orig.DescaleTensors
    interface.FlashAttentionForwardSm100 = mod.FlashAttentionForwardSm100


def _deterministic():
    return (torch.are_deterministic_algorithms_enabled()
            or os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1') == '0')


# FA4 picks its kernel variant (Q tiles per CTA, single-block and persistent-scheduler flags, ...) from the host-side
# max_seqlen hint, and JIT-compiles every new variant at first use. The hint is only an upper bound for the
# scheduler (the lengths are read from cu_seqlens on the device), so it is raised to at least _MIN_HINT: one variant
# per module (vision windows, language model), compiled in the first iteration instead of whenever a packed sequence
# with only tiny sub-samples comes along.
_MIN_HINT = 512


def _hint(max_seqlen):
    if max_seqlen is None or torch.is_tensor(max_seqlen):
        return max_seqlen
    return max(int(max_seqlen), _MIN_HINT)


def _same_tensor(a, b):
    """The same cu_seqlens for q and kv (the same tensor or a view of the same memory; no device read)."""
    if a is b:
        return True
    return (isinstance(b, torch.Tensor) and a.data_ptr() == b.data_ptr() and a.dtype == b.dtype
            and a.shape == b.shape and a.stride() == b.stride())


def _softmax_scale(module, head_dim):
    """The scale Transformer Engine uses (kept by its backend sub-modules; default 1/sqrt(kv_channels))."""
    for name in ('flash_attention', 'fused_attention', 'unfused_attention'):
        sub = getattr(module, name, None)
        scale = getattr(sub, 'softmax_scale', None)
        if scale is not None:
            return float(scale)
    import math
    return 1.0 / math.sqrt(head_dim)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    original = TEDotProductAttention.forward
    _cute()                                   # import once at model build time (not inside an iteration)

    def forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias=None,
                packed_seq_params=None):
        psp = packed_seq_params
        if (psp is not None and psp.qkv_format == 'thd' and attention_bias is None
                and self.config.context_parallel_size == 1
                and self.config.position_embedding_type != 'alibi'
                and getattr(self.config, 'window_size', None) is None
                and float(getattr(self, 'attention_dropout', 0.0) or 0.0) == 0.0
                and getattr(psp, 'cu_seqlens_q_padded', None) is None
                and getattr(psp, 'cu_seqlens_kv_padded', None) is None
                and psp.cu_seqlens_q is not None
                and _same_tensor(psp.cu_seqlens_q, psp.cu_seqlens_kv)
                and query.dim() == 3 and query.dtype in (torch.bfloat16, torch.float16)
                and query.shape[-1] <= 128 and key.shape[-1] == query.shape[-1] == value.shape[-1]
                and attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal, AttnMaskType.no_mask,
                                       AttnMaskType.padding)):
            causal = attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal)
            vision_ok = (_SCOPE != 'lm' or (_VISION_FROM > 0 and int(getattr(self, 'layer_number', 0) or 0) >= _VISION_FROM))
            if (not causal and not vision_ok) or (_SCOPE == 'vision' and causal):
                return original(self, query, key, value, attention_mask, attn_mask_type,
                                attention_bias=attention_bias, packed_seq_params=packed_seq_params)
            cu = psp.cu_seqlens_q
            out = _cute().flash_attn_varlen_func(
                query, key, value, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=_hint(psp.max_seqlen_q), max_seqlen_k=_hint(psp.max_seqlen_kv),
                softmax_scale=_softmax_scale(self, query.shape[-1]), causal=causal, deterministic=_deterministic())
            if isinstance(out, tuple):
                out = out[0]
            if _QKV_RECOMPUTE:
                # OV2_OPT_QKV_RECOMPUTE: q / k / v freed now, rebuilt before this call's backward
                from aiak_training_llm.models.custom.common.fused_qk_norm_rope import take_qkv_recompute
                rq = take_qkv_recompute(query)
                if rq is not None:
                    rq.discard(out)
            return out.reshape(out.shape[0], -1)
        return original(self, query, key, value, attention_mask, attn_mask_type, attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params)

    TEDotProductAttention.forward = forward
    _INSTALLED = True
