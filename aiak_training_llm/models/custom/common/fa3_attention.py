"""FlashAttention-3 (Hopper sm90 only, optional package flash_attn_3: optional_kernels.load_fa3,
tools/kernels/build_fa3.sh) for the packed (thd) self-attention of the vision encoder and/or the language model
(switch OV2_OPT_FA3_ATTN=all|lm|vision; off by default, experimental: on Hopper it was not faster than FA4's sm90
kernels).

install() wraps Megatron's TEDotProductAttention.forward (on top of whatever forward is installed: Transformer
Engine's, or FA4's from OV2_OPT_FA4_ATTN): a call in scope on packed sequences (qkv_format 'thd', the same cu_seqlens
for q and kv, no attention bias, no context parallelism, no dropout, no sliding window, bf16, head dim 64 or 128) runs
flash_attn_interface.flash_attn_varlen_func instead: the same function (softmax(q k^T * softmax_scale) v inside each
sub-sample, causal for the language model, bidirectional for the vision windows), bf16 inputs and outputs, fp32
softmax statistics and accumulation, the running row maximum updated at every new maximum (as flash-attn 2). Calls
out of scope fall through to the previous forward.

OV2_OPT_FA3_VISION_FROM=<k> (with scope lm): also the vision encoder's attention of layers >= k (1-based).

Deterministic mode (torch.are_deterministic_algorithms_enabled() or NVTE_ALLOW_NONDETERMINISTIC_ALGO=0): the same
kernels with FA3's deterministic backward (deterministic=True), same precision.
"""
import math
import os

import torch

_FA3 = None
_INSTALLED = False
_QKV_RECOMPUTE = os.environ.get('OV2_OPT_QKV_RECOMPUTE', '0') == '1'
_SCOPE = os.environ.get('OV2_OPT_FA3_ATTN', 'all')
if _SCOPE == '1':
    _SCOPE = 'all'
_VISION_FROM = int(os.environ.get('OV2_OPT_FA3_VISION_FROM', '0') or 0)


def _fa3():
    global _FA3
    if _FA3 is None:
        from aiak_training_llm.models.custom.common.optional_kernels import load_fa3
        _FA3 = load_fa3()
    return _FA3


def _deterministic():
    return (torch.are_deterministic_algorithms_enabled()
            or os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1') == '0')


def _same_tensor(a, b):
    if a is b:
        return True
    return (isinstance(b, torch.Tensor) and a.data_ptr() == b.data_ptr() and a.dtype == b.dtype
            and a.shape == b.shape and a.stride() == b.stride())


def _softmax_scale(module, head_dim):
    for name in ('flash_attention', 'fused_attention', 'unfused_attention'):
        scale = getattr(getattr(module, name, None), 'softmax_scale', None)
        if scale is not None:
            return float(scale)
    return 1.0 / math.sqrt(head_dim)


def _max_seqlen(m, cu):
    if m is None or torch.is_tensor(m):
        return int((cu[1:] - cu[:-1]).max())      # device read: only if the caller gave no host value
    return int(m)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    if _SCOPE not in ('all', 'lm', 'vision'):
        raise ValueError(f'OV2_OPT_FA3_ATTN={_SCOPE}: expected all | lm | vision')
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    previous = TEDotProductAttention.forward
    _fa3()                                    # import once at model build time (not inside an iteration)

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
                and query.dim() == 3 and query.dtype == torch.bfloat16
                and query.shape[-1] in (64, 128) and key.shape[-1] == query.shape[-1] == value.shape[-1]
                and attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal, AttnMaskType.no_mask,
                                       AttnMaskType.padding)):
            causal = attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal)
            if causal:
                in_scope = _SCOPE in ('all', 'lm')
            else:
                in_scope = _SCOPE in ('all', 'vision') or (
                    _VISION_FROM > 0 and int(getattr(self, 'layer_number', 0) or 0) >= _VISION_FROM)
            if in_scope:
                cu = psp.cu_seqlens_q
                mq = _max_seqlen(psp.max_seqlen_q, cu)
                out = _fa3().flash_attn_varlen_func(
                    query, key, value, cu, cu, mq, mq, softmax_scale=_softmax_scale(self, query.shape[-1]),
                    causal=causal, deterministic=_deterministic())
                if isinstance(out, tuple):
                    out = out[0]
                if _QKV_RECOMPUTE:
                    # OV2_OPT_QKV_RECOMPUTE: q / k / v freed now, rebuilt before this call's backward
                    from aiak_training_llm.models.custom.common.fused_qk_norm_rope import take_qkv_recompute
                    rq = take_qkv_recompute(query)
                    if rq is not None:
                        rq.discard(out)
                return out.reshape(out.shape[0], -1)
        return previous(self, query, key, value, attention_mask, attn_mask_type, attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params)

    TEDotProductAttention.forward = forward
    _INSTALLED = True
