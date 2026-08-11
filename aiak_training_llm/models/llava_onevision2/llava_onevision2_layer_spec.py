import torch
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from aiak_training_llm.models.llava_onevision1_5.llava_onevision1_5_layer_spec import (
    get_adapeter_layer_with_spec,
    get_qwen_layer_with_te_spec,
)


def rotate_half(x):
    """
    Perform interleaved rotation for rotary position embedding.

    Rotates half of the dimensions in an interleaved pattern to match the source model's implementation.
    Transforms (x1, x2, x3, x4) -> (-x2, x1, -x4, x3).

    Args:
        x (torch.Tensor): Input tensor to rotate.

    Returns:
        torch.Tensor: Rotated tensor with the same shape as input.
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


# Cos/sin cache for ViT RoPE. The freqs tensor is the same Python object across
# all 24 ViT layers and both forward/backward calls within a single encoder
# step, so a cache hit skips ~96 redundant cos/sin/repeat computations per
# step. Keyed by (data_ptr, shape, cu_seqlens, t.dtype) to be robust against
# pointer reuse after free; bounded to one entry by clearing before insert so
# no encoder-side cleanup is required.
_ROPE_CACHE = {}


def apply_rotary_pos_emb_vision(t, freqs, config, cu_seqlens=None, rotary_interleaved=False):
    """
    Apply rotary position embedding to vision tokens.

    Args:
        t (torch.Tensor): Input tensor to apply rotary embedding.
        freqs (torch. Tensor): Frequency tensor for rotary embedding.
        config:  Model configuration.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for packed sequences.
        rotary_interleaved (bool, optional): Whether to use interleaved rotation pattern.

    Returns:
        torch.Tensor: Tensor with rotary position embedding applied.
    """
    key = (freqs.data_ptr(), tuple(freqs.shape), cu_seqlens is not None, t.dtype)
    cs = _ROPE_CACHE.get(key)
    if cs is None:
        f = freqs.squeeze(1) if cu_seqlens is not None else freqs
        rep = (1, 1, 2) if cu_seqlens is not None else (1, 1, 1, 2)
        # Match t.dtype for cos/sin (bf16) instead of fp32: saves the
        # bf16<->fp32 autocast round-trip per call. End-result quantization
        # error is bf16 either way (the original code also cast back to bf16).
        cos_ = f.cos().to(t.dtype).repeat(*rep)
        sin_ = f.sin().to(t.dtype).repeat(*rep)
        _ROPE_CACHE.clear()
        _ROPE_CACHE[key] = (cos_, sin_)
        cs = (cos_, sin_)
    cos_, sin_ = cs
    return (t * cos_) + (rotate_half(t) * sin_)


def get_vision_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb_vision,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
