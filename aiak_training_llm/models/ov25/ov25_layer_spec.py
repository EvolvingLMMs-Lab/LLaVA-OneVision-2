# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 layer spec (ported to AIAK-Megatron / mcore 0.12.0rc0)."""

from copy import deepcopy

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec

from .rmsnorm import Qwen3NextRMSNorm
from .gated_deltanet import GatedDeltaNet
from .gated_attention import Qwen3NextSelfAttention
from .ov25_rope import apply_rotary_pos_emb_ov25_llm

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _get_dense_mlp_module_spec():
    """Get module spec for dense (non-MoE) MLP.

    Uses TEColumnParallelLinear (non-fused) instead of TELayerNormColumnParallelLinear
    because Qwen3.5 uses zero-centered RMSNorm (Qwen3NextRMSNorm) whose weight format
    (initialized to zeros, computes (1+w)*norm(x)) differs from TE's fused layernorm
    (initialized to ones). The layernorm is handled separately by pre_mlp_layernorm.
    """
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )


def get_local_layer_specs(config, layer_specs, vp_stage=None):
    """Slice the full per-layer spec list for the local pipeline stage.

    ADAPTATION: our fork's ``get_num_layers_to_build`` / ``get_transformer_layer_offset``
    take only ``config`` (no ``vp_stage``). If anything is unavailable we fall back to
    the FULL list (parent will handle PP later).
    """
    try:
        num_layers_to_build = get_num_layers_to_build(config)
        if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
            from megatron.core.transformer.enums import LayerType
            local_layer_specs = [
                layer_specs[layer_id]
                for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                    layer_type=LayerType.decoder, vp_stage=vp_stage
                )
            ]
        else:
            offset = get_transformer_layer_offset(config)
            local_layer_specs = layer_specs[offset:offset + num_layers_to_build]
    except Exception:
        # Fallback: build the full list; parent handles PP slicing later.
        local_layer_specs = layer_specs
    return local_layer_specs


def get_qwen3_5_transformer_layer_spec(config, vp_stage=None):
    """Helper function to get module spec for Qwen3_5.

    Returns ``(block_spec, mtp_block_spec)`` where ``mtp_block_spec`` is always ``None``
    (MTP intentionally skipped for this port).
    """
    if not HAVE_TE:
        raise ImportError(
            "Qwen3_5 layer spec requires Transformer Engine. "
            "Please install it with: pip install transformer-engine"
        )

    layer_norm_impl = Qwen3NextRMSNorm
    num_moe_experts = getattr(config, "num_moe_experts", None)
    is_dense = num_moe_experts is None

    base_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_moe_experts,
        moe_grouped_gemm=getattr(config, "moe_grouped_gemm", False),
        qk_layernorm=getattr(config, "qk_layernorm", False),
        multi_latent_attention=getattr(config, "multi_latent_attention", False),
        moe_use_legacy_grouped_gemm=getattr(config, "moe_use_legacy_grouped_gemm", False),
    )

    if is_dense:
        mlp = _get_dense_mlp_module_spec()
    else:
        mlp = get_moe_module_spec(
            use_te=HAVE_TE,
            num_experts=num_moe_experts,
            moe_grouped_gemm=getattr(config, "moe_grouped_gemm", False),
            moe_use_legacy_grouped_gemm=getattr(config, "moe_use_legacy_grouped_gemm", False),
        )
        # Qwen3.5-MoE (qwen3_5_moe) scales the shared expert by sigmoid(shared_expert_gate(x)).
        # The stock mcore spec builds SharedExpertMLP with gate=False; turn it on unless
        # explicitly disabled with OV25_SHARED_EXPERT_GATE=0.
        if __import__("os").environ.get("OV25_SHARED_EXPERT_GATE", "1") != "0":
            mlp.submodules.shared_experts.params = {"gate": True}

    full_attention_interval = getattr(config, "full_attention_interval", 4)

    # Build per-layer specs based on layer_types pattern
    layer_types = [
        'full_attention' if (i + 1) % full_attention_interval == 0 else 'linear_attention'
        for i in range(config.num_layers)
    ]

    layer_specs = []
    for layer_type in layer_types:
        layer_spec = deepcopy(base_layer_spec)
        mlp_spec = deepcopy(mlp)

        if layer_type == 'linear_attention':
            layer_spec.submodules.self_attention.module = GatedDeltaNet
        elif layer_type == 'full_attention':
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
            layer_spec.submodules.self_attention.params = {"attn_mask_type": AttnMaskType.causal}
            # Cos/sin-cached RoPE wrapper for the unfused path. When
            # apply_rope_fusion=True (default for Qwen3.5 LLM backbone),
            # the wrapper just delegates to the mcore fused kernel; when
            # False, it caches cos/sin across the 2 * num_full_attention
            # layer calls per step. Mirrors the ViT-side pattern in commit
            # 1b9d69b.
            layer_spec.submodules.self_attention.submodules.apply_rotary_fn = (
                apply_rotary_pos_emb_ov25_llm
            )

        # Replace all layernorms with Qwen3NextRMSNorm (zero-centered)
        layer_spec.submodules.input_layernorm = layer_norm_impl
        # For dense models, pre_mlp_layernorm must be explicitly set because the dense MLP
        # uses TEColumnParallelLinear (non-fused), so the layernorm is not inside linear_fc1.
        # For MoE models, the base spec already provides a non-IdentityOp pre_mlp_layernorm.
        if is_dense:
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        elif hasattr(layer_spec.submodules, 'pre_mlp_layernorm') and \
                layer_spec.submodules.pre_mlp_layernorm is not IdentityOp:
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
            layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
            layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl

        layer_spec.submodules.mlp = mlp_spec
        layer_specs.append(layer_spec)

    # Slice layer specs for pipeline parallelism
    local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    # MTP intentionally skipped for this port.
    mtp_block_spec = None

    return block_spec, mtp_block_spec
