import torch
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention, TELayerNormColumnParallelLinear, TELinear,
    TERowParallelLinear)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import (SelfAttention,
                                                 SelfAttentionSubmodules)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer, TransformerLayerSubmodules)

from aiak_training_llm.models.custom.common.local_norm import LocalNorm
from aiak_training_llm.models.dispatch import multiacc_modules
from aiak_training_llm.models.qwen_vl.qwen2_vl_layer_spec import \
    get_adapeter_layer_with_spec
from aiak_training_llm.utils import is_te_min_version

from .rice_vision_model import apply_rotary_pos_emb_vision


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


def _get_mlp_module_spec(
    num_experts: int=None,
    moe_grouped_gemm: bool=False
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    if num_experts is None:
        # Dense MLP w/ TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
                linear_fc2=multiacc_modules.TERowParallelLinear,
                bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
            ),
        )

    # moe mlp
    if moe_grouped_gemm:
        # use TEGroupedLinear
        assert multiacc_modules.TEColumnParallelGroupedLinear is not None
        expert_module = TEGroupedMLP
        linear_fc1 = multiacc_modules.TEColumnParallelGroupedLinear
        linear_fc2 = multiacc_modules.TERowParallelGroupedLinear
    else:
        expert_module = SequentialMLP
        linear_fc1 = multiacc_modules.TEColumnParallelLinear
        linear_fc2 = multiacc_modules.TERowParallelLinear

    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=ModuleSpec(
                module=expert_module,
                submodules=MLPSubmodules(
                    linear_fc1=linear_fc1,
                    linear_fc2=linear_fc2,
                    bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
                )
            )
        )
    )


def get_qwen_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    # To simplify the code, temporarily remove the compatibility with MoE/MLA.
    # If there is a new version in the future, add and test it separately.
    assert not config.multi_latent_attention, "Not supporting multi-latent attention for Qwen model yet."

    mlp = _get_mlp_module_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    # qk_norm = multiacc_modules.TENorm if is_te_min_version("1.9.0") else multiacc_modules.LocalNorm
    qk_norm = multiacc_modules.LocalNorm

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TELayerNormColumnParallelLinear,
                    core_attention=multiacc_modules.DotProductAttention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    q_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=(
                multiacc_modules.TENorm if config.num_moe_experts else IdentityOp
            ),
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        )
    )

# def get_qwen_layer_with_spec(qk_layernorm: bool = False) -> ModuleSpec:
#     """
#     Use this spec for an implementation using transformer, local or multi-accel engine
#     """
#     return ModuleSpec(
#         module=TransformerLayer,
#         submodules=TransformerLayerSubmodules(
#             input_layernorm=IdentityOp,
#             self_attention=ModuleSpec(
#                 module=SelfAttention,
#                 params={"attn_mask_type": AttnMaskType.causal},
#                 submodules=SelfAttentionSubmodules(
#                     linear_qkv=TELayerNormColumnParallelLinear,
#                     core_attention=TEDotProductAttention,
#                     linear_proj=TERowParallelLinear,
#                     q_layernorm=LocalNorm if qk_layernorm else IdentityOp,
#                     k_layernorm=LocalNorm if qk_layernorm else IdentityOp,
#                     apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
#                 ),
#             ),
#             self_attn_bda=get_bias_dropout_add,
#             pre_mlp_layernorm=IdentityOp,
#             mlp=ModuleSpec(
#                 module=MLP,
#                 submodules=MLPSubmodules(
#                     linear_fc1=TELayerNormColumnParallelLinear,
#                     linear_fc2=TERowParallelLinear,
#                 ),
#             ),
#             mlp_bda=get_bias_dropout_add,
#             sharded_state_dict_keys_map={
#                 'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
#                 'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
#             },
#         ),
#     )