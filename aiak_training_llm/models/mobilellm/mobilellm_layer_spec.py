"""MobileLLM layer specification - Standard LLaMA architecture"""

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from aiak_training_llm.models.dispatch import multiacc_modules


def get_mobilellm_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Get MobileLLM layer specification using Transformer Engine modules.
    
    MobileLLM-R1-140M uses standard LLaMA architecture:
    - Grouped Query Attention (9 heads, 3 KV heads)
    - SwiGLU activation (gated_linear_unit=True)
    - RMSNorm
    - No QK LayerNorm
    
    Args:
        config: Transformer configuration with MobileLLM parameters
        
    Returns:
        ModuleSpec for MobileLLM transformer layer
    """
    # Standard dense MLP with SwiGLU
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
            linear_fc2=multiacc_modules.TERowParallelLinear,
            bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
        ),
    )

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
                    q_layernorm=IdentityOp,  # MobileLLM doesn't use QK LayerNorm
                    k_layernorm=IdentityOp,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )
