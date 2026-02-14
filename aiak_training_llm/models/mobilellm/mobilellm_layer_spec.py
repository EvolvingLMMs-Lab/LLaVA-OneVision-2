"""MobileLLM layer specification - Standard LLaMA architecture"""

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from aiak_training_llm.models.dispatch import multiacc_modules


def _is_te_min_version(version: str) -> bool:
    """Check if Transformer Engine version is at least the specified version."""
    try:
        import transformer_engine
        from packaging import version as pkg_version
        te_version = getattr(transformer_engine, '__version__', '0.0.0')
        return pkg_version.parse(te_version) >= pkg_version.parse(version)
    except (ImportError, AttributeError):
        return False


def get_mobilellm_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Get MobileLLM layer specification using Transformer Engine modules.
    
    MobileLLM-R1-140M architecture:
    - Grouped Query Attention (9 heads, 3 KV heads)
    - SwiGLU activation (gated_linear_unit=True)
    - RMSNorm
    - QK LayerNorm (use_qk_norm: true in official config.json)
    
    Args:
        config: Transformer configuration with MobileLLM parameters
        
    Returns:
        ModuleSpec for MobileLLM transformer layer
    """
    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = multiacc_modules.TENorm if _is_te_min_version("1.9.0") else multiacc_modules.LocalNorm
    
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
                    q_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )
