"""Gemma4-VL layer specs.

Three exports:
  - get_vision_layer_with_spec: returns ``None`` — Gemma4 vision tower is a
    plain ``nn.Module`` stack (HF-style verbatim port), not a Megatron
    ``ModuleSpec``-driven ``TransformerBlock``. Kept as a no-op so
    ``gemma4_vl_provider`` can call it uniformly with the LlavaOnevision2
    provider; the returned ``None`` is forwarded to ``Gemma4VL.__init__`` and
    ignored there.
  - get_adapter_layer_with_spec: same story — Gemma4 adapter is a plain
    ``nn.Module`` (RMSNorm + Linear, see ``gemma4_adapter.py``).
  - get_gemma4_layer_with_te_spec: Gemma4 LLM layer wiring
        Gemma4TransformerLayer + Gemma4SelfAttention + Gemma4ParallelDenseMoE
"""

from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from aiak_training_llm.models.dispatch import multiacc_modules
from aiak_training_llm.models.gemma4_vl.gemma4_attention import Gemma4SelfAttention
from aiak_training_llm.models.gemma4_vl.gemma4_mlp import (
    Gemma4ParallelDenseMoE,
    Gemma4ParallelDenseMoESubmodules,
)
from aiak_training_llm.models.gemma4_vl.gemma4_moe_layer import (
    Gemma4MoELayer,
    Gemma4MoESubmodules,
)
from aiak_training_llm.models.gemma4_vl.gemma4_transformer_layer import (
    Gemma4TransformerLayer,
    Gemma4TransformerLayerSubmodules,
)
from aiak_training_llm.utils import is_te_min_version


__all__ = [
    "get_vision_layer_with_spec",
    "get_adapter_layer_with_spec",
    "get_gemma4_layer_with_te_spec",
]


def get_vision_layer_with_spec():
    """Gemma4 vision tower is not Megatron-spec-driven; return ``None``.

    The provider calls this for API parity with the LlavaOnevision2 provider
    and forwards the result to ``Gemma4VL.__init__``, which builds the vision
    tower via ``Gemma4VisionTower(...)`` directly and ignores this argument.
    """
    return None


def get_adapter_layer_with_spec():
    """Gemma4 adapter is not Megatron-spec-driven; return ``None``.

    Same rationale as ``get_vision_layer_with_spec``. The Gemma4 adapter is
    constructed as a plain ``nn.Module`` (see ``Gemma4Adapter``) inside
    ``Gemma4VL.__init__``.
    """
    return None


def _gemma4_norm_module() -> type:
    return multiacc_modules.TENorm if is_te_min_version("1.9.0") else multiacc_modules.LocalNorm


def _gemma4_dense_mlp_spec() -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=multiacc_modules.TEColumnParallelLinear,
            linear_fc2=multiacc_modules.TERowParallelLinear,
            bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
        ),
    )


def _gemma4_moe_spec(moe_grouped_gemm: bool) -> ModuleSpec:
    if moe_grouped_gemm:
        assert multiacc_modules.TEColumnParallelGroupedLinear is not None
        expert_module = TEGroupedMLP
        linear_fc1 = multiacc_modules.TEColumnParallelGroupedLinear
        linear_fc2 = multiacc_modules.TERowParallelGroupedLinear
    else:
        expert_module = SequentialMLP
        linear_fc1 = multiacc_modules.TEColumnParallelLinear
        linear_fc2 = multiacc_modules.TERowParallelLinear

    experts_spec = ModuleSpec(
        module=expert_module,
        submodules=MLPSubmodules(
            linear_fc1=linear_fc1,
            linear_fc2=linear_fc2,
            bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
        ),
    )
    # HF Gemma4 MoE has NO shared experts; the dense MLP is implemented as a
    # peer branch in Gemma4ParallelDenseMoE (see plan v5 §531 R5). We must
    # leave moe_shared_expert_intermediate_size unset (None) in the config —
    # any other value, including 0, makes MoELayer's ``use_shared_expert``
    # flag True (it checks ``is not None``) and triggers a build of the
    # shared-experts branch which we did not wire (would crash).
    return ModuleSpec(
        module=Gemma4MoELayer,
        submodules=Gemma4MoESubmodules(
            experts=experts_spec,
            pre_feedforward_layernorm_2=_gemma4_norm_module(),
        ),
    )


def _gemma4_parallel_dense_moe_spec(moe_grouped_gemm: bool) -> ModuleSpec:
    norm = _gemma4_norm_module()
    return ModuleSpec(
        module=Gemma4ParallelDenseMoE,
        submodules=Gemma4ParallelDenseMoESubmodules(
            dense=_gemma4_dense_mlp_spec(),
            moe=_gemma4_moe_spec(moe_grouped_gemm=moe_grouped_gemm),
            pre_feedforward_layernorm=norm,
            post_feedforward_layernorm_1=norm,
            post_feedforward_layernorm_2=norm,
        ),
    )


def _gemma4_self_attention_submodules(qk_norm) -> SelfAttentionSubmodules:
    return SelfAttentionSubmodules(
        linear_qkv=multiacc_modules.TELayerNormColumnParallelLinear,
        core_attention=multiacc_modules.DotProductAttention,
        linear_proj=multiacc_modules.TERowParallelLinear,
        q_layernorm=qk_norm,
        k_layernorm=qk_norm,
        apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
    )


def get_gemma4_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    assert not config.multi_latent_attention, (
        "Gemma4-VL does not use multi-latent attention; got "
        "config.multi_latent_attention=True"
    )

    qk_norm = _gemma4_norm_module()
    mlp = _gemma4_parallel_dense_moe_spec(moe_grouped_gemm=config.moe_grouped_gemm)

    # ``pre_mlp_layernorm = IdentityOp`` because Gemma4ParallelDenseMoE +
    # Gemma4MoELayer between them own all four pre-merge RMSNorms internally
    # and must receive the raw post-attention residual (router math depends
    # on this — see gemma4_mlp.py module docstring).
    return ModuleSpec(
        module=Gemma4TransformerLayer,
        submodules=Gemma4TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Gemma4SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=_gemma4_self_attention_submodules(qk_norm=qk_norm),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
            post_feedforward_layernorm=_gemma4_norm_module(),
        ),
    )
