"""Gemma4 parallel dense + MoE FFN block.

Each Gemma4 transformer layer runs TWO FFN branches in parallel from the same
*raw post-attention residual* and sums their outputs. The HF reference
(``transformers/models/gemma4/modeling_gemma4.py``,
``Gemma4TextDecoderLayer.forward`` lines 1374-1409) reads::

    residual = h_after_attn
    hidden_states = pre_feedforward_layernorm(residual)
    hidden_states = self.mlp(hidden_states)                       # dense
    if enable_moe_block:
        hidden_states_1 = post_feedforward_layernorm_1(hidden_states)
        hidden_states_flat = residual.reshape(-1, H)
        _, top_w, top_i = self.router(hidden_states_flat)         # router on RAW residual
        hidden_states_2 = pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(hidden_states_2, top_i, top_w)
        hidden_states_2 = post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = hidden_states_1 + hidden_states_2
    hidden_states = post_feedforward_layernorm(hidden_states)     # done in Gemma4TransformerLayer
    hidden_states = residual + hidden_states                      # done by Megatron mlp_bda

Megatron's standard ``TransformerLayer`` always applies
``self.pre_mlp_layernorm`` and feeds *that* to ``self.mlp``. To preserve the
HF requirement that **dense, experts and router each see** the raw residual
(through their own independent norms — or, for the router, no norm at all
beyond its own internal RMSNorm), the Gemma4 layer spec sets
``pre_mlp_layernorm = IdentityOp`` and this block holds three pre-merge norms:

  - ``pre_feedforward_layernorm``    (in front of dense, with weight)
  - ``post_feedforward_layernorm_1`` (after dense MLP, with weight)
  - ``post_feedforward_layernorm_2`` (after experts, with weight)

The fourth pre-merge norm (``pre_feedforward_layernorm_2``, the experts'
input norm) lives on :class:`~aiak_training_llm.models.gemma4_vl.gemma4_moe_layer.Gemma4MoELayer`
because :class:`MoELayer.forward` feeds the same tensor to both ``self.router``
(which must see RAW residual) and ``self.token_dispatcher`` (which must see
the normed residual). Splitting the two requires an override that lives next
to that forward, not in this composer.

The fifth norm ``post_feedforward_layernorm`` (post-merge, pre-residual-add)
lives on :class:`~aiak_training_llm.models.gemma4_vl.gemma4_transformer_layer.Gemma4TransformerLayer`
because Megatron's ``mlp_bda`` is what performs the residual add and there is
no slot between ``self.mlp(...)`` and ``mlp_bda(...)`` that we can populate
without subclassing the layer.

Forward contract matches Megatron's ``MLP`` / ``MoELayer``::

    forward(hidden_states) -> (output, bias_or_None)

Bias handling: Gemma4 sets ``add_bias_linear=False`` everywhere, so both
branches return ``bias=None``. We assert this and return ``None`` for the
composed bias to refuse silently-wrong fallbacks.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Gemma4ParallelDenseMoESubmodules:
    """Submodule spec for :class:`Gemma4ParallelDenseMoE`."""

    dense: Union[ModuleSpec, type] = None
    moe: Union[ModuleSpec, type] = None
    pre_feedforward_layernorm: Union[ModuleSpec, type] = None
    post_feedforward_layernorm_1: Union[ModuleSpec, type] = None
    post_feedforward_layernorm_2: Union[ModuleSpec, type] = None


class Gemma4ParallelDenseMoE(MegatronModule):
    """Parallel dense + MoE FFN composition for Gemma4.

    Receives the raw post-attention residual (Megatron's ``pre_mlp_layernorm``
    is forced to ``IdentityOp`` in the Gemma4 layer spec). The dense branch
    norms locally; the MoE branch passes the raw residual straight through to
    :class:`Gemma4MoELayer`, which owns the router-vs.-experts norm split.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Gemma4ParallelDenseMoESubmodules,
        layer_number: Optional[int] = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number

        self.pre_feedforward_layernorm = build_module(
            submodules.pre_feedforward_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm_1 = build_module(
            submodules.post_feedforward_layernorm_1,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm_2 = build_module(
            submodules.post_feedforward_layernorm_2,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        self.dense = build_module(submodules.dense, config=config)
        self.moe = build_module(submodules.moe, config=config)
        if layer_number is not None and hasattr(self.moe, "set_layer_number"):
            self.moe.set_layer_number(layer_number)

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        if hasattr(self.moe, "set_layer_number"):
            self.moe.set_layer_number(layer_number)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states

        dense_in = self.pre_feedforward_layernorm(residual)
        dense_out, dense_bias = self.dense(dense_in)
        dense_out = self.post_feedforward_layernorm_1(dense_out)

        moe_out, moe_bias = self.moe(residual)
        moe_out = self.post_feedforward_layernorm_2(moe_out)

        assert dense_bias is None, (
            "Gemma4ParallelDenseMoE: dense branch returned non-None bias; "
            "Gemma4 expects add_bias_linear=False everywhere."
        )
        assert moe_bias is None, (
            "Gemma4ParallelDenseMoE: MoE branch returned non-None mlp_bias; "
            "Gemma4 expects add_bias_linear=False everywhere."
        )
        return dense_out + moe_out, None

    @property
    def use_shared_expert(self) -> bool:
        return getattr(self.moe, "use_shared_expert", False)

    @property
    def shared_expert_overlap(self) -> bool:
        return getattr(self.moe, "shared_expert_overlap", False)

    @property
    def shared_experts(self):
        return getattr(self.moe, "shared_experts", None)

    @property
    def router(self):
        return self.moe.router

    @property
    def experts(self):
        return self.moe.experts

    @property
    def local_experts(self):
        return getattr(self.moe.experts, "local_experts", self.moe.experts)

    @property
    def token_dispatcher(self):
        return self.moe.token_dispatcher
