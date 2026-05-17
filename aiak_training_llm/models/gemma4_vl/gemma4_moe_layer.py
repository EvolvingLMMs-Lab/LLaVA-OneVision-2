"""Gemma4-VL MoE layer (subclass of Megatron's :class:`MoELayer`).

Two reasons we subclass :class:`MoELayer`:

1. **Router rebind.** Megatron's :class:`MoELayer.__init__` hard-codes
   ``self.router = TopKRouter(config)`` and ignores anything you might pass
   through :class:`MoESubmodules`. Gemma4 needs its own router with a
   parameter-free RMSNorm + ``scale[H]`` + ``per_expert_scale[E]``
   (see ``gemma4_router.py``), so we rebind ``self.router`` after the parent
   has finished wiring the experts / dispatcher / shared experts.

2. **Router input contract.** HF Gemma4 routes on the *raw post-attention
   residual* but feeds the experts a separately-normed view of that same
   residual (see ``modeling_gemma4.py`` lines 1394-1405)::

       hidden_states_flat = residual.reshape(-1, H)              # raw
       _, top_w, top_i  = self.router(hidden_states_flat)        # router on raw
       hidden_states_2  = pre_feedforward_layernorm_2(hidden_states_flat)
       hidden_states_2  = self.experts(hidden_states_2, top_i, top_w)

   Megatron's :class:`MoELayer.forward` instead feeds the same tensor to both
   ``self.router`` and ``self.token_dispatcher.token_permutation`` — making it
   impossible to reproduce HF's split-input behavior from a wrapper that lives
   *outside* :class:`MoELayer`. We therefore own ``pre_feedforward_layernorm_2``
   here and override :meth:`forward` to thread raw vs. normed correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import utils

from aiak_training_llm.models.gemma4_vl.gemma4_router import Gemma4Router


@dataclass
class Gemma4MoESubmodules(MoESubmodules):
    """Extends :class:`MoESubmodules` with Gemma4's expert pre-norm slot."""

    pre_feedforward_layernorm_2: Union[ModuleSpec, type] = None


class Gemma4MoELayer(MoELayer):
    """:class:`MoELayer` with Gemma4 router + split router/expert input norm."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Gemma4MoESubmodules = None,
        layer_number: int | None = None,
    ) -> None:
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)
        self.router = Gemma4Router(config=self.config)
        if layer_number is not None:
            self.router.set_layer_number(layer_number)

        if submodules is None or submodules.pre_feedforward_layernorm_2 is None:
            raise ValueError(
                "Gemma4MoELayer requires submodules.pre_feedforward_layernorm_2 to be set; "
                "the layer spec must wire this norm in."
            )
        self.pre_feedforward_layernorm_2 = build_module(
            submodules.pre_feedforward_layernorm_2,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism "
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)

            expert_input = self.pre_feedforward_layernorm_2(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                expert_input, probs, routing_map
            )

            if self.config.enable_mem_monitor:
                utils.monitor_max_memory_usage()
                utils.monitor_max_dispatcher_tokens(tokens_per_expert)
                if self.config.mem_monitor_log is not None:
                    utils.write_monitor_data_to_file(
                        self.config.mem_monitor_log, self.config.print_mem_monitor_interval
                    )

            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)

            if self.use_shared_expert and not self.shared_expert_overlap:
                raise RuntimeError(
                    "Gemma4MoELayer was built with shared experts enabled; HF Gemma4 has none. "
                    "Check that moe_shared_expert_intermediate_size is unset (None)."
                )
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


__all__ = ["Gemma4MoELayer", "Gemma4MoESubmodules"]
