""" Adapters """

import torch
from dataclasses import dataclass
from typing import Union
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""
    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None

class Adapter(MegatronModule):
    """ Adaptor  """
    def __init__(self, 
        config: TransformerConfig,
        submodules: AdapterSubmodules,
        input_size: int,
        output_size: int,
        spatial_merge_size: int = 2
    ) -> None:
        super().__init__(config=config)
        # self.hidden_size = input_size * (spatial_merge_size**2)  # OLD: assumes spatial merging
        # Since input is already [seq_len, input_size], no spatial merging needed
        self.hidden_size = input_size  # Use input_size directly

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )
        
        self.activation_func = config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

    def forward(self, x: torch.Tensor, window_index: torch.LongTensor = None) -> torch.Tensor:
        """ Forward pass."""
        # Input is already [seq_len, input_size], no spatial merging needed
        x = self.layernorm(x)
        x, _ = self.linear_fc1(x)
        x = self.activation_func(x)
        x, _ = self.linear_fc2(x)
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            x = x[reverse_indices, :].contiguous()
        return x