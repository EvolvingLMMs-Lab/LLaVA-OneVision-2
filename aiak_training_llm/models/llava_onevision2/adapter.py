import os
from dataclasses import dataclass
from typing import Union

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


def get_adapter_type() -> str:
    """Return the adapter implementation selected by ADAPTER_TYPE."""
    adapter_type = os.environ.get("ADAPTER_TYPE", "LINEAR").upper()
    if adapter_type not in {"LINEAR", "TP_LINEAR"}:
        raise ValueError(
            f"Unsupported ADAPTER_TYPE={adapter_type!r}. Supported values are: LINEAR, TP_LINEAR."
        )
    return adapter_type


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class Adapter(MegatronModule):
    """Adaptor"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AdapterSubmodules,
        input_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__(config=config)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = input_size * (spatial_merge_size**2)
        self.adapter_type = get_adapter_type()

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )

        if self.adapter_type == "TP_LINEAR":
            self.linear_fc1 = build_module(
                submodules.linear_fc1,
                self.hidden_size,
                self.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="fc1",
                skip_weight_param_allocation=False,
            )
        else:
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

        if self.adapter_type == "TP_LINEAR":
            self.linear_fc2 = build_module(
                submodules.linear_fc2,
                self.hidden_size,
                output_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="fc2",
            )
        else:
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

        if self.adapter_type == "TP_LINEAR" and not isinstance(self.linear_fc1, TEColumnParallelLinear):
            raise TypeError("ADAPTER_TYPE=TP_LINEAR requires linear_fc1 to be TEColumnParallelLinear.")
        if self.adapter_type == "TP_LINEAR" and not isinstance(self.linear_fc2, TERowParallelLinear):
            raise TypeError("ADAPTER_TYPE=TP_LINEAR requires linear_fc2 to be TERowParallelLinear.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.layernorm(x).view(-1, self.hidden_size)
        x, _ = self.linear_fc1(x)
        x = self.activation_func(x)
        x, _ = self.linear_fc2(x)
        return x
