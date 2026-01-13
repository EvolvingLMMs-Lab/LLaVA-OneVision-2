# Copyright (c) 2024, BAIDU CORPORATION.  All rights reserved.
"""Dispatch Module"""

from typing import Any
from dataclasses import dataclass
from functools import cached_property

from aiak_training_llm.utils import get_args


# TODO: aiak-accelerator should be removed in the future. 
@dataclass
class MultiAccModules:
    """MultiAccModules"""
    # dense linear impl
    TELayerNormColumnParallelLinear: Any = None
    TEColumnParallelLinear: Any = None
    TERowParallelLinear: Any = None
    # group-gemm linear impl
    TEColumnParallelGroupedLinear: Any = None
    TERowParallelGroupedLinear: Any = None
    # local linear
    ColumnParallelLinear: Any = None
    RowParallelLinear: Any = None
    # attention impl
    DotProductAttention: Any = None
    # norm impl
    TENorm: Any = None
    LocalNorm: Any = None
    # other ops impl
    get_bias_dropout_add: Any = None
    apply_rotary_pos_emb: Any = None
    bias_activation_func_impl: Any = None
    # some flags
    separate_layernorm_and_collinear: bool = False
    TELinear: Any = None


def _gpu_backend_transformer_layer_modules() -> MultiAccModules:
    """define gpu transformer layer modules"""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
        TENorm,
        TELinear,
    )
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
    from aiak_training_llm.models.custom.common.local_norm import LocalNorm

    args = get_args()
    
    # Use local DotProductAttention if transformer_impl is "local", otherwise use TE version
    # DotProductAttention=TEDotProductAttention (old hardcoded version, now commented)
    use_te_attn = args.transformer_impl != "local"
    attn_module = TEDotProductAttention if use_te_attn else DotProductAttention
    
    return MultiAccModules(
        # dense linear
        TELayerNormColumnParallelLinear=TELayerNormColumnParallelLinear,
        TEColumnParallelLinear=TEColumnParallelLinear,
        TERowParallelLinear=TERowParallelLinear,
        # group-gemm linear
        TEColumnParallelGroupedLinear=TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear=TERowParallelGroupedLinear,
        # local linear
        ColumnParallelLinear=ColumnParallelLinear,
        RowParallelLinear=RowParallelLinear,
        # attn
        DotProductAttention=attn_module,
        # norm
        TENorm=TENorm,
        LocalNorm=LocalNorm,
        # ops
        get_bias_dropout_add=get_bias_dropout_add,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        bias_activation_func_impl=None,
        separate_layernorm_and_collinear=args.separate_layernorm_and_collinear,
        TELinear=TELinear,
    )


def _xpu_backend_transformer_layer_modules() -> MultiAccModules:
    """define xpu transformer layer modules"""
    args = get_args()
    separate_layernorm_and_collinear = args.separate_layernorm_and_collinear

    from aiak_accelerator.multiacc_engine import multiacc_get_bias_dropout_add
    from aiak_accelerator.multiacc_engine import multiacc_apply_rotary_pos_emb
    from aiak_accelerator.multiacc_engine import multiacc_bias_activation_func_impl
    from aiak_accelerator.multiacc_engine import MultiAccTELayerNormColumnParallelLinear
    from aiak_accelerator.multiacc_engine import MultiAccTEColumnParallelLinear
    from aiak_accelerator.multiacc_engine import MultiAccTERowParallelLinear
    from aiak_accelerator.multiacc_engine import MultiAccColumnParallelLinear
    from aiak_accelerator.multiacc_engine import MultiAccRowParallelLinear
    from aiak_accelerator.multiacc_engine import MultiAccDotProductAttention
    from aiak_accelerator.multiacc_engine import MultiAccNorm
    from aiak_accelerator.multiacc_engine import MultiAccTELinear

    if MultiAccTELayerNormColumnParallelLinear is None:
        separate_layernorm_and_collinear = True

    return MultiAccModules(
        # dense linear
        TELayerNormColumnParallelLinear=MultiAccTELayerNormColumnParallelLinear,
        TEColumnParallelLinear=MultiAccTEColumnParallelLinear,
        TERowParallelLinear=MultiAccTERowParallelLinear,
        # group-gemm linear
        TEColumnParallelGroupedLinear=None, # TODO: add support for grouped linear
        TERowParallelGroupedLinear=None, # TODO: add support for grouped linear
        # local linear
        ColumnParallelLinear=MultiAccColumnParallelLinear,
        RowParallelLinear=MultiAccRowParallelLinear,
        # attn
        DotProductAttention=MultiAccDotProductAttention, # TODO: add support for var length attention
        # norm
        TENorm=MultiAccNorm,
        LocalNorm=MultiAccNorm,
        # ops
        get_bias_dropout_add=multiacc_get_bias_dropout_add,
        apply_rotary_pos_emb=multiacc_apply_rotary_pos_emb,
        bias_activation_func_impl=multiacc_bias_activation_func_impl,
        separate_layernorm_and_collinear=separate_layernorm_and_collinear,
        TELinear=MultiAccTELinear,
    )


def is_gpu_accelerator_backend():
    """check if gpu accelerator backend"""
    # NOTE: aiak-accelerator will be removed in the future, please minimize the calls.
    try:
        # if install aiak_accelerator, use aiak_accelerator backend
        from aiak_accelerator import get_accelerator
        return get_accelerator().backend == "NvidiaGpu"
    except ImportError:
        # not install, default gpu backend
        return True


class Dispatch:
    """dispatch transformer layer module"""
    @cached_property
    def settings(self):
        """stting attr"""
        if is_gpu_accelerator_backend():
            return _gpu_backend_transformer_layer_modules()
        # else xpu
        return _xpu_backend_transformer_layer_modules() 

    def __getattr__(self, name):
        return getattr(self.settings, name)


multiacc_modules: MultiAccModules = Dispatch()
