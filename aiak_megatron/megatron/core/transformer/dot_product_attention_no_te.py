# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


class ScaleMaskSoftmaxFallback:
    """
    Pure-PyTorch scale + mask + softmax used when fused kernels are unavailable.
    """

    def __init__(self, softmax_in_fp32: bool = False, attn_mask_type=None, scale: Optional[float] = None):
        self.softmax_in_fp32 = softmax_in_fp32
        self.attn_mask_type = attn_mask_type
        self.scale = scale

    def __call__(self, attention_scores: torch.Tensor, attention_mask: Optional[torch.Tensor], attn_mask_type: Optional[object] = None):
        if self.scale is not None:
            attention_scores = attention_scores * float(self.scale)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                add_mask = (~attention_mask).to(attention_scores.dtype) * -1e9
                attention_scores = attention_scores + add_mask
            else:
                attention_scores = attention_scores + attention_mask

        if self.softmax_in_fp32:
            orig_dtype = attention_scores.dtype
            probs = torch.softmax(attention_scores.float(), dim=-1).to(orig_dtype)
        else:
            probs = torch.softmax(attention_scores, dim=-1)

        return probs


class DotProductAttentionNoTE(MegatronModule):
    """
    Pure-PyTorch dot-product attention that avoids transformer-engine / fused softmax.
    Shapes follow Megatron's convention: query/key/value are expected as [sq/sk, b, ng/np, hn].
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert self.config.context_parallel_size == 1, "Context parallelism is only supported by TEDotProductAttention!"
        assert self.config.window_size is None, "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        self.scale_mask_softmax = ScaleMaskSoftmaxFallback(
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            attn_mask_type=self.attn_mask_type,
            scale=coeff,
        )

        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        packed_seq_params = None
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttentionNoTE."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttentionNoTE."

        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        key = key.reshape(output_size[3], output_size[0] * output_size[1], -1)

        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),
            key.transpose(0, 1).transpose(1, 2),
            beta=0.0,
            alpha=self.softmax_scale,
        )

        attention_scores = matmul_result.view(*output_size)

        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask, attn_mask_type)

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        b, np_per_part, sq_len, _sk_len = attention_probs.size()
        head_dim = value.size(-1)
        output_size = (b, np_per_part, sq_len, head_dim)

        value = value.reshape(value.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.reshape(output_size[0] * output_size[1], output_size[2], -1)

        context = torch.bmm(attention_probs, value.transpose(0, 1))

        context = context.reshape(*output_size)
        context = context.permute(2, 0, 1, 3).contiguous()

        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.reshape(*new_context_shape)

        return context
