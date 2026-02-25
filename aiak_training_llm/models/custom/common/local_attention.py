"""megatron local attention"""

import math
import torch

from torch import Tensor
from megatron.core.utils import divide
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention


try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    import rearrange
    HAVE_FLASH_ATTN = True
except Exception:
    HAVE_FLASH_ATTN = False


class FlashSelfAttention(MegatronModule):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        """__init__"""
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        # TODO(changtao02): directly mock flash_attn_varlen-func to enable this assertation
        # assert HAVE_FLASH_ATTN is True,
        # ('Please implement an available flashattn kernel for the underlying accelerator')

        self.layer_number = max(1, layer_number) # unused
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)

        self.causal = True
        self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        self.dropout_p = self.config.attention_dropout if attention_dropout is None else attention_dropout
        self.nheads = self.num_attention_heads_per_partition


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (query, key, value)))
        assert all((i.is_cuda for i in (query, key, value)))

        batch_size, seqlen_q = query.shape[0], query.shape[1]
        seqlen_k = key.shape[1]

        query, key, value = [rearrange(x, 'b s ... -> (b s) ...') for x in [query, key, value]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=query.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=query.device)
            self.dropout_p = 0

        output = flash_attn_varlen_func(
            query, key, value, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            self.dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class LocalAttention:
    """
    A conditional wrapper to initialize an instance of Megatron Local `CoreAttention`
    or `FlashSelfAttention` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        """__new__"""
        # if config.normalization == "LayerNorm":
        # TODO(changtao02): add argument to switch between core_attn and flash_attn
        instance = FlashSelfAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout
        )
        # else:
        #     raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
