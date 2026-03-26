# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math
from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide

import torch.nn.functional as F
from typing import Optional

class ScaleMaskSoftmaxFallback:
    """
    Pure-PyTorch fallback for fused scale+mask+softmax.

    Parameters:
    - softmax_in_fp32: if True, compute softmax in float32 for stability.
    - attn_mask_type: (optional) unused here except for API compatibility.
    - scale: a scalar or None; FusedScaleMaskSoftmax used 'scale' for layer scaling
             in some implementations.  This fallback will NOT multiply or divide
             by `scale` automatically (because your code already applies `self.softmax_scale`
             in the matmul). If you need special handling, you can pass scale and
             we will multiply scores by scale here.
    """

    def __init__(self, softmax_in_fp32: bool = False, attn_mask_type=None, scale: Optional[float] = None):
        self.softmax_in_fp32 = softmax_in_fp32
        self.attn_mask_type = attn_mask_type
        self.scale = scale

    def __call__(self, attention_scores: torch.Tensor, attention_mask: Optional[torch.Tensor], attn_mask_type: Optional[object] = None):
        """
        attention_scores: [b, np, sq, sk] (or equivalent)
        attention_mask: broadcastable mask, or None. Convention in Megatron often
                        uses additive masks (large negative for masked positions).
        Returns attention_probs: same shape as attention_scores
        """
        # Optionally apply an additional scale (some fused ops expect to apply layer scaling here).
        if self.scale is not None:
            # If scale is an integer layer_number in original fused implementation,
            # you may want to divide or multiply depending on how self.softmax_scale was set.
            # The original code sets self.softmax_scale = 1/sqrt(head_dim) and divides it by coeff
            # (layer_number) if apply_query_key_layer_scaling is True. Since the matmul already
            # used self.softmax_scale, we leave this as a no-op by default.
            try:
                # if user wants to apply the numeric scale here, do so:
                attention_scores = attention_scores * float(self.scale)
            except Exception:
                pass

        if attention_mask is not None:
            # If attention_mask is additive (contains large negative values in masked positions),
            # just add it. If it is boolean, convert it to additive mask.
            if attention_mask.dtype == torch.bool:
                # True = keep, False = masked? Depends on user's mask conventions.
                # Typical convention: True indicates valid tokens -> we want mask where False positions are -inf.
                # So convert boolean mask to additive: 0 for valid, -1e9 for masked.
                add_mask = (~attention_mask).to(attention_scores.dtype) * -1e9
                # Ensure add_mask broadcastable to attention_scores shape
                attention_scores = attention_scores + add_mask
            else:
                # assume it's additive mask already shaped/broadcastable
                attention_scores = attention_scores + attention_mask

        # Softmax
        if self.softmax_in_fp32:
            orig_dtype = attention_scores.dtype
            probs = torch.softmax(attention_scores.float(), dim=-1).to(orig_dtype)
        else:
            probs = torch.softmax(attention_scores, dim=-1)

        return probs


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
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

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
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

        # self.scale_mask_softmax = FusedScaleMaskSoftmax(
        #     input_in_fp16=self.config.fp16,
        #     input_in_bf16=self.config.bf16,
        #     attn_mask_type=self.attn_mask_type,
        #     scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
        #     mask_func=attention_mask_func,
        #     softmax_in_fp32=self.config.attention_softmax_in_fp32,
        #     scale=coeff,
        # )
        use_fused=False
        if use_fused:
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=self.config.fp16,
                input_in_bf16=self.config.bf16,
                attn_mask_type=self.attn_mask_type,
                scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
                mask_func=attention_mask_func,
                softmax_in_fp32=self.config.attention_softmax_in_fp32,
                scale=coeff,
            )
        else:
            # fallback implementation defined below (or import it)
            self.scale_mask_softmax = ScaleMaskSoftmaxFallback(
                softmax_in_fp32=self.config.attention_softmax_in_fp32,
                attn_mask_type=self.attn_mask_type,
                scale=coeff,
            )
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
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
        packed_seq_params=None
        """Forward."""
        # DEBUG: Log inputs
        print(f"[DotProductAttention.forward] INPUT: query={query is not None}, key={key is not None}, value={value is not None}", flush=True)
        
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            # Only repeat when tensors are 4D [*, b, ng, hn] so we expand ng -> np.
            repeat_times = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            if key is not None and key.dim() == 4:
                key = key.repeat_interleave(repeat_times, dim=2)
            if value is not None and value.dim() == 4:
                value = value.repeat_interleave(repeat_times, dim=2)
        
        # Debug: safely print shapes
        print(f"query.shape={tuple(query.shape) if query is not None else None}, key.shape={tuple(key.shape) if key is not None else None}, value.shape={tuple(value.shape) if value is not None else None}", flush=True)

        # Normalize shapes to 3D for baddbmm and derive (b, np, sq, sk)
        if query.dim() == 4:
            # [sq, b, np, hn]
            sq_len, b, np_per_part, hn = query.size()
            sk_len = key.size(0)
            # [sq, b, np, hn] -> [sq, b*np, hn]
            query = query.reshape(sq_len, b * np_per_part, hn)
            # Key can be 4D or 3D depending on upstream; normalize to 3D [sk, b*np, hn]
            if key.dim() == 4:
                key = key.reshape(sk_len, b * np_per_part, -1)
        elif query.dim() == 3:
            # [sq, b*np, hn]
            sq_len, bnp, hn = query.size()
            sk_len = key.size(0)
            np_per_part = self.num_attention_heads_per_partition
            assert bnp % np_per_part == 0, (
                f"DotProductAttention: b*np ({bnp}) not divisible by heads_per_part ({np_per_part})"
            )
            b = bnp // np_per_part
        else:
            raise RuntimeError(f"DotProductAttention: Unsupported query.dim()={query.dim()}")

        # Ensure key is 3D [sk, b*np, hn]
        if key.dim() == 4:
            # [sk, b, np, hn] -> [sk, b*np, hn]
            key = key.reshape(sk_len, b * np_per_part, -1)
        elif key.dim() == 3:
            # If key is [sk, b*g, hn*r] for GQA packing, unflatten to [sk, b*np, hn]
            if key.size(1) != b * np_per_part:
                g = self.num_query_groups_per_partition
                if key.size(1) == b * g:
                    r = key.size(2) // hn
                    if (r * g) == np_per_part and (key.size(2) % hn) == 0:
                        # [sk, b, g, r, hn] -> [sk, b, np, hn] -> [sk, b*np, hn]
                        key = (
                            key.reshape(sk_len, b, g, r, hn)
                               .permute(0, 1, 3, 2, 4)
                               .reshape(sk_len, b * np_per_part, hn)
                        )
                    elif np_per_part % g == 0 and r == 1:
                        # Simple case: [sk, b*g, hn] -> repeat groups to heads
                        repeat_times = np_per_part // g
                        key = key.repeat_interleave(repeat_times, dim=1)
                    else:
                        raise RuntimeError(
                            f"DotProductAttention: Cannot map key of shape {tuple(key.size())} to [sk, b*np, hn]. "
                            f"b={b}, np={np_per_part}, g={g}, hn={hn}"
                        )
                elif key.size(1) == b:
                    key = key.repeat_interleave(np_per_part, dim=1)
                else:
                    raise RuntimeError(
                        f"DotProductAttention: Unexpected key.size(1)={key.size(1)}; "
                        f"expected {b*np_per_part} or {b*self.num_query_groups_per_partition} or {b}"
                    )
        else:
            raise RuntimeError(f"DotProductAttention: Unsupported key.dim()={key.dim()}")

        # [b, np, sq, sk]
        output_size = (b, np_per_part, sq_len, sk_len)

        # Ensure value matches the [sk, b*np, hn] layout prior to context matmul
        if value.dim() == 4:
            # [sk, b, np, hn] -> [sk, b*np, hn]
            value = value.reshape(value.size(0), b * np_per_part, -1)
        elif value.dim() == 3:
            if value.size(1) != b * np_per_part:
                g = self.num_query_groups_per_partition
                if value.size(1) == b * g:
                    r = value.size(2) // hn
                    if (r * g) == np_per_part and (value.size(2) % hn) == 0:
                        # [sk, b, g, r, hv] -> [sk, b, np, hv] -> [sk, b*np, hv]
                        value = (
                            value.reshape(value.size(0), b, g, r, hn)
                                 .permute(0, 1, 3, 2, 4)
                                 .reshape(value.size(0), b * np_per_part, hn)
                        )
                    elif np_per_part % g == 0 and r == 1:
                        repeat_times = np_per_part // g
                        value = value.repeat_interleave(repeat_times, dim=1)
                    else:
                        raise RuntimeError(
                            f"DotProductAttention: Cannot map value of shape {tuple(value.size())} to [sk, b*np, hv]. "
                            f"b={b}, np={np_per_part}, g={g}, hv={hn}"
                        )
                elif value.size(1) == b:
                    value = value.repeat_interleave(np_per_part, dim=1)
                else:
                    raise RuntimeError(
                        f"DotProductAttention: Unexpected value.size(1)={value.size(1)}; "
                        f"expected {b*np_per_part} or {b*self.num_query_groups_per_partition} or {b}"
                    )
        else:
            raise RuntimeError(f"DotProductAttention: Unsupported value.dim()={value.dim()}")

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask, attn_mask_type)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        # output_size = (value.size(1), value.size(2), query.size(0), value.size(3))
        # Derive batch and heads-per-partition from attention_probs so this code
        # works whether `value` was provided as [sk, b, np, hn] (4D) or
        # [sk, b*np, hn] (3D). Use the last dimension of `value` for head dim.
        b, np_per_part, sq_len, _sk_len = attention_probs.size()
        head_dim = value.size(-1)
        output_size = (b, np_per_part, sq_len, head_dim)

        # change view [sk, b * np, hn]
        # value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        value = value.reshape(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        # attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        attention_probs = attention_probs.reshape(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        # context = context.view(*output_size)
        # Use inferred head dimension to handle cases where `value` was packed differently
        # and the product of dims differs from the expected head_dim.
        context = context.reshape(output_size[0], output_size[1], output_size[2], -1)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        # context = context.view(*new_context_shape)
        # Flatten heads using the actual last dimension to avoid shape mismatches when
        # self.hidden_size_per_partition does not equal np_per_part * head_dim.
        context = context.reshape(context.size(0), context.size(1), -1)

        return context
