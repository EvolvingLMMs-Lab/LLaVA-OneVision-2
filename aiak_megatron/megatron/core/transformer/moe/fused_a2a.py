# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

"""fused alltoall"""
import os
import logging

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
    Buffer.set_num_sms(int(os.environ.get("DEEP_EP_SM_NUMS", 20)))
except ImportError:
    HAVE_DEEP_EP = False

import torch
# from transformer_engine.pytorch.constants import TE_DType
# Try to import TE_DType from Transformer Engine, but fall back if unavailable
logger = logging.getLogger(__name__)
try:
    from transformer_engine.pytorch.constants import TE_DType
except Exception as e:
    logger.warning(
        f"TE_DType import failed, FP8 communication will be disabled: {str(e)}"
    )
    TE_DType = {}

try:
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )
except Exception as e:
    logger.warning(
        f"Float8BlockQuantizer, Float8BlockwiseQTensor import failed, FP8 not available: {str(e)}"
    )
    Float8BlockQuantizer, Float8BlockwiseQTensor = None, None

_buffer = None

def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    """raw func for cast fp8 tensor to bf16
    """
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        enable_fp8_comm=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        tmp_x = x
        if enable_fp8_comm:
            _x_quantizer = Float8BlockQuantizer(
                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=False,
                block_scaling_dim=1,
            )
            x_fp8 = _x_quantizer(x)
            # Cause SFT scenario the input is not always aligns with 4 (FP8 gemm condition), quantizer automatically
            # pads to multiple of 4, so before `dispatch`, slicing back is need.
            x_q = x_fp8._rowwise_data.view(torch.float8_e4m3fn)
            x_s = x_fp8._rowwise_scale_inv.T[:x_q.size(0)].contiguous()
            tmp_x = (x_q, x_s)

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            tmp_x,
            topk_idx=token_indices,
            topk_weights=token_probs.float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        if enable_fp8_comm:
            tmp_recv_x = Float8BlockwiseQTensor(
                rowwise_data=recv_x[0],
                rowwise_scale_inv=recv_x[1].T.contiguous(),
                columnwise_data=None,
                columnwise_scale_inv=None,
                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                quantizer=_x_quantizer,
                is_2D_scaled=False,
                shape=recv_x[0].shape,
                dtype=torch.bfloat16
            )
            recv_x = tmp_recv_x.dequantize(dtype=torch.bfloat16)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False, enable_fp8_comm=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.enable_fp8_comm = enable_fp8_comm
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))

        tmp_grad = grad_output.contiguous()
        if ctx.enable_fp8_comm:
            _x_quantizer = Float8BlockQuantizer(
                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=False,
                block_scaling_dim=1,
            )
            x_fp8 = _x_quantizer(tmp_grad)
            # Cause SFT scenario the input is not always aligns with 4 (FP8 gemm condition), quantizer automatically
            # pads to multiple of 4, so before `dispatch`, slicing back is need.
            x_q = x_fp8._rowwise_data.view(torch.float8_e4m3fn)
            x_s = x_fp8._rowwise_scale_inv.T[:x_q.size(0)].contiguous()
            tmp_grad = (x_q, x_s)

        grad_x, _, _, _, _, after_event = buffer.dispatch(
            tmp_grad,
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()

        if ctx.enable_fp8_comm:
            tmp_grad_x = Float8BlockwiseQTensor(
                rowwise_data=grad_x[0],
                rowwise_scale_inv=grad_x[1].T.contiguous(),
                columnwise_data=None,
                columnwise_scale_inv=None,
                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                quantizer=_x_quantizer,
                is_2D_scaled=False,
                shape=grad_x[0].shape,
                dtype=torch.bfloat16
            )
            grad_x = tmp_grad_x.dequantize(dtype=torch.bfloat16)

        return grad_x, None, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        enable_fp8_comm=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
            enable_fp8_comm,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False, enable_fp8_comm=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream, enable_fp8_comm)

else:
    fused_dispatch = None
    fused_combine = None
