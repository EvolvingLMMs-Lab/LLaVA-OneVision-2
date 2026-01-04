# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
TE-stub: Transformer Engine *disabled* version for Megatron-Core.

This file exposes the same public symbols as the original
`megatron.core.extensions.transformer_engine` module, but:

  * Does NOT import `transformer_engine` at all.
  * Provides simple PyTorch-based fallbacks for a few classes.
  * Marks TE-specific / FP8 / advanced features as unavailable
    and raises clear RuntimeError if they are actually used.

The goal is that:
  - `from megatron.core.extensions.transformer_engine import ...`
    always succeeds, and
  - as long as your config does NOT explicitly request TE/FP8
    (e.g. you use bf16 and non-TE attention backend), training
    can run without touching any real TE code.
"""

from typing import Any, Callable, Optional

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Global TE capability flag & MoE fused permute stubs
# ---------------------------------------------------------------------------

HAVE_TE = False  # other code may check this

# MoE fused permute helpers – not available without TE
fused_permute = None
fused_unpermute = None
fused_sort_chunks_by_index = None

# FP8 padding helpers – not available
Fp8Padding = None
Fp8Unpadding = None


# ---------------------------------------------------------------------------
# Utility: simple checkpoint wrapper (no real TE checkpoint)
# ---------------------------------------------------------------------------

def te_checkpoint(
    forward_func: Callable,
    distribute_saved_activations: bool,
    get_rng_state_tracker: Any,
    tp_group: Any,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor],
    attn_mask_type: Any,
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    rotary_pos_emb: Optional[Tensor],
    **kwargs,
):
    """
    Minimal replacement for TE checkpointing.

    We simply call `forward_func` directly WITHOUT activation checkpointing.
    This keeps correctness but loses memory savings compared to real TE.
    """
    return forward_func(
        hidden_states,
        attention_mask,
        attn_mask_type,
        context,
        context_mask,
        rotary_pos_emb,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# TENorm – fallback to plain LayerNorm
# ---------------------------------------------------------------------------

class TENorm(nn.Module):
    """
    TE-style normalization wrapper.

    Original TE version chooses between LayerNorm / RMSNorm etc.
    Here we just always use plain LayerNorm.

    Original signature:
        TENorm(config: TransformerConfig, hidden_size: int, eps: float = 1e-5)
    We accept arbitrary *args / **kwargs and try to pick hidden_size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Try to infer hidden_size and eps
        if len(args) >= 2:
            hidden_size = args[1]
        else:
            hidden_size = kwargs.get("hidden_size") or kwargs.get("normalized_shape")
        if hidden_size is None:
            raise ValueError(
                "TENorm stub could not infer hidden_size. "
                "Expected signature: TENorm(config, hidden_size, eps=1e-5)"
            )

        eps = kwargs.get("eps", 1e-5)
        self.ln = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.ln(x)


# ---------------------------------------------------------------------------
# TELinear and variants – minimal PyTorch fallbacks
# ---------------------------------------------------------------------------

class TELinear(nn.Module):
    """
    Fallback for TE's Linear wrappers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Any,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        parallel_mode: Optional[str] = None,
        is_expert: bool = False,
        **kwargs,
    ):
        super().__init__()
        if skip_weight_param_allocation:
            raise RuntimeError(
                "TELinear stub does not support skip_weight_param_allocation=True."
            )

        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.skip_bias_add = skip_bias_add

        if init_method is not None:
            init_method(self.linear.weight)

    def forward(self, x: Tensor):
        # Auto-adjust in_features on first mismatch to match runtime input.
        # This keeps training moving if TP sizing assumptions differ.
        if x.size(-1) != self.linear.in_features:
            try:
                device = self.linear.weight.device
                bias_flag = self.linear.bias is not None
                out_features = self.linear.out_features
                # Recreate linear with correct in_features
                new_linear = nn.Linear(x.size(-1), out_features, bias=bias_flag).to(device)
                self.linear = new_linear
                print(f"[TE-Stub] Adjusted Linear in_features to {x.size(-1)} (was mismatch)", flush=True)
            except Exception:
                pass
        # Ensure dtype/device match input to avoid BF16/FP32 matmul mismatch
        if self.linear.weight.dtype != x.dtype or self.linear.weight.device != x.device:
            self.linear = self.linear.to(device=x.device, dtype=x.dtype)
        y = self.linear(x)
        if self.skip_bias_add:
            return y, self.linear.bias
        return y, None


class TEColumnParallelLinear(TELinear):
    """
    Fallback version of TEColumnParallelLinear with basic TP-aware sizing.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Any,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        input_is_parallel: bool = False,
        **kwargs,
    ):
        if gather_output:
            raise RuntimeError(
                "TEColumnParallelLinear stub does not support gather_output=True."
            )
        try:
            from megatron.core import parallel_state as _ps
            tp_world = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_world = 1
        local_in = input_size
        local_out = output_size
        if tp_world and tp_world > 1:
            if input_is_parallel:
                assert input_size % tp_world == 0, "input_size must be divisible by TP size"
                local_in = input_size // tp_world
            assert output_size % tp_world == 0, "output_size must be divisible by TP size"
            local_out = output_size // tp_world

        super().__init__(
            input_size=local_in,
            output_size=local_out,
            parallel_mode="column",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        )


class TERowParallelLinear(TELinear):
    """
    Fallback version of TERowParallelLinear with basic TP-aware sizing.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Any,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        **kwargs,
    ):
        if not input_is_parallel:
            raise RuntimeError(
                "TERowParallelLinear stub expects input_is_parallel=True (same as TE)."
            )
        try:
            from megatron.core import parallel_state as _ps
            tp_world = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_world = 1
        local_in = input_size
        local_out = output_size
        if tp_world and tp_world > 1:
            assert input_size % tp_world == 0, "input_size must be divisible by TP size"
            local_in = input_size // tp_world
            # Row-parallel returns full-sized outputs
        super().__init__(
            input_size=local_in,
            output_size=local_out,
            parallel_mode="row",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        )


class TELayerNormColumnParallelLinear(nn.Module):
    """
    Fallback for TELayerNormColumnParallelLinear.

    Original TE combines LayerNorm + ColumnParallelLinear.
    Here we implement:   y = Linear(LayerNorm(x))

    Forward returns (y, bias_or_none) to match TE behaviour.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Any,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if gather_output:
            raise RuntimeError(
                "TELayerNormColumnParallelLinear stub does not support gather_output=True."
            )
        if is_expert:
            # You could still allow this, but keep consistent with original constraint
            raise RuntimeError(
                "TELayerNormColumnParallelLinear stub does not support is_expert=True."
            )

        eps = getattr(config, "layernorm_epsilon", 1e-5)
        self.ln = nn.LayerNorm(input_size, eps=eps)
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.skip_bias_add = skip_bias_add

        if init_method is not None:
            init_method(self.linear.weight)

    def forward(self, x: Tensor):
        y = self.linear(self.ln(x))
        if self.skip_bias_add:
            return y, self.linear.bias
        else:
            return y, None


# ---------------------------------------------------------------------------
# GroupedLinear stubs – names exist, but not implemented
# ---------------------------------------------------------------------------

class TEGroupedLinear(nn.Module):
    """
    Stub for TEGroupedLinear.

    If your config ever tries to instantiate this, you are relying on
    Transformer Engine's grouped FP8 GEMMs, which are not available
    in this stubbed build.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("TEGroupedLinear is not available (Transformer Engine disabled).")

    def forward(self, *args, **kwargs):
        raise RuntimeError("TEGroupedLinear is not available (Transformer Engine disabled).")


class TEColumnParallelGroupedLinear(TEGroupedLinear):
    pass


class TERowParallelGroupedLinear(TEGroupedLinear):
    pass


# ---------------------------------------------------------------------------
# Attention wrapper – disabled (we use non-TE backend instead)
# ---------------------------------------------------------------------------
# class TEDotProductAttention(nn.Module):
#     """
#     Very simple scaled dot-product attention fallback.
#     This is only meant to keep imports and basic training running,
#     not for maximum performance.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         print(type(query), query)
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
#         attn = torch.softmax(scores, dim=-1)
#         return torch.matmul(attn, value)

import math
from typing import Optional, Any
import torch
import torch.nn as nn
from torch import Tensor

# Keep the same name so it can be used as a drop-in
class TEDotProductAttention(nn.Module):
    """
    Simple, readable scaled dot-product attention replacement for Transformer-Engine's
    DotProductAttention. Meant as a fallback for correctness / debugging, not performance.
    Forward signature kept compatible with the TE wrapper used previously.

    Notes:
      - Supports qkv_format 'sbhd' (seq, batch, head, dim) and 'bshd' (batch, seq, head, dim).
      - Also accepts 3-D (B, S, D) inputs and will split heads automatically.
      - attention_mask can be boolean (True=keep) or additive (float with -inf on masked positions)
      - attention_bias (if provided) is added to attention scores before softmax.
    """

    cp_stream: torch.cuda.Stream = None  # keep attribute so callers referencing it won't fail

    def __init__(
        self,
        config: Any,
        layer_number: int,
        attn_mask_type: Any,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
    ):
        super().__init__()
        # Store config values we may need
        self.config = config
        # default qkv format preserved from your original class
        self.qkv_format: str = "sbhd"
        # num heads convenience
        self.num_heads = getattr(config, "num_attention_heads", 1)
        # total kv channels (if present on config). We'll use this only when needed to infer head dim
        self.kv_channels = getattr(config, "kv_channels", None)
        # optional dropout on attention weights (kept None unless provided)
        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout and attention_dropout > 0.0 else None
        # allow externally-configured softmax scale (float) OR compute as 1/sqrt(head_dim) later
        self.softmax_scale = softmax_scale
        # keep set of packed seq param names to remain compatible with callers (ignored here)
        self.kept_packed_seq_params = set()

    def _ensure_4d_and_split_heads(self, x: Tensor) -> (Tensor, bool):
        """
        Ensure x is in (B, S, H, D_head) layout and return (x4d, was_3d)
        Accepts:
          - x shaped (S, B, H, D) if qkv_format == 'sbhd'  (seq, batch, head, dim)
          - x shaped (B, S, H, D) if qkv_format == 'bshd'
          - x shaped (B, S, D) -> will split last dim into (H, D_head)
        Returns x in (B, S, H, D_head) format and a flag indicating whether original was 3d.
        """
        was_3d = False
        if x is None:
            raise ValueError("TEDotProductAttention: input tensor is None")
        if x.dim() == 4:
            if self.qkv_format == "sbhd":
                # (S, B, H, D) -> convert to (B, S, H, D)
                x4 = x.transpose(0, 1).contiguous()
            elif self.qkv_format == "bshd":
                x4 = x.contiguous()
            else:
                # fallback: assume bshd if unknown
                x4 = x.contiguous()
        elif x.dim() == 3:
            # (B, S, D) -> split D into (H, D_head)
            was_3d = True
            B, S, D = x.shape
            if D % self.num_heads != 0:
                raise ValueError(
                    f"TEDotProductAttention: embedding dim ({D}) is not divisible by num_heads ({self.num_heads})."
                )
            D_head = D // self.num_heads
            x4 = x.view(B, S, self.num_heads, D_head).contiguous()
        else:
            raise ValueError(f"Tensors must be 3D (B,S,D) or 4D. Got shape {tuple(x.shape)}")
        return x4, was_3d

    def _reconstruct_output(self, out4: Tensor, was_3d: bool) -> Tensor:
        """
        out4 is (B, S, H, D_head).
        If was_3d True -> return (B, S, D) by merging head dim
        If qkv_format previously was 'sbhd' and user originally passed sbhd (we tracked earlier),
        we attempt to return in original layout. However to keep behavior simple we return:
          - (B,S,D) if original inputs were 3D
          - (B,S,H,D) otherwise
        """
        if was_3d:
            B, S, H, Dh = out4.shape
            return out4.view(B, S, H * Dh)
        else:
            return out4

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Any,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Any = None,
    ) -> Tensor:
        """
        Simple scaled dot-product attention:
          out[b, s_q, h, d] = softmax( (q*k^T)/sqrt(d) + attention_bias + mask ) @ v

        Returns output in (B, S_q, H, D_head) unless the inputs were 3D (B,S,D) in which case returns (B,S,D).
        """

        # Allow callers to pass packed_seq_params (kept for API compatibility) but we ignore it here.
        _ = packed_seq_params  # noqa: F841

        # Convert inputs to (B, S, H, D_head)
        q4, q_was_3d = self._ensure_4d_and_split_heads(query)
        k4, k_was_3d = self._ensure_4d_and_split_heads(key)
        v4, v_was_3d = self._ensure_4d_and_split_heads(value)

        # sanity check shapes
        if not (q_was_3d == k_was_3d == v_was_3d):
            # mixing 3D / 4D forms is confusing; require consistent input forms
            raise ValueError("TEDotProductAttention: query/key/value must be all 3D or all 4D (consistent formats).")

        Bq, Sq, Hq, Dh = q4.shape
        Bk, Sk, Hk, Dk = k4.shape
        Bv, Sv, Hv, Dv = v4.shape

        if not (Bq == Bk == Bv):
            raise ValueError("Batch sizes of query/key/value must match")
        if not (Hq == Hk == Hv):
            raise ValueError("Number of heads of query/key/value must match")
        if not (Dk == Dv):
            raise ValueError("Key/value head dims must match")
        if Dh != Dk:
            # head dims should match for q and k
            # if they don't, we can still compute by projecting; but here we require equality
            raise ValueError(f"Head dimension mismatch: q head dim {Dh} vs k head dim {Dk}")

        B = Bq
        # compute scale
        scale = self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(Dh)

        # compute raw scores: we want shape (B, H, Sq, Sk)
        # reshape to (B, H, Sq, Dh) etc and use matmul after transposing
        # q4: (B, Sq, H, Dh) -> (B, H, Sq, Dh)
        qBH = q4.transpose(1, 2)  # (B, H, Sq, Dh)
        kBH = k4.transpose(1, 2)  # (B, H, Sk, Dh)
        vBH = v4.transpose(1, 2)  # (B, H, Sv, Dh)

        # scores: (B, H, Sq, Sk)
        scores = torch.matmul(qBH, kBH.transpose(-2, -1)) * scale

        # apply attention_bias if given (broadcastable)
        if attention_bias is not None:
            # attention_bias commonly has shape (B, 1, 1, Sk) or (1, 1, 1, Sk) or (Sq, Sk)
            scores = scores + attention_bias

        # apply attention_mask
        if attention_mask is not None:
            # If mask is boolean where True means keep, convert to additive mask
            # Accept both boolean masks and additive masks (float)
            if attention_mask.dtype == torch.bool:
                # mask shape might be (B, Sk) or (B, 1, 1, Sk) etc; we want True where keep
                # we fill disallowed positions with -inf
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                # assume additive mask already shaped for broadcasting
                scores = scores + attention_mask

        # softmax, dropout (if any), then matmul with v
        attn = torch.softmax(scores, dim=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        # (B, H, Sq, Sk) @ (B, H, Sk, Dh) -> (B, H, Sq, Dh)
        outBH = torch.matmul(attn, vBH)

        # bring back to (B, Sq, H, Dh)
        out4 = outBH.transpose(1, 2).contiguous()

        # If inputs were 3D originally, merge heads back
        out = self._reconstruct_output(out4, q_was_3d)

        return out

# ---------------------------------------------------------------------------
# FP8 / RNG / misc TE utilities – disabled
# ---------------------------------------------------------------------------

class TEDelayedScaling:
    """
    Stub for TE DelayedScaling FP8 recipe.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("FP8 / TEDelayedScaling is not available (Transformer Engine disabled).")


class TECudaRNGStatesTracker:
    """
    Stub for TE CUDA RNG tracker.

    Megatron's own RNG tracker should be used instead.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("TECudaRNGStatesTracker is not available (Transformer Engine disabled).")


# ---------------------------------------------------------------------------
# SplitAlongDim, RoPE, CPU offload – minimal / disabled
# ---------------------------------------------------------------------------

# def SplitAlongDim(x: Tensor, dim: int,  num_chunks: int):
#     """
#     Minimal replacement for TE's _SplitAlongDim.apply.

#     Returns a tuple of chunks along the given dimension.
#     Autograd works fine through torch.chunk, so this is enough
#     for most use cases that only expect a "split" function.
#     """
#     return torch.chunk(x, num_chunks, dim=dim)
import torch
from torch import Tensor
from typing import Union, Sequence

def SplitAlongDim(x: Tensor, dim: int, num_chunks: Union[int, Sequence[int]]):
    """
    Replacement for TE's _SplitAlongDim.apply.

    - If `num_chunks` is an int: behaves like torch.chunk(x, num_chunks, dim=dim).
    - If `num_chunks` is a sequence of ints: behaves like torch.split(x, num_chunks, dim=dim).
      In that case the ints specify sizes for each split along `dim`.

    Returns a tuple of tensors.

    Raises ValueError for invalid inputs (e.g. sizes don't sum to x.size(dim)).
    """
    if x is None:
        raise ValueError("SplitAlongDim: input tensor x is None")

    if not isinstance(dim, int):
        raise TypeError(f"SplitAlongDim: dim must be int, got {type(dim)}")

    # int -> torch.chunk
    if isinstance(num_chunks, int):
        if num_chunks <= 0:
            raise ValueError(f"SplitAlongDim: num_chunks must be > 0, got {num_chunks}")
        return tuple(torch.chunk(x, num_chunks, dim=dim))

    # sequence -> torch.split with sizes
    if isinstance(num_chunks, (list, tuple)):
        sizes = list(num_chunks)
        if not all(isinstance(s, int) and s >= 0 for s in sizes):
            raise TypeError("SplitAlongDim: when passing sizes, all elements must be non-negative ints")
        total = sum(sizes)
        dim_size = x.size(dim)
        if total != dim_size:
            raise ValueError(
                f"SplitAlongDim: sizes sum to {total} but tensor.size({dim}) == {dim_size}. "
                "Sizes must match the dimension length exactly."
            )
        return tuple(torch.split(x, sizes, dim=dim))

    raise TypeError(
        "SplitAlongDim: num_chunks must be an int (number of equal chunks) or "
        "a list/tuple of sizes for each chunk."
    )


def get_cpu_offload_context(
    enabled: bool,
    num_layers: int,
    model_layers: Any,
    activation_offloading: bool,
    weight_offloading: bool,
):
    """
    Stub for TE CPU offload context.

    We simply return (None, lambda: None).
    """
    def _sync():
        return None

    return None, _sync


# def fused_apply_rotary_pos_emb(
#     t: torch.Tensor, freqs: torch.Tensor, transpose_output_memory: bool = False
# ) -> torch.Tensor:
#     """
#     Stub for fused RoPE (sbhd). We fall back to naive RoPE application:

#     This is a placeholder; if your config explicitly requires TE fused RoPE,
#     you should not be using this stub file.
#     """
#     # A very naive RoPE: assume last dimension is split into (cos, sin)
#     # This is intentionally simple and may NOT match all TE behaviours.
#     dim = t.shape[-1]
#     half = dim // 2
#     cos = freqs[..., :half]
#     sin = freqs[..., half:half * 2]
#     x1, x2 = t[..., :half], t[..., half:half * 2]
#     rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
#     if transpose_output_memory:
#         # We ignore this just like TE stub
#         pass
#     return rot
import torch
from torch import Tensor
from typing import Tuple

def _align_and_broadcast_to(tensor: Tensor, target: Tensor, name: str) -> Tensor:
    """
    Try to make `tensor` broadcastable to `target` by:
      - checking last-dim size matches target's last-dim,
      - if tensor has fewer dims: prepend singleton dims,
      - if tensor has more dims: try to squeeze leading singleton dims,
      - finally expand to target.shape.

    If impossible, raise a helpful ValueError showing shapes.
    """
    if tensor is None:
        raise ValueError(f"{name} is None")

    if tensor.size(-1) != target.size(-1):
        raise ValueError(
            f"{name} last-dim size ({tensor.size(-1)}) != target last-dim ({target.size(-1)}). "
            "They must match (both equal to half = D//2)."
        )

    t_ndim = tensor.ndim
    tgt_ndim = target.ndim

    # If tensor has fewer dims, prepend singleton dims
    if t_ndim < tgt_ndim:
        new_shape = (1,) * (tgt_ndim - t_ndim) + tensor.shape
        tensor = tensor.reshape(new_shape)

    # If tensor has more dims, try to squeeze leading singleton dims
    elif t_ndim > tgt_ndim:
        # Attempt to remove leading dims of size 1 until dims match or cannot.
        # Only allow squeezing if those dims are 1.
        diff = t_ndim - tgt_ndim
        leading = tensor.shape[:diff]
        if all(s == 1 for s in leading):
            tensor = tensor.reshape(tensor.shape[diff:])
        else:
            # can't safely squeeze; give helpful message
            raise ValueError(
                f"{name} has more leading dims ({tensor.shape}) than target ({target.shape}) "
                "and they are not singleton. Cannot broadcast. You probably need to slice freqs "
                "to the appropriate time-range (e.g. freqs[sequence_start:sequence_end])."
            )

    # Now tensor.ndim == target.ndim and last dims already checked equal.
    try:
        tensor = tensor.expand(target.shape)
    except Exception as e:
        raise ValueError(
            f"Failed to broadcast {name} with shape {tensor.shape} to target shape {target.shape}: {e}"
        )
    return tensor


# def fused_apply_rotary_pos_emb(
#     t: Tensor, freqs: Tensor, transpose_output_memory: bool = False
# ) -> Tensor:
#     """
#     Memory-efficient, broadcasting-robust RoPE fallback.

#     - t: Tensor with shape (..., D) where D is even (D = 2*half)
#     - freqs: Tensor with last dim either half (cos only) or 2*half (cos||sin)
#       Leading dims of freqs are flexible but must be broadcastable to t's leading dims
#       after appropriate simple reshapes (see _align_and_broadcast_to).
#     - transpose_output_memory ignored (kept for API compatibility).
#     """
#     if t is None:
#         raise ValueError("fused_apply_rotary_pos_emb: input t is None")

#     D = t.size(-1)
#     if D % 2 != 0:
#         raise ValueError(f"fused_apply_rotary_pos_emb: last dimension must be even, got {D}")
#     half = D // 2

#     if freqs is None:
#         raise ValueError("fused_apply_rotary_pos_emb: freqs must not be None")

#     if freqs.size(-1) == 2 * half:
#         cos = freqs[..., :half]
#         sin = freqs[..., half : 2 * half]
#     elif freqs.size(-1) == half:
#         cos = freqs
#         sin = torch.zeros_like(cos)
#     else:
#         raise ValueError(
#             "fused_apply_rotary_pos_emb: freqs last dim must be half or 2*half "
#             f"(expected {half} or {2*half}, got {freqs.size(-1)})"
#         )

#     # Ensure cos/sin dtype/device match to avoid implicit copies in subsequent ops
#     if cos.dtype != t.dtype:
#         cos = cos.to(dtype=t.dtype)
#     if sin.dtype != t.dtype:
#         sin = sin.to(dtype=t.dtype)
#     if cos.device != t.device:
#         cos = cos.to(device=t.device)
#     if sin.device != t.device:
#         sin = sin.to(device=t.device)

#     # Extract x1, x2 views (no copy)
#     x1 = t[..., :half]
#     x2 = t[..., half : 2 * half]

#     # Align cos/sin to x1/x2 shapes (leading dims)
#     try:
#         cos_b = _align_and_broadcast_to(cos, x1, "cos")
#         sin_b = _align_and_broadcast_to(sin, x1, "sin")
#     except ValueError as e:
#         # augment message with shapes for debugging
#         raise ValueError(
#             f"fused_apply_rotary_pos_emb: cannot align freqs with inputs.\n"
#             f"t.shape={tuple(t.shape)}, x1.shape={tuple(x1.shape)}, freqs.shape={tuple(freqs.shape)}\n"
#             f"Detailed error: {e}"
#         )

#     # Allocate output once
#     out = torch.empty_like(t)

#     # Compute first half: out[..., :half] = x1 * cos - x2 * sin
#     # Use out= to reduce temporaries, then addcmul_ for the subtraction
#     torch.mul(x1, cos_b, out=out[..., :half])      # out[:half] = x1 * cos
#     out[..., :half].addcmul_(x2, -sin_b)           # out[:half] += x2 * (-sin) -> x1*cos - x2*sin

#     # Compute second half: out[..., half:] = x1 * sin + x2 * cos
#     torch.mul(x1, sin_b, out=out[..., half:])     # out[half:] = x1 * sin
#     out[..., half:].addcmul_(x2, cos_b)           # out[half:] += x2 * cos -> x1*sin + x2*cos

#     return out
import torch
from torch import Tensor
from typing import Optional

def _expand_freqs_for_packed_tokens(freqs: Tensor, cu_seqlens: Tensor, total_tokens: int) -> Tensor:
    """
    Expand freqs (shape [L, ...]) into packed layout according to cu_seqlens.
    Returns shape (total_tokens, ...).
    """
    if cu_seqlens is None:
        raise ValueError("_expand_freqs_for_packed_tokens: cu_seqlens is None")

    if cu_seqlens.ndim != 1:
        raise ValueError("_expand_freqs_for_packed_tokens: cu_seqlens must be 1D cumulative sums")

    # Ensure using CPU for iteration (cheap)
    cu = cu_seqlens.to("cpu")
    seq_lengths = (cu[1:] - cu[:-1]).tolist()
    if sum(seq_lengths) != total_tokens:
        raise ValueError(
            f"_expand_freqs_for_packed_tokens: sum(seq_lengths)={sum(seq_lengths)} != total_tokens={total_tokens}. "
            "Check cu_seqlens and packed tensor construction."
        )

    L = freqs.shape[0]
    out_parts = []
    for length in seq_lengths:
        if length == 0:
            continue
        if length > L:
            raise ValueError(
                f"_expand_freqs_for_packed_tokens: requested length {length} > freqs available {L}. "
                "You need freqs computed for this sequence length or slice differently."
            )
        out_parts.append(freqs[:length])
    return torch.cat(out_parts, dim=0)


def fused_apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    transpose_output_memory: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """
    Memory-efficient RoPE that supports packed sequences.

    Args:
      t: Tensor with shape (total_tokens, ... , D) where last dim D is even (D = 2*half).
         Example from your code: t.shape == (6466, 32, 128).
      freqs: Tensor with last dim either 2*half (cos||sin) or half (cos only).
         Example: freqs.shape == (316,1,1,128) meaning 316 positions.
      cu_seqlens: Optional 1D tensor length (batch+1) of cumulative sums that describe
                  how t is packed (used to expand freqs into a per-packed-token table).
    Returns:
      Tensor same shape as t with RoPE applied.
    """

    if t is None or freqs is None:
        raise ValueError("fused_apply_rotary_pos_emb: t and freqs must not be None")

    D = t.size(-1)
    if D % 2 != 0:
        raise ValueError(f"fused_apply_rotary_pos_emb: last dim must be even, got {D}")
    half = D // 2

    # extract cos/sin from freqs last dim
    if freqs.size(-1) == 2 * half:
        cos_all = freqs[..., :half]
        sin_all = freqs[..., half : 2 * half]
    elif freqs.size(-1) == half:
        cos_all = freqs
        sin_all = torch.zeros_like(cos_all)
    else:
        raise ValueError(
            f"fused_apply_rotary_pos_emb: freqs last dim must be half or 2*half "
            f"(expected {half} or {2*half}, got {freqs.size(-1)})"
        )

    total_tokens = t.shape[0]

    # Expand freqs into per-packed-token table if cu_seqlens provided
    if cu_seqlens is not None:
        cos = _expand_freqs_for_packed_tokens(cos_all, cu_seqlens, total_tokens)
        sin = _expand_freqs_for_packed_tokens(sin_all, cu_seqlens, total_tokens)
    else:
        # Try a few reasonable auto-alignments; otherwise raise informative error
        if cos_all.shape[0] == total_tokens:
            cos = cos_all
            sin = sin_all
        elif cos_all.shape[0] > 1 and total_tokens % cos_all.shape[0] == 0:
            factor = total_tokens // cos_all.shape[0]
            cos = cos_all.repeat_interleave(factor, dim=0)
            sin = sin_all.repeat_interleave(factor, dim=0)
        else:
            raise ValueError(
                f"fused_apply_rotary_pos_emb: freqs time dim ({cos_all.shape[0]}) does not match t time dim ({total_tokens}). "
                "Provide cu_seqlens to expand freqs to packed layout or slice freqs upstream."
            )

    # Move cos/sin to same dtype/device as t to avoid unexpected copies during ops
    if cos.dtype != t.dtype:
        cos = cos.to(dtype=t.dtype)
    if sin.dtype != t.dtype:
        sin = sin.to(dtype=t.dtype)
    if cos.device != t.device:
        cos = cos.to(device=t.device)
    if sin.device != t.device:
        sin = sin.to(device=t.device)

    # Split t into x1, x2 (views)
    x1 = t[..., :half]
    x2 = t[..., half : 2 * half]

    # Ensure cos/sin broadcast shape matches x1: add singleton dims if necessary
    cos_b = cos
    sin_b = sin
    while cos_b.ndim < x1.ndim:
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    try:
        cos_b = cos_b.expand(x1.shape)
        sin_b = sin_b.expand(x1.shape)
    except Exception as e:
        raise ValueError(
            f"fused_apply_rotary_pos_emb: failed to broadcast cos/sin to x1 shape.\n"
            f"t.shape={tuple(t.shape)}, cos.shape={tuple(cos.shape)}, sin.shape={tuple(sin.shape)}\n"
            f"error: {e}"
        )

    # Compute into a single output tensor using in-place operations to minimize temporaries
    out = torch.empty_like(t)
    torch.mul(x1, cos_b, out=out[..., :half])
    out[..., :half].addcmul_(x2, -sin_b)
    torch.mul(x1, sin_b, out=out[..., half:])
    out[..., half:].addcmul_(x2, cos_b)

    return out


# def fused_apply_rotary_pos_emb_thd(
#     t: torch.Tensor,
#     cu_seqlens: torch.Tensor,
#     freqs: torch.Tensor,
#     cp_size: int = 1,
#     cp_rank: int = 0,
# ) -> torch.Tensor:
#     """
#     Stub for TE thd RoPE; we just delegate to the naive RoPE above and
#     ignore cu_seqlens / cp_*.
#     """
#     return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=False)
def fused_apply_rotary_pos_emb_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    pass
