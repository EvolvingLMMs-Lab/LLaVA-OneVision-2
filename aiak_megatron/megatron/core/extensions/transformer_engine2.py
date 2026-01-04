# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""transformer engine utilities - PURE PYTORCH FALLBACK (no transformer_engine).

This module mirrors the public API used elsewhere (TENorm, TELinear, TEColumnParallelLinear,
TERowParallelLinear, TEDotProductAttention, TEGroupedLinear, etc.) but implements them
with pure PyTorch fallbacks so Transformer-Engine is not required.
"""

import dataclasses
import io
import os
import pickle
import warnings
from typing import Any, Callable, Optional, Tuple, Dict

import torch
import torch.nn as nn
import math
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter

# Megatron imports retained for compatibility (these should exist in your environment)
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.random import get_data_parallel_rng_tracker_name
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_te_version, is_te_min_version  # these helpers are still useful

# ------------------------
# Small helpers & fallbacks
# ------------------------

def _get_extra_te_kwargs(config: TransformerConfig):
    """Return minimal kwargs to mimic transformer-engine behavior. For fallback this is minimal."""
    # Provide dtype and device guidance like TE would
    params_dtype = getattr(config, "params_dtype", torch.float32)
    extra = {"params_dtype": params_dtype}
    # Expose device selection hints for CPU/meta initialization flags
    if getattr(config, "use_cpu_initialization", False):
        extra["device"] = "cpu"
    elif getattr(config, "init_model_with_meta_device", False):
        extra["device"] = "meta"
    else:
        # default to current cuda device if available
        if torch.cuda.is_available():
            extra["device"] = torch.cuda.current_device()
        else:
            extra["device"] = "cpu"
    return extra


def condition_init_method(config, init_method):
    """Condition TE init_method on config.perform_initialization (mirror TE wrapper)."""
    return init_method if getattr(config, "perform_initialization", True) else (lambda w: None)


class _RMSNorm(nn.Module):
    """Simple RMSNorm implementation used by fallback TENorm."""
    def __init__(self, hidden_size: int, eps: float = 1e-8, zero_centered_gamma: bool = False, use_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.zeros(hidden_size) if zero_centered_gamma else torch.ones(hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        out = x_norm * self.weight
        if self.use_bias:
            out = out + self.bias
        return out


class TENorm:
    """Factory that returns LayerNorm or RMSNorm fallback matching TE behavior."""
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        zero_centered = getattr(config, "layernorm_zero_centered_gamma", False)
        normalization = getattr(config, "normalization", "LayerNorm")
        if normalization == "LayerNorm":
            ln = nn.LayerNorm(normalized_shape=hidden_size, eps=eps, elementwise_affine=True)
            # match TE zero-centered gamma behavior
            with torch.no_grad():
                ln.weight.fill_(0.0 if zero_centered else 1.0)
                ln.bias.zero_()
            return ln
        elif normalization == "RMSNorm":
            return _RMSNorm(hidden_size=hidden_size, eps=eps, zero_centered_gamma=zero_centered, use_bias=False)
        else:
            raise ValueError("Only LayerNorm and RMSNorm are currently supported in TENorm fallback.")


def condition_init_method(config, init_method: Callable) -> Callable:
    """Return provided init_method or fallback initializer (compat helper)."""
    if callable(init_method):
        return init_method
    def _f(tensor: torch.Tensor):
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
    return _f


# ------------------------
# TELinear pure-PyTorch fallback
# ------------------------

def _apply_linear_with_multiplicity(x_flat: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], in_features: int, out_features: int, te_return_bias: bool):
    """
    Robust linear application that handles last_dim equal to in_features or
    last_dim divisible by in_features (treat as M blocks, apply linear to each block and average).
    Returns (out, bias_or_none) where out has same leading dims as x_flat except last -> out_features.
    """
    last_dim = x_flat.shape[-1]
    if last_dim == in_features:
        out = torch.nn.functional.linear(x_flat, weight, bias)
        return out, (bias if te_return_bias and bias is not None else None)
    if last_dim % in_features == 0:
        M = last_dim // in_features
        lead = x_flat.shape[:-1]
        # reshape: (..., M, in_features)
        x_blocks = x_flat.reshape(*lead, M, in_features)
        prod_lead = 1
        for s in lead:
            prod_lead *= s
        x_blocks_flat = x_blocks.reshape(prod_lead * M, in_features)
        out_blocks_flat = torch.nn.functional.linear(x_blocks_flat, weight, bias)
        out_blocks = out_blocks_flat.view(*lead, M, out_features)
        out = out_blocks.mean(dim=-2)
        warnings.warn(
            f"_apply_linear_with_multiplicity: last_dim {last_dim} interpreted as {M} blocks of {in_features}; "
            "applied linear to each and averaged results.",
            UserWarning,
        )
        return out, (bias if te_return_bias and bias is not None else None)
    raise RuntimeError(
        f"TELinear/_apply_linear_with_multiplicity: last_dim ({last_dim}) is not equal to in_features ({in_features}) "
        "and is not divisible by in_features."
    )


class TELinear(nn.Module):
    """Pure-PyTorch replacement implementing the TE Linear wrapper API surface."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
    ):
        super().__init__()
        self.config = config
        self.parallel_mode = parallel_mode
        self.is_expert = is_expert

        if skip_weight_param_allocation:
            raise ValueError("skip_weight_param_allocation unsupported in TELinear fallback")

        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = getattr(config, "disable_parameter_transpose_cache", False)

        if getattr(config, "sequence_parallel", False):
            warnings.warn("sequence_parallel requested but unsupported in TELinear fallback; ignored.")

        if self.parallel_mode not in (None, "duplicated", "column", "row"):
            warnings.warn(f"Unknown parallel_mode={self.parallel_mode}; proceeding without TP semantics")

        self.in_features = input_size
        self.out_features = output_size

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), dtype=getattr(config, "params_dtype", torch.float32)))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, dtype=getattr(config, "params_dtype", torch.float32)))
        else:
            self.register_parameter("bias", None)

        init_fn = condition_init_method(config, init_method)
        try:
            init_fn(self.weight)
        except Exception:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if getattr(self, "bias", None) is not None:
            nn.init.zeros_(self.bias)

        for p in (p for p in (self.weight, getattr(self, "bias", None)) if p is not None):
            if is_expert:
                expert_parallel_size = getattr(self.config, "expert_model_parallel_size", 1)
                setattr(p, "allreduce", not (expert_parallel_size > 1))
            else:
                setattr(p, "allreduce", True)
                if parallel_mode == "duplicated":
                    setattr(p, "sequence_parallel", getattr(self.config, "sequence_parallel", False))

    def set_tensor_parallel_group(self, tp_group):
        warnings.warn("set_tensor_parallel_group called — no TP support in TELinear fallback", UserWarning)
        self.tp_group = tp_group

    def forward(self, x: torch.Tensor):
        if x is None:
            raise RuntimeError("TELinear.forward received x=None")

        orig_shape = tuple(x.shape)

        def to_batch_first(t: torch.Tensor):
            if t.ndim < 2:
                return t, False
            if t.shape[0] > t.shape[1]:
                return t.transpose(0, 1), True
            return t, False

        if x.ndim == 4:
            xb, was_seq_first = to_batch_first(x)
            B, S, H, D = xb.shape
            x_flat = xb.reshape(B, S, H * D)
            restore_permute = was_seq_first
            lead = (B, S)
        elif x.ndim == 3:
            xb, was_seq_first = to_batch_first(x)
            B, S, HP = xb.shape
            x_flat = xb
            restore_permute = was_seq_first
            lead = (B, S)
        elif x.ndim == 2:
            x_flat = x
            restore_permute = False
            lead = (x_flat.shape[0],)
        else:
            x_flat = x.contiguous().view(-1, x.shape[-1])
            restore_permute = False
            lead = x.shape[:-1]

        out, returned_bias = _apply_linear_with_multiplicity(x_flat, self.weight, getattr(self, "bias", None), self.in_features, self.out_features, self.te_return_bias)

        # restore layout
        if x.ndim == 4:
            if out.ndim == 2 and out.shape[0] == (lead[0] * lead[1]):
                out = out.view(lead[0], lead[1], self.out_features)
            if restore_permute:
                out = out.transpose(0, 1)
        elif x.ndim == 3:
            if out.ndim == 2 and out.shape[0] == (lead[0] * lead[1]):
                out = out.view(lead[0], lead[1], self.out_features)
            if restore_permute:
                out = out.transpose(0, 1)
        elif x.ndim == 2:
            pass
        else:
            out = out.view(*lead, self.out_features)

        if self.te_return_bias:
            return out, returned_bias
        return out, None

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        assert self.parallel_mode in (None, "duplicated"), "sharded_state_dict only supported for duplicated in fallback"
        return self.state_dict(prefix=prefix)

    def backward_dw(self):
        if getattr(self.config, "split_bw", False):
            warnings.warn("split_bw True but not implemented in TELinear fallback; no-op")


# ------------------------
# TELayerNormColumnParallelLinear fallback
# ------------------------

class TELayerNormColumnParallelLinear(nn.Module):
    """Fallback for LayerNorm + Column-Parallel Linear combination (no TP)."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
    ):
        super().__init__()
        self.config = config

        if gather_output:
            raise ValueError("gather_output=True unsupported in fallback")
        if is_expert:
            raise ValueError("MoE (is_expert=True) unsupported in fallback")
        if skip_weight_param_allocation:
            raise ValueError("skip_weight_param_allocation unsupported in fallback")

        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = getattr(config, "disable_parameter_transpose_cache", False)

        self.in_features = input_size
        self.out_features = output_size
        self.use_bias = bias

        normalization = getattr(config, "normalization", "LayerNorm")
        eps = getattr(config, "layernorm_epsilon", 1e-5)
        zero_centered = getattr(config, "layernorm_zero_centered_gamma", False)
        if normalization == "LayerNorm":
            self.layernorm = nn.LayerNorm(normalized_shape=input_size, eps=eps, elementwise_affine=True)
            if zero_centered:
                with torch.no_grad():
                    self.layernorm.weight.zero_()
                    self.layernorm.bias.zero_()
        elif normalization == "RMSNorm":
            self.layernorm = _RMSNorm(input_size, eps=eps, zero_centered_gamma=zero_centered, use_bias=False)
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        self.weight = Parameter(torch.empty((self.out_features, self.in_features), dtype=getattr(config, "params_dtype", torch.float32)))
        if self.use_bias:
            self.bias = Parameter(torch.empty(self.out_features, dtype=getattr(config, "params_dtype", torch.float32)))
        else:
            self.register_parameter("bias", None)

        init_fn = condition_init_method(config, init_method)
        try:
            init_fn(self.weight)
        except Exception:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if getattr(self, "bias", None) is not None:
            nn.init.zeros_(self.bias)

        for p in (self.weight, self.bias) if self.bias is not None else (self.weight,):
            setattr(p, "allreduce", True)
            setattr(p, "sequence_parallel", getattr(config, "sequence_parallel", False))

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        _is_first_microbatch = None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        ln_out = self.layernorm(x)

        # Flatten to apply linear helper robustly
        lead = ln_out.shape[:-1]
        flat = ln_out.contiguous().view(-1, ln_out.shape[-1])
        out_flat, returned_bias = _apply_linear_with_multiplicity(flat, self.weight, getattr(self, "bias", None), self.in_features, self.out_features, self.te_return_bias)
        out = out_flat.view(*lead, self.out_features)

        self.is_first_microbatch = False
        if self.te_return_bias:
            return out, returned_bias
        return out, None

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self.state_dict(prefix=prefix)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        out = self.out_features
        per_shard = divide(out, world_size)
        start = rank * per_shard
        end = start + per_shard if rank != (world_size - 1) else out
        sd = {}
        sd[prefix + "weight"] = self.weight.data[start:end].clone()
        if self.use_bias and "bias" in self._parameters:
            sd[prefix + "bias"] = self.bias.data[start:end].clone()
        return sd

    def __repr__(self):
        return f"{type(self).__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"

    def backward_dw(self):
        if getattr(self.config, "split_bw", False):
            warnings.warn("split_bw True but not implemented in fallback; no-op")


# Column/Row wrappers (reuse TELinear)
class TEColumnParallelLinear(TELinear):
    def __init__(self, input_size: int, output_size: int, *, config: ModelParallelConfig, init_method: Callable, gather_output: bool, bias: bool, skip_bias_add: bool, is_expert: bool, skip_weight_param_allocation: bool = False, tp_comm_buffer_name: Optional[str] = None):
        if gather_output:
            raise ValueError('gather_output True unsupported in fallback')
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(condition_init_method(config, init_method) if not getattr(config, "use_cpu_initialization", False) else (lambda w: None)),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        )


class TERowParallelLinear(TELinear):
    def __init__(self, input_size: int, output_size: int, *, config: ModelParallelConfig, init_method: Callable, bias: bool, input_is_parallel: bool, skip_bias_add: bool, is_expert: bool, tp_comm_buffer_name: Optional[str] = None):
        if not input_is_parallel:
            raise ValueError("input_is_parallel=False unsupported in fallback")
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(condition_init_method(config, init_method) if not getattr(config, "use_cpu_initialization", False) else (lambda w: None)),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        )


# ------------------------
# Chunked attention helper + TEDotProductAttention fallback
# ------------------------

def sdp_chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    use_sdp: bool,
    mha_module: Optional[nn.MultiheadAttention] = None,
    chunk_size: int = 64,
):
    """
    Compute attention in chunks along the sequence axis to reduce peak memory.

    q,k,v: (B, S, H, D)
    attention_mask: SDP-compatible or None
    chunk_size: number of query tokens per chunk
    """
    B, S, H, D = q.shape
    out = torch.empty_like(q)

    if use_sdp:
        k_flat = k.reshape(B, S, H * D)
        v_flat = v.reshape(B, S, H * D)
    else:
        embed_dim = H * D
        k_flat = k.reshape(B, S, embed_dim)
        v_flat = v.reshape(B, S, embed_dim)

    have_full_mask = attention_mask is not None and attention_mask.ndim >= 2

    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        q_chunk = q[:, start:end]  # (B, chunk, H, D)
        if use_sdp:
            q_flat = q_chunk.reshape(B, end - start, H * D)
            if have_full_mask:
                try:
                    attn_mask_chunk = attention_mask[:, start:end, :]
                except Exception:
                    attn_mask_chunk = None
            else:
                attn_mask_chunk = None

            out_flat_chunk = torch.nn.functional.scaled_dot_product_attention(
                q_flat, k_flat, v_flat, attn_mask=attn_mask_chunk, dropout_p=dropout_p, is_causal=is_causal
            )
            out[:, start:end] = out_flat_chunk.reshape(B, end - start, H, D)
            del q_flat, attn_mask_chunk, out_flat_chunk
            torch.cuda.empty_cache()
        else:
            q_flat = q_chunk.reshape(B, end - start, H * D)
            if attention_mask is not None:
                warnings.warn("attention_mask ignored in chunked MHA fallback. Consider using SDP for masks.")
            out_flat_chunk, _ = mha_module(q_flat, k_flat, v_flat, attn_mask=None)
            out[:, start:end] = out_flat_chunk.reshape(B, end - start, H, D)
            del q_flat, out_flat_chunk
            torch.cuda.empty_cache()

    return out


class TEDotProductAttention(nn.Module):
    """
    Chunked, robust fallback attention wrapper. Accepts (B,S,H,D), (B,S,E), (S,B,H,D), (S,B,E).
    """
    cp_stream: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout if attention_dropout is not None else getattr(config, "attention_dropout", 0.0)
        self.softmax_scale = softmax_scale
        self.kv_channels = (k_channels, v_channels) if (k_channels is not None and v_channels is not None) else getattr(config, "kv_channels", None)
        self.num_heads = getattr(config, "num_attention_heads_per_partition", None) or getattr(config, "num_attention_heads", None)
        self.qkv_format = 'sbhd'
        self.te_forward_mask_type = False

        self.kept_packed_seq_params = set(field.name for field in dataclasses.fields(PackedSeqParams))
        self._use_pytorch_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self._mha: Optional[nn.MultiheadAttention] = None

        # chunk size tuneable from config
        self.attn_chunk_size = getattr(config, "attn_chunk_size", 64)

    def _to_batch_first(self, t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if t.ndim < 3:
            raise ValueError(f"Unsupported tensor ndim {t.ndim} in attention; expected >=3")
        if t.shape[0] > t.shape[1]:
            return t.transpose(0, 1), True
        return t, False

    def _ensure_b_s_h_d(self, tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, int, int, int, int]:
        t_bsf, _ = self._to_batch_first(tensor)
        if t_bsf.ndim == 4:
            B, S, H, D = t_bsf.shape
            return t_bsf, B, S, H, D
        if t_bsf.ndim == 3:
            if self.num_heads is None:
                raise RuntimeError(f"num_attention_heads unknown; cannot reshape 3D {name} tensor")
            B, S, E = t_bsf.shape
            H = self.num_heads
            if E % H != 0:
                raise ValueError(
                    f"Embedding dim {E} (for {name}) is not divisible by num_heads {H}."
                )
            D = E // H
            t_bsf = t_bsf.view(B, S, H, D)
            return t_bsf, B, S, H, D
        raise ValueError(f"Unsupported tensor ndim for {name}: {tensor.ndim}")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        q, B, S, H, D = self._ensure_b_s_h_d(query, "query")
        k, _, _, _, _ = self._ensure_b_s_h_d(key, "key")
        v, _, _, _, _ = self._ensure_b_s_h_d(value, "value")

        if q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
            raise RuntimeError("Empty q/k/v passed to attention")

        is_causal = attn_mask_type == AttnMaskType.causal

        if not self._use_pytorch_sdp and self._mha is None:
            if self.num_heads is None:
                raise RuntimeError("num_attention_heads unknown for MultiheadAttention fallback")
            embed_dim = H * D
            self._mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.num_heads, dropout=self.attention_dropout, batch_first=True)

        chunk_size = max(1, int(getattr(self, "attn_chunk_size", 64)))
        out = sdp_chunked_attention(
            q, k, v,
            attention_mask,
            dropout_p=self.attention_dropout,
            is_causal=is_causal,
            use_sdp=self._use_pytorch_sdp,
            mha_module=self._mha,
            chunk_size=chunk_size,
        )
        return out


# ------------------------
# Grouped linear fallbacks
# ------------------------

class TEGroupedLinear(nn.Module):
    """Grouped linear fallback implemented as ModuleList of TELinear."""
    def __init__(self, num_gemms: int, input_size: int, output_size: int, *, parallel_mode: Optional[str], config: ModelParallelConfig, init_method: Callable, bias: bool, skip_bias_add: bool, is_expert: bool = False, tp_comm_buffer_name: Optional[str] = None):
        super().__init__()
        self.num_gemms = num_gemms
        self.config = config
        self.submodules = nn.ModuleList(
            [
                TELinear(input_size, output_size, parallel_mode=parallel_mode, config=config, init_method=init_method, bias=bias, skip_bias_add=skip_bias_add, skip_weight_param_allocation=False, tp_comm_buffer_name=tp_comm_buffer_name, is_expert=is_expert)
                for _ in range(num_gemms)
            ]
        )
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = getattr(config, "disable_parameter_transpose_cache", False)

    def forward(self, x, m_splits=None):
        outs = []
        for module in self.submodules:
            out = module(x)
            outs.append(out[0] if isinstance(out, tuple) else out)
        stacked = torch.stack(outs, dim=0)
        if self.te_return_bias:
            biases = [m.bias for m in self.submodules if getattr(m, "bias", None) is not None]
            return stacked, biases
        return stacked, None


class TEColumnParallelGroupedLinear(TEGroupedLinear):
    def __init__(self, num_gemms: int, input_size: int, output_size: int, *, config: ModelParallelConfig, init_method: Callable, bias: bool, skip_bias_add: bool, is_expert: bool, tp_comm_buffer_name: Optional[str] = None):
        super().__init__(num_gemms=num_gemms, input_size=input_size, output_size=output_size, parallel_mode="column", config=config, init_method=condition_init_method(config, init_method), bias=bias, skip_bias_add=skip_bias_add, is_expert=is_expert, tp_comm_buffer_name=tp_comm_buffer_name)


class TERowParallelGroupedLinear(TEGroupedLinear):
    def __init__(self, num_gemms: int, input_size: int, output_size: int, *, config: ModelParallelConfig, init_method: Callable, bias: bool, skip_bias_add: bool, is_expert: bool, tp_comm_buffer_name: Optional[str] = None):
        super().__init__(num_gemms=num_gemms, input_size=input_size, output_size=output_size, parallel_mode="row", config=config, init_method=condition_init_method(config, init_method), bias=bias, skip_bias_add=skip_bias_add, is_expert=is_expert, tp_comm_buffer_name=tp_comm_buffer_name)


# ------------------------
# Stubs and TE-like utilities
# ------------------------

class TEDelayedScaling:
    """Stub for TE DelayedScaling; FP8 not implemented in fallback."""
    def __init__(self, config: ModelParallelConfig, fp8_format: int, override_linear_precision: tuple = (False, False, False)):
        warnings.warn("TEDelayedScaling stub: FP8 not implemented in pure-PyTorch fallback.", UserWarning)
        self.config = config


class TECudaRNGStatesTracker:
    """Small wrapper mimicking TE's RNG tracker surface."""
    def __init__(self):
        self._is_initialized = False
    def reset(self):
        self._is_initialized = False
    def is_initialized(self):
        return self._is_initialized
    def set_states(self, states):
        self._is_initialized = True
    def add(self, name, seed):
        self._is_initialized = True


def fused_apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, transpose_output_memory: bool = False) -> torch.Tensor:
    """Fallback non-fused RoPE. This is a simple placeholder; keep if your code expects the function to exist."""
    warnings.warn("Using non-fused RoPE fallback. Consider replacing with optimized implementation.", UserWarning)
    # naive elementwise application for sbhd layout: t: (..., H, D) and freqs shaped accordingly.
    # This fallback simply returns input unchanged (safe default). If you need actual RoPE behavior,
    # provide your project's RoPE implementation here.
    return t


def fused_apply_rotary_pos_emb_thd(t: torch.Tensor, cu_seqlens: torch.Tensor, freqs: torch.Tensor, cp_size: int = 1, cp_rank: int = 0) -> torch.Tensor:
    warnings.warn("Using non-fused RoPE-thd fallback. Consider replacing with optimized implementation.", UserWarning)
    return t


fused_permute = None
fused_unpermute = None
fused_sort_chunks_by_index = None

Fp8Padding = None
Fp8Unpadding = None


def get_cpu_offload_context(enabled, num_layers, model_layers, activation_offloading, weight_offloading):
    warnings.warn("get_cpu_offload_context fallback: no TE CPU-offload context available.", UserWarning)
    return None, (lambda: None)


def te_checkpoint(
    forward_func,
    distribute_saved_activations,
    get_rng_state_tracker,
    tp_group,
    hidden_states,
    attention_mask,
    attn_mask_type,
    context,
    context_mask,
    rotary_pos_emb,
    **kwargs,
):
    """Fallback checkpointing using torch.utils.checkpoint (best-effort)."""
    import torch.utils.checkpoint as cp
    # Best-effort wrapper: TE's checkpoint signature differs across versions; we use a simple approach
    return cp.checkpoint(forward_func, hidden_states, attention_mask, context)


# End of pure-PyTorch fallback module
