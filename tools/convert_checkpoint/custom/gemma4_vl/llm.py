#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""Gemma4-VL LLM body standalone converter (HF <-> Megatron Core).

Mirrors ``custom/llava_onevision2/`` pattern but targets Gemma4-26B-A4B-it
LLM body (30 layers, MoE+dense hybrid, dual RoPE, KV-tying global layers).

Per-layer schema (data-driven from HF ``text_config.layer_types``):
  - sliding layers (25): head_dim=256, num_kv=8, has v_proj
      HF q/k/v_proj ([4096|2048|2048, 2816]) -> MG linear_qkv ([8192, 2816]) concat dim=0
  - global layers (5: indices 5/11/17/23/29): head_dim=512, num_kv=2, KV-tying (no v_proj)
      HF q/k_proj ([8192|1024, 2816]) -> MG linear_qkv ([9216, 2816]) concat dim=0

Common per-layer keys (HF -> MG):
  input_layernorm.weight                -> self_attention.linear_qkv.layer_norm_weight
  post_attention_layernorm.weight       -> self_attention.post_attention_layernorm.weight
  pre_feedforward_layernorm.weight      -> mlp._inner_mlp.pre_feedforward_layernorm.weight
  pre_feedforward_layernorm_2.weight    -> mlp._inner_mlp.moe.pre_feedforward_layernorm_2.weight
  post_feedforward_layernorm.weight     -> mlp._post_ffn.weight  AND  post_feedforward_layernorm.weight (alias)
  post_feedforward_layernorm_1.weight   -> mlp._inner_mlp.post_feedforward_layernorm_1.weight
  post_feedforward_layernorm_2.weight   -> mlp._inner_mlp.post_feedforward_layernorm_2.weight
  layer_scalar                          -> layer_scalar
  self_attn.q_norm.weight               -> self_attention.q_layernorm.weight
  self_attn.k_norm.weight               -> self_attention.k_layernorm.weight
  self_attn.o_proj.weight               -> self_attention.linear_proj.weight
  mlp.gate_proj.weight + mlp.up_proj.weight -> mlp._inner_mlp.dense.linear_fc1.weight (concat dim=0, gate first)
  mlp.down_proj.weight                  -> mlp._inner_mlp.dense.linear_fc2.weight
  experts.gate_up_proj [128, fc1_out, 2816] -> 128 x mlp._inner_mlp.moe.experts.local_experts.{i}.linear_fc1.weight
  experts.down_proj    [128, 2816, fc2_in]  -> 128 x mlp._inner_mlp.moe.experts.local_experts.{i}.linear_fc2.weight
  router.proj.weight [128, 2816]        -> mlp._inner_mlp.moe.router.weight (NO transpose, verified)
  router.scale                          -> mlp._inner_mlp.moe.router.scale
  router.per_expert_scale               -> mlp._inner_mlp.moe.router.per_expert_scale

Non-layer (HF -> MG):
  model.language_model.embed_tokens.weight -> language_model.embedding.word_embeddings.weight
  model.language_model.norm.weight         -> language_model.decoder.final_layernorm.weight

Tied weights: tie_word_embeddings=True; MG ``output_layer`` writes _extra_state=None placeholder
(no .weight key); HF ``lm_head.weight`` is reconstructed from embed_tokens at runtime, not in safetensors.

_extra_state placeholders: every TE-managed metadata sibling is written as
``None``. Transformer Engine 2.2 treats tensor-valued extra_state as pickled
bytes; an empty tensor therefore raises EOFError during load.
"""

from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from os.path import dirname

import torch


SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.arguments import parse_args  # noqa: E402
from convert_checkpoint.custom.llava_onevision2.util import (  # noqa: E402
    load_huggingface_checkpoint,
    load_megatron_checkpoint,
    load_megatron_checkpoint_tp_ep,
    load_megatron_checkpoint_tp_pp_ep,
    save_huggingface_checkpoint,
    save_megatron_checkpoint,
    save_megatron_checkpoint_tp_ep,
    save_megatron_checkpoint_tp_pp_ep,
)


HF_LLM_PREFIX = "model.language_model."
MG_LLM_PREFIX = "language_model."
MG_LAYER_PREFIX = "language_model.decoder.layers."

# HF non-layer LLM key mapping
NON_LAYER_HF_TO_MG = {
    "model.language_model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
    "model.language_model.norm.weight": "language_model.decoder.final_layernorm.weight",
}

# Per-layer norm/layer-scalar mappings: HF tail -> MG tail
NORM_AND_SCALAR_MAP = {
    "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
    "post_attention_layernorm.weight": "self_attention.post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight": "mlp._inner_mlp.pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2.weight": "mlp._inner_mlp.moe.pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm_1.weight": "mlp._inner_mlp.post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight": "mlp._inner_mlp.post_feedforward_layernorm_2.weight",
    "self_attn.q_norm.weight": "self_attention.q_layernorm.weight",
    "self_attn.k_norm.weight": "self_attention.k_layernorm.weight",
    "self_attn.o_proj.weight": "self_attention.linear_proj.weight",
    "layer_scalar": "layer_scalar",
}

# Router 3-tuple: HF tail -> MG tail (NO transpose, verified [128, 2816] both sides)
ROUTER_MAP = {
    "router.proj.weight": "mlp._inner_mlp.moe.router.weight",
    "router.scale": "mlp._inner_mlp.moe.router.scale",
    "router.per_expert_scale": "mlp._inner_mlp.moe.router.per_expert_scale",
}

# Per-layer _extra_state keys to emit (one per fused TE op or normalization).
# Discovered from dump_full_v3.txt layer 0: 9 weight ops with sibling _extra_state.
# Plus self_attention.core_attention._extra_state (single, no weight pair).
PER_LAYER_EXTRA_STATE_TAILS = (
    "mlp._inner_mlp.dense.linear_fc1._extra_state",
    "mlp._inner_mlp.dense.linear_fc2._extra_state",
    "mlp._inner_mlp.moe.pre_feedforward_layernorm_2._extra_state",
    "mlp._inner_mlp.post_feedforward_layernorm_1._extra_state",
    "mlp._inner_mlp.post_feedforward_layernorm_2._extra_state",
    "mlp._inner_mlp.pre_feedforward_layernorm._extra_state",
    "mlp._post_ffn._extra_state",
    "post_feedforward_layernorm._extra_state",
    "self_attention.core_attention._extra_state",
    "self_attention.k_layernorm._extra_state",
    "self_attention.linear_proj._extra_state",
    "self_attention.linear_qkv._extra_state",
    "self_attention.post_attention_layernorm._extra_state",
    "self_attention.q_layernorm._extra_state",
)
# Per-expert _extra_state (128 experts * 2 fc = 256 keys/layer)
PER_EXPERT_EXTRA_STATE_TAILS = (
    "linear_fc1._extra_state",
    "linear_fc2._extra_state",
)

# Non-layer _extra_state (from dump line 590-593)
NON_LAYER_EXTRA_STATE = {
    "language_model.decoder.final_layernorm._extra_state": "tensor",
    "language_model.output_layer._extra_state": "none",
}


def _read_hf_text_config(hf_dir: str) -> dict:
    cfg_path = os.path.join(hf_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["text_config"]


def _resolve_hf_dir(args) -> str:
    """Return the HF directory for config inspection regardless of direction."""
    if args.load_platform == "huggingface":
        return args.load_ckpt_path
    return args.save_ckpt_path


def _resolve_layer_meta(text_config: dict) -> tuple[int, int, list[str]]:
    """Return (num_layers, num_experts, layer_types). All data-driven from HF config."""
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["num_experts"])
    layer_types = list(text_config["layer_types"])
    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types length {len(layer_types)} != num_hidden_layers {num_layers}"
        )
    return num_layers, num_experts, layer_types


def _is_global_layer(layer_types: list[str], layer_idx: int) -> bool:
    return layer_types[layer_idx] == "full_attention"


def _empty_extra_state() -> None:
    """Placeholder extra_state compatible with Transformer Engine 2.2 loaders."""
    return None


def _qkv_pack_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden: int,
) -> torch.Tensor:
    """Pack HF q/k/v projections into Megatron's per-query-group interleaved layout.

    Megatron's :class:`Gemma4SelfAttention.get_query_key_value_tensors` reshapes
    ``mixed_qkv`` as ``(num_kv_heads, (heads_per_group + extra) * head_dim)``
    where ``extra=2`` (sliding, with V) or ``extra=1`` (global, KV-tied, no V).
    That reshape only round-trips correctly when each query group's heads sit
    contiguous with their K (and V) head along the row axis. A naive
    ``torch.cat([Q, K, V], dim=0)`` produces block layout that this reshape
    interprets as scrambled Q/K/V (see ``.cache/v6_qkv_layout_test.py``,
    BLOCK diff vs HF reference ~14, INTERLEAVED diff = 0).
    """
    heads_per_group = num_q_heads // num_kv_heads
    parts = [q.view(num_kv_heads, heads_per_group, head_dim, hidden), k.view(num_kv_heads, 1, head_dim, hidden)]
    if v is not None:
        parts.append(v.view(num_kv_heads, 1, head_dim, hidden))
    return torch.cat(parts, dim=1).reshape(-1, hidden)


def _qkv_unpack_interleaved(
    qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden: int,
    has_v: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Inverse of :func:`_qkv_pack_interleaved` — split MG QKV back into HF q/k/[v]."""
    heads_per_group = num_q_heads // num_kv_heads
    extra = 2 if has_v else 1
    qkv_resh = qkv.view(num_kv_heads, heads_per_group + extra, head_dim, hidden)
    q = qkv_resh[:, :heads_per_group, :, :].reshape(num_q_heads * head_dim, hidden).contiguous()
    k = qkv_resh[:, heads_per_group:heads_per_group + 1, :, :].reshape(num_kv_heads * head_dim, hidden).contiguous()
    if has_v:
        v = qkv_resh[:, heads_per_group + 1:, :, :].reshape(num_kv_heads * head_dim, hidden).contiguous()
        return q, k, v
    return q, k, None


def _hf_to_mg_layer(
    source: dict,
    target: dict,
    layer_idx: int,
    num_experts: int,
    is_global: bool,
) -> None:
    """Convert one LLM layer from HF to MG, in-place into ``target``."""
    hf_pfx = f"{HF_LLM_PREFIX}layers.{layer_idx}."
    mg_pfx = f"{MG_LAYER_PREFIX}{layer_idx}."

    # 1. Norms + layer_scalar + o_proj (1:1 rename)
    for hf_tail, mg_tail in NORM_AND_SCALAR_MAP.items():
        target[mg_pfx + mg_tail] = source[hf_pfx + hf_tail]

    # 2. post_feedforward_layernorm: HF single key -> MG dual alias
    pffn = source[hf_pfx + "post_feedforward_layernorm.weight"]
    target[mg_pfx + "mlp._post_ffn.weight"] = pffn
    target[mg_pfx + "post_feedforward_layernorm.weight"] = pffn

    # 3. Attention QKV concat dim=0
    q = source[hf_pfx + "self_attn.q_proj.weight"]
    k = source[hf_pfx + "self_attn.k_proj.weight"]
    if is_global:
        v = None
        num_kv = _GLOBAL_NUM_KV
        head_dim = _GLOBAL_HEAD_DIM
    else:
        v = source[hf_pfx + "self_attn.v_proj.weight"]
        num_kv = _SLIDING_NUM_KV
        head_dim = _SLIDING_HEAD_DIM
    qkv = _qkv_pack_interleaved(q, k, v, _NUM_Q_HEADS, num_kv, head_dim, _HIDDEN)
    target[mg_pfx + "self_attention.linear_qkv.weight"] = qkv

    # 4. Dense MLP fused fc1 (gate + up concat dim=0)
    gate = source[hf_pfx + "mlp.gate_proj.weight"]
    up = source[hf_pfx + "mlp.up_proj.weight"]
    target[mg_pfx + "mlp._inner_mlp.dense.linear_fc1.weight"] = torch.cat([gate, up], dim=0)
    target[mg_pfx + "mlp._inner_mlp.dense.linear_fc2.weight"] = source[
        hf_pfx + "mlp.down_proj.weight"
    ]

    # 5. MoE experts: 3D HF tensor -> 128 per-expert 2D tensors
    gate_up_3d = source[hf_pfx + "experts.gate_up_proj"]  # [128, fc1_out, hidden]
    down_3d = source[hf_pfx + "experts.down_proj"]  # [128, hidden, fc2_in]
    if gate_up_3d.shape[0] != num_experts or down_3d.shape[0] != num_experts:
        raise ValueError(
            f"layer {layer_idx}: expert dim mismatch "
            f"gate_up={gate_up_3d.shape}, down={down_3d.shape}, num_experts={num_experts}"
        )
    expert_pfx = mg_pfx + "mlp._inner_mlp.moe.experts.local_experts."
    for e in range(num_experts):
        target[f"{expert_pfx}{e}.linear_fc1.weight"] = gate_up_3d[e].contiguous()
        target[f"{expert_pfx}{e}.linear_fc2.weight"] = down_3d[e].contiguous()

    # 6. Router 3-tuple (no transpose, verified)
    for hf_tail, mg_tail in ROUTER_MAP.items():
        target[mg_pfx + mg_tail] = source[hf_pfx + hf_tail]

    # 7. _extra_state placeholders (per-layer common)
    for tail in PER_LAYER_EXTRA_STATE_TAILS:
        target[mg_pfx + tail] = _empty_extra_state()
    # 7b. _extra_state per-expert
    for e in range(num_experts):
        for tail in PER_EXPERT_EXTRA_STATE_TAILS:
            target[f"{expert_pfx}{e}.{tail}"] = _empty_extra_state()


def _mg_to_hf_layer(
    source: dict,
    target: dict,
    layer_idx: int,
    num_experts: int,
    is_global: bool,
) -> None:
    """Convert one LLM layer from MG to HF, in-place into ``target``."""
    hf_pfx = f"{HF_LLM_PREFIX}layers.{layer_idx}."
    mg_pfx = f"{MG_LAYER_PREFIX}{layer_idx}."

    # 1. Norms + layer_scalar + o_proj (1:1 inverse rename)
    for hf_tail, mg_tail in NORM_AND_SCALAR_MAP.items():
        target[hf_pfx + hf_tail] = source[mg_pfx + mg_tail]

    # 2. post_feedforward_layernorm: pick canonical alias source (mlp._post_ffn.weight)
    target[hf_pfx + "post_feedforward_layernorm.weight"] = source[
        mg_pfx + "mlp._post_ffn.weight"
    ]

    # 3. Attention QKV split dim=0 (sliding head_dim=256, global head_dim=512)
    qkv = source[mg_pfx + "self_attention.linear_qkv.weight"]
    if is_global:
        q, k, _ = _qkv_unpack_interleaved(
            qkv, _NUM_Q_HEADS, _GLOBAL_NUM_KV, _GLOBAL_HEAD_DIM, _HIDDEN, has_v=False,
        )
        target[hf_pfx + "self_attn.q_proj.weight"] = q
        target[hf_pfx + "self_attn.k_proj.weight"] = k
    else:
        q, k, v = _qkv_unpack_interleaved(
            qkv, _NUM_Q_HEADS, _SLIDING_NUM_KV, _SLIDING_HEAD_DIM, _HIDDEN, has_v=True,
        )
        target[hf_pfx + "self_attn.q_proj.weight"] = q
        target[hf_pfx + "self_attn.k_proj.weight"] = k
        target[hf_pfx + "self_attn.v_proj.weight"] = v

    # 4. Dense MLP: split fused fc1 -> gate + up (dim=0, gate first)
    fc1 = source[mg_pfx + "mlp._inner_mlp.dense.linear_fc1.weight"]
    half = fc1.shape[0] // 2
    if fc1.shape[0] != 2 * half:
        raise ValueError(
            f"layer {layer_idx}: dense linear_fc1 dim0 {fc1.shape[0]} not divisible by 2"
        )
    gate, up = torch.split(fc1, [half, half], dim=0)
    target[hf_pfx + "mlp.gate_proj.weight"] = gate.contiguous()
    target[hf_pfx + "mlp.up_proj.weight"] = up.contiguous()
    target[hf_pfx + "mlp.down_proj.weight"] = source[
        mg_pfx + "mlp._inner_mlp.dense.linear_fc2.weight"
    ]

    # 5. MoE experts: 128 per-expert 2D -> 3D HF tensor
    expert_pfx = mg_pfx + "mlp._inner_mlp.moe.experts.local_experts."
    gate_up_list = [source[f"{expert_pfx}{e}.linear_fc1.weight"] for e in range(num_experts)]
    down_list = [source[f"{expert_pfx}{e}.linear_fc2.weight"] for e in range(num_experts)]
    target[hf_pfx + "experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0).contiguous()
    target[hf_pfx + "experts.down_proj"] = torch.stack(down_list, dim=0).contiguous()

    # 6. Router 3-tuple (inverse, no transpose)
    for hf_tail, mg_tail in ROUTER_MAP.items():
        target[hf_pfx + hf_tail] = source[mg_pfx + mg_tail]


# Module-level QKV split sizes; resolved from HF text_config at startup.
_SLIDING_Q_DIM = 0
_SLIDING_K_DIM = 0
_SLIDING_V_DIM = 0
_GLOBAL_Q_DIM = 0
_GLOBAL_K_DIM = 0
_NUM_Q_HEADS = 0
_HIDDEN = 0
_SLIDING_NUM_KV = 0
_SLIDING_HEAD_DIM = 0
_GLOBAL_NUM_KV = 0
_GLOBAL_HEAD_DIM = 0


def _resolve_qkv_dims(text_config: dict) -> None:
    """Set module-level QKV split sizes from HF text_config (data-driven)."""
    global _SLIDING_Q_DIM, _SLIDING_K_DIM, _SLIDING_V_DIM, _GLOBAL_Q_DIM, _GLOBAL_K_DIM
    global _NUM_Q_HEADS, _HIDDEN
    global _SLIDING_NUM_KV, _SLIDING_HEAD_DIM, _GLOBAL_NUM_KV, _GLOBAL_HEAD_DIM
    sliding_head_dim = int(text_config["head_dim"])
    global_head_dim = int(text_config.get("global_head_dim", sliding_head_dim))
    num_q_heads = int(text_config["num_attention_heads"])
    num_kv_heads = int(text_config["num_key_value_heads"])
    num_global_kv = int(text_config.get("num_global_key_value_heads", num_kv_heads))
    hidden = int(text_config["hidden_size"])
    _SLIDING_Q_DIM = num_q_heads * sliding_head_dim
    _SLIDING_K_DIM = num_kv_heads * sliding_head_dim
    _SLIDING_V_DIM = num_kv_heads * sliding_head_dim
    _GLOBAL_Q_DIM = num_q_heads * global_head_dim
    _GLOBAL_K_DIM = num_global_kv * global_head_dim
    _NUM_Q_HEADS = num_q_heads
    _HIDDEN = hidden
    _SLIDING_NUM_KV = num_kv_heads
    _SLIDING_HEAD_DIM = sliding_head_dim
    _GLOBAL_NUM_KV = num_global_kv
    _GLOBAL_HEAD_DIM = global_head_dim


def _convert_hf_to_mg(args, text_config: dict) -> dict:
    """Build the full MG state_dict (single rank, TP=PP=EP=1) from HF safetensors."""
    num_layers, num_experts, layer_types = _resolve_layer_meta(text_config)
    print(
        f" > LLM body: num_layers={num_layers}, num_experts={num_experts}, "
        f"global_layers={[i for i, t in enumerate(layer_types) if t == 'full_attention']}"
    )

    source = load_huggingface_checkpoint(args.load_ckpt_path)
    target: dict = {}

    # 1. Non-layer keys (embed, final norm)
    for hf_key, mg_key in NON_LAYER_HF_TO_MG.items():
        target[mg_key] = source[hf_key]

    # 2. Per-layer body
    for i in range(num_layers):
        is_global = _is_global_layer(layer_types, i)
        _hf_to_mg_layer(source, target, i, num_experts, is_global)

    # 3. Non-layer _extra_state placeholders (final_layernorm tensor, output_layer None)
    target["language_model.decoder.final_layernorm._extra_state"] = _empty_extra_state()
    target["language_model.output_layer._extra_state"] = None

    print(f" > MG LLM keys produced: {len(target)}")
    return target


def _convert_mg_to_hf(args, text_config: dict, source: dict) -> dict:
    """Build the full HF state_dict from a single-rank MG ``model`` dict."""
    num_layers, num_experts, layer_types = _resolve_layer_meta(text_config)

    target: dict = {}
    # 1. Non-layer keys
    for hf_key, mg_key in NON_LAYER_HF_TO_MG.items():
        target[hf_key] = source[mg_key]

    # 2. Per-layer body
    for i in range(num_layers):
        is_global = _is_global_layer(layer_types, i)
        _mg_to_hf_layer(source, target, i, num_experts, is_global)

    print(f" > HF LLM keys produced: {len(target)}")
    return target


def _get_non_ep_model_source(state_dict):
    first = state_dict[0]
    if isinstance(first, dict):
        return first["model"]
    first_rank = first[0]
    if isinstance(first_rank, dict):
        return first_rank["model"]
    raise TypeError("Unsupported non-EP checkpoint structure")


# =====================================================================
# P2.8 PP partitioning + per-stage view construction.
#
# Layer numbering contract (verified against Megatron Core
# ``transformer_block.py:691-696`` and ``transformer_layer.py:372``):
#   - TransformerLayer.layer_number is GLOBAL (1-based) and equals
#     ``local + get_transformer_layer_offset(config)``.
#   - When state_dict keys are constructed, TransformerBlock subtracts the
#     stage offset, so on-disk keys use LOCAL indices: each PP rank's shard
#     stores ``language_model.decoder.layers.{0..local_num-1}.*`` regardless
#     of where those layers sit in the global model.
#
# Non-layer routing across PP stages:
#   - ``language_model.embedding.word_embeddings.weight``     -> PP stage 0 only
#   - ``language_model.decoder.final_layernorm.weight``       -> last PP stage only
#   - ``language_model.decoder.final_layernorm._extra_state`` -> last PP stage only
#   - ``language_model.output_layer._extra_state`` (= None)   -> last PP stage only
#     (Gemma4 has tie_word_embeddings=True, so output_layer carries no .weight,
#      only the None placeholder.)
#   - vit / adapter / vision_patch live solely on PP stage 0; this file does
#     NOT touch them — they are merged in by ``merge_megatron.py``.
# =====================================================================


def _partition_layers_for_pp(
    num_layers: int, pp: int, custom: list[int] | None
) -> list[tuple[int, int]]:
    """Return the (start, end) global layer range for each PP stage.

    ``end`` is exclusive; ``end - start`` equals the stage's local layer count.
    When ``custom`` is None the layers are split evenly (Megatron's default
    even-split policy); otherwise ``custom`` must list per-stage counts that
    sum to ``num_layers``.
    """
    if pp <= 0:
        raise ValueError(f"pp must be positive, got {pp}")
    if custom is not None:
        if len(custom) != pp:
            raise ValueError(
                f"custom_pipeline_layers length {len(custom)} != pp {pp}"
            )
        if sum(custom) != num_layers:
            raise ValueError(
                f"custom_pipeline_layers sum {sum(custom)} != num_layers {num_layers}"
            )
        sizes = list(custom)
    else:
        if num_layers % pp != 0:
            raise ValueError(
                f"num_layers ({num_layers}) must be divisible by pp ({pp}) when no "
                "custom_pipeline_layers is supplied; pass --custom_pipeline_layers "
                "to override (e.g. '8,8,7,7')."
            )
        per_stage = num_layers // pp
        sizes = [per_stage] * pp

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        ranges.append((cursor, cursor + size))
        cursor += size
    return ranges


def _per_stage_target_view(
    full_target: dict, stage_idx: int, pp: int, ranges: list[tuple[int, int]]
) -> dict:
    """Return the LOCAL-indexed slice of ``full_target`` belonging to PP stage ``stage_idx``.

    Layer keys ``language_model.decoder.layers.{global}.*`` belonging to this
    stage are renamed to ``language_model.decoder.layers.{local}.*`` where
    ``local = global - ranges[stage_idx][0]``. Non-layer keys are routed:
    embeddings -> stage 0 only; final_layernorm + output_layer -> last stage only.
    """
    start, end = ranges[stage_idx]
    is_first = stage_idx == 0
    is_last = stage_idx == pp - 1
    layer_re = re.compile(
        r"^(language_model\.decoder\.layers\.)(\d+)(\..*)$"
    )

    out: dict = {}
    for key, value in full_target.items():
        m = layer_re.match(key)
        if m is not None:
            gid = int(m.group(2))
            if start <= gid < end:
                local = gid - start
                out[f"{m.group(1)}{local}{m.group(3)}"] = value
            continue
        if key == "language_model.embedding.word_embeddings.weight":
            if is_first:
                out[key] = value
            continue
        if key.startswith("language_model.decoder.final_layernorm.") or key.startswith(
            "language_model.output_layer."
        ):
            if is_last:
                out[key] = value
            continue
        # Defensive: any other top-level LLM key is treated as PP-stage-0-only
        # (matches embedding's policy). Add explicit branches above if new
        # non-layer keys appear in future Megatron versions.
        if is_first:
            out[key] = value
    return out


def _shard_target_to_tp_pp_ep(
    full_target: dict,
    tp: int,
    pp: int,
    ep: int,
    num_experts: int,
    pp_ranges: list[tuple[int, int]],
) -> list[list[list[dict]]]:
    """Return a 3D ``[pp][tp][ep]`` nested list of per-rank ``{"model": ...}`` dicts.

    Each PP stage gets its LOCAL-indexed slice via ``_per_stage_target_view``,
    then that slice is TP+EP-sharded by reusing ``_shard_target_to_tp_ep``.
    """
    state_dict: list[list[list[dict]]] = []
    for p in range(pp):
        per_stage_target = _per_stage_target_view(full_target, p, pp, pp_ranges)
        tp_ep_2d = _shard_target_to_tp_ep(per_stage_target, tp, ep, num_experts)
        state_dict.append(tp_ep_2d)
    return state_dict


def _gather_full_target_from_tp_pp_ep(
    state_dict: list[list[list[dict]]],
    num_experts: int,
    pp_ranges: list[tuple[int, int]],
) -> dict:
    """Inverse of ``_shard_target_to_tp_pp_ep``.

    For each PP stage, un-shards the [tp][ep] 2D view back to a per-stage
    LOCAL-indexed target via ``_unshard_tp_ep_to_target``, then re-maps LOCAL
    layer indices back to GLOBAL by adding ``pp_ranges[stage_idx][0]``.
    Non-layer keys (embedding / final_layernorm / output_layer) are taken from
    the stage where they live.
    """
    pp = len(state_dict)
    layer_re = re.compile(
        r"^(language_model\.decoder\.layers\.)(\d+)(\..*)$"
    )

    full: dict = {}
    for p in range(pp):
        per_stage = _unshard_tp_ep_to_target(state_dict[p], num_experts)
        start, _ = pp_ranges[p]
        for key, value in per_stage.items():
            m = layer_re.match(key)
            if m is not None:
                local = int(m.group(2))
                gid = local + start
                full[f"{m.group(1)}{gid}{m.group(3)}"] = value
            else:
                full[key] = value
    return full


def _parse_custom_pipeline_layers(raw: str | None) -> list[int] | None:
    if raw is None or raw == "":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


# =====================================================================
# P2.7 TP + EP sharding / un-sharding helpers.
#
# Sharding contract (validated against Megatron-Core layer specs and the
# Qwen3-30B-A3B reference under custom/llava_onevision2/):
#
#   TP axis splits per the mapping below; EP axis splits the local-experts
#   list (contiguous chunks of expert IDs).
#
#   Layout decision tree (P2.8):
#       pp==1 ep==1  -> 1D  ``mp_rank_TT/``
#       pp==1 ep>1   -> 2D  ``mp_rank_TT_EEE/``
#       pp>1  any ep -> 3D  ``mp_rank_TT_PPP_EEE/`` (always 3D when PP>1,
#                            with EP=1 collapsing to a single ep slot — this
#                            follows the Qwen3 reference convention).
#
#   Replicated across BOTH TP and EP (every shard holds an identical copy):
#       norms, layer_scalar, router.{proj.weight, scale, per_expert_scale},
#       all *._extra_state placeholders, output_layer._extra_state (None).
#
#   TP-sharded, EP-replicated (every EP shard holds the same TP slice):
#       embedding.word_embeddings.weight              -> chunk dim=0
#       self_attention.linear_qkv.weight              -> chunk dim=0
#                                                        (INTERLEAVED layout;
#                                                         requires tp | num_kv_heads)
#       self_attention.linear_proj.weight             -> chunk dim=1
#       mlp._inner_mlp.dense.linear_fc1.weight        -> per-half (gate,up)
#                                                        chunk dim=0 then re-cat
#       mlp._inner_mlp.dense.linear_fc2.weight        -> chunk dim=1
#
#   EP-sharded (each EP shard owns 1/EP of the experts), with the per-expert
#   tensors themselves TP-sharded:
#       mlp._inner_mlp.moe.experts.local_experts.{i}.linear_fc1.weight
#           -> per-half chunk dim=0 (gate+up fused)
#       mlp._inner_mlp.moe.experts.local_experts.{i}.linear_fc2.weight
#           -> chunk dim=1
#       (per-expert ``*._extra_state`` placeholders also follow EP sharding,
#        but stay TP-replicated as plain placeholders.)
#
#   Local expert indexing: each EP rank stores its local experts as
#   ``local_experts.{0..exp_per_ep-1}`` (re-indexed from 0), matching
#   Megatron's ``SequentialMLP`` convention.
# =====================================================================


def _expert_id_map(num_experts: int, ep: int) -> list[list[int]]:
    if num_experts % ep != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep ({ep}); "
            "Gemma4-VL ships 128 experts which divides evenly by EP=1/2/4/8/16/32/64/128."
        )
    exp_per_ep = num_experts // ep
    return [list(range(e * exp_per_ep, (e + 1) * exp_per_ep)) for e in range(ep)]


def _split_fused_gate_up(weight: torch.Tensor, tp: int, t: int) -> torch.Tensor:
    """Split a fused ``cat([gate, up], dim=0)`` weight for TP rank ``t``.

    Naive ``chunk(tp, dim=0)`` would assign rank 0 the entire ``gate`` half and
    rank 1 the entire ``up`` half (for tp=2), which is wrong: each TP rank must
    receive a slice of BOTH gate and up so its local SwiGLU sees matching column
    ranges. The Megatron contract is ``cat([gate_chunk_t, up_chunk_t], dim=0)``.
    """
    fc1_out = weight.shape[0]
    if fc1_out % 2 != 0:
        raise ValueError(f"fused fc1 dim0 ({fc1_out}) is not even")
    half = fc1_out // 2
    gate, up = weight[:half], weight[half:]
    if half % tp != 0:
        raise ValueError(
            f"fused fc1 half ({half}) not divisible by tp ({tp}); "
            "intermediate_size must be divisible by tp"
        )
    return torch.cat([gate.chunk(tp, dim=0)[t], up.chunk(tp, dim=0)[t]], dim=0).contiguous()


def _gather_fused_gate_up(shards: list[torch.Tensor]) -> torch.Tensor:
    halves = [s.shape[0] // 2 for s in shards]
    if any(s.shape[0] != 2 * h for s, h in zip(shards, halves)):
        raise ValueError("each TP shard of fused fc1 must have an even dim0")
    gates = [s[: halves[i]] for i, s in enumerate(shards)]
    ups = [s[halves[i] :] for i, s in enumerate(shards)]
    return torch.cat(gates + ups, dim=0).contiguous()


def _shard_value_for_tp(key: str, value, tp: int, t: int):
    if not isinstance(value, torch.Tensor) or value.numel() == 0 or tp == 1:
        return value

    if key == "language_model.embedding.word_embeddings.weight":
        return value.chunk(tp, dim=0)[t].contiguous()

    if key.endswith(".self_attention.linear_qkv.weight"):
        # INTERLEAVED layout: dim=0 chunk valid only because tp | num_kv_heads
        # (Gemma4 sliding=8, global=2, both divisible by tp=2).
        return value.chunk(tp, dim=0)[t].contiguous()

    if key.endswith(".self_attention.linear_proj.weight"):
        return value.chunk(tp, dim=1)[t].contiguous()

    if key.endswith(".mlp._inner_mlp.dense.linear_fc1.weight"):
        return _split_fused_gate_up(value, tp, t)

    if key.endswith(".mlp._inner_mlp.dense.linear_fc2.weight"):
        return value.chunk(tp, dim=1)[t].contiguous()

    if key.endswith(".linear_fc1.weight") and ".local_experts." in key:
        return _split_fused_gate_up(value, tp, t)

    if key.endswith(".linear_fc2.weight") and ".local_experts." in key:
        return value.chunk(tp, dim=1)[t].contiguous()

    return value


def _shard_target_to_tp_ep(
    target: dict, tp: int, ep: int, num_experts: int
) -> list[list[dict]]:
    if num_experts % ep != 0:
        raise ValueError(f"num_experts ({num_experts}) not divisible by ep ({ep})")
    expert_map = _expert_id_map(num_experts, ep)
    expert_owner = [-1] * num_experts
    expert_local = [-1] * num_experts
    for e_idx, ids in enumerate(expert_map):
        for local_idx, gid in enumerate(ids):
            expert_owner[gid] = e_idx
            expert_local[gid] = local_idx

    expert_key_re = re.compile(
        r"(.*\.mlp\._inner_mlp\.moe\.experts\.local_experts\.)(\d+)(\..*)"
    )

    state_dict: list[list[dict]] = []
    for t in range(tp):
        ep_list: list[dict] = []
        for e in range(ep):
            shard: dict = {}
            for key, value in target.items():
                m = expert_key_re.match(key)
                if m is not None:
                    gid = int(m.group(2))
                    if expert_owner[gid] != e:
                        continue
                    local_id = expert_local[gid]
                    new_key = f"{m.group(1)}{local_id}{m.group(3)}"
                    shard[new_key] = _shard_value_for_tp(key, value, tp, t)
                else:
                    shard[key] = _shard_value_for_tp(key, value, tp, t)
            ep_list.append({"model": shard})
        state_dict.append(ep_list)
    return state_dict


def _gather_value_for_tp(key: str, shards: list, tp: int):
    head = shards[0]
    if not isinstance(head, torch.Tensor) or head.numel() == 0 or tp == 1:
        return head

    if key == "language_model.embedding.word_embeddings.weight":
        return torch.cat(shards, dim=0).contiguous()

    if key.endswith(".self_attention.linear_qkv.weight"):
        return torch.cat(shards, dim=0).contiguous()

    if key.endswith(".self_attention.linear_proj.weight"):
        return torch.cat(shards, dim=1).contiguous()

    if key.endswith(".mlp._inner_mlp.dense.linear_fc1.weight"):
        return _gather_fused_gate_up(shards)

    if key.endswith(".mlp._inner_mlp.dense.linear_fc2.weight"):
        return torch.cat(shards, dim=1).contiguous()

    if key.endswith(".linear_fc1.weight") and ".local_experts." in key:
        return _gather_fused_gate_up(shards)

    if key.endswith(".linear_fc2.weight") and ".local_experts." in key:
        return torch.cat(shards, dim=1).contiguous()

    return head


def _unshard_tp_ep_to_target(
    state_dict: list[list[dict]], num_experts: int
) -> dict:
    tp = len(state_dict)
    ep = len(state_dict[0])
    if num_experts % ep != 0:
        raise ValueError(f"num_experts ({num_experts}) not divisible by ep ({ep})")
    exp_per_ep = num_experts // ep

    expert_local_re = re.compile(
        r"(.*\.mlp\._inner_mlp\.moe\.experts\.local_experts\.)(\d+)(\..*)"
    )

    target: dict = {}
    canonical_keys: list[str] = []
    for k in state_dict[0][0]["model"].keys():
        if expert_local_re.match(k) is not None:
            continue
        canonical_keys.append(k)
    for key in canonical_keys:
        shards = [state_dict[t][0]["model"][key] for t in range(tp)]
        target[key] = _gather_value_for_tp(key, shards, tp)

    sample_expert_keys = [
        k for k in state_dict[0][0]["model"].keys() if expert_local_re.match(k) is not None
    ]
    expert_tails: set[tuple[str, str]] = set()
    for k in sample_expert_keys:
        m = expert_local_re.match(k)
        if m is not None:
            expert_tails.add((m.group(1), m.group(3)))

    for e in range(ep):
        for local_id in range(exp_per_ep):
            gid = e * exp_per_ep + local_id
            for prefix, suffix in expert_tails:
                local_key = f"{prefix}{local_id}{suffix}"
                global_key = f"{prefix}{gid}{suffix}"
                shards = [state_dict[t][e]["model"][local_key] for t in range(tp)]
                target[global_key] = _gather_value_for_tp(local_key, shards, tp)

    return target


# =====================================================================
# Entry point: dispatch on (load_platform, save_platform).
# =====================================================================
args = parse_args()
text_config = _read_hf_text_config(_resolve_hf_dir(args))
_resolve_qkv_dims(text_config)


if (args.load_platform, args.save_platform) == ("huggingface", "mcore"):
    print(" ====== convert Gemma4-VL LLM body from HuggingFace to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size or 1
    ep = args.expert_parallel_size
    custom = _parse_custom_pipeline_layers(getattr(args, "custom_pipeline_layers", None))
    target = _convert_hf_to_mg(args, text_config)
    num_experts = text_config["num_experts"]
    num_layers = int(text_config["num_hidden_layers"])
    if pp > 1:
        ep_eff = ep if ep is not None and ep > 0 else 1
        pp_ranges = _partition_layers_for_pp(num_layers, pp, custom)
        state_dict = _shard_target_to_tp_pp_ep(target, tp, pp, ep_eff, num_experts, pp_ranges)
        save_megatron_checkpoint_tp_pp_ep(
            state_dict, os.path.join(args.save_ckpt_path, "release")
        )
    elif ep is None or ep == 1:
        state_dict = [{"model": deepcopy(target)} for _ in range(tp)]
        save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, "release"))
    else:
        state_dict = _shard_target_to_tp_ep(target, tp, ep, num_experts)
        save_megatron_checkpoint_tp_ep(state_dict, os.path.join(args.save_ckpt_path, "release"))

elif (args.load_platform, args.save_platform) == ("mcore", "huggingface"):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert Gemma4-VL LLM body from Megatron Core to HuggingFace ======")
    pp = args.pipeline_model_parallel_size or 1
    ep = args.expert_parallel_size
    custom = _parse_custom_pipeline_layers(getattr(args, "custom_pipeline_layers", None))
    num_experts = text_config["num_experts"]
    num_layers = int(text_config["num_hidden_layers"])
    if pp > 1:
        ep_eff = ep if ep is not None and ep > 0 else 1
        pp_ranges = _partition_layers_for_pp(num_layers, pp, custom)
        state_dict = load_megatron_checkpoint_tp_pp_ep(args.load_ckpt_path)
        source = _gather_full_target_from_tp_pp_ep(state_dict, num_experts, pp_ranges)
    elif ep is not None and ep > 1:
        state_dict = load_megatron_checkpoint_tp_ep(args.load_ckpt_path)
        source = _unshard_tp_ep_to_target(state_dict, num_experts)
    else:
        state_dict = load_megatron_checkpoint(args.load_ckpt_path)
        source = _get_non_ep_model_source(state_dict)
    target = _convert_mg_to_hf(args, text_config, source)
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ("mcore", "mcore"):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert Gemma4-VL LLM body from Megatron Core to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size or 1
    ep = args.expert_parallel_size
    custom = _parse_custom_pipeline_layers(getattr(args, "custom_pipeline_layers", None))
    num_experts = text_config["num_experts"]
    num_layers = int(text_config["num_hidden_layers"])
    state_dict = load_megatron_checkpoint(args.load_ckpt_path)
    source = _get_non_ep_model_source(state_dict)
    if pp > 1:
        ep_eff = ep if ep is not None and ep > 0 else 1
        pp_ranges = _partition_layers_for_pp(num_layers, pp, custom)
        new_state_dict = _shard_target_to_tp_pp_ep(
            source, tp, pp, ep_eff, num_experts, pp_ranges
        )
        save_megatron_checkpoint_tp_pp_ep(
            new_state_dict, os.path.join(args.save_ckpt_path, "release")
        )
    elif ep is None or ep == 1:
        new_state_dict = [{"model": deepcopy(source)} for _ in range(tp)]
        save_megatron_checkpoint(new_state_dict, os.path.join(args.save_ckpt_path, "release"))
    else:
        new_state_dict = _shard_target_to_tp_ep(source, tp, ep, num_experts)
        save_megatron_checkpoint_tp_ep(
            new_state_dict, os.path.join(args.save_ckpt_path, "release")
        )

else:
    raise NotImplementedError(
        f"Unsupported (load, save) = ({args.load_platform}, {args.save_platform})"
    )
