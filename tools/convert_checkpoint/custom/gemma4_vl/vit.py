#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""Gemma4-VL vision-tower body standalone converter.

Mirrors ``custom/llava_onevision2/vision_model.py`` for OV2 but targets
the Gemma4 27-layer ViT. Differences from generic ViT converters:

- The MG ViT module is the entire ``Gemma4VisionTower`` (top-level
  ``vision_model`` attribute on ``Gemma4VL``); per-layer keys live under
  ``vision_model.layers.{i}.*``.
- The HF ViT module is ``Gemma4VisionEncoder`` nested inside
  ``Gemma4VisionModel``; per-layer keys live under
  ``model.vision_tower.encoder.layers.{i}.*``.
- All projection weights (q/k/v/o, gate/up/down) are wrapped in
  ``Gemma4ClippableLinear`` whose underlying ``nn.Linear`` lives at
  attribute ``.linear``, so HF and MG keys both end with ``.linear.weight``.
- Every projection is a plain ``nn.Linear`` (NOT ColumnParallel/RowParallel),
  so weights are TP-replicated, NOT sharded.
- Each layer has 5 RMSNorm tensors (input / post_attention / pre_feedforward /
  post_feedforward / q_norm + k_norm = 7 with attn norms), all replicated.
- Patch embedder, std_bias, std_scale are handled by ``vision_patch.py``;
  this file ONLY converts ``vision_model.layers.{i}.*``.

Per-layer key set (13 keys, identical between HF and MG modulo prefix):
    input_layernorm.weight
    post_attention_layernorm.weight
    pre_feedforward_layernorm.weight
    post_feedforward_layernorm.weight
    self_attn.{q,k,v,o}_proj.linear.weight   (4)
    self_attn.{q,k}_norm.weight              (2)
    mlp.{gate,up,down}_proj.linear.weight    (3)

Total: 27 layers * 13 keys = 351 keys. Plus 4 patch keys (vision_patch.py)
gives 355, matching the HF vision_tower.* count.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from os.path import dirname


SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.arguments import parse_args  # noqa: E402
from convert_checkpoint.custom.llava_onevision2.util import (  # noqa: E402
    load_huggingface_checkpoint,
    load_megatron_checkpoint,
    load_megatron_checkpoint_tp_ep,
    save_huggingface_checkpoint,
    save_megatron_checkpoint,
)


HF_PREFIX = "model.vision_tower.encoder.layers."
MG_PREFIX = "vision_model.layers."

PER_LAYER_TAILS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_proj.linear.weight",
    "self_attn.k_proj.linear.weight",
    "self_attn.v_proj.linear.weight",
    "self_attn.o_proj.linear.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.linear.weight",
    "mlp.up_proj.linear.weight",
    "mlp.down_proj.linear.weight",
)


args = parse_args()


def _read_vit_num_layers_from_hf_config(hf_dir: str) -> int:
    cfg_path = os.path.join(hf_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["vision_config"]["num_hidden_layers"])


def _resolve_vit_num_layers() -> int:
    if (args.load_platform, args.save_platform) == ("huggingface", "mcore"):
        return _read_vit_num_layers_from_hf_config(args.load_ckpt_path)
    if (args.load_platform, args.save_platform) == ("mcore", "huggingface"):
        return _read_vit_num_layers_from_hf_config(args.save_ckpt_path)
    return _read_vit_num_layers_from_hf_config(args.load_ckpt_path)


def _get_non_ep_model_source(state_dict):
    first = state_dict[0]
    if isinstance(first, dict):
        return first["model"]
    first_rank = first[0]
    if isinstance(first_rank, dict):
        return first_rank["model"]
    raise TypeError("Unsupported non-EP checkpoint structure")


def _build_layer_mapping(num_layers: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in range(num_layers):
        for tail in PER_LAYER_TAILS:
            pairs.append((f"{MG_PREFIX}{i}.{tail}", f"{HF_PREFIX}{i}.{tail}"))
    return pairs


if (args.load_platform, args.save_platform) == ("mcore", "huggingface"):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert Gemma4-VL ViT body from Megatron Core to HuggingFace ======")
    ep = args.expert_parallel_size
    if ep is not None and ep > 1:
        state_dict = load_megatron_checkpoint_tp_ep(args.load_ckpt_path)
        source = state_dict[0][0]["model"]
    else:
        state_dict = load_megatron_checkpoint(args.load_ckpt_path)
        source = _get_non_ep_model_source(state_dict)

    num_layers = _resolve_vit_num_layers()
    target = {}
    for mg_key, hf_key in _build_layer_mapping(num_layers):
        target[hf_key] = source[mg_key]
    print(f" > converted {len(target)} ViT body keys ({num_layers} layers x {len(PER_LAYER_TAILS)} tails)")
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ("huggingface", "mcore"):
    print(" ====== convert Gemma4-VL ViT body from HuggingFace to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    source = load_huggingface_checkpoint(args.load_ckpt_path)

    num_layers = _resolve_vit_num_layers()
    target = {}
    for mg_key, hf_key in _build_layer_mapping(num_layers):
        target[mg_key] = source[hf_key]
    print(f" > converted {len(target)} ViT body keys ({num_layers} layers x {len(PER_LAYER_TAILS)} tails)")

    # ViT/adapter/patch always save 1D [tp] (mp_rank_TT) regardless of PP/EP, mirroring
    # the OV2 Qwen3 convention; merge_megatron.py broadcasts them across PP stage 0
    # (and across EP) at merge time, since these modules live solely on PP stage 0.
    state_dict = [{"model": deepcopy(target)} for _ in range(tp)]
    save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, "release"))

elif (args.load_platform, args.save_platform) == ("mcore", "mcore"):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert Gemma4-VL ViT body from Megatron Core to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    state_dict = load_megatron_checkpoint(args.load_ckpt_path)
    source = _get_non_ep_model_source(state_dict)

    target = deepcopy(source)
    new_state_dict = [{"model": deepcopy(target)} for _ in range(tp)]
    save_megatron_checkpoint(new_state_dict, os.path.join(args.save_ckpt_path, "release"))

else:
    raise NotImplementedError
