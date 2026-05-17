#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""Gemma4-VL adapter standalone converter.

Mirrors ``custom/llava_onevision2/adapter.py`` but targets Gemma4-VL.
Differences from the OV2 adapter:

- Gemma4 ``Gemma4Adapter`` has only ONE trainable parameter:
  ``embedding_projection.weight``. The ``embedding_pre_projection_norm`` is a
  ``Gemma4RMSNorm`` with ``with_scale=False`` so it has no learnable weight,
  and is therefore absent from both HF and Megatron state_dict.
- ``embedding_projection`` is a plain ``nn.Linear``, NOT a ColumnParallelLinear,
  so the weight is replicated across TP ranks (no sharding).
- No ``_extra_state`` blobs are written: the Gemma4 adapter does not contain
  TransformerEngine modules.

The mapping is loaded from ``args.common_config_path``
(Megatron-key → HF-key); the only entry is ``adapter.embedding_projection.weight
-> model.embed_vision.embedding_projection.weight``.
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


args = parse_args()
name_map: dict[str, str] = {}
with open(args.common_config_path, "r", encoding="utf-8") as f:
    name_map = json.loads(f.read())


def _get_non_ep_model_source(state_dict):
    first = state_dict[0]
    if isinstance(first, dict):
        return first["model"]
    first_rank = first[0]
    if isinstance(first_rank, dict):
        return first_rank["model"]
    raise TypeError("Unsupported non-EP checkpoint structure")


if (args.load_platform, args.save_platform) == ("mcore", "huggingface"):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert Gemma4-VL adapter from Megatron Core to HuggingFace ======")
    ep = args.expert_parallel_size
    target = {}
    if ep is not None and ep > 1:
        state_dict = load_megatron_checkpoint_tp_ep(args.load_ckpt_path)
        source = state_dict[0][0]["model"]
    else:
        state_dict = load_megatron_checkpoint(args.load_ckpt_path)
        source = _get_non_ep_model_source(state_dict)
    for mg_key, hf_key in name_map.items():
        target[hf_key] = source[mg_key]
        print(f" > {mg_key} -> {hf_key}  (shape {list(target[hf_key].shape)})")
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ("huggingface", "mcore"):
    print(" ====== convert Gemma4-VL adapter from HuggingFace to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    source = load_huggingface_checkpoint(args.load_ckpt_path)
    target = {}
    for mg_key, hf_key in name_map.items():
        target[mg_key] = source[hf_key]
        print(f" > {hf_key} -> {mg_key}  (shape {list(target[mg_key].shape)})")

    state_dict = [{"model": deepcopy(target)} for _ in range(tp)]
    save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, "release"))

elif (args.load_platform, args.save_platform) == ("mcore", "mcore"):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert Gemma4-VL adapter from Megatron Core to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    state_dict = load_megatron_checkpoint(args.load_ckpt_path)
    source = _get_non_ep_model_source(state_dict)
    target = deepcopy(source)
    new_state_dict = [{"model": deepcopy(target)} for _ in range(tp)]
    save_megatron_checkpoint(new_state_dict, os.path.join(args.save_ckpt_path, "release"))

else:
    raise NotImplementedError
