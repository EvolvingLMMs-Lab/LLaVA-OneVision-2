#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""Gemma4-VL Megatron checkpoint merger.

Merges 4 separately-converted Megatron checkpoint trees produced by
``llm.py``, ``vit.py``, ``vision_patch.py`` and ``adapter.py`` into one
Megatron release tree.

Layout selection (mirrors the OV2 ``merge_megatron_qwen3_30b_a3b.py`` template):

  - PP=1, EP=1            -> 1D ``mp_rank_TT``
  - PP=1, EP>1            -> 2D ``mp_rank_TT_EEE`` (this file's P2.7 path)
  - PP>1                  -> 3D ``mp_rank_TT_PPP_EEE`` (staged for P2.8)

The LLM body uses the appropriate EP-aware loader; the non-LLM modules
(ViT, adapter, vision_patch) are always saved as 1D ``mp_rank_TT`` by
their standalone converters and broadcast across the EP axis at merge
time, matching the OV2 Qwen3 convention.

Inputs (each is a directory containing ``mp_rank_*``):
    --language_model_path  : Gemma4 LLM body
    --vision_model_path    : Gemma4 ViT body (1D TP only)
    --vision_patch         : Gemma4 ViT patch+pos+std (1D TP only)
    --adapter_path         : Gemma4 vision-language adapter (1D TP only)

Output:
    --save_ckpt_path       : Merged Megatron release tree

Usage (TP=PP=EP=1, the P2.5 first-round target):
    python merge_megatron.py \\
        --language_model_path /tmp/gemma4_llm/release \\
        --vision_model_path   /tmp/gemma4_vit/release \\
        --vision_patch        /tmp/gemma4_patch/release \\
        --adapter_path        /tmp/gemma4_adapter/release \\
        --save_ckpt_path      /ov2/pretrain_models/megatron/gemma-4-26B-A4B-it-tp1pp1ep1/release \\
        --tensor_model_parallel_size 1 \\
        --pipeline_model_parallel_size 1 \\
        --expert_parallel_size 1
"""

import argparse
import os
import sys
from os.path import dirname


SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.custom.llava_onevision2.util import (  # noqa: E402
    load_megatron_checkpoint,
    load_megatron_checkpoint_tp_ep,
    load_megatron_checkpoint_tp_pp_ep,
    save_megatron_checkpoint,
    save_megatron_checkpoint_tp_ep,
    save_megatron_checkpoint_tp_pp_ep,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma4-VL Megatron Merger", allow_abbrev=False)
    group = parser.add_argument_group(title="checkpoint")
    group.add_argument("--language_model_path", type=str, required=True,
                       help="Path to language model release dir (mp_rank_*).")
    group.add_argument("--vision_model_path", type=str, required=True,
                       help="Path to vision model release dir (mp_rank_*).")
    group.add_argument("--vision_patch", type=str, required=True,
                       help="Path to vision patch release dir (mp_rank_*).")
    group.add_argument("--adapter_path", type=str, required=True,
                       help="Path to adapter release dir (mp_rank_*).")
    group.add_argument("--save_ckpt_path", type=str, required=True,
                       help="Path to save merged checkpoint.")
    group.add_argument("--megatron_path", type=str, default=None,
                       help="Base directory of Megatron repository.")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1,
                       help="Tensor parallel size.")
    group.add_argument("--pipeline_model_parallel_size", type=int, default=1,
                       help="Pipeline parallel size.")
    group.add_argument("--expert_parallel_size", type=int, default=1,
                       help="Expert parallel size (1 = disabled).")
    return parser.parse_args()


def merge_dict(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value


def merge_module_1d_into_language_2d(language_model, module, module_name):
    """Broadcast a 1D ``[tp]`` non-LLM module into all EP shards of each TP rank."""
    tp_size = len(language_model)
    assert tp_size > 0, "language_model tp dimension is empty"
    ep_size = len(language_model[0])
    assert ep_size > 0, "language_model ep dimension is empty"

    assert isinstance(module, list), f"{module_name} should be a TP-sharded list"
    assert len(module) == tp_size, (
        f"{module_name} tp shards ({len(module)}) mismatch language_model tp shards ({tp_size})"
    )

    for t in range(tp_size):
        src = module[t]
        assert "model" in src, f"{module_name}[{t}] missing 'model' key"
        for e in range(ep_size):
            dst = language_model[t][e]
            assert "model" in dst, f"language_model[tp={t}][ep={e}] missing 'model' key"
            merge_dict(src["model"], dst["model"])


def merge_module_1d_into_language_3d_stage0(language_model, module, module_name):
    """Broadcast a 1D ``[tp]`` non-LLM module into PP stage 0 only of a 3D LLM tree.

    ``language_model`` is shaped ``[pp][tp][ep]``. The non-LLM modules
    (ViT / adapter / vision_patch) live solely on PP stage 0 in the runtime
    Megatron model, so we only inject them into ``language_model[0][t][e]``;
    later PP stages are untouched.
    """
    pp_size = len(language_model)
    assert pp_size > 0, "language_model pp dimension is empty"
    tp_size = len(language_model[0])
    assert tp_size > 0, "language_model tp dimension is empty"
    ep_size = len(language_model[0][0])
    assert ep_size > 0, "language_model ep dimension is empty"

    assert isinstance(module, list), f"{module_name} should be a TP-sharded list"
    assert len(module) == tp_size, (
        f"{module_name} tp shards ({len(module)}) mismatch language_model tp shards ({tp_size})"
    )

    for t in range(tp_size):
        src = module[t]
        assert "model" in src, f"{module_name}[{t}] missing 'model' key"
        for e in range(ep_size):
            dst = language_model[0][t][e]
            assert "model" in dst, (
                f"language_model[pp=0][tp={t}][ep={e}] missing 'model' key"
            )
            merge_dict(src["model"], dst["model"])


args = parse_args()
if args.megatron_path is not None:
    sys.path.insert(0, args.megatron_path)

print("===== merge gemma4-vl megatron checkpoints ======")

pp_active = args.pipeline_model_parallel_size > 1
ep_active = args.expert_parallel_size is not None and args.expert_parallel_size > 1

if pp_active:
    language_model = load_megatron_checkpoint_tp_pp_ep(args.language_model_path)
elif ep_active:
    language_model = load_megatron_checkpoint_tp_ep(args.language_model_path)
else:
    language_model = load_megatron_checkpoint(args.language_model_path)

vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

if pp_active:
    for module_name, module in [("vision", vision_model), ("adapter", adapter), ("patch", patch)]:
        merge_module_1d_into_language_3d_stage0(language_model, module, module_name)
    print(
        f" > total LLM PP/TP/EP shards: {len(language_model)}/{len(language_model[0])}"
        f"/{len(language_model[0][0])}"
    )
    save_megatron_checkpoint_tp_pp_ep(language_model, args.save_ckpt_path)
elif ep_active:
    for module_name, module in [("vision", vision_model), ("adapter", adapter), ("patch", patch)]:
        merge_module_1d_into_language_2d(language_model, module, module_name)
    print(f" > total LLM TP/EP shards: {len(language_model)}/{len(language_model[0])}")
    save_megatron_checkpoint_tp_ep(language_model, args.save_ckpt_path)
else:
    for module_name, module in [("vision", vision_model), ("adapter", adapter), ("patch", patch)]:
        assert len(module) == len(language_model), (
            f"{module_name} TP shards ({len(module)}) != language_model TP shards ({len(language_model)})"
        )
        for i in range(len(module)):
            merge_dict(module[i]["model"], language_model[i]["model"])
    print(f" > total LLM TP shards: {len(language_model)}")
    save_megatron_checkpoint(language_model, args.save_ckpt_path)

print("===== merge complete ======")
