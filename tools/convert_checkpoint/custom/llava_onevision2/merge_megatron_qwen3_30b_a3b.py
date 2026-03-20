#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

from convert_checkpoint.custom.llava_onevision2.util import (  # noqa: E402
    load_megatron_checkpoint,
    load_megatron_checkpoint_tp_ep,
    load_megatron_checkpoint_tp_pp_ep,
    save_megatron_checkpoint_tp_ep,
    save_megatron_checkpoint_tp_pp_ep,
)


def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Merger Arguments", allow_abbrev=False)
    group = parser.add_argument_group(title="checkpoint")
    group.add_argument("--language_model_path", type=str, help="Path to language model.")
    group.add_argument("--vision_model_path", type=str, help="Path to vision model.")
    group.add_argument("--vision_patch", type=str, help="Path to vision patch.")
    group.add_argument("--adapter_path", type=str, help="Path to adapter.")
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1, help="Tensor parallel size.")
    group.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline parallel size.")

    return parser.parse_args()


def merge_dict(src_dict, dst_dict):
    """Merge src_dict into dst_dict recursively without overwriting existing leaves."""
    for key, value in src_dict.items():
        if key in dst_dict and isinstance(value, dict) and isinstance(dst_dict[key], dict):
            merge_dict(value, dst_dict[key])
        elif key not in dst_dict:
            dst_dict[key] = value


def merge_module_1d_into_language_2d(language_model, module, module_name):
    """Broadcast one TP-sharded module into all EP ranks of each TP rank."""
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


def merge_module_into_language_3d(language_model, module, module_name):
    """Merge module into language_model[pp][tp][ep].

    module can be:
    - 1D TP list: module[tp], broadcast over all PP and EP
    - 2D PP x TP list: module[pp][tp], broadcast over EP
    """
    pp_size = len(language_model)
    assert pp_size > 0, "language_model pp dimension is empty"
    tp_size = len(language_model[0])
    assert tp_size > 0, "language_model tp dimension is empty"
    ep_size = len(language_model[0][0])
    assert ep_size > 0, "language_model ep dimension is empty"

    assert isinstance(module, list), f"{module_name} should be a list"

    is_2d_pp_tp = bool(module) and isinstance(module[0], list)

    if is_2d_pp_tp:
        assert len(module) == pp_size, (
            f"{module_name} pp shards ({len(module)}) mismatch language_model pp shards ({pp_size})"
        )
        for p in range(pp_size):
            assert len(module[p]) == tp_size, (
                f"{module_name}[pp={p}] tp shards ({len(module[p])}) mismatch tp size ({tp_size})"
            )
            for t in range(tp_size):
                src = module[p][t]
                assert "model" in src, f"{module_name}[pp={p}][tp={t}] missing 'model' key"
                for e in range(ep_size):
                    dst = language_model[p][t][e]
                    assert "model" in dst, f"language_model[pp={p}][tp={t}][ep={e}] missing 'model' key"
                    merge_dict(src["model"], dst["model"])
    else:
        assert len(module) == tp_size, (
            f"{module_name} tp shards ({len(module)}) mismatch language_model tp shards ({tp_size})"
        )
        for t in range(tp_size):
            src = module[t]
            assert "model" in src, f"{module_name}[tp={t}] missing 'model' key"
            for p in range(pp_size):
                for e in range(ep_size):
                    dst = language_model[p][t][e]
                    assert "model" in dst, f"language_model[pp={p}][tp={t}][ep={e}] missing 'model' key"
                    merge_dict(src["model"], dst["model"])


args = parse_args()
if args.megatron_path is not None:
    sys.path.insert(0, args.megatron_path)

print("===== merge megatron checkpoints (qwen3-30b-a3b) ======")

# LLM: PP=1 -> 2D(tp, ep), PP>1 -> 3D(pp, tp, ep)
if args.pipeline_model_parallel_size > 1:
    language_model = load_megatron_checkpoint_tp_pp_ep(args.language_model_path)
else:
    language_model = load_megatron_checkpoint_tp_ep(args.language_model_path)

# Other modules are typically TP-only; also support PPxTP if provided.
vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

for module_name, module in [("vision", vision_model), ("adapter", adapter), ("patch", patch)]:
    if args.pipeline_model_parallel_size > 1:
        # vision/adapter/patch only belong to PP stage 0; pass a 1-stage view
        merge_module_into_language_3d(language_model[0:1], module, module_name)
    else:
        merge_module_1d_into_language_2d(language_model, module, module_name)

if args.pipeline_model_parallel_size > 1:
    save_megatron_checkpoint_tp_pp_ep(language_model, args.save_ckpt_path)
else:
    save_megatron_checkpoint_tp_ep(language_model, args.save_ckpt_path)
