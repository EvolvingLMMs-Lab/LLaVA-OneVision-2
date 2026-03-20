#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import json
import os
import sys
from os.path import dirname

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"model-{i:05d}-of-{num_checkpoints:05d}.safetensors")
        current_chunk = load_file(checkpoint_path)
        state_dict.update(current_chunk)
    return state_dict


def load_huggingface_checkpoint(load_path):
    """load ckpt"""
    state_dict = {}
    sub_dirs = [x for x in os.listdir(load_path) if x.endswith("safetensors")]
    if len(sub_dirs) == 1:
        checkpoint_name = "model.safetensors"
        state_dict = load_file(os.path.join(load_path, checkpoint_name), device="cpu")
    else:
        num_checkpoints = len(sub_dirs)
        state_dict = merge_transformers_sharded_states(load_path, num_checkpoints)
    return state_dict


def save_huggingface_checkpoint(state_dict, save_path):
    """save ckpt"""
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "model.safetensors")

    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    for shard_file, tensors in state_dict_split.filename_to_tensors.items():
        shard = {}
        for tensor in tensors:
            shard[tensor] = state_dict[tensor].contiguous()
            del state_dict[tensor]
        shard_path = os.path.join(save_path, shard_file)
        save_file(shard, shard_path, metadata={"format": "pt"})
        print(f"Saving HuggingFace shard to: {shard_path}")

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(save_path, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def load_megatron_checkpoint(load_path):
    """load ckpt"""
    state_dict = []
    sub_dirs = sorted([x for x in os.listdir(load_path) if x.startswith("mp_rank")])
    last_dir = sub_dirs[-1].split("_")
    if len(last_dir) == 4:
        tp = int(last_dir[-2]) + 1
        pp = int(last_dir[-1]) + 1
        for p in range(pp):
            state_dict.append([])
            for t in range(tp):
                checkpoint_name = f"mp_rank_{t:02d}_{p:03d}/model_optim_rng.pt"
                ckpt = torch.load(os.path.join(load_path, checkpoint_name), map_location="cpu", weights_only=False)
                state_dict[p].append(ckpt)
        return state_dict
    else:
        for t in range(len(sub_dirs)):
            checkpoint_name = f"mp_rank_{t:02d}/model_optim_rng.pt"
            ckpt = torch.load(os.path.join(load_path, checkpoint_name), map_location="cpu", weights_only=False)
            state_dict.append(ckpt)
        return state_dict


def save_megatron_checkpoint(state_dict, save_path):
    """save ckpt"""
    if isinstance(state_dict[0], list):
        for p in range(len(state_dict)):
            for t in range(len(state_dict[p])):
                sub_dir_name = f"mp_rank_{t:02d}_{p:03d}"
                os.makedirs(os.path.join(save_path, sub_dir_name), exist_ok=True)
                checkpoint_path = os.path.join(save_path, sub_dir_name, "model_optim_rng.pt")
                torch.save(state_dict[p][t], checkpoint_path)
                print(f"Saving Megatron shard to: {checkpoint_path}")
    else:
        for t in range(len(state_dict)):
            sub_dir_name = f"mp_rank_{t:02d}"
            os.makedirs(os.path.join(save_path, sub_dir_name), exist_ok=True)
            checkpoint_path = os.path.join(save_path, sub_dir_name, "model_optim_rng.pt")
            torch.save(state_dict[t], checkpoint_path)
            print(f"Saving Megatron shard to: {checkpoint_path}")


import os
import re
from typing import Dict, List, Tuple

import torch


_MP3D_DIR_RE = re.compile(r"^mp_rank_(\d+)_(\d+)_(\d+)$")
_MP2D_DIR_RE = re.compile(r"^mp_rank_(\d+)_(\d+)$")
_MP1D_DIR_RE = re.compile(r"^mp_rank_(\d+)$")
_SHARD_FILE = "model_optim_rng.pt"


def _scan_mp_dirs(load_path: str) -> Tuple[Dict[Tuple[int, int, int], str], List[int], List[int], List[int]]:
    """
    Scan load_path for 3D mp_rank directories and return:
    - dir_map: mapping from (tp, pp, ep) -> directory name (not full path)
    - t_vals, p_vals, e_vals: sorted unique indices found for tp, pp, ep.

    Raises:
        FileNotFoundError: if load_path does not exist.
        ValueError: if no 3D mp_rank_*_*_* directories are found.
    """
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Directory not found: {load_path}")

    sub_dirs = [d for d in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, d))]
    dir_map_3d: Dict[Tuple[int, int, int], str] = {}
    t_set, p_set, e_set = set(), set(), set()

    # Prefer strict 3D directories
    for d in sub_dirs:
        m = _MP3D_DIR_RE.match(d)
        if m:
            t, p, e = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            dir_map_3d[(t, p, e)] = d
            t_set.add(t)
            p_set.add(p)
            e_set.add(e)

    if not dir_map_3d:
        # Helpful diagnostics if the tree is not 3D
        has_2d = any(_MP2D_DIR_RE.match(d) for d in sub_dirs)
        has_1d = any(_MP1D_DIR_RE.match(d) for d in sub_dirs)
        if has_2d or has_1d:
            raise ValueError(
                "Expected 3D mp_rank_{tp}_{pp}_{ep} directory layout, but found 1D/2D.\n"
                "Use the original load_megatron_checkpoint for 1D/2D or convert your checkpoints."
            )
        raise ValueError(f"No mp_rank_* directories found in: {load_path}")

    t_vals = sorted(t_set)
    p_vals = sorted(p_set)
    e_vals = sorted(e_set)
    return dir_map_3d, t_vals, p_vals, e_vals


def load_megatron_checkpoint_tp_pp_ep(load_path: str):
    """
    Load Megatron checkpoints organized in 3D layout: mp_rank_{tp}_{pp}_{ep}/model_optim_rng.pt

    Returns:
        A 3-level nested list organized as state_dict[pp_idx][tp_idx][ep_idx],
        where indices follow the ascending order of discovered pp, tp, ep values.

        Each leaf is the object loaded from torch.load(...):
        typically a dict with keys like 'model', 'optimizer', etc.

    Notes:
        - All shards are loaded onto CPU (map_location='cpu').
        - This function is strict for 3D; it will error if only 1D/2D layout is found.
        - Directory numeric widths (zero-padding) are not assumed; we reuse the actual directory names found.
    """
    dir_map_3d, t_vals, p_vals, e_vals = _scan_mp_dirs(load_path)

    # Build nested structure: [pp][tp][ep]
    state_dict: List[List[List[dict]]] = []
    for p in p_vals:
        pp_list: List[List[dict]] = []
        for t in t_vals:
            tp_list: List[dict] = []
            for e in e_vals:
                sub_dir = dir_map_3d.get((t, p, e))
                if sub_dir is None:
                    raise FileNotFoundError(f"Missing shard directory for (tp={t}, pp={p}, ep={e}) under {load_path}")
                checkpoint_path = os.path.join(load_path, sub_dir, _SHARD_FILE)
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(f"Shard file not found: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                tp_list.append(ckpt)
            pp_list.append(tp_list)
        state_dict.append(pp_list)

    return state_dict


def save_megatron_checkpoint_tp_pp_ep(state_dict, save_path: str, pad_tp: int = 2, pad_pp: int = 3, pad_ep: int = 3):
    """
    Save a 3D Megatron checkpoint layout to save_path as:
        mp_rank_{tp:0{pad_tp}d}_{pp:0{pad_pp}d}_{ep:0{pad_ep}d}/model_optim_rng.pt

    Args:
        state_dict: 3-level nested list organized as state_dict[pp_idx][tp_idx][ep_idx].
                    Each leaf element is the checkpoint object to be torch.save'd.
        save_path:  Target root directory.
        pad_tp:     Zero-padding width for 'tp' in directory names (default 2).
        pad_pp:     Zero-padding width for 'pp' in directory names (default 3).
        pad_ep:     Zero-padding width for 'ep' in directory names (default 3).

    Raises:
        ValueError: if state_dict is not a 3-level nested list.
    """
    if not isinstance(state_dict, list) or not state_dict:
        raise ValueError("state_dict must be a non-empty 3-level nested list: [pp][tp][ep].")

    if not isinstance(state_dict[0], list) or not state_dict[0]:
        raise ValueError("state_dict must be a 3-level nested list: [pp][tp][ep].")
    if not isinstance(state_dict[0][0], list) or not state_dict[0][0]:
        raise ValueError("state_dict must be a 3-level nested list: [pp][tp][ep].")

    os.makedirs(save_path, exist_ok=True)

    for p_idx, pp_list in enumerate(state_dict):
        if not isinstance(pp_list, list):
            raise ValueError(f"Expected state_dict[{p_idx}] to be a list (tp dimension).")
        for t_idx, tp_list in enumerate(pp_list):
            if not isinstance(tp_list, list):
                raise ValueError(f"Expected state_dict[{p_idx}][{t_idx}] to be a list (ep dimension).")
            for e_idx, ckpt in enumerate(tp_list):
                sub_dir_name = f"mp_rank_{t_idx:0{pad_tp}d}_{p_idx:0{pad_pp}d}_{e_idx:0{pad_ep}d}"
                full_dir = os.path.join(save_path, sub_dir_name)
                os.makedirs(full_dir, exist_ok=True)
                checkpoint_path = os.path.join(full_dir, _SHARD_FILE)
                torch.save(ckpt, checkpoint_path)
                print(f"Saving Megatron shard to: {checkpoint_path}")


def _scan_mp_dirs_2d(load_path: str) -> Tuple[Dict[Tuple[int, int], str], List[int], List[int]]:
    """
    Scan load_path for 2D mp_rank directories and return:
    - dir_map: mapping from (tp, ep) -> directory name (not full path)
    - t_vals, e_vals: sorted unique indices found for tp, ep.

    Raises:
        FileNotFoundError: if load_path does not exist.
        ValueError: if no 2D mp_rank_*_* directories are found.
    """
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Directory not found: {load_path}")

    sub_dirs = [d for d in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, d))]
    dir_map_2d: Dict[Tuple[int, int], str] = {}
    t_set, e_set = set(), set()

    # Look for 2D directories
    for d in sub_dirs:
        m = _MP2D_DIR_RE.match(d)
        if m:
            t, e = (int(m.group(1)), int(m.group(2)))
            dir_map_2d[(t, e)] = d
            t_set.add(t)
            e_set.add(e)

    if not dir_map_2d:
        raise ValueError(f"No mp_rank_*_* directories found in: {load_path}")

    t_vals = sorted(t_set)
    e_vals = sorted(e_set)
    return dir_map_2d, t_vals, e_vals


def load_megatron_checkpoint_tp_ep(load_path: str):
    """
    Load Megatron checkpoints organized in 2D layout: mp_rank_{tp}_{ep}/model_optim_rng.pt

    Returns:
        A 2-level nested list organized as state_dict[tp_idx][ep_idx],
        where indices follow the ascending order of discovered tp, ep values.

        Each leaf is the object loaded from torch.load(...):
        typically a dict with keys like 'model', 'optimizer', etc.

    Notes:
        - All shards are loaded onto CPU (map_location='cpu').
        - This function is for 2D layout (tensor parallel + expert parallel).
        - Directory numeric widths (zero-padding) are not assumed; we reuse the actual directory names found.
    """
    dir_map_2d, t_vals, e_vals = _scan_mp_dirs_2d(load_path)

    # Build nested structure: [tp][ep]
    state_dict: List[List[dict]] = []
    for t in t_vals:
        tp_list: List[dict] = []
        for e in e_vals:
            sub_dir = dir_map_2d.get((t, e))
            if sub_dir is None:
                raise FileNotFoundError(f"Missing shard directory for (tp={t}, ep={e}) under {load_path}")
            checkpoint_path = os.path.join(load_path, sub_dir, _SHARD_FILE)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Shard file not found: {checkpoint_path}")

            print(f"Loading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            tp_list.append(ckpt)
        state_dict.append(tp_list)

    print(f"Loaded {len(t_vals)} tensor parallel ranks and {len(e_vals)} expert parallel ranks")
    return state_dict


def save_megatron_checkpoint_tp_ep(state_dict, save_path: str, pad_tp: int = 2, pad_ep: int = 3):
    """
    Save a 2D Megatron checkpoint layout to save_path as:
        mp_rank_{tp:0{pad_tp}d}_{ep:0{pad_ep}d}/model_optim_rng.pt

    Args:
        state_dict: 2-level nested list organized as state_dict[tp_idx][ep_idx].
                    Each leaf element is the checkpoint object to be torch.save'd.
        save_path:  Target root directory.
        pad_tp:     Zero-padding width for 'tp' in directory names (default 2).
        pad_ep:     Zero-padding width for 'ep' in directory names (default 3).

    Raises:
        ValueError: if state_dict is not a 2-level nested list.
    """
    if not isinstance(state_dict, list) or not state_dict:
        raise ValueError("state_dict must be a non-empty 2-level nested list: [tp][ep].")

    if not isinstance(state_dict[0], list) or not state_dict[0]:
        raise ValueError("state_dict must be a 2-level nested list: [tp][ep].")

    os.makedirs(save_path, exist_ok=True)

    for t_idx, tp_list in enumerate(state_dict):
        if not isinstance(tp_list, list):
            raise ValueError(f"Expected state_dict[{t_idx}] to be a list (ep dimension).")
        for e_idx, ckpt in enumerate(tp_list):
            sub_dir_name = f"mp_rank_{t_idx:0{pad_tp}d}_{e_idx:0{pad_ep}d}"
            full_dir = os.path.join(save_path, sub_dir_name)
            os.makedirs(full_dir, exist_ok=True)
            checkpoint_path = os.path.join(full_dir, _SHARD_FILE)
            torch.save(ckpt, checkpoint_path)
            print(f"Saved Megatron shard to: {checkpoint_path}")


# # 使用示例
# if __name__ == "__main__":
#     # 加载checkpoint
#     load_path = "/path/to/your/checkpoint/dir"
#     state_dict = load_megatron_checkpoint_tp_ep(load_path)

#     # 保存checkpoint
#     save_path = "/path/to/save/checkpoint/dir"
#     save_megatron_checkpoint_tp_ep(state_dict, save_path)
