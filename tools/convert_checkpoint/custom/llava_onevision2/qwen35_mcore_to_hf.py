#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert the Qwen3.5 hybrid-attention LANGUAGE part of a Megatron-Core (AIAK)
LLaVA-OneVision2 checkpoint back to the merged-HF layout.

This is the exact inverse of ``qwen35_hf_to_mcore.py``:
  * un-shard the per-rank tensor-parallel shards back to a single TP1 language
    state dict (inverse of ``shard_language``), then
  * invert each per-decoder-layer weight rule via
    ``aiak_training_llm.models.qwen35.hf_convert.mcore_to_hf_{linear,full}_attn_layer``
    (which invert convert_{linear,full}_attn_layer, cos=1.0 / bit-identical proven).

TP UN-SHARD (inverse of shard_language), applied per parameter suffix:
  * COLUMN-PARALLEL (was chunk dim0)  -> torch.cat(shards, dim=0):
        embedding.word_embeddings.weight, self_attention.linear_qkv.weight
  * ROW-PARALLEL (was chunk dim1)     -> torch.cat(shards, dim=1):
        self_attention.linear_proj.weight, mlp.linear_fc2.weight
  * SwiGLU mlp.linear_fc1.weight      -> cat(shards, dim=0) then
        transpose_shape0(x, tp, 2) == inverse of the forward transpose_shape0(x, 2, tp).
        (This is exactly the un-transpose the generic Qwen3 mcore->hf converter uses,
         mcore_checkpoint.py: tp_transpose_shape=(self.tp, 2).)
  * REPLICATED (all GatedDeltaNet params + all norms + _extra_state) -> take rank 0.

HF names emitted (prefix ``model.language_model.``):
    embed_tokens.weight            <- embedding.word_embeddings.weight
    norm.weight                    <- decoder.final_layernorm.weight
    layers.{i}.input_layernorm.weight            <- input_layernorm.weight
    layers.{i}.post_attention_layernorm.weight   <- pre_mlp_layernorm.weight
    layers.{i}.mlp.gate_proj.weight / up_proj.weight  <- split(mlp.linear_fc1.weight)
    layers.{i}.mlp.down_proj.weight              <- mlp.linear_fc2.weight
    layers.{i}.linear_attn.* / self_attn.*       <- mcore_to_hf_{linear,full}_attn_layer(...)
Plus tied ``lm_head.weight`` = embed_tokens.weight.

Layer type rule (matches forward): (i+1) % full_attention_interval == 0 -> full-attn.
Primary target: PP=1 (1D mp_rank_{t:02d} layout), TP in {1,2,4}.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file

from aiak_training_llm.models.qwen35.hf_convert import (
    mcore_to_hf_linear_attn_layer,
    mcore_to_hf_full_attn_layer,
)

MC_LM = "language_model"          # mcore language prefix
HF_LM = "model.language_model"    # HF merged-model language prefix


def transpose_shape0(param, m, n):
    """Verbatim from tools/convert_checkpoint/utils.py::transpose_shape0.
    Note transpose_shape0(transpose_shape0(x, m, n), n, m) == x."""
    _shape = param.size()
    current_shape = (m, n, _shape[0] // (m * n)) + tuple(_shape[1:])
    return param.view(*current_shape).transpose(0, 1).contiguous().view(*_shape)


# Suffix rules -- MUST mirror qwen35_hf_to_mcore.shard_language.
_COL_DIM0 = (
    "embedding.word_embeddings.weight",
    "self_attention.linear_qkv.weight",
)
_ROW_DIM1 = (
    "self_attention.linear_proj.weight",
    "mlp.linear_fc2.weight",
)
_FC1 = ("mlp.linear_fc1.weight",)


def load_language_shards(load_path, tp):
    """Load the `tp` per-rank mcore shards (PP=1, 1D mp_rank_{t:02d} layout) and
    return a list of `tp` flat dicts of language_model.* -> tensor (weights only,
    _extra_state dropped)."""
    shards = []
    for t in range(tp):
        sub = f"mp_rank_{t:02d}"
        ckpt_path = os.path.join(load_path, sub, "model_optim_rng.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"missing shard: {ckpt_path}")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        lang = {
            k: v for k, v in model.items()
            if k.startswith(MC_LM + ".") and not k.endswith("_extra_state")
        }
        assert lang, f"no language_model.* weights in {ckpt_path}"
        shards.append(lang)
    return shards


def unshard_language(shards, tp):
    """Inverse of shard_language: reconstruct the full TP1 language state dict."""
    if tp == 1:
        return dict(shards[0])
    full = {}
    for k in shards[0]:
        v0 = shards[0][k]
        if not isinstance(v0, torch.Tensor):
            full[k] = v0
            continue
        if any(k.endswith(s) for s in _COL_DIM0):
            full[k] = torch.cat([shards[t][k] for t in range(tp)], dim=0).contiguous()
        elif any(k.endswith(s) for s in _ROW_DIM1):
            full[k] = torch.cat([shards[t][k] for t in range(tp)], dim=1).contiguous()
        elif any(k.endswith(s) for s in _FC1):
            reordered = torch.cat([shards[t][k] for t in range(tp)], dim=0)
            full[k] = transpose_shape0(reordered, tp, 2).contiguous()
        else:
            full[k] = shards[0][k].clone()  # replicated -> rank 0
    return full


def convert_language_mcore_to_hf(lang_sd, num_layers, full_attention_interval,
                                 orig_vocab_size=None):
    """Full TP1 mcore language dict -> HF ``model.language_model.*`` (+ lm_head)."""
    out = {}

    embed = lang_sd[f"{MC_LM}.embedding.word_embeddings.weight"]
    if orig_vocab_size is not None and embed.shape[0] > orig_vocab_size:
        embed = embed[:orig_vocab_size, :]
    out[f"{HF_LM}.embed_tokens.weight"] = embed.clone().contiguous()
    out[f"{HF_LM}.norm.weight"] = lang_sd[f"{MC_LM}.decoder.final_layernorm.weight"].clone()
    out["lm_head.weight"] = embed.clone().contiguous()  # tied

    n_full, n_lin = 0, 0
    for i in range(num_layers):
        mpre = f"{MC_LM}.decoder.layers.{i}."
        layer_sd = {k[len(mpre):]: v for k, v in lang_sd.items() if k.startswith(mpre)}
        assert layer_sd, f"no mcore weights for layer {i} (prefix {mpre})"
        hpre = f"{HF_LM}.layers.{i}."

        out[hpre + "input_layernorm.weight"] = layer_sd["input_layernorm.weight"].clone()
        out[hpre + "post_attention_layernorm.weight"] = layer_sd["pre_mlp_layernorm.weight"].clone()

        # SwiGLU: mcore fc1 = cat([gate, up], dim0) -> split back
        fc1 = layer_sd["mlp.linear_fc1.weight"]
        assert fc1.shape[0] % 2 == 0, f"layer {i}: fc1 dim0 {fc1.shape[0]} not even"
        gate, up = torch.chunk(fc1, 2, dim=0)
        out[hpre + "mlp.gate_proj.weight"] = gate.clone().contiguous()
        out[hpre + "mlp.up_proj.weight"] = up.clone().contiguous()
        out[hpre + "mlp.down_proj.weight"] = layer_sd["mlp.linear_fc2.weight"].clone()

        is_full = ((i + 1) % full_attention_interval == 0)
        if is_full:
            attn = mcore_to_hf_full_attn_layer(layer_sd)
            n_full += 1
        else:
            attn = mcore_to_hf_linear_attn_layer(layer_sd)
            n_lin += 1
        for k, v in attn.items():
            out[hpre + k] = v

    print(f"[qwen35_mcore_to_hf] converted {num_layers} layers: {n_full} full-attn, {n_lin} linear")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_ckpt_path", required=True,
                    help="mcore checkpoint dir containing mp_rank_* (e.g. <ckpt>/release)")
    ap.add_argument("--save_ckpt_path", required=True,
                    help="output HF language dir (writes model.safetensors)")
    ap.add_argument("--num_layers", type=int, default=32)
    ap.add_argument("--full_attention_interval", type=int, default=4)
    ap.add_argument("--tensor_model_parallel_size", type=int, default=1)
    ap.add_argument("--orig_vocab_size", type=int, default=None,
                    help="if the mcore embedding was vocab-padded, cut back to this size")
    args = ap.parse_args()

    tp = args.tensor_model_parallel_size
    assert tp >= 1

    print(f"[qwen35_mcore_to_hf] loading {tp} shard(s) from: {args.load_ckpt_path}")
    shards = load_language_shards(args.load_ckpt_path, tp)
    lang = unshard_language(shards, tp)
    hf = convert_language_mcore_to_hf(lang, args.num_layers, args.full_attention_interval,
                                      orig_vocab_size=args.orig_vocab_size)

    os.makedirs(args.save_ckpt_path, exist_ok=True)
    out_path = os.path.join(args.save_ckpt_path, "model.safetensors")
    save_file({k: v.contiguous() for k, v in hf.items()}, out_path, metadata={"format": "pt"})
    print(f"[qwen35_mcore_to_hf] wrote {len(hf)} HF language weights -> {out_path}")


if __name__ == "__main__":
    main()
