#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert the Qwen3.5 hybrid-attention LANGUAGE part of a merged LLaVA-OneVision2
HuggingFace checkpoint to a Megatron-Core (AIAK) checkpoint (TP=1, PP=1).

This is the Qwen3.5 counterpart of ``tools/convert_checkpoint/model.py`` +
``config/llava-onevision2-4b/qwen3.json`` (which only handle a *standard* Qwen3
LLM).  The Qwen3.5 backbone is a hybrid stack (linear gated-delta-net layers +
periodic full-attention layers, interleaved 3D M-RoPE, partial rotary 0.25) whose
per-decoder-layer weight math lives in the validated module
``aiak_training_llm.models.qwen35.hf_convert``
(``convert_linear_attn_layer`` / ``convert_full_attn_layer``, both cos=1.0 proven).

It writes ``<save>/release/mp_rank_00/model_optim_rng.pt`` = ``{"model": {...}}``
using ONLY ``language_model.*`` mcore keys, exactly matching the parameter names of
the model the Phase-3 provider builds (arch ``llava-onevision2-4b-qwen35``):

    language_model.embedding.word_embeddings.weight        <- model.language_model.embed_tokens.weight
    language_model.decoder.final_layernorm.weight          <- model.language_model.norm.weight
    language_model.decoder.layers.{i}.input_layernorm.weight     <- layers.{i}.input_layernorm.weight
    language_model.decoder.layers.{i}.pre_mlp_layernorm.weight   <- layers.{i}.post_attention_layernorm.weight
    language_model.decoder.layers.{i}.mlp.linear_fc1.weight      <- cat[gate_proj, up_proj]  (dim 0)
    language_model.decoder.layers.{i}.mlp.linear_fc2.weight      <- layers.{i}.mlp.down_proj.weight
    language_model.decoder.layers.{i}.self_attention.*           <- convert_{linear,full}_attn_layer(...)

Notes:
  * The Qwen3.5 spec uses a SEPARATE (non-fused) ``input_layernorm`` for every
    decoder layer of BOTH types -- it is NOT the TE-fused
    ``self_attention.linear_qkv.layer_norm_weight`` used by the vision tower.
  * Embeddings are tied (lm_head is bit-identical to embed_tokens); the built model
    has ``share_embeddings_and_output_weights=True`` so ``output_layer`` carries no
    weight -- we therefore do NOT emit ``language_model.output_layer.weight``.
  * TE ``_extra_state`` entries are intentionally omitted (loaded with strict=False).

Layer type rule (matches HF ``text_config.layer_types`` / full_attention_interval):
    (i + 1) % full_attention_interval == 0  -> full-attention  (convert_full_attn_layer)
    otherwise                                -> linear gated-delta-net (convert_linear_attn_layer)
"""
import argparse
import io
import os
import sys

import torch
from safetensors.torch import load_file

# aiak_training_llm must be importable (repo root on PYTHONPATH).
from aiak_training_llm.models.qwen35.hf_convert import (
    convert_linear_attn_layer,
    convert_full_attn_layer,
)

HF_LM = "model.language_model"          # prefix of the LLM inside the merged HF model
MC_LM = "language_model"                # mcore prefix expected by the built model


def load_hf_state_dict(load_path):
    """Load a (possibly sharded) HF safetensors checkpoint into one dict."""
    files = sorted(f for f in os.listdir(load_path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {load_path}")
    sd = {}
    for fn in files:
        sd.update(load_file(os.path.join(load_path, fn), device="cpu"))
    return sd


def convert_language(hf_sd, num_layers, full_attention_interval, add_extra_state=True):
    """Convert the Qwen3.5 language weights from a merged HF state dict to mcore.

    Returns a flat dict of ``language_model.*`` -> tensor.
    """
    out = {}

    # ---- embeddings + final norm (output_layer tied -> not emitted) ----------
    out[f"{MC_LM}.embedding.word_embeddings.weight"] = hf_sd[f"{HF_LM}.embed_tokens.weight"].clone()
    out[f"{MC_LM}.decoder.final_layernorm.weight"] = hf_sd[f"{HF_LM}.norm.weight"].clone()

    # ---- decoder layers ------------------------------------------------------
    n_full, n_lin = 0, 0
    for i in range(num_layers):
        pre = f"{HF_LM}.layers.{i}."
        layer_sd = {k[len(pre):]: v for k, v in hf_sd.items() if k.startswith(pre)}
        assert layer_sd, f"no HF weights found for layer {i} (prefix {pre})"
        mpre = f"{MC_LM}.decoder.layers.{i}."

        # non-attention: separate input_layernorm + pre_mlp_layernorm + gated MLP
        out[mpre + "input_layernorm.weight"] = layer_sd["input_layernorm.weight"].clone()
        out[mpre + "pre_mlp_layernorm.weight"] = layer_sd["post_attention_layernorm.weight"].clone()
        out[mpre + "mlp.linear_fc1.weight"] = torch.cat(
            [layer_sd["mlp.gate_proj.weight"], layer_sd["mlp.up_proj.weight"]], dim=0
        ).clone()
        out[mpre + "mlp.linear_fc2.weight"] = layer_sd["mlp.down_proj.weight"].clone()

        # attention: dispatch on layer type
        is_full = ((i + 1) % full_attention_interval == 0)
        if is_full:
            attn = convert_full_attn_layer(layer_sd)
            n_full += 1
        else:
            attn = convert_linear_attn_layer(layer_sd)
            n_lin += 1
        for k, v in attn.items():
            out[mpre + "self_attention." + k] = v

    print(f"[qwen35_hf_to_mcore] converted {num_layers} layers: {n_full} full-attn, {n_lin} linear")

    if add_extra_state:
        # TE modules carry a serialized ``_extra_state`` blob; emit None-blobs so a
        # strict load has nothing to complain about (values are irrelevant at load).
        def blob():
            b = io.BytesIO()
            torch.save(None, b)
            return b
        te_suffixes = []
        for i in range(num_layers):
            mpre = f"{MC_LM}.decoder.layers.{i}."
            is_full = ((i + 1) % full_attention_interval == 0)
            te_suffixes += [mpre + "mlp.linear_fc1._extra_state", mpre + "mlp.linear_fc2._extra_state"]
            if is_full:
                te_suffixes += [
                    mpre + "self_attention.linear_qkv._extra_state",
                    mpre + "self_attention.linear_proj._extra_state",
                    mpre + "self_attention.core_attention._extra_state",
                ]
            else:
                te_suffixes += [
                    mpre + "self_attention.in_proj_qkvz._extra_state",
                    mpre + "self_attention.in_proj_ba._extra_state",
                    mpre + "self_attention.out_proj._extra_state",
                ]
        for k in te_suffixes:
            out[k] = blob()
        out[f"{MC_LM}.output_layer._extra_state"] = blob()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_ckpt_path", required=True, help="merged HF checkpoint dir")
    ap.add_argument("--save_ckpt_path", required=True, help="output dir (a 'release/' subdir is created)")
    ap.add_argument("--num_layers", type=int, default=32)
    ap.add_argument("--full_attention_interval", type=int, default=4)
    ap.add_argument("--no_extra_state", action="store_true")
    args = ap.parse_args()

    print(f"[qwen35_hf_to_mcore] loading HF checkpoint: {args.load_ckpt_path}")
    hf_sd = load_hf_state_dict(args.load_ckpt_path)
    lang = convert_language(hf_sd, args.num_layers, args.full_attention_interval,
                            add_extra_state=not args.no_extra_state)

    save_dir = os.path.join(args.save_ckpt_path, "release", "mp_rank_00")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "model_optim_rng.pt")
    torch.save({"model": lang}, out_path)
    n_w = sum(1 for k in lang if not k.endswith("_extra_state"))
    print(f"[qwen35_hf_to_mcore] wrote {n_w} language weights (+extra_state) -> {out_path}")


if __name__ == "__main__":
    main()
