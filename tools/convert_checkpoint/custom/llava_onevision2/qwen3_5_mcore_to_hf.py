#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert the Qwen3.5 hybrid-attention LANGUAGE part of a Megatron-Core (AIAK)
LLaVA-OneVision2 checkpoint back to the merged-HF layout.

This is the exact inverse of ``qwen3_5_hf_to_mcore.py``:
  * un-shard the per-rank tensor-parallel shards back to a single TP1 language
    state dict (inverse of ``shard_language``), then
  * invert each per-decoder-layer weight rule via
    ``aiak_training_llm.models.qwen3_5.hf_convert.mcore_to_hf_{linear,full}_attn_layer``
    (which invert convert_{linear,full}_attn_layer, cos=1.0 / bit-identical proven).

TP UN-SHARD (inverse of shard_language), applied per parameter suffix:
  * COLUMN-PARALLEL (was chunk dim0)  -> torch.cat(shards, dim=0):
        embedding.word_embeddings.weight, output_layer.weight (untied only),
        self_attention.linear_qkv.weight
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
Plus ``lm_head.weight``: the un-sharded ``output_layer.weight`` for untied models
(9B/27B), otherwise a copy of embed_tokens.weight (tied 2B/4B).

Layer type rule (matches forward): (i+1) % full_attention_interval == 0 -> full-attn.
Primary target: PP=1 (1D mp_rank_{t:02d} layout), TP in {1,2,4}.

MoE (Qwen3.5-MoE / qwen3_5_moe, e.g. the 35B-A3B backbone): layers carrying
``mlp.router.weight`` take the MoE path instead of the dense SwiGLU one, and
``--expert_model_parallel_size`` un-shards the expert-parallel layout written by
``qwen3_5_split_ep.py`` (routed experts are gathered from mp_rank_*_{e:03d} and
renumbered local->global; everything else is replicated, rank 0 wins).
MoE is TP=1 only -- qwen3_5_hf_to_mcore.py does not TP-shard expert weights, so a
TP>1 MoE checkpoint is rejected rather than silently mis-assembled.
"""
import argparse
import os
import sys

import re
import torch
from safetensors.torch import save_file

from aiak_training_llm.models.qwen3_5.hf_convert import (
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


# Suffix rules -- MUST mirror qwen3_5_hf_to_mcore.shard_language.
_COL_DIM0 = (
    "embedding.word_embeddings.weight",
    "output_layer.weight",                # untied models only (vocab dim0)
    "self_attention.linear_qkv.weight",
)
_ROW_DIM1 = (
    "self_attention.linear_proj.weight",
    "mlp.linear_fc2.weight",
)
_FC1 = ("mlp.linear_fc1.weight",)

# Routed-expert weights of a Qwen3.5-MoE layer: mlp.experts.linear_fc{1,2}.weight{expert}
_EXPERT_RE = re.compile(r"\.mlp\.experts\.linear_fc([12])\.weight(\d+)$")
# Every MoE parameter that is NOT a routed expert (replicated across EP ranks).
_MOE_NON_EXPERT = (
    "mlp.router.weight",
    "mlp.shared_experts.linear_fc1.weight",
    "mlp.shared_experts.linear_fc2.weight",
    "mlp.shared_experts.gate_weight",
)


def _count_local_experts(model):
    """Number of routed experts stored in one EP shard (None if it has no MoE keys)."""
    ids = {int(m.group(2)) for m in (_EXPERT_RE.search(k) for k in model) if m}
    return max(ids) + 1 if ids else None


def _globalize_expert_key(key, ep_rank, n_local):
    """mlp.experts.linear_fcX.weight{local} -> ...weight{local + ep_rank * n_local}."""
    m = _EXPERT_RE.search(key)
    return key[: m.start()] + f".mlp.experts.linear_fc{m.group(1)}.weight{int(m.group(2)) + ep_rank * n_local}"


def _shard_subdir(t, pstage, e, pp, ep):
    """Megatron checkpoint sub-directory name (get_checkpoint_name, PP/EP aware).

    pp==1, ep==1 -> mp_rank_{t:02d}
    pp>1,  ep==1 -> mp_rank_{t:02d}_{p:03d}
    pp==1, ep>1  -> mp_rank_{t:02d}_{e:03d}
    pp>1,  ep>1  -> mp_rank_{t:02d}_{p:03d}_{e:03d}
    NOTE the 2-part name is ambiguous between a PP and an EP checkpoint; the
    caller must say which via --pipeline_model_parallel_size / --expert_model_parallel_size.
    """
    name = f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{pstage:03d}"
    return name if ep == 1 else f"{name}_{e:03d}"


def load_language_shards(load_path, tp, pp=1, ep=1, return_untie_flag=False):
    """Load the mcore language shards and return a list of tp flat dicts of
    language_model.* -> tensor (weights only, _extra_state dropped).

    EP UN-SHARD (inverse of qwen3_5_split_ep.py): routed-expert weights
    ``mlp.experts.linear_fc{1,2}.weight{local}`` live only on the EP rank that
    owns them and are numbered LOCALLY (0..num_experts/ep-1); they are gathered
    and renumbered back to global ids. Every other parameter (router, shared
    expert, gate, attention, norms) is replicated across EP ranks, so EP rank 0
    wins.

    With return_untie_flag=True also return the checkpoint's saved
    ``args.untie_embeddings_and_output_weights`` (None if the checkpoint carries
    no megatron args, e.g. a converted ``release``).

    PP=1: 1D mp_rank_{t:02d} layout. PP>1: 2D mp_rank_{t:02d}_{p:03d} layout --
    pipeline stages are merged per TP rank: each stage stores decoder.layers.N
    with LOCAL indices, so stage p's layers are renumbered by the cumulative
    layer count of the preceding stages. language_model.output_layer.weight
    lives on the last stage: for untied models it is the real lm_head; for tied
    models it is only a synced copy of the embedding (cf. reshard_pp2.py, which
    materializes it there). It is kept either way; convert_language_mcore_to_hf
    decides whether it is used.
    """
    layer_re = re.compile(r"(language_model\.decoder\.layers\.)(\d+)(\..*)$")
    shards = []
    untie_flag = None
    n_local_experts = None
    for t in range(tp):
        lang = {}
        offset = 0
        for pstage in range(pp):
            stage_layers = set()
            for e in range(ep):
                sub = _shard_subdir(t, pstage, e, pp, ep)
                ckpt_path = os.path.join(load_path, sub, "model_optim_rng.pt")
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError(f"missing shard: {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if t == 0 and pstage == 0 and e == 0 and state.get("args") is not None:
                    untie_flag = getattr(state["args"], "untie_embeddings_and_output_weights", None)
                model = state["model"]
                del state
                if ep > 1:
                    local = _count_local_experts(model)
                    if local is not None:
                        if n_local_experts is None:
                            n_local_experts = local
                        elif n_local_experts != local:
                            raise ValueError(
                                f"EP rank {e} holds {local} local experts but rank 0 holds "
                                f"{n_local_experts}; uneven expert sharding is not supported")
                for k, v in model.items():
                    if not k.startswith(MC_LM + ".") or k.endswith("_extra_state"):
                        continue
                    is_expert = _EXPERT_RE.search(k) is not None
                    # non-expert params are replicated across EP ranks -> rank 0 wins
                    if e > 0 and not is_expert:
                        continue
                    if is_expert and ep > 1:
                        k = _globalize_expert_key(k, e, n_local_experts)
                    lm = layer_re.match(k)
                    if lm:
                        stage_layers.add(int(lm.group(2)))
                        lang[f"{lm.group(1)}{int(lm.group(2)) + offset}{lm.group(3)}"] = v
                    else:
                        lang[k] = v
            offset += (max(stage_layers) + 1) if stage_layers else 0
        assert lang, f"no language_model.* weights under {load_path} (tp{t})"
        shards.append(lang)
    if return_untie_flag:
        return shards, untie_flag
    return shards


def unshard_language(shards, tp):
    """Inverse of shard_language: reconstruct the full TP1 language state dict."""
    if tp == 1:
        return dict(shards[0])
    if any(_EXPERT_RE.search(k) or k.endswith(_MOE_NON_EXPERT) for k in shards[0]):
        raise NotImplementedError(
            "MoE checkpoints are only supported at TP=1 (expert parallelism is handled "
            "separately via --expert_model_parallel_size). qwen3_5_hf_to_mcore.py does not "
            "shard expert/shared-expert weights across TP either, so a TP>1 MoE checkpoint "
            "produced by it would be wrong; refusing to un-shard one here.")
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


def _convert_moe_mlp_to_hf(layer_sd, hpre, out, fused_experts=False):
    """mcore MoELayer -> HF qwen3_5_moe layer MLP. Inverse of
    qwen3_5_hf_to_mcore._convert_moe_mlp.

    mcore:
      mlp.router.weight                    [E, H]
      mlp.experts.linear_fc1.weight{e}     [2I, H] = cat[gate_e, up_e]
      mlp.experts.linear_fc2.weight{e}     [H, I]
      mlp.shared_experts.linear_fc1.weight [2I, H], linear_fc2.weight [H, I]
      mlp.shared_experts.gate_weight       [1, H]
    HF:
      mlp.gate.weight                                  <- router
      mlp.experts.{e}.{gate,up,down}_proj.weight       (default; what transformers
                                                        writes on save and what the
                                                        merged 35B model carries), or
      mlp.experts.{gate_up_proj,down_proj}             [E, 2I, H] / [E, H, I] when
                                                        fused_experts=True (the layout
                                                        of the original Qwen3.5/3.6
                                                        release). Both reload fine.
      mlp.shared_expert.{gate,up,down}_proj.weight
      mlp.shared_expert_gate.weight                    <- shared_experts.gate_weight
    Returns the number of routed experts converted.
    """
    out[hpre + "mlp.gate.weight"] = layer_sd["mlp.router.weight"].clone().contiguous()

    ids = sorted(int(m.group(2)) for m in (_EXPERT_RE.search("." + k) for k in layer_sd) if m
                 and m.group(1) == "1")
    assert ids, f"{hpre}: MoE layer without routed experts"
    assert ids == list(range(len(ids))), (
        f"{hpre}: routed expert ids are not contiguous from 0: "
        f"{ids[:4]}..{ids[-4:]} (EP un-shard incomplete? check --expert_model_parallel_size)")
    n_exp = len(ids)

    gates, ups, downs = [], [], []
    for e in ids:
        fc1 = layer_sd[f"mlp.experts.linear_fc1.weight{e}"]
        fc2 = layer_sd[f"mlp.experts.linear_fc2.weight{e}"]
        assert fc1.shape[0] % 2 == 0, f"{hpre}: expert {e} fc1 dim0 {fc1.shape[0]} not even"
        g, u = torch.chunk(fc1, 2, dim=0)
        if fused_experts:
            gates.append(fc1)
            downs.append(fc2)
        else:
            ep_pre = hpre + f"mlp.experts.{e}."
            out[ep_pre + "gate_proj.weight"] = g.clone().contiguous()
            out[ep_pre + "up_proj.weight"] = u.clone().contiguous()
            out[ep_pre + "down_proj.weight"] = fc2.clone().contiguous()
    if fused_experts:
        out[hpre + "mlp.experts.gate_up_proj"] = torch.stack(gates, dim=0).contiguous()
        out[hpre + "mlp.experts.down_proj"] = torch.stack(downs, dim=0).contiguous()

    sh_fc1 = layer_sd["mlp.shared_experts.linear_fc1.weight"]
    assert sh_fc1.shape[0] % 2 == 0, f"{hpre}: shared expert fc1 dim0 {sh_fc1.shape[0]} not even"
    sg, su = torch.chunk(sh_fc1, 2, dim=0)
    out[hpre + "mlp.shared_expert.gate_proj.weight"] = sg.clone().contiguous()
    out[hpre + "mlp.shared_expert.up_proj.weight"] = su.clone().contiguous()
    out[hpre + "mlp.shared_expert.down_proj.weight"] = \
        layer_sd["mlp.shared_experts.linear_fc2.weight"].clone().contiguous()
    gate_w = layer_sd.get("mlp.shared_experts.gate_weight")
    if gate_w is None:
        raise KeyError(
            f"{hpre}: mlp.shared_experts.gate_weight missing. Qwen3.5-MoE scales the shared "
            "expert by sigmoid(shared_expert_gate(x)); a checkpoint trained with "
            "QWEN3_5_SHARED_EXPERT_GATE=0 cannot be exported to the HF layout.")
    out[hpre + "mlp.shared_expert_gate.weight"] = gate_w.clone().contiguous()
    return n_exp


def convert_language_mcore_to_hf(lang_sd, num_layers, full_attention_interval,
                                 orig_vocab_size=None, dims=None, ckpt_untie_flag=None,
                                 fused_experts=False):
    """Full TP1 mcore language dict -> HF ``model.language_model.*`` (+ lm_head).

    lm_head is decided from the weights themselves: the model is untied iff an
    ``output_layer.weight`` exists and differs from the embedding.
      * tied, PP=1  -> no output_layer                       -> lm_head = embed copy
      * tied, PP>1  -> last stage holds a bit-identical copy -> lm_head = embed copy
      * untied      -> distinct output_layer                 -> lm_head = output_layer
    Neither the merged HF config (which may say tie_word_embeddings=False even for
    tied models) nor megatron args (the wrapper injects synthesized args into
    arg-less checkpoints) are trusted for this. ``ckpt_untie_flag`` (the saved
    ``args.untie_embeddings_and_output_weights``) is only used to fail loudly when
    an untied checkpoint is missing its output_layer.
    """
    out = {}

    embed = lang_sd[f"{MC_LM}.embedding.word_embeddings.weight"]
    if orig_vocab_size is not None and embed.shape[0] > orig_vocab_size:
        embed = embed[:orig_vocab_size, :]
    out[f"{HF_LM}.embed_tokens.weight"] = embed.clone().contiguous()
    out[f"{HF_LM}.norm.weight"] = lang_sd[f"{MC_LM}.decoder.final_layernorm.weight"].clone()

    output_w = lang_sd.get(f"{MC_LM}.output_layer.weight")
    if output_w is not None and orig_vocab_size is not None and output_w.shape[0] > orig_vocab_size:
        output_w = output_w[:orig_vocab_size, :]
    if ckpt_untie_flag and output_w is None:
        raise ValueError(
            "checkpoint args say untie_embeddings_and_output_weights=True but there is "
            f"no {MC_LM}.output_layer.weight; refusing to write lm_head = embedding"
        )
    untied = output_w is not None and not torch.equal(output_w, embed)
    if untied:
        out["lm_head.weight"] = output_w.clone().contiguous()
    else:
        out["lm_head.weight"] = embed.clone().contiguous()  # tied
    print(f"[qwen3_5_mcore_to_hf] lm_head: {'untied (output_layer)' if untied else 'tied (embed copy)'}")

    n_full, n_lin, n_moe, n_exp = 0, 0, 0, 0
    for i in range(num_layers):
        mpre = f"{MC_LM}.decoder.layers.{i}."
        layer_sd = {k[len(mpre):]: v for k, v in lang_sd.items() if k.startswith(mpre)}
        assert layer_sd, f"no mcore weights for layer {i} (prefix {mpre})"
        hpre = f"{HF_LM}.layers.{i}."

        out[hpre + "input_layernorm.weight"] = layer_sd["input_layernorm.weight"].clone()
        out[hpre + "post_attention_layernorm.weight"] = layer_sd["pre_mlp_layernorm.weight"].clone()

        if "mlp.router.weight" in layer_sd:  # Qwen3.5-MoE layer
            n_exp = _convert_moe_mlp_to_hf(layer_sd, hpre, out, fused_experts=fused_experts)
            n_moe += 1
        else:
            # SwiGLU: mcore fc1 = cat([gate, up], dim0) -> split back
            fc1 = layer_sd["mlp.linear_fc1.weight"]
            assert fc1.shape[0] % 2 == 0, f"layer {i}: fc1 dim0 {fc1.shape[0]} not even"
            gate, up = torch.chunk(fc1, 2, dim=0)
            out[hpre + "mlp.gate_proj.weight"] = gate.clone().contiguous()
            out[hpre + "mlp.up_proj.weight"] = up.clone().contiguous()
            out[hpre + "mlp.down_proj.weight"] = layer_sd["mlp.linear_fc2.weight"].clone()

        is_full = ((i + 1) % full_attention_interval == 0)
        if is_full:
            attn = mcore_to_hf_full_attn_layer(layer_sd, dims=dims)
            n_full += 1
        else:
            attn = mcore_to_hf_linear_attn_layer(layer_sd, dims=dims)
            n_lin += 1
        for k, v in attn.items():
            out[hpre + k] = v

    moe_note = (f", {n_moe} MoE ({n_exp} routed experts"
                f"{', fused' if fused_experts else ', per-expert'})" if n_moe else "")
    print(f"[qwen3_5_mcore_to_hf] converted {num_layers} layers: "
          f"{n_full} full-attn, {n_lin} linear{moe_note}")
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
    ap.add_argument("--pipeline_model_parallel_size", type=int, default=1)
    ap.add_argument("--expert_model_parallel_size", type=int, default=1,
                    help="EP size of the checkpoint (MoE only). Shards are read from "
                         "mp_rank_{t:02d}[_{p:03d}]_{e:03d} and routed experts are "
                         "renumbered back to global ids.")
    ap.add_argument("--fused_experts", action="store_true",
                    help="emit packed mlp.experts.{gate_up_proj,down_proj} [E,...] tensors "
                         "(original Qwen3.5/3.6 release layout) instead of per-expert keys")
    ap.add_argument("--hf_config_path", type=str, default=None,
                    help="HF model dir whose config.json provides hybrid-attn dims "
                         "(e.g. the reference merged model); default = Qwen3.5-4B dims")
    ap.add_argument("--orig_vocab_size", type=int, default=None,
                    help="if the mcore embedding was vocab-padded, cut back to this size")
    args = ap.parse_args()

    tp = args.tensor_model_parallel_size
    assert tp >= 1

    print(f"[qwen3_5_mcore_to_hf] loading {tp} shard(s) from: {args.load_ckpt_path}")
    shards, untie_flag = load_language_shards(
        args.load_ckpt_path, tp, pp=args.pipeline_model_parallel_size,
        ep=args.expert_model_parallel_size, return_untie_flag=True)
    lang = unshard_language(shards, tp)
    from aiak_training_llm.models.qwen3_5.hf_convert import dims_from_hf_config
    dims = dims_from_hf_config(args.hf_config_path) if args.hf_config_path else None
    print(f"[qwen3_5_mcore_to_hf] dims: {dims or 'default (4B)'}")
    print(f"[qwen3_5_mcore_to_hf] ckpt args untie_embeddings_and_output_weights={untie_flag}")
    hf = convert_language_mcore_to_hf(lang, args.num_layers, args.full_attention_interval,
                                      orig_vocab_size=args.orig_vocab_size, dims=dims,
                                      ckpt_untie_flag=untie_flag,
                                      fused_experts=args.fused_experts)

    os.makedirs(args.save_ckpt_path, exist_ok=True)
    out_path = os.path.join(args.save_ckpt_path, "model.safetensors")
    save_file({k: v.contiguous() for k, v in hf.items()}, out_path, metadata={"format": "pt"})
    print(f"[qwen3_5_mcore_to_hf] wrote {len(hf)} HF language weights -> {out_path}")


if __name__ == "__main__":
    main()
