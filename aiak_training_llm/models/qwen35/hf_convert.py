# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Qwen3.5 -> mcore (AIAK-Megatron) single-decoder-layer weight converters.

Reshape math is copied verbatim from LoongForge
tools/convert_checkpoint/huggingface/util/{hf_mixer_attn_converter.py, hf_attn_converter.py}
(cat_qkv_z_to_qkvz / cat_b_a_to_ba / cat_attn_qgkv), specialized to the
Qwen3.5-4B-LLM (Qwen3_5TextConfig) shapes.

HF param names verified against Qwen3.5-4B-LLM:
  linear_attn (Qwen3_5GatedDeltaNet):
    A_log (nv,), dt_bias (nv,), conv1d.weight (conv_dim,1,k)  [no conv1d.bias],
    in_proj_qkv.weight (2*nk*dk + nv*dv, H), in_proj_z.weight (nv*dv, H),
    in_proj_b.weight (nv, H), in_proj_a.weight (nv, H),
    norm.weight (dv,), out_proj.weight (H, nv*dv)
  self_attn (Qwen3_5Attention):
    q_proj.weight (2*nh*hd, H), k_proj.weight (nkv*hd, H), v_proj.weight (nkv*hd, H),
    o_proj.weight (H, nh*hd), q_norm.weight (hd,), k_norm.weight (hd,)
"""

import torch


# ----- Qwen3.5-4B-LLM linear-attention dims -----
LIN_NK = 16   # linear_num_key_heads
LIN_NV = 32   # linear_num_value_heads
LIN_DK = 128  # linear_key_head_dim
LIN_DV = 128  # linear_value_head_dim

# ----- Qwen3.5-4B-LLM full-attention dims -----
FULL_NH = 16   # num_attention_heads
FULL_NKV = 4   # num_key_value_heads
FULL_HD = 256  # head_dim
HIDDEN = 2560


def cat_qkv_z_to_qkvz(qkv_weight, z_weight, nk=LIN_NK, nv=LIN_NV, dk=LIN_DK, dv=LIN_DV):
    """HF [in_proj_qkv, in_proj_z] -> mcore interleaved in_proj_qkvz.

    Verbatim from HfMixerAttnConverter.cat_qkv_z_to_qkvz.
    """
    r = nv // nk  # value heads per key-head group
    H = qkv_weight.shape[1]

    q = qkv_weight[:nk * dk].reshape(nk, dk, H)
    k = qkv_weight[nk * dk:2 * nk * dk].reshape(nk, dk, H)
    v = qkv_weight[2 * nk * dk:].reshape(nk, r * dv, H)
    z = z_weight.reshape(nk, r * dv, H)

    qkvz = torch.cat([q, k, v, z], dim=1).reshape(-1, H)
    return qkvz


def cat_b_a_to_ba(b_weight, a_weight, nk=LIN_NK, nv=LIN_NV):
    """HF [in_proj_b, in_proj_a] -> mcore interleaved in_proj_ba.

    Verbatim from HfMixerAttnConverter.cat_b_a_to_ba.
    """
    r = nv // nk
    H = b_weight.shape[1]

    b = b_weight.reshape(nk, r, H)
    a = a_weight.reshape(nk, r, H)

    ba = torch.cat([b, a], dim=1).reshape(-1, H)
    return ba


def cat_attn_qgkv(q, k, v, num_key_value_heads=FULL_NKV, hidden_size=HIDDEN):
    """HF [q_proj, k_proj, v_proj] (q_proj carries interleaved query|gate per head)
    -> mcore gated linear_qkv.

    Verbatim from HfAttnGateQkvConverter.cat_attn_qgkv.
    """
    value = torch.cat([
        q.reshape((num_key_value_heads, -1, hidden_size)),
        k.reshape((num_key_value_heads, -1, hidden_size)),
        v.reshape((num_key_value_heads, -1, hidden_size)),
    ], dim=1).reshape((-1, hidden_size))
    return value


def convert_linear_attn_layer(hf_sd_layer):
    """Convert one HF linear-attention layer's ``linear_attn.*`` params to mcore
    ``self_attention.*`` params (GatedDeltaNet).

    Args:
        hf_sd_layer: dict mapping HF param name -> tensor. Names may be the bare
            names (``A_log``, ``in_proj_qkv.weight`` ...) or ``linear_attn.``-prefixed.

    Returns:
        dict of mcore param name -> tensor (no module prefix), keys:
            A_log, dt_bias, conv1d.weight, in_proj_qkvz.weight, in_proj_ba.weight,
            out_norm.weight, out_proj.weight  (+ conv1d.bias iff present in HF).
    """
    def g(name):
        if name in hf_sd_layer:
            return hf_sd_layer[name]
        return hf_sd_layer["linear_attn." + name]

    out = {}
    # verbatim copies
    out["A_log"] = g("A_log").clone()
    out["dt_bias"] = g("dt_bias").clone()
    out["conv1d.weight"] = g("conv1d.weight").clone()
    if "conv1d.bias" in hf_sd_layer or "linear_attn.conv1d.bias" in hf_sd_layer:
        out["conv1d.bias"] = g("conv1d.bias").clone()
    out["out_norm.weight"] = g("norm.weight").clone()
    out["out_proj.weight"] = g("out_proj.weight").clone()
    # interleaved projections
    out["in_proj_qkvz.weight"] = cat_qkv_z_to_qkvz(g("in_proj_qkv.weight"), g("in_proj_z.weight"))
    out["in_proj_ba.weight"] = cat_b_a_to_ba(g("in_proj_b.weight"), g("in_proj_a.weight"))
    return out


def convert_full_attn_layer(hf_sd_layer):
    """Convert one HF full-attention layer's ``self_attn.*`` params to mcore
    ``self_attention.*`` params (Qwen3NextSelfAttention).

    Returns dict keys:
        linear_qkv.weight, linear_proj.weight, q_layernorm.weight, k_layernorm.weight.
    """
    def g(name):
        if name in hf_sd_layer:
            return hf_sd_layer[name]
        return hf_sd_layer["self_attn." + name]

    out = {}
    out["linear_qkv.weight"] = cat_attn_qgkv(
        g("q_proj.weight"), g("k_proj.weight"), g("v_proj.weight")
    )
    out["linear_proj.weight"] = g("o_proj.weight").clone()
    out["q_layernorm.weight"] = g("q_norm.weight").clone()
    out["k_layernorm.weight"] = g("k_norm.weight").clone()
    return out
