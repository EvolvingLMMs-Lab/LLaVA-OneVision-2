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


# =============================================================================
# REVERSE (mcore -> HuggingFace) single-decoder-layer weight converters.
# Exact inverses of the forward converters above. Do NOT change the forward funcs.
# The de-interleave reshapes mirror LoongForge
# huggingface/util/{hf_mixer_attn_converter.py, hf_attn_converter.py}
# (split_qkvz_to_qkv_z / split_ba_to_b_a / split_attn_qgkv).
# =============================================================================


def split_qkvz_to_qkv_z(qkvz, nk=LIN_NK, nv=LIN_NV, dk=LIN_DK, dv=LIN_DV):
    """Inverse of cat_qkv_z_to_qkvz.

    mcore interleaved in_proj_qkvz (per key-head group [q,k,v,z] stacked on dim1)
    -> HF (in_proj_qkv, in_proj_z).
    """
    r = nv // nk  # value heads per key-head group
    H = qkvz.shape[1]
    block = 2 * dk + 2 * r * dv  # per-group rows: q(dk)+k(dk)+v(r*dv)+z(r*dv)
    x = qkvz.reshape(nk, block, H)
    q = x[:, :dk, :]
    k = x[:, dk:2 * dk, :]
    v = x[:, 2 * dk:2 * dk + r * dv, :]
    z = x[:, 2 * dk + r * dv:, :]
    in_proj_qkv = torch.cat(
        [q.reshape(nk * dk, H), k.reshape(nk * dk, H), v.reshape(nk * r * dv, H)], dim=0
    ).contiguous()
    in_proj_z = z.reshape(nk * r * dv, H).contiguous()
    return in_proj_qkv, in_proj_z


def split_ba_to_b_a(ba, nk=LIN_NK, nv=LIN_NV):
    """Inverse of cat_b_a_to_ba: mcore in_proj_ba -> HF (in_proj_b, in_proj_a)."""
    r = nv // nk
    H = ba.shape[1]
    x = ba.reshape(nk, 2 * r, H)
    b = x[:, :r, :].reshape(nk * r, H).contiguous()
    a = x[:, r:, :].reshape(nk * r, H).contiguous()
    return b, a


def split_attn_qgkv(linear_qkv, num_key_value_heads=FULL_NKV, head_dim=FULL_HD):
    """Inverse of cat_attn_qgkv: mcore gated linear_qkv -> HF (q_proj, k_proj, v_proj).

    linear_qkv rows are grouped per key-value head as [q|gate (block-2*hd), k (hd),
    v (hd)]; the query+gate interleave lives inside the q slice and is preserved
    verbatim (we do NOT de-interleave query vs gate here).
    """
    H = linear_qkv.shape[1]
    total = linear_qkv.shape[0]
    block = total // num_key_value_heads
    qg_dim = block - 2 * head_dim
    x = linear_qkv.reshape(num_key_value_heads, block, H)
    q = x[:, :qg_dim, :].reshape(-1, H).contiguous()
    k = x[:, qg_dim:qg_dim + head_dim, :].reshape(-1, H).contiguous()
    v = x[:, qg_dim + head_dim:, :].reshape(-1, H).contiguous()
    return q, k, v


def mcore_to_hf_linear_attn_layer(mcore_layer):
    """Inverse of convert_linear_attn_layer. Returns HF ``linear_attn.*`` params.

    Accepts mcore names either bare (``A_log`` ...) or ``self_attention.``-prefixed.
    """
    def g(name):
        if name in mcore_layer:
            return mcore_layer[name]
        return mcore_layer["self_attention." + name]

    out = {}
    out["linear_attn.A_log"] = g("A_log").clone()
    out["linear_attn.dt_bias"] = g("dt_bias").clone()
    out["linear_attn.conv1d.weight"] = g("conv1d.weight").clone()
    if "conv1d.bias" in mcore_layer or "self_attention.conv1d.bias" in mcore_layer:
        out["linear_attn.conv1d.bias"] = g("conv1d.bias").clone()
    out["linear_attn.norm.weight"] = g("out_norm.weight").clone()
    out["linear_attn.out_proj.weight"] = g("out_proj.weight").clone()
    in_proj_qkv, in_proj_z = split_qkvz_to_qkv_z(g("in_proj_qkvz.weight"))
    out["linear_attn.in_proj_qkv.weight"] = in_proj_qkv
    out["linear_attn.in_proj_z.weight"] = in_proj_z
    b, a = split_ba_to_b_a(g("in_proj_ba.weight"))
    out["linear_attn.in_proj_b.weight"] = b
    out["linear_attn.in_proj_a.weight"] = a
    return out


def mcore_to_hf_full_attn_layer(mcore_layer):
    """Inverse of convert_full_attn_layer. Returns HF ``self_attn.*`` params.

    Accepts mcore names either bare (``linear_qkv.weight`` ...) or
    ``self_attention.``-prefixed.
    """
    def g(name):
        if name in mcore_layer:
            return mcore_layer[name]
        return mcore_layer["self_attention." + name]

    out = {}
    q, k, v = split_attn_qgkv(g("linear_qkv.weight"))
    out["self_attn.q_proj.weight"] = q
    out["self_attn.k_proj.weight"] = k
    out["self_attn.v_proj.weight"] = v
    out["self_attn.o_proj.weight"] = g("linear_proj.weight").clone()
    out["self_attn.q_norm.weight"] = g("q_layernorm.weight").clone()
    out["self_attn.k_norm.weight"] = g("k_layernorm.weight").clone()
    return out
