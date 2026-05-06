import torch


def remap_vit(sd: dict) -> dict:
    out, qkv_w, qkv_b = {}, {}, {}
    for k, v in sd.items():
        k = k.replace("encoder._orig_mod.", "encoder.", 1)
        if k.endswith(".inv_freq") or k.startswith(("head.", "layernorm_post.")):
            continue
        for proj in ("q_proj", "k_proj", "v_proj"):
            if f".self_attn.{proj}." in k:
                prefix = k.rsplit(f".{proj}.", 1)[0]
                bucket = qkv_b if k.endswith(".bias") else qkv_w
                bucket.setdefault(prefix, {})[proj[0]] = v
                break
        else:
            new_k = k.replace(".self_attn.out_proj.", ".self_attn.proj.")
            out[f"model.visual.{new_k}"] = v
    for prefix, parts in qkv_w.items():
        if {"q", "k", "v"} <= parts.keys():
            out[f"model.visual.{prefix}.qkv.weight"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
    for prefix, parts in qkv_b.items():
        if {"q", "k", "v"} <= parts.keys():
            out[f"model.visual.{prefix}.qkv.bias"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
    return out


def remap_adapter(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(".inv_freq"):
            continue
        new_k = k.replace("model.mm_projector", "model.visual.merger", 1) if k.startswith("model.mm_projector") else k
        out[new_k] = v
    return out


def remap_llm(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(".inv_freq"):
            continue
        new_k = "model.language_model." + k[len("model.") :] if k.startswith("model.") else k
        out[new_k] = v
    if "lm_head.weight" not in out and "model.language_model.embed_tokens.weight" in out:
        out["lm_head.weight"] = out["model.language_model.embed_tokens.weight"]
    return out
