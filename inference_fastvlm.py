#!/usr/bin/env python3
"""
FastVLM End-to-End Inference Test
Tests FastViT + Adapter + MobileLLM-R1-140M pipeline.

Run with a single GPU (no torchrun needed, handles dist init internally):
  CUDA_VISIBLE_DEVICES=0 /home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/python3 \
      inference_fastvlm.py \
      [--image /path/to/image.jpg] \
      [--prompt "Describe what you see."] \
      [--checkpoint stage_1_alignment_mobilellm_140m/iter_0000019] \
      [--max-new-tokens 150]
"""

# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — Handle sys.argv BEFORE any imports that touch argparse / Megatron.
# We split our inference args out of argv and rebuild argv for Megatron.
# ──────────────────────────────────────────────────────────────────────────────
import sys
import os
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Our inference args ────────────────────────────────────────────────────────
_inf_parser = argparse.ArgumentParser(add_help=False)
_inf_parser.add_argument("--image", type=str, default=None,
                         help="Path to image file (PNG/JPG). Defaults to random noise.")
_inf_parser.add_argument("--prompt", type=str,
                         default="<|image_pad|>\nDescribe what you see in this image.",
                         help="Prompt. Keep <|image_pad|> where the image should go.")
_inf_parser.add_argument("--checkpoint", type=str, default=None,
                         help="Megatron checkpoint dir. Defaults to latest stage-1 ckpt.")
_inf_parser.add_argument("--max-new-tokens", type=int, default=150)
_inf_parser.add_argument("--temperature", type=float, default=1.0)
our_args, _ = _inf_parser.parse_known_args()

# ── Resolve checkpoint path ───────────────────────────────────────────────────
_STAGE1_CKPT = os.path.join(REPO_ROOT, "stage_1_alignment_mobilellm_140m/iter_0000019")
_MERGED_CKPT = os.path.join(REPO_ROOT, "checkpoints/mobilellm-fastvit-merged-tp1-pp1/release")

if our_args.checkpoint:
    _LOAD_CKPT = our_args.checkpoint if os.path.isabs(our_args.checkpoint) \
                 else os.path.join(REPO_ROOT, our_args.checkpoint)
elif os.path.exists(_STAGE1_CKPT):
    _LOAD_CKPT = _STAGE1_CKPT
else:
    _LOAD_CKPT = _MERGED_CKPT

_DATA_PATH = os.path.join(REPO_ROOT, "data/LLaVA-558K-Webdataset")

# ── Rebuild sys.argv for Megatron argument parser ────────────────────────────
# (Megatron parses sys.argv directly via argparse)
sys.argv = [
    sys.argv[0],
    # ── model ──────────────────────────────────────────────────────────────
    "--model-name",           "llava-ov-mobilellm-140m",
    # ── tokenizer ──────────────────────────────────────────────────────────
    "--tokenizer-type",       "HFTokenizer",
    "--hf-tokenizer-path",    "facebook/MobileLLM-R1-140M",
    # ── vision ─────────────────────────────────────────────────────────────
    "--use-fastvit",
    "--fastvit-image-size",   "1024",
    "--vision-tower-name",    "mobileclip_l_1024",
    "--image-aspect-ratio",   "pad",
    # ── training phase (sets dataloader-type=external) ─────────────────────
    "--training-phase",       "sft",
    "--chat-template",        "llama3",
    "--trainable-modules",    "adapter",
    # ── required training dimensions ───────────────────────────────────────
    "--micro-batch-size",     "1",
    "--global-batch-size",    "1",
    "--train-iters",          "1",
    "--seq-length",           "4096",
    "--max-position-embeddings", "32768",
    # ── architecture ───────────────────────────────────────────────────────
    "--attention-backend",    "local",
    "--transformer-impl",     "local",
    "--no-gradient-accumulation-fusion",
    "--no-rope-fusion",
    "--norm-epsilon",         "1e-05",
    "--init-method-std",      "0.02",
    "--training-rice-vl-max-answer-length", "32768",
    # ── precision ──────────────────────────────────────────────────────────
    "--bf16",
    # ── model parallelism ──────────────────────────────────────────────────
    "--tensor-model-parallel-size", "1",
    "--pipeline-model-parallel-size", "1",
    "--use-distributed-optimizer",
    "--distributed-backend",  "nccl",
    # ── checkpoint ─────────────────────────────────────────────────────────
    "--pretrained-checkpoint", _LOAD_CKPT,
    "--no-load-optim",
    "--no-load-rng",
    # ── data (dummy — not used for inference) ──────────────────────────────
    "--data-path",            _DATA_PATH,
    "--split",                "100,0,0",
    "--num-workers",          "0",
    "--dataloader-type",      "external",
    # ── logging ────────────────────────────────────────────────────────────
    "--log-interval",         "1",
]

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Add repo to path, then import Megatron and model code.
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "aiak_megatron"))

# Single-GPU distributed bootstrap (needed before Megatron init)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29600")

import torch
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Initialize Megatron (parses sys.argv, inits distributed, tokenizer…)
# ──────────────────────────────────────────────────────────────────────────────
def megatron_init():
    from aiak_training_llm.train.arguments import (
        aiak_extra_train_args_provider,
        validate_aiak_extra_args,
    )
    from aiak_training_llm.utils.initialize import (
        parse_arguments,
        initialize_aiak_megatron,
    )
    # parse_arguments sets global Megatron args; validate_aiak_extra_args
    # also applies MobileLLM config from the model registry.
    args = parse_arguments(
        extra_args_provider=aiak_extra_train_args_provider,
        validate_extra_args_provider=validate_aiak_extra_args,
    )
    initialize_aiak_megatron(args)
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Build model with the existing provider (same as training)
# ──────────────────────────────────────────────────────────────────────────────
def build_model():
    from megatron.training.training import get_model
    from aiak_training_llm.models import get_model_provider, get_model_family
    from aiak_training_llm.utils import get_args

    args = get_args()
    model_family = get_model_family(args.model_name)
    provider = get_model_provider(model_family)

    # get_model wraps provider in DDP etc., returns a list of model chunks
    models = get_model(provider, model_type=None, wrap_with_ddp=False)
    # Unwrap list/tuple; for PP=1 there is exactly one chunk
    model = models[0] if isinstance(models, (list, tuple)) else models
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Load checkpoint weights into the model
# ──────────────────────────────────────────────────────────────────────────────
def load_weights(model, ckpt_dir: str):
    """
    Direct weight copy from a Megatron checkpoint.
    Supports both the merged pre-train checkpoint and stage-1 checkpoints.
    """
    ckpt_file = os.path.join(ckpt_dir, "mp_rank_00", "model_optim_rng.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"\n[ckpt] Loading weights from:\n       {ckpt_file}")
    raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    saved = raw["model"]

    # Stage-1 checkpoints have 'language_model.*' prefix.
    # Merged pre-train has top-level keys without that prefix.
    has_lm_prefix = any(k.startswith("language_model.") for k in saved)

    def remap(k):
        if has_lm_prefix:
            return k
        if k.startswith(("embedding.", "decoder.", "output_layer.")):
            return "language_model." + k
        return k

    remapped = {remap(k): v for k, v in saved.items()}

    model_sd   = model.state_dict()
    loaded_n   = 0
    skipped    = []
    missing    = []

    for name, param in model_sd.items():
        if name in remapped:
            src = remapped[name].to(dtype=param.dtype)
            if src.shape == param.shape:
                model_sd[name].copy_(src)
                loaded_n += 1
            else:
                skipped.append(f"{name}: saved {src.shape} vs model {param.shape}")
        else:
            missing.append(name)

    model.load_state_dict(model_sd, strict=False)
    print(f"[ckpt] Loaded  : {loaded_n} / {len(model_sd)} tensors")
    if missing:
        pfx = sorted(set(n.split(".")[0] for n in missing))
        print(f"[ckpt] Missing : {len(missing)} tensors (kept random init) — prefixes: {pfx}")
    if skipped:
        print(f"[ckpt] Skipped (shape mismatch): {skipped[:5]}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(path, size, device, dtype):
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        print(f"[img] Loaded  : {path}  original size {img.size}")
    else:
        if path:
            print(f"[img] WARNING: {path!r} not found — using random noise")
        arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        print(f"[img] Using random noise image ({size}×{size})")

    img_resized = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # [H, W, 3] in [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)         # [3, H, W]
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)
    print(f"[img] Tensor   : {tuple(tensor.shape)}  "
          f"min={tensor.min():.3f}  max={tensor.max():.3f}")
    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Tokenisation
# ──────────────────────────────────────────────────────────────────────────────
def tokenise_prompt(prompt, tokenizer, image_token_id, device):
    # tokenizer is the AutoTokenizerFromHF wrapper; use inner HF tokenizer for encode/decode
    inner = tokenizer.tokenizer
    ids = inner.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    n_img = (input_ids == image_token_id).sum().item()
    print(f"[tok] Prompt tokens : {input_ids.shape[1]}  |  image placeholders : {n_img}")
    print(f"[tok] Decoded       : {inner.decode(ids)!r}")
    return input_ids


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Dimension diagnostic (one forward pass, no generation)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def dimension_check(model, images, input_ids, tokenizer, image_token_id):
    print("\n" + "═" * 68)
    print("  DIMENSION CHECK")
    print("═" * 68)

    # Monkey-patch feature_select to print the raw spatial shape before pooling
    vt = model.vision_model.vision_tower
    _orig = vt.feature_select
    def _patched(out):
        raw = out["image_embeddings"]
        B, C, H, W = raw.shape
        print(f"  [dim] FastViT spatial (pre-pool) : {(B, C, H, W)}  "
              f"= {B} images × {H*W} patches × {C}-dim")
        result = _orig(out)
        print(f"  [dim] After global avg pool      : {tuple(result.shape)}  "
              f"= {result.shape[0]} images × {result.shape[1]}-dim")
        return result
    vt.feature_select = _patched

    seq_len = input_ids.shape[1]
    pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    logits = model(
        images=images,
        image_grid_thw=None,
        input_ids=input_ids,
        position_ids=pos_ids,
        attention_mask=None,
        attn_mask_type=None,
        labels=None,
        packed_seq_params=None,
    )

    # Normalise shape to [batch, seq, vocab]
    if logits.dim() == 3 and logits.shape[0] != input_ids.shape[0]:
        logits = logits.transpose(0, 1).contiguous()

    print(f"\n  Final logits : {tuple(logits.shape)}")

    # Top-5 predicted next tokens
    top5 = logits[0, -1, :].topk(5)
    print(f"\n  Top-5 predicted next tokens (after prompt):")
    _inner = tokenizer.tokenizer  # AutoTokenizerFromHF → inner HF tokenizer
    for val, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        tok_text = _inner.decode([idx], skip_special_tokens=False)
        print(f"    token_id={idx:6d}  logit={val:8.3f}  decoded={tok_text!r}")

    vt.feature_select = _orig  # restore
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Step 8 — Autoregressive generation (greedy, full recompute — no KV cache)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, images, input_ids, tokenizer, image_token_id,
             max_new_tokens=150, temperature=1.0):
    """
    Greedy decode without KV cache.

    On every step we pass `images` so the model correctly substitutes
    image token positions with visual features (those positions persist in
    input_ids throughout generation).  This is correct but slow; it is
    fine for a pipeline sanity-check.
    """
    eos_ids = {tokenizer.eos_token_id}
    # Llama3 / MobileLLM may have multiple EOS ids
    if hasattr(tokenizer, "eos_token_ids"):
        eos_ids |= set(tokenizer.eos_token_ids)

    generated_ids = []
    print(f"\n[gen] Generating up to {max_new_tokens} tokens …")
    print("─" * 60)
    print("RESPONSE: ", end="", flush=True)

    for step in range(max_new_tokens):
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        logits = model(
            images=images,           # always pass images (for correct visual sub.)
            image_grid_thw=None,
            input_ids=input_ids,
            position_ids=pos_ids,
            attention_mask=None,
            attn_mask_type=None,
            labels=None,
            packed_seq_params=None,
        )

        # Normalise to [batch, seq, vocab]
        if logits.dim() == 3 and logits.shape[0] != 1:
            logits = logits.transpose(0, 1).contiguous()

        last_logit = logits[0, -1, :]   # [vocab]
        if temperature != 1.0:
            last_logit = last_logit / temperature

        next_id = int(last_logit.argmax())
        generated_ids.append(next_id)

        tok_text = tokenizer.decode([next_id], skip_special_tokens=False)
        print(tok_text, end="", flush=True)

        if next_id in eos_ids:
            print()
            break

        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

    print("\n" + "─" * 60)
    return generated_ids


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 68)
    print("  FastVLM Inference — FastViT + Adapter + MobileLLM-R1-140M")
    print("═" * 68 + "\n")

    # 1. Init Megatron
    print("[init] Initialising Megatron (single GPU mode) …")
    args = megatron_init()
    print(f"[init] Done.  dtype=bfloat16, TP=1, PP=1, device=cuda:0")

    device = torch.device("cuda:0")
    dtype  = torch.bfloat16

    # 2. Tokenizer (initialised by Megatron, just fetch it)
    from aiak_training_llm.utils import get_tokenizer
    tokenizer = get_tokenizer()
    # AutoTokenizerFromHF wraps the inner HF tokenizer in .tokenizer
    _inner_tok = tokenizer.tokenizer
    IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    EOS_ID = _inner_tok.eos_token_id
    print(f"[tok] image_token_id = {IMAGE_TOKEN_ID}  (eos={EOS_ID})")

    # 3. Build model
    print("\n[model] Building LlavaOnevision1_5 (FastViT + Adapter + MobileLLM) …")
    model = build_model()
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] Total parameters : {total:.1f}M")

    # 4. Load checkpoint
    load_weights(model, _LOAD_CKPT)
    print(f"[ckpt] Source: {_LOAD_CKPT}")

    # 5. Image
    images = preprocess_image(our_args.image, 1024, device, dtype)

    # 6. Prompt
    prompt = our_args.prompt
    if "<|image_pad|>" not in prompt:
        prompt = "<|image_pad|>\n" + prompt
        print("[tok] Prepended <|image_pad|> to prompt")
    input_ids = tokenise_prompt(prompt, tokenizer, IMAGE_TOKEN_ID, device)
    _inner_tok_ref = tokenizer.tokenizer

    # 7. Dimension check
    _ = dimension_check(model, images, input_ids, tokenizer, IMAGE_TOKEN_ID)
    # (dimension_check only uses tokenizer.tokenizer for decode internally)

    # 8. Generation
    print("\n" + "═" * 68)
    print("  GENERATION")
    print("═" * 68)
    print(f"PROMPT: {prompt!r}\n")

    gen_ids = generate(
        model          = model,
        images         = images,
        input_ids      = input_ids,
        tokenizer      = _inner_tok,
        image_token_id = IMAGE_TOKEN_ID,
        max_new_tokens = our_args.max_new_tokens,
        temperature    = our_args.temperature,
    )

    response = _inner_tok.decode(gen_ids, skip_special_tokens=True)
    print(f"\nFull decoded response:\n  {response!r}")

    # 9. Summary
    print("\n" + "═" * 68)
    print("  SUMMARY")
    print("═" * 68)
    print(f"  Checkpoint    : {_LOAD_CKPT}")
    print(f"  Image         : {our_args.image or 'random noise (1024×1024)'}")
    print(f"  Prompt tokens : {input_ids.shape[1]}")
    print(f"  Gen tokens    : {len(gen_ids)}")
    print(f"  Vision        : FastViT MobileCLIP-L @1024 → global avg pool → 3072-dim")
    print(f"  Adapter       : 3072 → 576  (2-layer MLP, SiLU)")
    print(f"  Language      : MobileLLM-R1-140M (15 layers, 576-dim, GQA 9/3 heads)")
    print()


if __name__ == "__main__":
    main()
