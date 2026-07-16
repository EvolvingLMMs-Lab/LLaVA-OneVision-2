#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_token_consistency.py

Sanity check that token lengths recorded by s2 match what the actual HF
processor would emit when fed the *post-preprocess* image bytes from the
generated WebDataset tar.

Pipeline:
    s2 token_info.txt  ──┐
    s4 webdataset tar ──┴──> for N random bins, decode JSON + images,
                              run AutoProcessor, compare token length.

Usage:
    python verify_token_consistency.py \
        --token-info /path/output/token_info.txt \
        --webdataset-dir /path/output/webdataset \
        --bins-file /path/output/bins_32klen.pkl \
        --model-path /ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33 \
        --n-samples 32
"""
from __future__ import annotations
import argparse
import io
import json
import os
import pickle
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
from jinja2 import Template
from PIL import Image, ImageOps
from transformers import AutoProcessor


CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] | replace('<image>', "
    "'<|vision_start|><|image_pad|><|vision_end|>') }}"
    "<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def load_token_info(path: str) -> dict[str, int]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ":" not in ln:
                continue
            name, tok = ln.rsplit(":", 1)
            out[name] = int(tok)
    return out


def iter_tar_samples(tar_path: Path):
    with tarfile.open(tar_path, "r") as tf:
        groups: dict[str, dict] = {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            key, _, ext = m.name.partition(".")
            buf = tf.extractfile(m).read()
            groups.setdefault(key, {})[ext] = buf
        for k, v in groups.items():
            yield k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-info", required=True)
    ap.add_argument("--webdataset-dir", required=True)
    ap.add_argument("--bins-file", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--tolerance", type=int, default=0,
                    help="Allowed |sum(token_info) - hf_processor| diff per bin (default 0)")
    args = ap.parse_args()

    print(f"# loading token_info: {args.token_info}", flush=True)
    token_info = load_token_info(args.token_info)

    print(f"# loading bins: {args.bins_file}", flush=True)
    with open(args.bins_file, "rb") as f:
        bins = pickle.load(f)

    print(f"# loading HF processor: {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    template = Template(CHAT_TEMPLATE)
    ip = processor.image_processor
    _patch = getattr(ip, "patch_size", "?")
    _merge = getattr(ip, "merge_size", 2)
    _size = getattr(ip, "size", {}) or {}
    _minp = getattr(ip, "min_pixels", None) or (_size.get("shortest_edge") if isinstance(_size, dict) else None)
    _maxp = getattr(ip, "max_pixels", None) or (_size.get("longest_edge") if isinstance(_size, dict) else None)
    print(f"# processor: patch={_patch} merge={_merge} min_pixels={_minp} max_pixels={_maxp}", flush=True)
    merge_factor = int(_merge) ** 2

    tars = sorted(Path(args.webdataset_dir).glob("*.tar"))
    print(f"# found {len(tars)} tar shards", flush=True)
    if not tars:
        sys.exit(1)

    # Build (bin_key -> tar) index by scanning first members of each tar
    # to map ps_XXXXXXXX -> tar (we sample randomly across all tars).
    candidates = []
    rng = random.Random(0)
    rng.shuffle(tars)
    for t in tars:
        for key, files in iter_tar_samples(t):
            if "json" in files:
                candidates.append((t, key, files))
                if len(candidates) >= args.n_samples * 4:
                    break
        if len(candidates) >= args.n_samples * 4:
            break

    rng.shuffle(candidates)
    candidates = candidates[: args.n_samples]
    print(f"# verifying {len(candidates)} packed bins", flush=True)

    ok = 0
    diffs = []
    for tar_path, key, files in candidates:
        payload = json.loads(files["json"].decode("utf-8"))
        # Sum recorded token lengths for sub-samples in this bin via bin idx.
        # Bin key format: ps_{bin_index:08d}
        bin_index = int(key.split("_")[1])
        bin_items = bins[bin_index]
        # bin items are tuples (orig_idx, token_len, name)
        def _tlen(it):
            # supports tuple, list, numpy.void (structured record), or dict
            if isinstance(it, dict):
                return int(it.get("len", it.get("token_len", 0)))
            try:
                return int(it[1])
            except Exception:
                return 0
        recorded_sum = sum(_tlen(it) for it in bin_items)

        # Per-sub-sample token computation via the processor.
        total_hf = 0
        for sub_idx, (prompts, captions, img_names) in enumerate(
            zip(payload["prompts"], payload["captions"], payload["images"])
        ):
            messages = []
            # Reconstruct messages: assume alternating user/assistant pairs.
            n_turns = max(len(prompts), len(captions))
            for ti in range(n_turns):
                if ti < len(prompts):
                    messages.append({"role": "user", "content": prompts[ti]})
                if ti < len(captions):
                    messages.append({"role": "assistant", "content": captions[ti]})
            txt = template.render({"messages": messages}, tokenize=False, add_generation_prompt=False)

            imgs = []
            for img_name in img_names:
                base = img_name.replace("img", "img").replace("_sub", "_")  # img000_sub000.jpg -> img000_000.jpg
                # The actual tar key for the image: img{sample_idx}_{img_idx}.jpg
                # img_name = img{sample_idx:03d}_sub{img_idx:03d}.jpg
                stem = img_name[:-4]  # strip .jpg
                a, _, b = stem.partition("_sub")
                sample_part = a.lstrip("img").lstrip("0") or "0"
                img_part = b.lstrip("0") or "0"
                tar_img_key = f"img{int(sample_part)}_{int(img_part)}.jpg"
                if tar_img_key not in files:
                    continue
                im = Image.open(io.BytesIO(files[tar_img_key]))
                im.load()
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                imgs.append(im)

            inputs = processor(text=[txt], images=imgs if imgs else None,
                               videos=None, padding=True, return_tensors="pt")
            total_hf += int(inputs["input_ids"].shape[1])

        diff = total_hf - recorded_sum
        diffs.append(diff)
        flag = "OK" if abs(diff) <= args.tolerance else "MISMATCH"
        if flag == "OK":
            ok += 1
        print(f"  bin {bin_index:08d} (tar={tar_path.name}): "
              f"hf={total_hf}, recorded={recorded_sum}, diff={diff} -> {flag}", flush=True)

    print(f"\nSummary: {ok}/{len(candidates)} bins within tolerance={args.tolerance}", flush=True)
    if diffs:
        arr = np.asarray(diffs)
        print(f"  diff stats: mean={arr.mean():.1f} std={arr.std():.1f} "
              f"min={arr.min()} max={arr.max()}", flush=True)
    sys.exit(0 if ok == len(candidates) else 2)


if __name__ == "__main__":
    main()
