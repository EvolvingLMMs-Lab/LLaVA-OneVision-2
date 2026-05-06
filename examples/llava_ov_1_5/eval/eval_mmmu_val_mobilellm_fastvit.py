#!/usr/bin/env python3
"""
Evaluate the FastViT + MobileLLM-R1 checkpoint on MMMU validation.

This script is intentionally independent of lmms-eval because the current
checkpoint is Megatron/MCore format, not HuggingFace format.

Smoke test:
  CUDA_VISIBLE_DEVICES=4 /home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/python \
    examples/llava_ov_1_5/eval/eval_mmmu_val_mobilellm_fastvit.py \
    --checkpoint stage_1_alignment_mobilellm_140m/iter_0000100 \
    --subjects Math Physics \
    --max-samples 10 \
    --output-path eval_outputs/mmmu_val_mobilellm_fastvit_10.jsonl

Full validation:
  CUDA_VISIBLE_DEVICES=4 /home/ashaker/miniconda3/envs/llava-ov-4b-clean/bin/python \
    examples/llava_ov_1_5/eval/eval_mmmu_val_mobilellm_fastvit.py \
    --checkpoint stage_1_alignment_mobilellm_140m/iter_0000100 \
    --subjects all \
    --output-path eval_outputs/mmmu_val_mobilellm_fastvit_full.jsonl
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_our_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMMU-val evaluator for FastViT + MobileLLM")
    parser.add_argument("--checkpoint", default="stage_1_alignment_mobilellm_140m/iter_0000100")
    parser.add_argument("--output-path", default="eval_outputs/mmmu_val_mobilellm_fastvit.jsonl")
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--hf-cache", default=str(REPO_ROOT / "data" / "hf_cache"))
    parser.add_argument("--local-mmmu-path", default="", help="Optional local MMMU dataset root.")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--master-port", default="29610")
    parser.add_argument("--verbose-forward", action="store_true", help="Print per-token Megatron forward diagnostics.")
    return parser.parse_args()


OUR_ARGS = parse_our_args()


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


LOAD_CKPT = resolve_path(OUR_ARGS.checkpoint)
DATA_PATH = str(REPO_ROOT / "data" / "LLaVA-558K-Webdataset")


def configure_megatron_argv() -> None:
    sys.argv = [
        sys.argv[0],
        "--model-name", "llava-ov-mobilellm-140m",
        "--tokenizer-type", "HFTokenizer",
        "--hf-tokenizer-path", "facebook/MobileLLM-R1-140M",
        "--use-fastvit",
        "--fastvit-image-size", str(OUR_ARGS.image_size),
        "--vision-tower-name", f"mobileclip_l_{OUR_ARGS.image_size}",
        "--image-aspect-ratio", "pad",
        "--training-phase", "sft",
        "--chat-template", "llama3",
        "--trainable-modules", "adapter",
        "--micro-batch-size", "1",
        "--global-batch-size", "1",
        "--train-iters", "1",
        "--seq-length", "4096",
        "--max-position-embeddings", "32768",
        "--attention-backend", "local",
        "--transformer-impl", "local",
        "--no-gradient-accumulation-fusion",
        "--no-rope-fusion",
        "--norm-epsilon", "1e-05",
        "--init-method-std", "0.02",
        "--training-rice-vl-max-answer-length", "32768",
        "--bf16",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--use-distributed-optimizer",
        "--distributed-backend", "nccl",
        "--pretrained-checkpoint", LOAD_CKPT,
        "--no-load-optim",
        "--no-load-rng",
        "--data-path", DATA_PATH,
        "--split", "100,0,0",
        "--num-workers", "0",
        "--dataloader-type", "external",
        "--log-interval", "1",
    ]


configure_megatron_argv()

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "aiak_megatron"))

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", OUR_ARGS.master_port)
os.environ.setdefault("HF_DATASETS_CACHE", OUR_ARGS.hf_cache)

import numpy as np
import torch
from PIL import Image


SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]


def megatron_init():
    from aiak_training_llm.train.arguments import (
        aiak_extra_train_args_provider,
        validate_aiak_extra_args,
    )
    from aiak_training_llm.utils.initialize import initialize_aiak_megatron, parse_arguments

    args = parse_arguments(
        extra_args_provider=aiak_extra_train_args_provider,
        validate_extra_args_provider=validate_aiak_extra_args,
    )
    initialize_aiak_megatron(args)
    return args


def build_model():
    from megatron.training.training import get_model
    from aiak_training_llm.models import get_model_family, get_model_provider
    from aiak_training_llm.utils import get_args

    args = get_args()
    model_family = get_model_family(args.model_name)
    provider = get_model_provider(model_family)
    models = get_model(provider, model_type=None, wrap_with_ddp=False)
    return models[0] if isinstance(models, (list, tuple)) else models


def load_weights(model, ckpt_dir: str) -> None:
    ckpt_file = Path(ckpt_dir) / "mp_rank_00" / "model_optim_rng.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    print(f"[ckpt] Loading {ckpt_file}")
    raw = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    saved = raw["model"]

    has_lm_prefix = any(k.startswith("language_model.") for k in saved)

    def remap(k: str) -> str:
        if has_lm_prefix:
            return k
        if k.startswith(("embedding.", "decoder.", "output_layer.")):
            return "language_model." + k
        return k

    remapped = {remap(k): v for k, v in saved.items()}
    model_sd = model.state_dict()
    loaded = 0
    for name, param in model_sd.items():
        src = remapped.get(name)
        if src is not None and src.shape == param.shape:
            model_sd[name].copy_(src.to(dtype=param.dtype))
            loaded += 1
    model.load_state_dict(model_sd, strict=False)
    print(f"[ckpt] Loaded {loaded}/{len(model_sd)} tensors")


def parse_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, list):
        return [str(x) for x in options]
    if isinstance(options, str):
        try:
            parsed = ast.literal_eval(options)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [x.strip() for x in options.split("\n") if x.strip()]
    return []


def get_images(sample: dict[str, Any]) -> list[Image.Image]:
    images = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            images.append(img.convert("RGB"))
    if not images and sample.get("image") is not None:
        images.append(sample["image"].convert("RGB"))
    return images


def expand2square(image: Image.Image, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    square = Image.new(image.mode, (side, side), background)
    square.paste(image, ((side - width) // 2, (side - height) // 2))
    return square


def preprocess_images(images: list[Image.Image], size: int, device, dtype) -> torch.Tensor:
    tensors = []
    for img in images:
        img = expand2square(img)
        resized = img.resize((size, size), Image.BICUBIC)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors).to(device=device, dtype=dtype)


def build_prompt(sample: dict[str, Any], num_images: int) -> str:
    question = sample["question"].strip()
    options = parse_options(sample.get("options"))
    tokens_per_image = (OUR_ARGS.image_size // 64) ** 2
    image_tokens = "\n".join(["<|image_pad|>" * tokens_per_image] * max(1, num_images))

    if sample.get("question_type") == "multiple-choice" and options:
        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        options_text = "\n".join(f"{letter}. {option}" for letter, option in zip(option_letters, options))
        instruction = "Answer with only the option letter."
        return f"{image_tokens}\n{question}\n{options_text}\n{instruction}"

    return f"{image_tokens}\n{question}\nAnswer the question directly."


def tokenise_prompt(prompt: str, tokenizer, image_token_id: int, device) -> torch.Tensor:
    ids = tokenizer.tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    count = (input_ids == image_token_id).sum().item()
    if count == 0:
        raise ValueError("Prompt does not contain an image token after tokenization.")
    return input_ids


@torch.no_grad()
def generate(model, images, input_ids, tokenizer, max_new_tokens: int, temperature: float) -> str:
    eos_ids = {tokenizer.eos_token_id}
    generated = []
    for _ in range(max_new_tokens):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        quiet_context = contextlib.nullcontext()
        if not OUR_ARGS.verbose_forward:
            quiet_context = contextlib.redirect_stdout(io.StringIO())
        with quiet_context:
            logits = model(
                images=images,
                image_grid_thw=None,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                attn_mask_type=None,
                labels=None,
                packed_seq_params=None,
            )
        if logits.dim() == 3 and logits.shape[0] != 1:
            logits = logits.transpose(0, 1).contiguous()
        next_logits = logits[0, -1, :]
        if temperature and temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))
        else:
            next_id = int(next_logits.argmax())
        generated.append(next_id)
        if next_id in eos_ids:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)],
            dim=1,
        )
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_choice(prediction: str, num_options: int) -> str:
    letters = [chr(ord("A") + i) for i in range(num_options)]
    pred = prediction.strip()
    match = re.search(r"\b([A-Z])\b", pred.upper())
    if match and match.group(1) in letters:
        return match.group(1)
    if pred[:1].upper() in letters:
        return pred[:1].upper()
    return pred


def score_prediction(sample: dict[str, Any], prediction: str) -> tuple[bool, str]:
    answer = str(sample["answer"]).strip()
    options = parse_options(sample.get("options"))
    if sample.get("question_type") == "multiple-choice" and options:
        pred = extract_choice(prediction, len(options))
        return pred == answer, pred
    return normalize_answer(prediction) == normalize_answer(answer), prediction


def load_mmmu_samples(subjects: list[str], max_samples: int) -> list[dict[str, Any]]:
    import datasets

    Path(OUR_ARGS.hf_cache).mkdir(parents=True, exist_ok=True)
    selected_subjects = SUBJECTS if subjects == ["all"] else subjects
    samples: list[dict[str, Any]] = []
    for subject in selected_subjects:
        dataset_name = "MMMU/MMMU"
        if OUR_ARGS.local_mmmu_path:
            dataset_name = str(Path(OUR_ARGS.local_mmmu_path) / subject)
            ds = datasets.load_dataset(
                dataset_name,
                split=datasets.Split.VALIDATION,
                cache_dir=OUR_ARGS.hf_cache,
                verification_mode="no_checks",
            )
        else:
            ds = datasets.load_dataset(
                dataset_name,
                subject,
                split=datasets.Split.VALIDATION,
                cache_dir=OUR_ARGS.hf_cache,
            )
        for sample in ds:
            if str(sample.get("id", "")).startswith("val"):
                sample = dict(sample)
                sample["subject"] = subject
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    return samples
    return samples


def main() -> None:
    print("[init] Initialising Megatron")
    megatron_init()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    from aiak_training_llm.utils import get_tokenizer

    tokenizer = get_tokenizer()
    inner_tokenizer = tokenizer.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    print("[model] Building model")
    model = build_model().to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    load_weights(model, LOAD_CKPT)

    print("[data] Loading MMMU validation samples")
    samples = load_mmmu_samples(OUR_ARGS.subjects, OUR_ARGS.max_samples)
    print(f"[data] Loaded {len(samples)} samples")

    output_path = Path(resolve_path(OUR_ARGS.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    mc_correct = 0
    mc_total = 0
    open_correct = 0
    open_total = 0

    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            images = get_images(sample)
            if not images:
                print(f"[warn] Skipping {sample.get('id')} with no image")
                continue

            prompt = build_prompt(sample, len(images))
            image_tensor = preprocess_images(images, OUR_ARGS.image_size, device, dtype)
            input_ids = tokenise_prompt(prompt, tokenizer, image_token_id, device)
            prediction = generate(
                model,
                image_tensor,
                input_ids,
                inner_tokenizer,
                OUR_ARGS.max_new_tokens,
                OUR_ARGS.temperature,
            )
            is_correct, parsed_prediction = score_prediction(sample, prediction)

            total += 1
            correct += int(is_correct)
            if sample.get("question_type") == "multiple-choice":
                mc_total += 1
                mc_correct += int(is_correct)
            else:
                open_total += 1
                open_correct += int(is_correct)

            row = {
                "id": sample.get("id"),
                "subject": sample.get("subject"),
                "question_type": sample.get("question_type"),
                "answer": sample.get("answer"),
                "prediction": prediction,
                "parsed_prediction": parsed_prediction,
                "correct": is_correct,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            running = 100.0 * correct / total
            print(f"[{idx}/{len(samples)}] {sample.get('id')} correct={is_correct} acc={running:.2f}% pred={parsed_prediction!r} ans={sample.get('answer')!r}")

    metrics = {
        "total": total,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "multiple_choice_total": mc_total,
        "multiple_choice_accuracy": 100.0 * mc_correct / mc_total if mc_total else 0.0,
        "open_total": open_total,
        "open_accuracy": 100.0 * open_correct / open_total if open_total else 0.0,
        "checkpoint": LOAD_CKPT,
        "output_path": str(output_path),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== MMMU validation summary =====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
