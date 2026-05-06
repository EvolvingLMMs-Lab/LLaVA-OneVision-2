#!/usr/bin/env python3
"""Create a repo-faithful FastViT + MobileLLM checkpoint for LLaVA-OV.

The original merged checkpoint stores MobileLLM weights under standalone
language-model names such as ``embedding.*`` and ``decoder.*``.  The wrapped
LLaVA-OneVision model expects those weights under ``language_model.*``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def _checkpoint_file(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "release" / "mp_rank_00" / "model_optim_rng.pt"


def _pad_embedding(
    weight: torch.Tensor,
    target_vocab_size: int,
    std: float,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    current_vocab_size, hidden_size = weight.shape
    if current_vocab_size == target_vocab_size:
        return weight, 0
    if current_vocab_size > target_vocab_size:
        raise ValueError(
            f"Embedding has {current_vocab_size} rows, larger than target {target_vocab_size}."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    extra = torch.empty(
        target_vocab_size - current_vocab_size,
        hidden_size,
        dtype=weight.dtype,
        device="cpu",
    )
    extra.normal_(mean=0.0, std=std, generator=generator)
    return torch.cat([weight.cpu(), extra], dim=0), extra.shape[0]


def _remap_language_key(key: str) -> str | None:
    if key == "embedding.word_embeddings.weight":
        return "language_model.embedding.word_embeddings.weight"
    if key == "decoder.final_layernorm.weight":
        return "language_model.decoder.final_layernorm.weight"
    if "_extra_state" in key:
        return None
    if not key.startswith("decoder.layers."):
        return None

    replacements = (
        (".self_attention.linear_qkv.weight", ".self_attention.linear_qkv.linear.weight"),
        (
            ".self_attention.linear_qkv.layer_norm_weight",
            ".self_attention.linear_qkv.ln.weight",
        ),
        (".self_attention.linear_proj.weight", ".self_attention.linear_proj.linear.weight"),
        (".mlp.linear_fc1.weight", ".mlp.linear_fc1.linear.weight"),
        (".mlp.linear_fc1.layer_norm_weight", ".mlp.linear_fc1.ln.weight"),
        (".mlp.linear_fc2.weight", ".mlp.linear_fc2.linear.weight"),
    )
    for old_suffix, new_suffix in replacements:
        if key.endswith(old_suffix):
            return "language_model." + key[: -len(old_suffix)] + new_suffix
    return None


def create_checkpoint(
    source_dir: Path,
    output_dir: Path,
    target_vocab_size: int,
    embedding_init_std: float,
    embedding_seed: int,
) -> None:
    source_file = _checkpoint_file(source_dir)
    output_file = _checkpoint_file(output_dir)
    if not source_file.exists():
        raise FileNotFoundError(source_file)
    if output_file.exists():
        raise FileExistsError(
            f"{output_file} already exists. Move it aside if you want to recreate it."
        )

    checkpoint = torch.load(source_file, map_location="cpu", weights_only=False)
    source_model: Dict[str, torch.Tensor] = checkpoint["model"]
    output_model: Dict[str, torch.Tensor] = {}

    counts = {
        "vision": 0,
        "language": 0,
        "language_ln_bias": 0,
        "skipped_extra_state": 0,
        "skipped_other": 0,
    }
    added_embedding_rows = 0

    for key, value in source_model.items():
        if key.startswith("vision_model."):
            output_model[key] = value.cpu() if torch.is_tensor(value) else value
            counts["vision"] += 1
            continue

        new_key = _remap_language_key(key)
        if new_key is None:
            if "_extra_state" in key:
                counts["skipped_extra_state"] += 1
            else:
                counts["skipped_other"] += 1
            continue

        tensor = value.cpu() if torch.is_tensor(value) else value
        if key == "embedding.word_embeddings.weight":
            tensor, added_embedding_rows = _pad_embedding(
                tensor,
                target_vocab_size=target_vocab_size,
                std=embedding_init_std,
                seed=embedding_seed,
            )
        output_model[new_key] = tensor
        counts["language"] += 1

        if new_key.endswith(".linear_qkv.ln.weight") or new_key.endswith(".linear_fc1.ln.weight"):
            bias_key = new_key[:-len("weight")] + "bias"
            output_model[bias_key] = torch.zeros_like(tensor)
            counts["language_ln_bias"] += 1

    if "language_model.embedding.word_embeddings.weight" not in output_model:
        raise RuntimeError("Failed to remap MobileLLM embedding weight.")
    if not any(key.startswith("vision_model.") for key in output_model):
        raise RuntimeError("No FastViT vision weights were copied.")
    if any(key.startswith(("embedding.", "decoder.")) for key in output_model):
        raise RuntimeError("Unwrapped language keys leaked into output checkpoint.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": output_model,
            "checkpoint_version": checkpoint.get("checkpoint_version", 3.0),
            "args": checkpoint.get("args", None),
            "iteration": 0,
        },
        output_file,
    )
    (output_dir / "latest_checkpointed_iteration.txt").write_text("release\n")

    print(f"source: {source_file}")
    print(f"output: {output_file}")
    print(f"keys written: {len(output_model)}")
    print(f"vision keys copied: {counts['vision']}")
    print(f"language tensors remapped: {counts['language']}")
    print(f"layernorm biases added: {counts['language_ln_bias']}")
    print(f"embedding rows added: {added_embedding_rows}")
    print(f"extra_state keys skipped: {counts['skipped_extra_state']}")
    print(f"other keys skipped: {counts['skipped_other']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("checkpoints/mobilellm-fastvit-merged-tp1-pp1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/mobilellm-fastvit-merged-faithful-tp1-pp1"),
    )
    parser.add_argument("--target-vocab-size", type=int, default=128384)
    parser.add_argument("--embedding-init-std", type=float, default=0.02)
    parser.add_argument("--embedding-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_checkpoint(
        source_dir=args.source,
        output_dir=args.output,
        target_vocab_size=args.target_vocab_size,
        embedding_init_std=args.embedding_init_std,
        embedding_seed=args.embedding_seed,
    )


if __name__ == "__main__":
    main()
