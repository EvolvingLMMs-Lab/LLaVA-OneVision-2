"""Build tiny deterministic dense/MoE merge fixtures.
Creates tiny-{variant} triplets for refactor smoke tests.
Processor assets are copied from the canonical OV1.5 processor source.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, try_to_load_from_cache
from safetensors.torch import load_file, save_file


DEFAULT_PROCESSOR_REPO = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
PROCESSOR_FAILURE_MESSAGE = (
    "cannot resolve LLaVA-OneVision-1.5-8B-Instruct processor: "
    "set OV2_REFERENCE_PROCESSOR=<local-dir-or-repo-id> or enable network"
)
MAX_LLM_FILE_BYTES = 5 * 1024 * 1024
MAX_ADAPTER_FILE_BYTES = 200 * 1024 * 1024
MAX_VIT_FILE_BYTES = 2 * 1024 * 1024 * 1024


class ProcessorResolutionError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["dense", "moe"], required=True)
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def ensure_transformers_impl_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    transformers_impl = repo_root / "transformers_impl"
    transformers_impl_str = str(transformers_impl)
    if transformers_impl_str not in sys.path:
        sys.path.insert(0, transformers_impl_str)
    return repo_root


def deterministic_tensor(shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    return torch.arange(int(torch.Size(shape).numel()), dtype=torch.float32).reshape(shape) * 1e-3


def deterministic_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: deterministic_tensor(tensor.shape).contiguous() for name, tensor in state_dict.items()}


VIT_HIDDEN_SIZE = 1024
VIT_INTERMEDIATE_SIZE = 4096
VIT_NUM_HIDDEN_LAYERS = 24
VIT_NUM_ATTENTION_HEADS = 16
VIT_PATCH_SIZE = 14
LLM_HIDDEN_SIZE = 64
SPATIAL_MERGE_SIZE = 2


def build_vit_state_dict() -> dict[str, torch.Tensor]:
    ensure_transformers_impl_on_path()
    from onevision_encoder import OneVisionEncoderConfig, OneVisionEncoderModel

    config = OneVisionEncoderConfig(
        hidden_size=VIT_HIDDEN_SIZE,
        intermediate_size=VIT_INTERMEDIATE_SIZE,
        num_hidden_layers=VIT_NUM_HIDDEN_LAYERS,
        num_attention_heads=VIT_NUM_ATTENTION_HEADS,
        image_size=448,
        patch_size=VIT_PATCH_SIZE,
        use_head=False,
        _attn_implementation="eager",
    )
    model = OneVisionEncoderModel(config)
    return deterministic_state_dict(model.state_dict())


def build_adapter_state_dict() -> dict[str, torch.Tensor]:
    proj_hidden_size = VIT_HIDDEN_SIZE * (SPATIAL_MERGE_SIZE**2)
    return {
        "model.mm_projector.ln_q.weight": deterministic_tensor((VIT_HIDDEN_SIZE,)),
        "model.mm_projector.ln_q.bias": deterministic_tensor((VIT_HIDDEN_SIZE,)),
        "model.mm_projector.mlp.0.weight": deterministic_tensor((proj_hidden_size, proj_hidden_size)),
        "model.mm_projector.mlp.0.bias": deterministic_tensor((proj_hidden_size,)),
        "model.mm_projector.mlp.2.weight": deterministic_tensor((LLM_HIDDEN_SIZE, proj_hidden_size)),
        "model.mm_projector.mlp.2.bias": deterministic_tensor((LLM_HIDDEN_SIZE,)),
    }


def build_dense_llm() -> tuple[object, dict[str, torch.Tensor]]:
    import transformers

    if hasattr(transformers, "Qwen3Config") and hasattr(transformers, "Qwen3ForCausalLM"):
        config_cls = transformers.Qwen3Config
        model_cls = transformers.Qwen3ForCausalLM
    else:
        config_cls = transformers.Qwen2Config
        model_cls = transformers.Qwen2ForCausalLM

    config = config_cls(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    config.architectures = [model_cls.__name__]
    model = model_cls(config)
    return config, deterministic_state_dict(model.state_dict())


def build_moe_llm() -> tuple[object, dict[str, torch.Tensor]]:
    import transformers

    config = transformers.Qwen3MoeConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    model = transformers.Qwen3MoeForCausalLM(config)
    return config, deterministic_state_dict(model.state_dict())


def verify_vit_fixture(vit_path: Path) -> None:
    loaded = load_file(vit_path)
    if not any(fnmatch.fnmatch(key, "*.self_attn.q_proj.weight") for key in loaded):
        raise AssertionError(f"missing pre-fuse q_proj key in {vit_path}")


def cached_snapshot_dir(repo_id: str) -> Path | None:
    cache_hit = try_to_load_from_cache(repo_id, "tokenizer_config.json")
    if isinstance(cache_hit, str):
        cache_path = Path(cache_hit)
        if cache_path.exists():
            return cache_path.parent

    snapshot_root = (
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    )
    if not snapshot_root.exists():
        return None

    candidates = sorted(
        (path for path in snapshot_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "tokenizer_config.json").exists():
            return candidate
    return None


def resolve_processor_source() -> Path:
    processor_override = os.environ.get("OV2_REFERENCE_PROCESSOR")
    if processor_override:
        override_path = Path(processor_override)
        if override_path.is_dir():
            return override_path.resolve()
        repo_id = processor_override
        try:
            return Path(
                snapshot_download(
                    repo_id,
                    allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer*", "*.model"],
                )
            )
        except Exception as exc:  # pragma: no cover - exercised when cache/network both unavailable.
            raise ProcessorResolutionError(PROCESSOR_FAILURE_MESSAGE) from exc

    cached = cached_snapshot_dir(DEFAULT_PROCESSOR_REPO)
    if cached is not None:
        return cached

    try:
        return Path(
            snapshot_download(
                DEFAULT_PROCESSOR_REPO,
                allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer*", "*.model"],
            )
        )
    except Exception as exc:  # pragma: no cover - exercised when cache/network both unavailable.
        raise ProcessorResolutionError(PROCESSOR_FAILURE_MESSAGE) from exc


def copy_processor_tree(processor_dir: Path) -> None:
    source_dir = resolve_processor_source()
    shutil.copytree(source_dir, processor_dir, dirs_exist_ok=True)
    delete_patterns = [
        "*.safetensors",
        "pytorch_model*.bin",
        "consolidated.*",
        "model.*",
        "model.safetensors.index.json",
    ]
    for pattern in delete_patterns:
        for path in processor_dir.rglob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()


def verify_processor_fixture(processor_dir: Path) -> None:
    from transformers import AutoProcessor, Qwen2Tokenizer

    Qwen2Tokenizer.from_pretrained(processor_dir, trust_remote_code=True, use_fast=True)
    processor = AutoProcessor.from_pretrained(processor_dir, use_fast=True)
    image_processor = processor.image_processor
    if image_processor is None:
        raise AttributeError(f"processor loaded from {processor_dir} has no image_processor")


def verify_file_sizes(root_dir: Path) -> None:
    caps = {"vit": MAX_VIT_FILE_BYTES, "adapter": MAX_ADAPTER_FILE_BYTES, "llm": MAX_LLM_FILE_BYTES}
    for sub, cap in caps.items():
        for path in (root_dir / sub).rglob("*"):
            if path.is_file() and path.stat().st_size >= cap:
                raise AssertionError(f"synthetic fixture file {path} exceeds cap of {cap} bytes")


def remove_existing(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def build_variant(variant: str, out_root: Path, force: bool) -> Path:
    torch.manual_seed(0)
    fixture_dir = out_root / f"tiny-{variant}"
    if fixture_dir.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {fixture_dir}")
        remove_existing(fixture_dir)

    vit_dir = fixture_dir / "vit"
    adapter_dir = fixture_dir / "adapter"
    llm_dir = fixture_dir / "llm"
    processor_dir = fixture_dir / "processor"
    for directory in (vit_dir, adapter_dir, llm_dir, processor_dir):
        directory.mkdir(parents=True, exist_ok=True)

    vit_path = vit_dir / "model.safetensors"
    save_file(build_vit_state_dict(), vit_path)
    verify_vit_fixture(vit_path)

    adapter_path = adapter_dir / "model.safetensors"
    save_file(build_adapter_state_dict(), adapter_path)

    llm_config, llm_state_dict = build_dense_llm() if variant == "dense" else build_moe_llm()
    save_file(llm_state_dict, llm_dir / "model.safetensors")
    llm_config.save_pretrained(llm_dir)

    copy_processor_tree(processor_dir)
    verify_processor_fixture(processor_dir)
    verify_file_sizes(fixture_dir)
    return fixture_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        fixture_dir = build_variant(args.variant, out_root, args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ProcessorResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(fixture_dir)
    print(fixture_dir / "vit" / "model.safetensors")
    print(fixture_dir / "adapter" / "model.safetensors")
    print(fixture_dir / "llm" / "model.safetensors")
    print(fixture_dir / "llm" / "config.json")
    print(fixture_dir / "processor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
