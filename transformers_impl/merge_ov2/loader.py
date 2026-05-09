from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from huggingface_hub import snapshot_download

from transformers import logging

from .io import iter_safetensors
from .remap import remap_adapter, remap_llm, remap_vit


logger = logging.get_logger(__name__)


@dataclass
class LoadReport:
    source: str
    loaded: int = 0
    missing_in_model: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    covered_param_names: set[str] = field(default_factory=set)

    def render(self) -> str:
        return (
            f"[{self.source}] loaded={self.loaded} "
            f"missing_in_model={len(self.missing_in_model)} "
            f"shape_mismatch={len(self.shape_mismatch)}"
        )


# Model paths (vit/llm/adapter/ckpt): pull the full repo minus legacy weight
# formats. We still need config.json, modeling_*.py, preprocessor_config.json,
# tokenizer files and *.safetensors because dense.py / moe.py read the ViT
# config and validators reload via AutoModel(trust_remote_code=True) /
# AutoTokenizer / CLIPImageProcessor — all of which need the full repo layout.
_HUB_MODEL_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.gguf",
    "*.pt",
    "*.pth",
    "*.pkl",
    "*.tflite",
    "*.ot",
)

# Processor paths (processor/qwen_processor): the consumers are
# AutoProcessor / AutoTokenizer / CLIPImageProcessor.from_pretrained — they
# only need tokenizer + processor + image_processor configs (KB-scale). Many
# repos used as a processor source (e.g. lmms-lab/LLaVA-OneVision-1.5-*) are
# full N-billion-param model repos with ~16GB of safetensors we never read.
# Whitelist exactly the file shapes the *_from_pretrained loaders need.
_HUB_PROCESSOR_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
    "tokenizer*",
    "processor_config*",
    "preprocessor_config*",
    "tokenization_*.py",
    "processing_*.py",
    "image_processing_*.py",
    "video_processing_*.py",
    "feature_extraction_*.py",
)


def _resolve_local_or_hub(path: str, *, kind: str = "model") -> str:
    if os.path.exists(path):
        return path
    if kind == "processor":
        return snapshot_download(path, allow_patterns=list(_HUB_PROCESSOR_ALLOW_PATTERNS))
    if kind != "model":
        msg = f"unknown kind {kind!r}; expected 'model' or 'processor'"
        raise ValueError(msg)
    return snapshot_download(path, ignore_patterns=list(_HUB_MODEL_IGNORE_PATTERNS))


def apply_weights(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    source: str,
    *,
    target_dtype: torch.dtype | None = None,
    strict_shape: bool = True,
    strict_missing: bool = True,
) -> LoadReport:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    report = LoadReport(source=source)

    for k, v in weights:
        target = params.get(k)
        if target is None:
            target = buffers.get(k)
        if target is None:
            report.missing_in_model.append(k)
            if strict_missing:
                msg = f"[{source}] unknown key not in model: {k}"
                raise RuntimeError(msg)
            logger.warning(f"[{source}] key not in model, skipping: {k}")
            continue
        if tuple(target.shape) != tuple(v.shape):
            report.shape_mismatch.append((k, tuple(target.shape), tuple(v.shape)))
            if strict_shape:
                msg = f"[{source}] shape mismatch for {k}: model={tuple(target.shape)} weights={tuple(v.shape)}"
                raise RuntimeError(msg)
            continue
        cast = v if target_dtype is None else v.to(dtype=target_dtype)
        with torch.no_grad():
            target.copy_(cast.to(dtype=target.dtype, device=target.device))
        report.covered_param_names.add(k)
        report.loaded += 1

    return report


def assert_full_coverage(
    model: torch.nn.Module,
    reports: Iterable[LoadReport],
    *,
    expected_uncovered: Iterable[str] = (),
) -> list[str]:
    covered: set[str] = set()
    for r in reports:
        covered |= r.covered_param_names
    expected = set(expected_uncovered)
    all_param_names = {n for n, _ in model.named_parameters()}
    uncovered = sorted((all_param_names - covered) - expected)
    return uncovered


def load_all_weights(
    model: torch.nn.Module,
    vit_path: str,
    adapter_path: str,
    llm_path: str,
    *,
    target_dtype: torch.dtype | None = None,
) -> list[LoadReport]:
    reports: list[LoadReport] = []

    vit_src = _resolve_local_or_hub(vit_path)
    vit_sd = remap_vit(dict(iter_safetensors(vit_src)))
    reports.append(apply_weights(model, vit_sd.items(), "ViT", target_dtype=target_dtype))

    if adapter_path:
        adapter_sd = remap_adapter(dict(iter_safetensors(adapter_path)))
        reports.append(apply_weights(model, adapter_sd.items(), "Adapter", target_dtype=target_dtype))

    llm_src = _resolve_local_or_hub(llm_path)
    llm_sd = remap_llm(dict(iter_safetensors(llm_src)))
    reports.append(apply_weights(model, llm_sd.items(), "LLM", target_dtype=target_dtype))

    return reports


def dry_run_report(
    model: torch.nn.Module,
    vit_path: str,
    adapter_path: str,
    llm_path: str,
) -> tuple[list[LoadReport], list[str]]:
    reports: list[LoadReport] = []
    params = {n: tuple(p.shape) for n, p in model.named_parameters()}
    buffers = {n: tuple(b.shape) for n, b in model.named_buffers()}

    def _check(weights: dict[str, torch.Tensor], source: str) -> None:
        report = LoadReport(source=source)
        for k, v in weights.items():
            target_shape = params.get(k)
            if target_shape is None:
                target_shape = buffers.get(k)
            if target_shape is None:
                report.missing_in_model.append(k)
                continue
            if target_shape != tuple(v.shape):
                report.shape_mismatch.append((k, target_shape, tuple(v.shape)))
                continue
            report.covered_param_names.add(k)
            report.loaded += 1
        reports.append(report)

    vit_src = _resolve_local_or_hub(vit_path)
    _check(remap_vit(dict(iter_safetensors(vit_src))), "ViT")

    if adapter_path:
        _check(remap_adapter(dict(iter_safetensors(adapter_path))), "Adapter")

    llm_src = _resolve_local_or_hub(llm_path)
    _check(remap_llm(dict(iter_safetensors(llm_src))), "LLM")

    uncovered = assert_full_coverage(model, reports)
    return reports, uncovered
