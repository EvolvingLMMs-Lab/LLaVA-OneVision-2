import argparse
import os
import sys

import torch

from transformers import logging

from .loader import _resolve_local_or_hub, load_all_weights
from .save import save_merged
from .utils import log_stage
from .variants import get_variant


logger = logging.get_logger(__name__)


_VALIDATORS = ("vit", "llm", "e2e")
_VIT_STRATEGIES = ("blockorder", "layerwise")
_LLM_STRATEGIES = ("parallel", "sequential")


def _add_variant_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--variant", required=True, choices=("dense", "moe"))


def _add_validation_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--qwen-processor", dest="qwen_processor_path", default=None)
    p.add_argument("--img", dest="img_path", default=None)
    p.add_argument("--sample-text", dest="sample_text", default=None)
    p.add_argument("--validate-skip", dest="validate_skip", action="append", default=[], choices=_VALIDATORS)
    p.add_argument("--vit-validator-strategy", dest="vit_strategy", default=None, choices=_VIT_STRATEGIES)
    p.add_argument("--llm-validator-strategy", dest="llm_strategy", default=None, choices=_LLM_STRATEGIES)


def _resolve_strategies(args: argparse.Namespace) -> None:
    if args.vit_strategy is None:
        args.vit_strategy = "layerwise" if args.variant == "moe" else "blockorder"
    if args.llm_strategy is None:
        args.llm_strategy = "sequential" if args.variant == "moe" else "parallel"
    # Validators reload from this path internally, so sharing --processor is safe:
    # each from_pretrained call materializes an independent instance the validator
    # mutates locally (do_resize=False, max_pixels override, ...) without touching
    # the production processor that ships with the saved checkpoint.
    if getattr(args, "qwen_processor_path", None) is None and getattr(args, "processor_path", None):
        args.qwen_processor_path = args.processor_path


# Path attrs that may be HF Hub repo IDs; resolved at CLI entry so downstream
# code stays Hub-agnostic. Add new path attrs here when adding new subcommands.
_HUB_RESOLVABLE_PATH_ATTRS: tuple[str, ...] = (
    "vit_path",
    "llm_path",
    "processor_path",
    "qwen_processor_path",
    "ckpt_path",
    "adapter_path",
)

# Subset of the above whose Hub source is consumed only by AutoProcessor /
# AutoTokenizer / CLIPImageProcessor.from_pretrained — no model weights needed.
# Routed to _resolve_local_or_hub(kind="processor") to skip multi-GB safetensors.
_HUB_PROCESSOR_PATH_ATTRS: frozenset[str] = frozenset({"processor_path", "qwen_processor_path"})


def _resolve_paths(args: argparse.Namespace) -> None:
    for attr in _HUB_RESOLVABLE_PATH_ATTRS:
        val = getattr(args, attr, None)
        if val and not os.path.exists(val):
            kind = "processor" if attr in _HUB_PROCESSOR_PATH_ATTRS else "model"
            setattr(args, attr, _resolve_local_or_hub(val, kind=kind))


def _reject_unsafe_combos(args: argparse.Namespace) -> None:
    if args.variant == "moe" and args.llm_strategy == "parallel":
        sys.stderr.write(
            "--llm-validator-strategy=parallel is not supported with --variant=moe "
            "(would OOM on large MoE LLMs); use 'sequential' instead\n"
        )
        raise SystemExit(2)


def _enforce_validation_required(args: argparse.Namespace, *, need_e2e: bool) -> None:
    skip = set(args.validate_skip)
    missing: list[str] = []
    if "vit" not in skip and not args.qwen_processor_path:
        missing.append("--qwen-processor")
    need_img = ("vit" not in skip) or (need_e2e and "e2e" not in skip)
    if need_img and not args.img_path:
        missing.append("--img")
    if "llm" not in skip and not args.sample_text:
        missing.append("--sample-text")
    if missing:
        sys.stderr.write(f"missing required validation flag(s): {', '.join(missing)}\n")
        raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merge_ov2", description="Merge ViT + Adapter + LLM into LlavaOnevision2.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="remap + load + (validate) + save")
    _add_variant_flag(m)
    m.add_argument("--vit", dest="vit_path", required=True)
    m.add_argument("--adapter", dest="adapter_path", default="")
    m.add_argument("--llm", dest="llm_path", required=True)
    m.add_argument("--processor", dest="processor_path", required=True)
    m.add_argument("--out", dest="output_path", required=True)
    m.add_argument("--spatial-merge-size", dest="spatial_merge_size", type=int, default=2, choices=[1, 2, 3])
    m.add_argument("--target-dtype", dest="target_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    m.add_argument("--device", type=int, default=0)
    _add_validation_flags(m)

    v = sub.add_parser("validate", help="validate an already-merged checkpoint")
    _add_variant_flag(v)
    v.add_argument("--ckpt", dest="ckpt_path", required=True)
    v.add_argument("--vit", dest="vit_path", required=True)
    v.add_argument("--llm", dest="llm_path", required=True)
    v.add_argument("--processor", dest="processor_path", required=True)
    v.add_argument("--device", type=int, default=0)
    _add_validation_flags(v)

    d = sub.add_parser("dry-run", help="remap only; report load coverage; no save")
    _add_variant_flag(d)
    d.add_argument("--vit", dest="vit_path", required=True)
    d.add_argument("--adapter", dest="adapter_path", default="")
    d.add_argument("--llm", dest="llm_path", required=True)
    d.add_argument("--processor", dest="processor_path", required=True)
    d.add_argument("--spatial-merge-size", dest="spatial_merge_size", type=int, default=2, choices=[1, 2, 3])
    return parser


_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _cmd_merge(args: argparse.Namespace) -> int:
    _resolve_strategies(args)
    _resolve_paths(args)
    _reject_unsafe_combos(args)
    _enforce_validation_required(args, need_e2e=True)

    variant = get_variant(args.variant)
    device = torch.device(f"cuda:{args.device}")
    target_dtype = _DTYPE[args.target_dtype]

    with log_stage("build_empty"):
        model, processor, tokenizer = variant.build_empty(
            args.llm_path,
            args.processor_path,
            spatial_merge_size=args.spatial_merge_size,
            target_dtype=target_dtype,
            vit_path=args.vit_path,
        )

    with log_stage("load_all_weights"):
        load_all_weights(
            model,
            args.vit_path,
            args.adapter_path,
            args.llm_path,
            target_dtype=target_dtype,
        )

    skip = set(args.validate_skip)
    if "vit" not in skip:
        from .validators import vit_blockorder, vit_layerwise

        runner = vit_layerwise.run if args.vit_strategy == "layerwise" else vit_blockorder.run
        with log_stage(f"VALIDATE: vit ({args.vit_strategy})"):
            runner(model, args.vit_path, args.qwen_processor_path, args.img_path, device)
    if "llm" not in skip:
        from .validators import llm_parallel, llm_sequential

        runner = llm_sequential.run if args.llm_strategy == "sequential" else llm_parallel.run
        with log_stage(f"VALIDATE: llm ({args.llm_strategy})"):
            runner(model, args.llm_path, args.sample_text)
    if "e2e" not in skip:
        from .validators import e2e

        with log_stage("VALIDATE: e2e"):
            e2e.run(model, processor, tokenizer, args.img_path, device)

    with log_stage("save"):
        save_merged(model, args.output_path, tokenizer, processor, variant=args.variant)
    logger.info("Done.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    _resolve_strategies(args)
    _resolve_paths(args)
    _reject_unsafe_combos(args)
    _enforce_validation_required(args, need_e2e=True)

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    variant = get_variant(args.variant)
    device = torch.device(f"cuda:{args.device}")
    _ = variant  # variant chosen via CLI; modeling files travel with the saved ckpt

    with log_stage("load_ckpt"):
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(args.processor_path, use_fast=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.processor_path, use_fast=True, trust_remote_code=True)

    skip = set(args.validate_skip)
    if "vit" not in skip:
        from .validators import vit_blockorder, vit_layerwise

        runner = vit_layerwise.run if args.vit_strategy == "layerwise" else vit_blockorder.run
        with log_stage(f"VALIDATE: vit ({args.vit_strategy})"):
            runner(model, args.vit_path, args.qwen_processor_path, args.img_path, device)
    if "llm" not in skip:
        from .validators import llm_parallel, llm_sequential

        runner = llm_sequential.run if args.llm_strategy == "sequential" else llm_parallel.run
        with log_stage(f"VALIDATE: llm ({args.llm_strategy})"):
            runner(model, args.llm_path, args.sample_text)
    if "e2e" not in skip:
        from .validators import e2e

        with log_stage("VALIDATE: e2e"):
            e2e.run(model, processor, tokenizer, args.img_path, device)
    logger.info("Validation done.")
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    from .loader import dry_run_report

    _resolve_paths(args)
    variant = get_variant(args.variant)
    with log_stage("build_empty"):
        model, _processor, _tokenizer = variant.build_empty(
            args.llm_path,
            args.processor_path,
            spatial_merge_size=args.spatial_merge_size,
            vit_path=args.vit_path,
        )

    with log_stage("dry_run"):
        reports, uncovered = dry_run_report(
            model,
            args.vit_path,
            args.adapter_path,
            args.llm_path,
        )

    total_missing = sum(len(r.missing_in_model) for r in reports)
    total_shape = sum(len(r.shape_mismatch) for r in reports)
    print(f"missing_in_model: {total_missing}")
    print(f"shape_mismatch: {total_shape}")
    print(f"uncovered_model_params: {len(uncovered)}")
    for r in reports:
        print(r.render())
    if total_missing or total_shape or uncovered:
        for r in reports:
            for k in r.missing_in_model:
                print(f"  missing  [{r.source}] {k}")
            for k, m, w in r.shape_mismatch:
                print(f"  shape    [{r.source}] {k}: model={m} weights={w}")
        for k in uncovered:
            print(f"  uncovered {k}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "merge":
        return _cmd_merge(args)
    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "dry-run":
        return _cmd_dry_run(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2
