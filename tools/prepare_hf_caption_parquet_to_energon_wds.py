#!/usr/bin/env python3
"""Prepare HF image-caption parquet data into the Energon WebDataset format.

This is intentionally small-slice friendly.  Use it for one Mid-Training-85M
folder at a time, e.g. ImageNet-EN part00, instead of cloning/downloading the
whole Hugging Face dataset.  It can either stream from Hugging Face or read an
already-downloaded local HF-style directory.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "aiak_megatron"))

from megatron.energon.epathlib import EPath  # noqa: E402
from megatron.energon.flavors import BaseWebdatasetFactory  # noqa: E402
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME  # noqa: E402


def _image_to_jpeg_bytes(image: Any) -> bytes:
    if image is None:
        raise ValueError("sample has no image")
    if isinstance(image, dict) and image.get("bytes") is not None:
        return image["bytes"]
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)

    # HF Image features usually decode to PIL.Image in non-streaming and
    # streaming modes.  Normalize to RGB JPEG so the Energon loader can use PIL.
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    raise TypeError(f"Unsupported image value: {type(image)!r}")


def _safe_key(raw_id: Any, index: int) -> str:
    text = str(raw_id or f"sample_{index:09d}")
    text = text.replace("/", "_").replace(".", "_")
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text)


def _sample_loader_template() -> str:
    return "\n".join(
        [
            "def sample_loader(sample: dict) -> dict:",
            "    data = sample['json']",
            "    images = [sample.get(f'img{i}.jpg') for i in range(len(data['images']))]",
            "    captions = data['captions']",
            "    prompts = data['prompts']",
            "    return dict(",
            "        __key__=sample['__key__'],",
            "        __restore_key__=sample['__restore_key__'],",
            "        captions=captions,",
            "        prompts=prompts,",
            "        images=images,",
            "    )",
            "def part_filter(part: str) -> bool:",
            "    return True",
            "",
        ]
    )


def _write_metadata(output_dir: Path, tar_names: list[str], workers: int) -> None:
    meta_dir = output_dir / MAIN_FOLDER_NAME
    meta_dir.mkdir(parents=True, exist_ok=True)

    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": "PackedCaptioningSample",
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader",
    }
    (meta_dir / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_definition, sort_keys=False),
        encoding="utf-8",
    )
    (meta_dir / "sample_loader.py").write_text(_sample_loader_template(), encoding="utf-8")

    BaseWebdatasetFactory.prepare_dataset(
        EPath(output_dir).absolute(),
        tar_names,
        split_parts_ratio=[("train", 1.0), ("val", 0.0), ("test", 0.0)],
        tar_index_only=False,
        workers=workers,
    )


def _resolve_local_files(local_data_root: Path, patterns: list[str]) -> list[str]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(p for p in local_data_root.glob(pattern) if p.is_file())
        if not matches:
            raise RuntimeError(f"No local files matched {local_data_root / pattern}")
        files.extend(matches)
    return [str(p) for p in files]


def _iter_hf_samples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    if args.local_data_root is not None:
        data_files = {"train": _resolve_local_files(args.local_data_root, args.data_files)}
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=True,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {"train": args.data_files}
        dataset = load_dataset(
            args.repo_id,
            data_files=data_files,
            split="train",
            streaming=True,
            cache_dir=args.cache_dir,
        )

    if args.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer_size,
        )
    return dataset


def convert(args: argparse.Namespace) -> None:
    import webdataset as wds
    from tqdm import tqdm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.output_dir).free / (1024**3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"{args.output_dir} has only {free_gb:.1f} GB free; "
            f"need at least {args.min_free_gb:.1f} GB before writing shards"
        )

    pattern = str(args.output_dir / f"{args.shard_prefix}-%06d.tar")
    written = 0

    with wds.ShardWriter(pattern, maxcount=args.maxcount, maxsize=args.maxsize) as sink:
        for index, row in enumerate(tqdm(_iter_hf_samples(args), desc="streaming HF rows")):
            if args.max_samples and written >= args.max_samples:
                break

            caption = row.get(args.caption_column)
            image = row.get(args.image_column)
            if not caption or image is None:
                continue

            key = _safe_key(row.get(args.id_column), index)
            payload = {
                "images": [f"{key}.img0.jpg"],
                "captions": [str(caption)],
                "prompts": [args.prompt],
            }
            try:
                image_bytes = _image_to_jpeg_bytes(image)
            except Exception as exc:
                print(f"[skip] {key}: {exc}", file=sys.stderr)
                continue

            sink.write(
                {
                    "__key__": key,
                    "img0.jpg": image_bytes,
                    "json": json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                }
            )
            written += 1

    tar_names = sorted(p.name for p in args.output_dir.glob(f"{args.shard_prefix}-*.tar"))
    if not tar_names:
        raise RuntimeError(f"No shards were written to {args.output_dir}")
    _write_metadata(args.output_dir, tar_names, args.index_workers)
    print(f"Wrote {written} samples to {args.output_dir}")
    print("Use as DATA_PATH for Stage 1.5 midtraining.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=["imagenet/EN/part00/*.parquet"],
        help="Remote parquet files/globs inside the HF dataset repo.",
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=None,
        help="Optional local root that mirrors the HF repo layout, e.g. data/LLaVA-OneVision-Mid-Training-EN.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/midtraining_imagenet_en_part00_webdataset"),
    )
    parser.add_argument("--cache-dir", default="data/hf_cache_midtraining_stream")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--maxcount", type=int, default=1000)
    parser.add_argument("--maxsize", type=int, default=2_000_000_000)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--shard-prefix", default="imagenet-en")
    parser.add_argument("--prompt", default="<image>\nDescribe the image in detail.")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--index-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
