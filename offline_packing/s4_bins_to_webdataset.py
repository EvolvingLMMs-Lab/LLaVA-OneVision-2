#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import webdataset as wds
from tqdm import tqdm

from s4_pack_samples import PackSamplesConfig, SamplePacker
from s5_convert_to_webdataset import WebDatasetConfig, WebDatasetConverter


def build_direct_sample(packer: SamplePacker, bin_index: int, bin_data: list[Any]) -> dict[str, Any]:
    sample_key = f"ps_{bin_index:08d}"
    sample: dict[str, Any] = {"__key__": sample_key}

    packed_images: list[list[str]] = []
    packed_prompts: list[list[str]] = []
    packed_captions: list[list[str]] = []
    packed_patch_positions: list[list[str]] = []
    packed_fps: list[Any] = []
    packed_timestamp_decimals: list[int] = []

    sample_names = [str(item["name"]) for item in bin_data]
    for sample_idx, raw_sample_name in enumerate(sample_names):
        sample_source_dir, sample_name = packer._find_sample_source_dir(raw_sample_name)
        if sample_source_dir is None:
            raise FileNotFoundError(f"Sample {raw_sample_name} not found in source dirs")

        json_path = os.path.join(sample_source_dir, f"{sample_name}.{packer.config.json_extension}")
        images, prompts, responses, patch_positions, fps, timestamp_decimal = packer._extract_bmr_content(
            json_path,
            image_base_dir=sample_source_dir,
        )

        sample_image_names: list[str] = []
        sample_patch_position_names: list[str] = []

        for img_idx, img_src in enumerate(images):
            if not img_src:
                continue

            img_payload_name = f"img{sample_idx:03d}_sub{img_idx:03d}.jpg"
            sample_image_names.append(img_payload_name)

            with open(img_src, "rb") as f:
                sample[f"img{sample_idx}_{img_idx}.jpg"] = f.read()

            pp_payload_name = ""
            if img_idx < len(patch_positions):
                pp_src = patch_positions[img_idx]
                if pp_src:
                    with open(pp_src, "rb") as f:
                        sample[f"img{sample_idx}_{img_idx}.npy"] = f.read()
                    pp_payload_name = f"img{sample_idx:03d}_sub{img_idx:03d}.npy"

            sample_patch_position_names.append(pp_payload_name)

        packed_images.append(sample_image_names)
        packed_patch_positions.append(sample_patch_position_names)
        packed_prompts.append(prompts if prompts else [""])
        packed_captions.append(responses if responses else [""])
        packed_fps.append(fps)
        packed_timestamp_decimals.append(timestamp_decimal)

    payload: dict[str, Any] = {
        "images": packed_images,
        "prompts": packed_prompts,
        "captions": packed_captions,
        "sample_count": len(packed_images),
        "patch_positions": packed_patch_positions,
    }
    if any(fps is not None for fps in packed_fps):
        payload["fps"] = packed_fps
    payload["timestamp_decimal"] = packed_timestamp_decimals

    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample


def process_chunk(
    chunk_id: int,
    bins_chunk: list[tuple[int, list[Any]]],
    config: PackSamplesConfig,
    output_dir: str,
    shard_prefix: str,
    max_samples_per_shard: int,
    max_shard_size: int,
) -> tuple[int, list[str]]:
    packer = SamplePacker(config)
    shard_pattern = os.path.join(output_dir, f"{shard_prefix}-{chunk_id:03d}-%03d.tar")
    processed = 0

    skipped = 0
    with wds.ShardWriter(
        shard_pattern,
        maxcount=max_samples_per_shard,
        maxsize=max_shard_size,
    ) as sink:
        for bin_index, bin_data in bins_chunk:
            try:
                sample = build_direct_sample(packer, bin_index, bin_data)
            except (FileNotFoundError, OSError) as exc:
                skipped += 1
                print(
                    f"# [chunk {chunk_id}] SKIP bin {bin_index}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            sink.write(sample)
            processed += 1

    if skipped:
        print(f"# [chunk {chunk_id}] skipped {skipped} bins out of {len(bins_chunk)}", flush=True)

    tar_files = sorted(str(path) for path in Path(output_dir).glob(f"{shard_prefix}-{chunk_id:03d}-*.tar"))
    return processed, tar_files


def build_chunks(indexed_bins: list[tuple[int, list[Any]]], chunk_count: int) -> list[list[tuple[int, list[Any]]]]:
    chunk_size = (len(indexed_bins) + chunk_count - 1) // chunk_count
    return [indexed_bins[i:i + chunk_size] for i in range(0, len(indexed_bins), chunk_size)]


def parse_chunk_ids(raw_value: str, max_chunk_id: int) -> list[int]:
    chunk_ids: set[int] = set()
    for part in raw_value.split(","):
        value = part.strip()
        if not value:
            continue
        if "-" in value:
            start_str, end_str = value.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            chunk_ids.update(range(start, end + 1))
        else:
            chunk_ids.add(int(value))
    return sorted(chunk_id for chunk_id in chunk_ids if 0 <= chunk_id <= max_chunk_id)


def count_samples_in_tar(tar_path: Path) -> int:
    try:
        with tarfile.open(tar_path, "r") as tf:
            return sum(1 for member in tf.getmembers() if member.isfile() and member.name.endswith(".json"))
    except tarfile.TarError:
        return 0


def get_chunk_progress(output_dir: Path, shard_prefix: str, chunk_count: int) -> dict[int, int]:
    progress = {chunk_id: 0 for chunk_id in range(chunk_count)}
    for chunk_id in range(chunk_count):
        for tar_path in output_dir.glob(f"{shard_prefix}-{chunk_id:03d}-*.tar"):
            progress[chunk_id] += count_samples_in_tar(tar_path)
    return progress


def clear_chunk_outputs(output_dir: Path, shard_prefix: str, chunk_id: int) -> None:
    for path in output_dir.glob(f"{shard_prefix}-{chunk_id:03d}-*.tar"):
        path.unlink()
    for path in output_dir.glob(f"{shard_prefix}-{chunk_id:03d}-*.tar.idx"):
        path.unlink()


def finalize_output(output_dir: Path, shard_prefix: str, sample_class_name: str, workers: int) -> None:
    created_tars = sorted(output_dir.glob(f"{shard_prefix}-*.tar"))
    chunk_tars = [path for path in created_tars if path.stem.count("-") == 2]
    if not chunk_tars:
        return

    for final_path in output_dir.glob(f"{shard_prefix}-[0-9][0-9][0-9][0-9][0-9][0-9].tar"):
        final_path.unlink()
    for final_idx in output_dir.glob(f"{shard_prefix}-[0-9][0-9][0-9][0-9][0-9][0-9].tar.idx"):
        final_idx.unlink()

    for new_idx, old_path in enumerate(chunk_tars):
        new_path = output_dir / f"{shard_prefix}-{new_idx:06d}.tar"
        if old_path != new_path:
            shutil.move(str(old_path), str(new_path))
        idx_path = old_path.with_suffix(old_path.suffix + ".idx")
        if idx_path.exists():
            idx_new_path = output_dir / f"{shard_prefix}-{new_idx:06d}.tar.idx"
            if idx_path != idx_new_path:
                shutil.move(str(idx_path), str(idx_new_path))

    wds_config = WebDatasetConfig(
        input_dir=str(output_dir),
        output_dir=str(output_dir),
        sample_class_name=sample_class_name,
        workers=workers,
    )
    WebDatasetConverter(wds_config).write_energon_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert bins directly to WebDataset")
    parser.add_argument("--bins-file", required=True)
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--source-dirs", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-count", type=int, default=0)
    parser.add_argument("--max-samples-per-shard", type=int, default=10000)
    parser.add_argument("--max-shard-size", type=int, default=3_000_000_000)
    parser.add_argument("--shard-prefix", default="pretrain")
    parser.add_argument("--sample-class-name", default="PackedCaptioningSample")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume-chunks", default="")
    parser.add_argument("--finalize-only", action="store_true", default=False)
    parser.add_argument("--skip-finalize", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dirs = [item.strip() for item in args.source_dirs.split(",") if item.strip()]
    config = PackSamplesConfig(
        bins_file=args.bins_file,
        source_dir=args.source_dir,
        source_dirs=source_dirs,
        output_dir=args.output_dir,
        task_type="bmr",
        clear_output_dir=False,
        workers=args.workers,
    )
    config.validate()

    output_path = Path(args.output_dir)
    chunk_count = args.chunk_count or args.workers

    with open(args.bins_file, "rb") as f:
        bins = pickle.load(f)

    indexed_bins = [(idx, bins[idx]) for idx in range(len(bins))]
    chunks = build_chunks(indexed_bins, chunk_count)
    max_chunk_id = len(chunks) - 1

    if args.finalize_only:
        finalize_output(output_path, args.shard_prefix, args.sample_class_name, args.workers)
        print(f"output_dir={args.output_dir}")
        return

    if output_path.exists() and not args.resume:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_chunk_ids = parse_chunk_ids(args.resume_chunks, max_chunk_id) if args.resume_chunks else list(range(len(chunks)))
    if args.resume and not args.resume_chunks:
        progress = get_chunk_progress(output_path, args.shard_prefix, len(chunks))
        selected_chunk_ids = []
        for chunk_id, chunk in enumerate(chunks):
            if progress.get(chunk_id, 0) < len(chunk):
                selected_chunk_ids.append(chunk_id)

    worker_count = max(1, min(args.workers, len(selected_chunk_ids)))
    if worker_count == 0:
        finalize_output(output_path, args.shard_prefix, args.sample_class_name, args.workers)
        print("processed_bins=0")
        print(f"output_dir={args.output_dir}")
        return

    for chunk_id in selected_chunk_ids:
        if args.resume:
            clear_chunk_outputs(output_path, args.shard_prefix, chunk_id)

    processed_total = 0
    created_tars: list[str] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_chunk,
                chunk_id,
                chunks[chunk_id],
                config,
                args.output_dir,
                args.shard_prefix,
                args.max_samples_per_shard,
                args.max_shard_size,
            ): chunk_id
            for chunk_id in selected_chunk_ids
        }

        with tqdm(total=len(futures), desc="Direct WDS chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                processed, tar_files = future.result()
                processed_total += processed
                created_tars.extend(tar_files)
                pbar.set_postfix(processed=processed_total, tars=len(created_tars))
                pbar.update(1)

    if not args.skip_finalize:
        finalize_output(output_path, args.shard_prefix, args.sample_class_name, args.workers)

    print(f"processed_bins={processed_total}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
