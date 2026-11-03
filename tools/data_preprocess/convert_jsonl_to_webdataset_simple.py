#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a JSONL file (no packing) to WebDataset for SFT video/image QA.

Expected JSONL fields (per line):
- id: str
- messages: List[{"role": "user|assistant|system", "content": str}]
- images: List[path] (frame images or still images)
- patch_positions: Optional[List[path]] (npy files)
- fps: float|int (required)
- timestamp_decimal: int (optional, 1 or 2, default 1)

This script writes WebDataset shards and a minimal Megatron-Energon config
for MultiMixQASample, with an auto-generated sample_loader.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import webdataset as wds
import yaml
from tqdm import tqdm


try:
    from megatron.energon.epathlib import EPath
    from megatron.energon.flavors import BaseWebdatasetFactory
    from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME

    ENERGON_AVAILABLE = True
except ImportError:
    ENERGON_AVAILABLE = False
    MAIN_FOLDER_NAME = ".nv-meta"

logger = logging.getLogger(__name__)


def sample_loader_template() -> str:
    """Return sample_loader.py content for MultiMixQASample."""
    return """# Auto-generated sample loader for JSONL->WebDataset (no packing)
import io
import numpy as np


def _load_npy(data):
	if data is None:
		return None
	if isinstance(data, np.ndarray):
		return data
	if isinstance(data, bytes):
		return np.load(io.BytesIO(data), allow_pickle=True)
	return np.asarray(data)


def sample_loader(sample: dict) -> dict:
	data = sample['json']

	# Load images
	images = [sample.get(name) for name in data.get('image_keys', [])]

	# Load patch_positions if present
	patch_positions = None
	if 'patch_positions_keys' in data:
		patch_positions = [_load_npy(sample.get(name)) for name in data['patch_positions_keys']]

	# Extract system + messages
	system = None
	messages = []
	for msg in data.get('messages', []):
		if msg.get('role') == 'system':
			system = msg.get('content')
			continue
		messages.append({'role': msg.get('role'), 'content': msg.get('content')})

	result = dict(
		__key__=sample['__key__'],
		__restore_key__=sample['__restore_key__'],
		messages=messages,
		system=system,
		image=images if len(images) > 0 else None,
		fps=data.get('fps'),
		timestamp_decimal=data.get('timestamp_decimal'),
		data_type=data.get('data_type'),
	)
	if patch_positions is not None:
		result['patch_positions'] = patch_positions
	return result


def part_filter(part: str) -> bool:
	return True
"""


def resolve_path(path: str, root_dir: str | None) -> str:
    if os.path.isabs(path) or not root_dir:
        return path
    return os.path.join(root_dir, path)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_config(output_dir: str) -> None:
    """Write minimal Energon config if available."""
    meta_dir = Path(output_dir) / MAIN_FOLDER_NAME
    meta_dir.mkdir(parents=True, exist_ok=True)

    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": "MultiMixQASample",
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader",
    }
    with (meta_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)
    with (meta_dir / "sample_loader.py").open("w", encoding="utf-8") as f:
        f.write(sample_loader_template())

    if ENERGON_AVAILABLE:
        path = EPath(output_dir).absolute()
        all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
        all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
        BaseWebdatasetFactory.prepare_dataset(
            path,
            all_tars,
            split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
            tar_index_only=False,
            workers=96,
        )


def _normalize_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return list(value)


def _build_sample(
    entry: dict, idx: int, image_root: str | None, sample_prefix: str, default_timestamp_decimal: int | None = None
) -> dict:
    """Build one sample."""
    sample_id = entry.get("id") or f"{sample_prefix}{idx}"
    sample_id = sample_id.replace(".", "_")
    messages = entry.get("messages", [])

    images = _normalize_list(entry.get("images") or entry.get("image"))
    patch_positions = _normalize_list(entry.get("patch_positions"))

    sample = {"__key__": str(sample_id)}

    image_keys: list[str] = []
    for img_idx, img_path in enumerate(images):
        full_path = resolve_path(img_path, image_root)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        key = f"img_{img_idx:03d}.jpg"
        with open(full_path, "rb") as f:
            sample[key] = f.read()
        image_keys.append(key)

    patch_positions_keys: list[str] = []
    for pp_idx, pp_path in enumerate(patch_positions):
        if not os.path.exists(pp_path):
            raise FileNotFoundError(f"Patch positions not found: {pp_path}")
        key = f"patch_positions_{pp_idx:03d}.npy"
        with open(pp_path, "rb") as f:
            sample[key] = f.read()
        patch_positions_keys.append(key)

    fps = entry.get("fps")
    if fps is None:
        raise ValueError(f"Missing required field 'fps' in sample: {sample_id}")

    # Per-sample JSONL field takes priority; fall back to CLI default
    timestamp_decimal = entry.get("timestamp_decimal") or default_timestamp_decimal

    payload = {
        "messages": messages,
        "image_keys": image_keys,
        "fps": fps,
    }
    if timestamp_decimal is not None:
        payload["timestamp_decimal"] = timestamp_decimal
    data_type = entry.get("data_type")
    if data_type is not None:
        payload["data_type"] = data_type
    logging.info(payload["fps"])
    if patch_positions_keys:
        payload["patch_positions_keys"] = patch_positions_keys
    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample


_COUNTER = None


def _init_worker(counter) -> None:
    global _COUNTER
    _COUNTER = counter


def _process_chunk(
    chunk_path: str,
    tar_pattern: str,
    maxcount: int,
    maxsize: int,
    image_root: str | None,
    sample_prefix: str,
    worker_id: int,
    default_timestamp_decimal: int | None = None,
) -> tuple[int, int]:
    count = 0
    with wds.ShardWriter(tar_pattern, maxcount=maxcount, maxsize=maxsize, verbose=0) as shard_writer:
        for idx, entry in enumerate(iter_jsonl(chunk_path)):
            sample = _build_sample(entry, idx, image_root, sample_prefix, default_timestamp_decimal)
            shard_writer.write(sample)
            count += 1
            if _COUNTER is not None and count % 50 == 0:
                with _COUNTER.get_lock():
                    _COUNTER.value += 50
        if _COUNTER is not None:
            remainder = count % 50
            if remainder:
                with _COUNTER.get_lock():
                    _COUNTER.value += remainder
    return worker_id, count


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _split_jsonl(jsonl_path: str, num_workers: int, tmp_dir: str) -> tuple[list[str], int]:
    total_lines = _count_lines(jsonl_path)
    if total_lines == 0:
        empty_path = os.path.join(tmp_dir, "chunk-00.jsonl")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        Path(empty_path).write_text("", encoding="utf-8")
        return [empty_path], 0

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    per_chunk = max(1, math.ceil(total_lines / num_workers))
    chunk_paths = [os.path.join(tmp_dir, f"chunk-{i:02d}.jsonl") for i in range(num_workers)]
    writers = [open(p, "w", encoding="utf-8") for p in chunk_paths]
    counts = [0] * num_workers

    current = 0
    written = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if written >= per_chunk and current < num_workers - 1:
                current += 1
                written = 0
            writers[current].write(line)
            written += 1
            counts[current] += 1

    for w in writers:
        w.close()

    return [p for p, c in zip(chunk_paths, counts, strict=False) if c > 0], total_lines


def convert_jsonl_to_wds(
    jsonl_path: str,
    output_dir: str,
    maxcount: int,
    maxsize: int,
    image_root: str | None,
    num_workers: int,
    keep_chunks: bool,
    default_timestamp_decimal: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if num_workers <= 1:
        tar_pattern = os.path.join(output_dir, "instruct_%06d.tar")
        with tqdm(total=_count_lines(jsonl_path)) as pbar:
            counter = mp.Value("i", 0)
            _init_worker(counter)
            _process_chunk(
                jsonl_path,
                tar_pattern,
                maxcount,
                maxsize,
                image_root,
                sample_prefix="sample_",
                worker_id=0,
                default_timestamp_decimal=default_timestamp_decimal,
            )
            with counter.get_lock():
                pbar.n = counter.value
            pbar.refresh()
    else:
        tmp_dir = os.path.join(output_dir, ".tmp_jsonl_chunks")
        chunk_paths, total_lines = _split_jsonl(jsonl_path, num_workers, tmp_dir)
        worker_count = len(chunk_paths)
        ctx = mp.get_context("fork")
        counter = ctx.Value("i", 0)
        with ctx.Pool(processes=worker_count, initializer=_init_worker, initargs=(counter,)) as pool:
            args_list = []
            for worker_id, chunk_path in enumerate(chunk_paths):
                tar_pattern = os.path.join(output_dir, f"instruct_{worker_id:02d}_%06d.tar")
                args_list.append(
                    (
                        chunk_path,
                        tar_pattern,
                        maxcount,
                        maxsize,
                        image_root,
                        f"sample_{worker_id:02d}_",
                        worker_id,
                        default_timestamp_decimal,
                    )
                )
            results_async = pool.starmap_async(_process_chunk, args_list)
            with tqdm(total=total_lines) as pbar:
                while not results_async.ready():
                    with counter.get_lock():
                        pbar.n = counter.value
                    pbar.refresh()
                    time.sleep(0.5)
                results = results_async.get()
                with counter.get_lock():
                    pbar.n = counter.value
                pbar.refresh()

        for worker_id, count in sorted(results):
            print(f"worker-{worker_id:02d}: samples={count}")

        if not keep_chunks:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    write_config(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Input jsonl path")
    parser.add_argument("--output_dir", required=True, help="Output directory for WDS shards")
    parser.add_argument("--maxcount", type=int, default=10000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=int, default=3_000_000_000, help="Max shard size in bytes")
    parser.add_argument("--image_root", default=None, help="Optional root dir for image paths")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for chunked writing",
    )
    parser.add_argument(
        "--keep_chunks",
        action="store_true",
        help="Keep temporary split jsonl chunks",
    )
    parser.add_argument(
        "--timestamp_decimal",
        type=int,
        default=None,
        choices=[1, 2],
        help="Default timestamp decimal places (1 or 2). Per-sample JSONL field overrides this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_jsonl_to_wds(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        image_root=args.image_root,
        num_workers=args.num_workers,
        keep_chunks=args.keep_chunks,
        default_timestamp_decimal=args.timestamp_decimal,
    )


if __name__ == "__main__":
    main()
