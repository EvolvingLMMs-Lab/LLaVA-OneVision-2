#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_split_json_to_samples.py

Split a JSON array or JSONL file into individual sample JSON files,
with automatic image copying and deduplication renaming.

This is the first step (s1) in the offline packing pipeline.

Features:
    - Supports both JSON (array) and JSONL input formats
    - Handles single or multiple images per sample (e.g., video frames)
    - Thread-safe filename deduplication across processes
    - Multi-process + multi-thread parallel processing
    - Automatic skip of samples with missing images

Usage:
    # Basic usage
    python s1_split_json_to_samples.py -i /path/to/data.jsonl -o /path/to/output

    # With image root directory
    python s1_split_json_to_samples.py -i /path/to/data.jsonl \\
        --image-root /path/to/images \\
        -o /path/to/output

    # With relative image path (relative to input JSON directory)
    python s1_split_json_to_samples.py -i /path/to/data.json \\
        --rel-img images/ \\
        -o /path/to/output

    # Overwrite existing output directory
    python s1_split_json_to_samples.py -i /path/to/data.jsonl \\
        -o /path/to/output --overwrite

    # Adjust parallelism
    python s1_split_json_to_samples.py -i /path/to/data.jsonl \\
        --chunk-size 500 -m 4

Author: LLaVA-OneVision Team
License: Apache 2.0
"""

import argparse
import json
import logging
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process, cpu_count
from typing import Any, Optional

# Pillow is only required when --jpeg-quality is enabled; import lazily inside the
# conversion helper so the module remains importable in environments without PIL.
_JPEG_RECODE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _unique_filename(name: str, name_counter: dict[str, int], name_lock) -> str:
    """
    Generate a unique filename by appending a counter suffix if needed.

    Thread-safe implementation using a shared counter and lock.

    Args:
        name: Original filename
        name_counter: Shared dict tracking filename occurrences
        name_lock: Lock for thread-safe access

    Returns:
        Unique filename (original or with _N suffix)
    """
    base, ext = os.path.splitext(name)
    with name_lock:
        count = name_counter.get(name, 0)
        name_counter[name] = count + 1
        if count == 0:
            return name
        return f"{base}_{count}{ext}"


def _resolve_image_path(path: str, base_dir: str, rel_img_path: Optional[str], image_root: Optional[str]) -> str:
    """
    Resolve image path to absolute path.

    Args:
        path: Original image path (absolute or relative)
        base_dir: Directory containing the input JSON file
        rel_img_path: Relative image directory (relative to base_dir)
        image_root: Absolute image root directory

    Returns:
        Resolved absolute path
    """
    if os.path.isabs(path):
        return os.path.normpath(path)
    if rel_img_path:
        return os.path.normpath(os.path.join(base_dir, rel_img_path, path))
    if image_root:
        return os.path.normpath(os.path.join(image_root, path))
    # Fallback: relative to input JSON directory
    return os.path.normpath(os.path.join(base_dir, path))


def _copy_or_recode_image(src_path: str, dst_path: str, jpeg_quality: int) -> str:
    """Copy ``src_path`` to ``dst_path``; when ``jpeg_quality >= 0`` and the source
    is a recodable raster image, re-encode as JPEG and return the actual dst path
    (extension forced to ``.jpg``). Otherwise behaves like ``shutil.copy2``.
    """
    if jpeg_quality < 0:
        shutil.copy2(src_path, dst_path)
        return dst_path

    src_ext = os.path.splitext(src_path)[1].lower()
    if src_ext not in _JPEG_RECODE_EXTS:
        shutil.copy2(src_path, dst_path)
        return dst_path

    from PIL import Image  # noqa: PLC0415 — lazy import; only needed when flag is on

    new_dst = os.path.splitext(dst_path)[0] + ".jpg"
    with Image.open(src_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(new_dst, format="JPEG", quality=int(jpeg_quality), optimize=False)
    return new_dst


def _process_single_item(args: tuple) -> Optional[str]:
    """
    Process a single data item: copy images, patch_position.npy files, and create JSON file.

    Args:
        args: Tuple of (item, base_dir, output_dir, rel_img_path,
              image_root, no_img_indices, name_counter, name_lock)

    Returns:
        Output filename (without extension) or None if skipped
    """
    (item, base_dir, output_dir, rel_img_path, image_root, no_img_indices, name_counter, name_lock, jpeg_quality) = args

    # Extract image paths from item
    original_image_paths: list[str] = []
    if item.get("images"):
        images = item["images"]
        original_image_paths = images if isinstance(images, list) else [images]
    elif item.get("image"):
        image = item["image"]
        original_image_paths = image if isinstance(image, list) else [image]

    # Extract patch_position paths from item if provided
    original_patch_position_paths: list[str] = []
    if item.get("patch_positions"):
        patch_positions = item["patch_positions"]
        original_patch_position_paths = patch_positions if isinstance(patch_positions, list) else [patch_positions]

    # Resolve all image paths
    resolved_paths = [_resolve_image_path(p, base_dir, rel_img_path, image_root) for p in original_image_paths]

    # Resolve patch_position paths if provided
    resolved_pp_paths = [_resolve_image_path(p, base_dir, rel_img_path, image_root) for p in original_patch_position_paths] if original_patch_position_paths else []

    # Pre-check: if any image is missing, skip the entire sample
    # (partial frames would cause <image> tag / patch_positions / timestamp mismatch)
    if resolved_paths:
        missing_images = [p for p in resolved_paths if not os.path.exists(p)]
        if missing_images:
            item_id = item.get("id", item.get("_orig_index", "unknown"))
            for mp in missing_images:
                logger.warning(f"Image not found: {mp} (sample id: {item_id})")
            logger.info(
                f"Skipping sample {item_id}: "
                f"{len(missing_images)}/{len(resolved_paths)} images missing"
            )
            return None

    # Pre-check: if any patch_positions npy is missing, skip the entire sample
    if resolved_pp_paths:
        missing_npys = [p for p in resolved_pp_paths if p and not os.path.exists(p)]
        if missing_npys:
            item_id = item.get("id", item.get("_orig_index", "unknown"))
            for mp in missing_npys:
                logger.warning(f"patch_positions not found: {mp} (sample id: {item_id})")
            logger.info(
                f"Skipping sample {item_id}: "
                f"{len(missing_npys)}/{len(resolved_pp_paths)} patch_positions missing"
            )
            return None

    # Copy images and track new names
    new_image_basenames: list[str] = []
    new_patch_position_basenames: list[str] = []
    for idx, src_path in enumerate(resolved_paths):
        old_name = os.path.basename(src_path)
        new_name = _unique_filename(old_name, name_counter, name_lock)

        dst_path = os.path.join(output_dir, new_name)
        try:
            actual_dst = _copy_or_recode_image(src_path, dst_path, jpeg_quality)
            if actual_dst != dst_path:
                new_name = os.path.basename(actual_dst)
        except Exception as e:
            logger.error(f"Failed to copy image: {src_path} -> {dst_path} | {e}")
        new_image_basenames.append(new_name)

        # Copy corresponding .npy file
        # First, check if explicit path was provided in the input JSON
        if idx < len(resolved_pp_paths) and resolved_pp_paths[idx]:
            npy_src_path = resolved_pp_paths[idx]
        elif not original_patch_position_paths:
            # Only fall back to co-located .npy when NO patch_positions field
            # was provided in the input JSON at all
            npy_src_path = os.path.splitext(src_path)[0] + ".npy"
        else:
            # patch_positions was provided but has fewer entries than images,
            # meaning remaining images don't have individual .npy files
            npy_src_path = None
        
        if npy_src_path is None:
            # patch_positions field exists but doesn't cover this image index; skip silently
            pass
        elif os.path.exists(npy_src_path):
            npy_new_name = os.path.splitext(new_name)[0] + ".npy"
            npy_dst_path = os.path.join(output_dir, npy_new_name)
            try:
                shutil.copy2(npy_src_path, npy_dst_path)
                new_patch_position_basenames.append(npy_new_name)
            except Exception as e:
                logger.error(f"Failed to copy patch_position: {npy_src_path} -> {npy_dst_path} | {e}")
                new_patch_position_basenames.append("")  # Empty string for failed copy
        else:
            item_id = item.get("id", item.get("_orig_index", "unknown"))
            logger.warning(f"patch_positions not found: {npy_src_path} (sample id: {item_id}, image idx: {idx})")
            new_patch_position_basenames.append("")  # Empty string for missing .npy

    # Update images and patch_positions fields in item
    item["images"] = new_image_basenames
    if any(new_patch_position_basenames):  # Only add field if at least one .npy exists
        item["patch_positions"] = new_patch_position_basenames

    # Generate JSON filename
    if new_image_basenames:
        json_name_root = os.path.splitext(new_image_basenames[0])[0]
    else:
        # No images - use special naming
        try:
            idx = no_img_indices.index(item["_orig_index"])
        except (ValueError, KeyError):
            idx = item.get("_orig_index", 0)
        json_name_root = f"__no_image_{idx:08d}"

    json_name = _unique_filename(json_name_root + ".json", name_counter, name_lock)
    json_path = os.path.join(output_dir, json_name)

    # Remove internal index before saving
    item.pop("_orig_index", None)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write JSON: {json_path} | {e}")
        return None

    return os.path.splitext(json_name)[0]


def _worker_process(
    job_queue,
    result_list,
    base_dir: str,
    output_dir: str,
    rel_img_path: Optional[str],
    image_root: Optional[str],
    num_threads: int,
    no_img_indices: list[int],
    name_counter,
    name_lock,
    jpeg_quality: int,
) -> None:
    """
    Worker process that processes chunks of data items.

    Args:
        job_queue: Queue of data chunks to process
        result_list: Shared list to collect results
        base_dir: Directory containing input JSON
        output_dir: Output directory
        rel_img_path: Relative image path
        image_root: Absolute image root
        num_threads: Number of threads per process
        no_img_indices: Indices of items without images
        name_counter: Shared filename counter
        name_lock: Lock for thread safety
    """
    while True:
        try:
            chunk = job_queue.get_nowait()
        except Exception:
            break

        logger.info(f"Process {os.getpid()} processing chunk ({len(chunk)} items)")

        # Build argument list for thread pool
        arg_list = [
            (item, base_dir, output_dir, rel_img_path, image_root, no_img_indices, name_counter, name_lock, jpeg_quality)
            for item in chunk
        ]

        valid_names: list[str] = []
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            results = pool.map(_process_single_item, arg_list)
            for result in tqdm(results, total=len(arg_list), desc=f"PID-{os.getpid()}", leave=False):
                if result is not None:
                    valid_names.append(result)

        result_list.extend(valid_names)


def split_json_to_samples(
    input_path: str,
    rel_img_path: Optional[str] = None,
    *,
    image_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    chunk_size: int = 1000,
    num_threads: int = 8,
    num_processes_override: Optional[int] = None,
    shuffle: bool = True,
    jpeg_quality: int = -1,
) -> set[str]:
    """
    Split a JSON array or JSONL file into individual sample files.

    Args:
        input_path: Path to input JSON or JSONL file
        rel_img_path: Relative image directory (relative to input file)
        image_root: Absolute image root directory
        output_dir: Output directory (default: <input_dir>/split_samples)
        overwrite: Whether to overwrite existing output directory
        chunk_size: Number of items per process chunk
        num_threads: Number of threads per process
        num_processes_override: If set (>0), cap process count to this value
            instead of cpu_count(); useful to reduce NFS pressure
        shuffle: Whether to shuffle data before processing

    Returns:
        Set of generated filenames (without extensions)
    """
    # Load data
    try:
        if input_path.lower().endswith(".jsonl"):
            data: list[dict[str, Any]] = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        if shuffle:
            random.shuffle(data)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return set()

    if not isinstance(data, list):
        logger.error("Input JSON root must be an array")
        return set()

    # Add original indices and collect no-image indices
    for i, item in enumerate(data):
        item["_orig_index"] = i

    no_img_indices = [i for i, item in enumerate(data) if not item.get("images") and not item.get("image")]

    # Prepare directories
    base_dir = os.path.dirname(os.path.abspath(input_path))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "split_samples")
    else:
        output_dir = os.path.abspath(output_dir)

    if os.path.exists(output_dir):
        if overwrite:
            logger.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create chunks
    total = len(data)
    num_chunks = (total + chunk_size - 1) // chunk_size
    chunks = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    if num_processes_override is not None and num_processes_override > 0:
        num_processes = min(num_chunks, num_processes_override)
    else:
        num_processes = min(num_chunks, cpu_count())
    logger.info(
        f"Total: {total} items, {num_chunks} chunks, "
        f"{num_processes} processes, {num_threads} threads each; "
        f"Output: {output_dir}"
    )

    # Process with multiprocessing
    with Manager() as manager:
        job_queue = manager.Queue()
        for chunk in chunks:
            job_queue.put(chunk)

        result_list = manager.list()
        name_counter = manager.dict()
        name_lock = manager.Lock()

        processes = [
            Process(
                target=_worker_process,
                args=(
                    job_queue,
                    result_list,
                    base_dir,
                    output_dir,
                    rel_img_path,
                    image_root,
                    num_threads,
                    no_img_indices,
                    name_counter,
                    name_lock,
                    jpeg_quality,
                ),
            )
            for _ in range(num_processes)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        all_valid_names = set(result_list)

    logger.info("Processing complete")
    return all_valid_names


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Split a JSON array or JSONL file into individual sample files "
            "with automatic image copying and deduplication."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with JSONL input
  python s1_split_json_to_samples.py -i data.jsonl -o output/

  # Specify image root directory
  python s1_split_json_to_samples.py -i data.jsonl --image-root /path/to/images -o output/

  # Use relative image path
  python s1_split_json_to_samples.py -i data.json --rel-img images/ -o output/

  # Overwrite existing output
  python s1_split_json_to_samples.py -i data.jsonl -o output/ --overwrite
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input JSON or JSONL file path")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="Output directory (default: <input_dir>/split_samples)"
    )
    parser.add_argument("--rel-img", default=None, help="Relative image directory (relative to input file)")
    parser.add_argument("--image-root", default=None, help="Absolute image root directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Number of items per process chunk (default: 1000)"
    )
    parser.add_argument("-m", "--num-threads", type=int, default=8, help="Number of threads per process (default: 8)")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle data before processing")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=-1,
        help="If >=0, re-encode raster images as JPEG with the given quality (e.g. 80). "
             "Output extension is forced to .jpg. Default -1 (disabled, byte-copy).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=0,
        help=(
            "Cap number of worker processes (default 0 = use cpu_count(), "
            "the historical behavior). Set to a small integer (e.g. 16) to "
            "reduce NFS pressure when running across many nodes."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Run processing
    result = split_json_to_samples(
        input_path=args.input,
        rel_img_path=args.rel_img,
        image_root=args.image_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
        num_threads=args.num_threads,
        num_processes_override=args.num_processes if args.num_processes > 0 else None,
        shuffle=not args.no_shuffle,
        jpeg_quality=args.jpeg_quality,
    )

    # Print summary
    output_path = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "split_samples")
    print(f"\nGenerated {len(result)} sample files in: {output_path}")


if __name__ == "__main__":
    main()
