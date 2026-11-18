#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Packed Samples to WebDataset Format.

This script is Step 5 of the offline packing pipeline. It converts the packed
samples from Step 4 into WebDataset (WDS) format for efficient distributed
training with frameworks like Megatron-Energon.

WebDataset is a format optimized for large-scale distributed training that:
- Stores samples in tar archives for sequential I/O
- Supports streaming and shuffling across shards
- Minimizes filesystem overhead with large files

Pipeline Overview:
    Step 1: s1_split_json_to_samples.py    - Split source JSON/JSONL files
    Step 2: s2_compute_token_lengths.py    - Calculate token lengths
    Step 3: s3_bin_packing.py              - Pack samples into bins
    Step 4: s4_pack_samples.py             - Generate final packed data
    Step 5: s5_convert_to_webdataset.py    - Convert to WebDataset (this script)

Input Directory Structure:
    packed_samples/
    ├── ps_00000000.img000_sub000.jpg    # Packed sample 0, sample 0, image 0
    ├── ps_00000000.img000_sub001.jpg    # Packed sample 0, sample 0, image 1
    ├── ps_00000000.img001_sub000.jpg    # Packed sample 0, sample 1, image 0
    ├── ps_00000000.json                  # Packed sample 0 metadata
    ├── ps_00000001.json
    ...

Input JSON Format (multi-sample packed):
    {
        "images": [
            ["img000_sub000.jpg", "img000_sub001.jpg"],  # Sample 0 images
            ["img001_sub000.jpg"]                         # Sample 1 images
        ],
        "prompts": [
            ["Question 1 turn 1", "Question 1 turn 2"],   # Sample 0 prompts
            ["Question 2 turn 1"]                          # Sample 1 prompts
        ],
        "captions": [
            ["Answer 1 turn 1", "Answer 1 turn 2"],       # Sample 0 responses
            ["Answer 2 turn 1"]                            # Sample 1 responses
        ],
        "timestamp_decimal": [1, 2]                        # Optional per-sample timestamp precision
    }

Output Format:
    WebDataset tar files with Megatron-Energon configuration:
    output/
    ├── pretrain-000000.tar
    ├── pretrain-000001.tar
    ├── .nv-meta/
    │   ├── dataset.yaml
    │   ├── sample_loader.py
    │   └── split.yaml

Usage:
    python s5_convert_to_webdataset.py --config s5_config.yaml

    # Or with command line arguments:
    python s5_convert_to_webdataset.py \\
        --input-dir /path/to/packed_samples \\
        --output-dir /path/to/webdataset \\
        --mode bmr_pack \\
        --max-samples-per-shard 10000 \\
        --max-shard-size 3000000000

Author: LLaVA-OneVision Team
License: Apache-2.0
"""

import argparse
import json
import os
import shutil
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import webdataset as wds
import yaml
from tqdm import tqdm


# Megatron-Energon imports for configuration generation
try:
    from megatron.energon.epathlib import EPath
    from megatron.energon.flavors import BaseWebdatasetFactory
    from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME

    ENERGON_AVAILABLE = True
except ImportError:
    ENERGON_AVAILABLE = False
    MAIN_FOLDER_NAME = ".nv-meta"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class WebDatasetConfig:
    """Configuration for WebDataset conversion."""

    # Input/Output paths
    input_dir: str = ""
    output_dir: str = ""

    # Conversion mode
    mode: str = "bmr_pack"  # caption_pack or bmr_pack

    # Shard settings
    max_samples_per_shard: int = 10000
    max_shard_size: int = 3_000_000_000  # 3GB

    # Media type
    media_type: str = "image"  # image, video, or mix

    # Output settings
    shard_prefix: str = "pretrain"
    sample_class_name: str = "PackedCaptioningSample"

    # Processing options
    workers: int = 32
    validate_samples: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "WebDatasetConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WebDatasetConfig":
        """Create configuration from command line arguments."""
        return cls(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            max_samples_per_shard=args.max_samples_per_shard,
            max_shard_size=args.max_shard_size,
            media_type=args.media_type,
            shard_prefix=args.shard_prefix,
            sample_class_name=args.sample_class_name,
            workers=args.workers,
            validate_samples=args.validate_samples,
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.input_dir:
            raise ValueError("input_dir is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.mode not in ("caption_pack", "bmr_pack"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be caption_pack or bmr_pack")
        if not os.path.isdir(self.input_dir):
            raise NotADirectoryError(f"Input directory not found: {self.input_dir}")


# =============================================================================
# Sample Loader Template
# =============================================================================


def generate_sample_loader_code() -> str:
    """
    Generate the sample_loader.py code for Megatron-Energon.

    This loader handles the nested multi-image structure used in packed samples.

    Returns:
        Python code string for sample_loader.py
    """
    return """# Auto-generated sample loader for packed multi-image samples
# This file is used by Megatron-Energon to parse WebDataset samples

import io
import numpy as np


def _load_npy(data) -> np.ndarray:
    \"\"\"
    Load numpy array from data.
    
    Handles both cases:
    - bytes: raw .npy file content (needs np.load)
    - np.ndarray: already decoded by WebDataset's decode() pipeline
    
    Returns None if data is None.
    \"\"\"
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        # Already decoded by WebDataset's automatic decoder
        return data
    if isinstance(data, bytes):
        return np.load(io.BytesIO(data), allow_pickle=True)
    # Unknown type, try to convert
    return np.asarray(data)


def sample_loader(sample: dict) -> dict:
    \"\"\"
    Load and parse a packed sample from WebDataset.

    Args:
        sample: Raw sample dict from WebDataset containing:
            - 'json': Encoded JSON with metadata
            - 'img{sample_idx}_{img_idx}.jpg': Image binary data
            - 'img{sample_idx}_{img_idx}.npy': Patch position data (optional)

    Returns:
        Parsed sample with decoded images, prompts, captions, and patch_positions.
    \"\"\"
    data = sample['json']

    # Dynamically load images based on the nested structure
    # images[i] = list of images for sample i
    # images[i][j] = j-th image of sample i
    images = [
        [sample[f'img{i}_{j}.jpg'] for j in range(len(data['images'][i]))]
        for i in range(len(data['images']))
    ]

    # Load patch_positions if available (decode .npy bytes to numpy arrays)
    patch_positions = None
    if 'patch_positions' in data:
        patch_positions = [
            [_load_npy(sample.get(f'img{i}_{j}.npy')) for j in range(len(data['patch_positions'][i]))]
            for i in range(len(data['patch_positions']))
        ]

    result = dict(
        __key__=sample['__key__'],
        __restore_key__=sample['__restore_key__'],
        images=images,
        prompts=data['prompts'],
        captions=data['captions'],
    )

    if patch_positions is not None:
        result['patch_positions'] = patch_positions

    if 'fps' in data:
        result['fps'] = data['fps']

    if 'timestamp_decimal' in data:
        result['timestamp_decimal'] = data['timestamp_decimal']

    if 'data_type' in data:
        result['data_type'] = data['data_type']

    return result


def part_filter(part: str) -> bool:
    \"\"\"Filter function for dataset parts. Returns True to include all parts.\"\"\"
    return True
"""


# =============================================================================
# WebDataset Converter
# =============================================================================


def _process_single_sample(json_path: Path, input_dir: str, validate: bool) -> dict[str, Any] | None:
    """
    Process a single sample for multiprocessing.

    This is a standalone function (not a method) to be picklable for multiprocessing.

    Args:
        json_path: Path to the JSON file.
        input_dir: Input directory path.
        validate: Whether to validate samples.

    Returns:
        WebDataset sample dict or None if failed.
    """
    sample_id = json_path.stem

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        prompts = data.get("prompts", [])
        captions = data.get("captions", [])
        patch_positions = data.get("patch_positions", [])
        fps = data.get("fps", None)
        timestamp_decimal = data.get("timestamp_decimal", None)
        data_type = data.get("data_type", None)

        # Determine sample count
        sample_count = 1
        if images and isinstance(images[0], list):
            sample_count = len(images)

        if timestamp_decimal is None:
            timestamp_decimal = [1] * sample_count

        # Validate if required
        if validate:
            if sample_count > 1:
                if (
                    len(images) != sample_count
                    or len(prompts) != sample_count
                    or len(captions) != sample_count
                    or (timestamp_decimal is not None and len(timestamp_decimal) != sample_count)
                ):
                    return None
            else:
                if images and isinstance(images[0], list):
                    if len(prompts) != 1 or len(captions) != 1:
                        return None

        # Build WebDataset sample
        sample = {"__key__": sample_id}

        if sample_count > 1:
            # Multi-sample: images is nested list
            for sample_idx, sample_images in enumerate(images):
                if not sample_images:
                    continue
                for img_idx, img_name in enumerate(sample_images):
                    img_path = os.path.join(input_dir, f"{sample_id}.{img_name}")
                    if os.path.exists(img_path):
                        img_key = f"img{sample_idx}_{img_idx}.jpg"
                        with open(img_path, "rb") as f:
                            sample[img_key] = f.read()
                    
                    # Load corresponding .npy file if exists
                    if sample_idx < len(patch_positions) and img_idx < len(patch_positions[sample_idx]):
                        pp_name = patch_positions[sample_idx][img_idx]
                        if pp_name:
                            pp_path = os.path.join(input_dir, f"{sample_id}.{pp_name}")
                            if os.path.exists(pp_path):
                                pp_key = f"img{sample_idx}_{img_idx}.npy"
                                with open(pp_path, "rb") as f:
                                    sample[pp_key] = f.read()
        else:
            # Single sample
            if images and isinstance(images[0], list):
                sample_images = images[0]
                for img_idx, img_name in enumerate(sample_images):
                    img_path = os.path.join(input_dir, f"{sample_id}.{img_name}")
                    if os.path.exists(img_path):
                        img_key = f"img0_{img_idx}.jpg"
                        with open(img_path, "rb") as f:
                            sample[img_key] = f.read()
                    
                    # Load corresponding .npy file if exists
                    if patch_positions and len(patch_positions) > 0 and img_idx < len(patch_positions[0]):
                        pp_name = patch_positions[0][img_idx]
                        if pp_name:
                            pp_path = os.path.join(input_dir, f"{sample_id}.{pp_name}")
                            if os.path.exists(pp_path):
                                pp_key = f"img0_{img_idx}.npy"
                                with open(pp_path, "rb") as f:
                                    sample[pp_key] = f.read()
            else:
                for img_idx, img_name in enumerate(images):
                    img_path = os.path.join(input_dir, f"{sample_id}.{img_name}")
                    if os.path.exists(img_path):
                        img_key = f"img0_{img_idx}.jpg"
                        with open(img_path, "rb") as f:
                            sample[img_key] = f.read()

        # Build JSON payload
        payload = {
            "images": images,
            "prompts": prompts,
            "captions": captions,
            "sample_count": sample_count,
        }
        if patch_positions:
            payload["patch_positions"] = patch_positions
        if fps is not None:
            payload["fps"] = fps
        if timestamp_decimal is not None:
            payload["timestamp_decimal"] = timestamp_decimal
        if data_type is not None:
            payload["data_type"] = data_type
        sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        return sample

    except Exception:
        return None


def _process_chunk(
    chunk_id: int,
    json_files: list[str],
    input_dir: str,
    output_dir: str,
    shard_prefix: str,
    max_samples_per_shard: int,
    max_shard_size: int,
    validate: bool,
) -> tuple[int, int, list[str]]:
    """
    Process a chunk of samples and write to temporary tar files.

    Each worker processes its own chunk independently and writes to its own tar files.
    This avoids the serialization bottleneck of the original approach.

    Args:
        chunk_id: Unique identifier for this chunk.
        json_files: List of JSON file paths to process.
        input_dir: Input directory path.
        output_dir: Temporary output directory for this chunk.
        shard_prefix: Prefix for shard filenames.
        max_samples_per_shard: Max samples per shard.
        max_shard_size: Max shard size in bytes.
        validate: Whether to validate samples.

    Returns:
        Tuple of (processed_count, skipped_count, list of created tar files).
    """
    processed = 0
    skipped = 0
    created_tars = []

    # Create chunk-specific output pattern
    shard_pattern = os.path.join(output_dir, f"{shard_prefix}-{chunk_id:03d}-%03d.tar")

    with wds.ShardWriter(
        shard_pattern,
        maxcount=max_samples_per_shard,
        maxsize=max_shard_size,
    ) as sink:
        for json_file in json_files:
            json_path = Path(json_file)
            sample = _process_single_sample(json_path, input_dir, validate)

            if sample is not None:
                sink.write(sample)
                processed += 1
            else:
                skipped += 1

    # Collect created tar files
    for tar_file in Path(output_dir).glob(f"{shard_prefix}-{chunk_id:03d}-*.tar"):
        created_tars.append(str(tar_file))

    return processed, skipped, created_tars


class WebDatasetConverter:
    """
    Convert packed samples to WebDataset format.

    This class handles reading packed samples and writing them as
    WebDataset tar shards with proper Megatron-Energon configuration.
    """

    def __init__(self, config: WebDatasetConfig) -> None:
        """
        Initialize the converter.

        Args:
            config: Conversion configuration.
        """
        self.config = config
        self.processed_count = 0
        self.skipped_count = 0

    def stream_samples(self) -> Iterator[dict[str, Any]]:
        """
        Stream samples from the input directory.

        Yields:
            Sample dictionaries with id, images, prompts, captions, patch_positions, and sample_count.
        """
        input_path = Path(self.config.input_dir)

        for json_path in sorted(input_path.glob("*.json")):
            sample_id = json_path.stem  # e.g., ps_00000000

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                images = data.get("images", [])
                prompts = data.get("prompts", [])
                captions = data.get("captions", [])
                patch_positions = data.get("patch_positions", [])
                fps = data.get("fps", None)
                timestamp_decimal = data.get("timestamp_decimal", None)
                data_type = data.get("data_type", None)

                # Determine sample count from structure
                sample_count = 1
                if images and isinstance(images[0], list):
                    sample_count = len(images)

                if timestamp_decimal is None:
                    timestamp_decimal = [1] * sample_count

                result = {
                    "id": sample_id,
                    "images": images,
                    "prompts": prompts,
                    "captions": captions,
                    "sample_count": sample_count,
                }
                if patch_positions:
                    result["patch_positions"] = patch_positions
                if fps is not None:
                    result["fps"] = fps
                if timestamp_decimal is not None:
                    result["timestamp_decimal"] = timestamp_decimal
                if data_type is not None:
                    result["data_type"] = data_type
                
                yield result

            except Exception as e:
                print(f"Warning: Failed to read {json_path}: {e}")
                continue

    def validate_sample(self, entry: dict[str, Any]) -> bool:
        """
        Validate sample structure for consistency.

        Args:
            entry: Sample entry to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not self.config.validate_samples:
            return True

        sample_count = entry.get("sample_count", 1)
        images = entry["images"]
        prompts = entry["prompts"]
        captions = entry["captions"]
        timestamp_decimal = entry.get("timestamp_decimal", None)
        if timestamp_decimal is None:
            timestamp_decimal = [1] * sample_count

        if sample_count > 1:
            # Multi-sample validation
            if len(images) != sample_count:
                print(f"Warning: {entry['id']} images count ({len(images)}) != sample count ({sample_count})")
                return False
            if len(prompts) != sample_count:
                print(f"Warning: {entry['id']} prompts count ({len(prompts)}) != sample count ({sample_count})")
                return False
            if len(captions) != sample_count:
                print(f"Warning: {entry['id']} captions count ({len(captions)}) != sample count ({sample_count})")
                return False
            if timestamp_decimal is not None and len(timestamp_decimal) != sample_count:
                print(
                    f"Warning: {entry['id']} timestamp_decimal count ({len(timestamp_decimal)}) != sample count ({sample_count})"
                )
                return False
            entry_data_type = entry.get("data_type", None)
            if entry_data_type is not None and len(entry_data_type) != sample_count:
                print(
                    f"Warning: {entry['id']} data_type count ({len(entry_data_type)}) != sample count ({sample_count})"
                )
                return False
        else:
            # Single sample validation
            if images and isinstance(images[0], list):
                if len(prompts) != 1 or len(captions) != 1:
                    print(f"Warning: {entry['id']} single sample format incorrect")
                    return False

        return True

    def build_wds_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        """
        Build a WebDataset sample from entry.

        Args:
            entry: Sample entry with id, images, prompts, captions, and optionally patch_positions.

        Returns:
            WebDataset sample dict with __key__, image binaries, patch_position binaries, and json.
        """
        sample = {"__key__": entry["id"]}
        input_dir = self.config.input_dir
        sample_count = entry.get("sample_count", 1)
        images = entry["images"]
        patch_positions = entry.get("patch_positions", [])

        if sample_count > 1:
            # Multi-sample: images is nested list
            for sample_idx, sample_images in enumerate(images):
                if not sample_images:
                    continue
                for img_idx, img_name in enumerate(sample_images):
                    img_path = os.path.join(input_dir, f"{entry['id']}.{img_name}")
                    if os.path.exists(img_path):
                        img_key = f"img{sample_idx}_{img_idx}.jpg"
                        with open(img_path, "rb") as f:
                            sample[img_key] = f.read()
                    else:
                        print(f"Warning: Image not found: {img_path}")
                    
                    # Load corresponding .npy file if exists
                    if sample_idx < len(patch_positions) and img_idx < len(patch_positions[sample_idx]):
                        pp_name = patch_positions[sample_idx][img_idx]
                        if pp_name:
                            pp_path = os.path.join(input_dir, f"{entry['id']}.{pp_name}")
                            if os.path.exists(pp_path):
                                pp_key = f"img{sample_idx}_{img_idx}.npy"
                                with open(pp_path, "rb") as f:
                                    sample[pp_key] = f.read()
        else:
            # Single sample
            if images and isinstance(images[0], list):
                # New format: nested list [["img1", "img2", ...]]
                sample_images = images[0]
                for img_idx, img_name in enumerate(sample_images):
                    img_path = os.path.join(input_dir, f"{entry['id']}.{img_name}")
                    if os.path.exists(img_path):
                        img_key = f"img0_{img_idx}.jpg"
                        with open(img_path, "rb") as f:
                            sample[img_key] = f.read()
                    else:
                        print(f"Warning: Image not found: {img_path}")
                    
                    # Load corresponding .npy file if exists
                    if patch_positions and len(patch_positions) > 0 and img_idx < len(patch_positions[0]):
                        pp_name = patch_positions[0][img_idx]
                        if pp_name:
                            pp_path = os.path.join(input_dir, f"{entry['id']}.{pp_name}")
                            if os.path.exists(pp_path):
                                pp_key = f"img0_{img_idx}.npy"
                                with open(pp_path, "rb") as f:
                                    sample[pp_key] = f.read()
            else:
                # Legacy format: flat list ["img1", "img2", ...]
                for img_idx, img_name in enumerate(images):
                    img_path = os.path.join(input_dir, f"{entry['id']}.{img_name}")
                    if os.path.exists(img_path):
                        img_key = f"img0_{img_idx}.jpg"
                        with open(img_path, "rb") as f:
                            sample[img_key] = f.read()
                    else:
                        print(f"Warning: Image not found: {img_path}")

        # Build JSON payload
        payload = {
            "images": entry["images"],
            "prompts": entry["prompts"],
            "captions": entry["captions"],
            "sample_count": sample_count,
        }
        if patch_positions:
            payload["patch_positions"] = patch_positions
        fps = entry.get("fps", None)
        if fps is not None:
            payload["fps"] = fps
        timestamp_decimal = entry.get("timestamp_decimal", None)
        if timestamp_decimal is None:
            timestamp_decimal = [1] * sample_count
        payload["timestamp_decimal"] = timestamp_decimal
        data_type = entry.get("data_type", None)
        if data_type is not None:
            payload["data_type"] = data_type
        sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        return sample

    def write_energon_config(self) -> None:
        """Write Megatron-Energon configuration files."""
        output_path = Path(self.config.output_dir)
        meta_dir = output_path / MAIN_FOLDER_NAME
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Find all tar files
        tar_files = sorted(output_path.glob("*.tar"))
        tar_names = [str(p.relative_to(output_path)) for p in tar_files]

        # Write dataset.yaml
        dataset_config = {
            "sample_type": {
                "__module__": "aiak_training_llm.data.multimodal",
                "__class__": self.config.sample_class_name,
            },
            "part_filter": "sample_loader.py:part_filter",
            "sample_loader": "sample_loader.py:sample_loader",
        }

        with (meta_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(dataset_config, f, sort_keys=False)

        # Write sample_loader.py
        with (meta_dir / "sample_loader.py").open("w", encoding="utf-8") as f:
            f.write(generate_sample_loader_code())

        # Use Energon to prepare dataset if available
        if ENERGON_AVAILABLE:
            try:
                BaseWebdatasetFactory.prepare_dataset(
                    EPath(self.config.output_dir).absolute(),
                    tar_names,
                    split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
                    tar_index_only=False,
                    workers=self.config.workers,
                )
            except Exception as e:
                print(f"Warning: Energon dataset preparation failed: {e}")
                print("Manual configuration may be required.")
        else:
            # Write basic split.yaml manually
            split_config = {
                "split": {
                    "train": tar_names,
                    "val": [],
                    "test": [],
                }
            }
            with (meta_dir / "split.yaml").open("w", encoding="utf-8") as f:
                yaml.dump(split_config, f, sort_keys=False)

    def run(self) -> None:
        """Execute the conversion process."""
        # Validate configuration
        self.config.validate()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Prepare shard writer pattern
        shard_pattern = os.path.join(self.config.output_dir, f"{self.config.shard_prefix}-%06d.tar")

        print("\n" + "=" * 70)
        print("WebDataset Conversion")
        print("=" * 70)
        print(f"  Input directory:  {self.config.input_dir}")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Mode:             {self.config.mode}")
        print(f"  Max samples/shard: {self.config.max_samples_per_shard:,}")
        print(f"  Max shard size:   {self.config.max_shard_size:,} bytes")
        print(f"  Workers:          {self.config.workers}")
        print("=" * 70)

        # Collect all JSON files
        input_path = Path(self.config.input_dir)
        json_files = sorted(input_path.glob("*.json"))
        total_files = len(json_files)
        print(f"\n  Found {total_files:,} samples to process")

        # Use multiprocessing for reading and building samples
        num_workers = min(self.config.workers, cpu_count(), total_files)

        if num_workers > 1 and total_files > 100:
            print(f"  Using {num_workers} workers for parallel processing\n")
            self._run_parallel(json_files, shard_pattern, num_workers)
        else:
            print("  Using single-threaded processing\n")
            self._run_sequential(shard_pattern)

        # Write Energon configuration
        print("\nWriting Megatron-Energon configuration...")
        self.write_energon_config()

        # Print summary
        print("\n" + "=" * 70)
        print("Conversion Complete")
        print("=" * 70)
        print(f"  Samples processed: {self.processed_count:,}")
        print(f"  Samples skipped:   {self.skipped_count:,}")
        print(f"  Output directory:  {self.config.output_dir}")

        # Count output shards
        shard_count = len(list(Path(self.config.output_dir).glob("*.tar")))
        print(f"  Shards created:    {shard_count}")
        print("=" * 70)

    def _run_sequential(self, shard_pattern: str) -> None:
        """Run conversion in single-threaded mode."""
        with wds.ShardWriter(
            shard_pattern,
            maxcount=self.config.max_samples_per_shard,
            maxsize=self.config.max_shard_size,
        ) as sink:
            for entry in tqdm(self.stream_samples(), desc="Converting samples", unit="sample"):
                # Validate sample structure
                if not self.validate_sample(entry):
                    self.skipped_count += 1
                    continue

                try:
                    sample = self.build_wds_sample(entry)
                    sink.write(sample)
                    self.processed_count += 1
                except Exception as e:
                    print(f"Error processing {entry['id']}: {e}")
                    self.skipped_count += 1

    def _run_parallel(self, json_files: list[Path], shard_pattern: str, num_workers: int) -> None:
        """
        Run conversion with multiprocessing using chunk-based parallelism.

        Each worker processes a chunk of samples and writes to its own tar files.
        This achieves true parallelism since both reading AND writing are parallel.
        After all workers complete, tar files are renamed to sequential order.
        """
        input_dir = self.config.input_dir
        output_dir = self.config.output_dir
        validate = self.config.validate_samples

        # Split files into chunks for each worker
        total_files = len(json_files)
        chunk_size = (total_files + num_workers - 1) // num_workers
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min(start + chunk_size, total_files)
            if start < end:
                chunks.append((i, [str(f) for f in json_files[start:end]]))

        print(f"  Split into {len(chunks)} chunks, ~{chunk_size:,} samples each\n")

        all_tar_files = []

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for chunk_id, chunk_files in chunks:
                future = executor.submit(
                    _process_chunk,
                    chunk_id=chunk_id,
                    json_files=chunk_files,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    shard_prefix=self.config.shard_prefix,
                    max_samples_per_shard=self.config.max_samples_per_shard,
                    max_shard_size=self.config.max_shard_size,
                    validate=validate,
                )
                futures[future] = chunk_id

            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing chunks", unit="chunk") as pbar:
                for future in as_completed(futures):
                    chunk_id = futures[future]
                    try:
                        processed, skipped, tar_files = future.result()
                        self.processed_count += processed
                        self.skipped_count += skipped
                        all_tar_files.extend(tar_files)
                        pbar.set_postfix(
                            processed=self.processed_count, skipped=self.skipped_count, tars=len(all_tar_files)
                        )
                    except Exception as e:
                        print(f"Error in chunk {chunk_id}: {e}")
                    pbar.update(1)

        # Rename tar files to sequential order
        print(f"\n  Renaming {len(all_tar_files)} tar files to sequential order...")
        all_tar_files.sort()
        for new_idx, old_path in enumerate(all_tar_files):
            new_name = f"{self.config.shard_prefix}-{new_idx:06d}.tar"
            new_path = os.path.join(output_dir, new_name)
            if old_path != new_path:
                shutil.move(old_path, new_path)


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert packed samples to WebDataset format (Step 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file:
    python s5_convert_to_webdataset.py --config s5_config.yaml

    # Using command line arguments:
    python s5_convert_to_webdataset.py \\
        --input-dir /path/to/packed_samples \\
        --output-dir /path/to/webdataset \\
        --mode bmr_pack \\
        --max-samples-per-shard 10000

    # With custom shard settings:
    python s5_convert_to_webdataset.py \\
        --input-dir ./packed_data \\
        --output-dir ./wds_output \\
        --shard-prefix train \\
        --max-shard-size 5000000000
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Directory containing packed samples from Step 4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for WebDataset shards",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["caption_pack", "bmr_pack"],
        default="bmr_pack",
        help="Conversion mode: caption_pack or bmr_pack (default: bmr_pack)",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=10000,
        help="Maximum samples per shard (default: 10000)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=3_000_000_000,
        help="Maximum shard size in bytes (default: 3GB)",
    )
    parser.add_argument(
        "--media-type",
        type=str,
        choices=["image", "video", "mix"],
        default="image",
        help="Media type (default: image)",
    )
    parser.add_argument(
        "--shard-prefix",
        type=str,
        default="pretrain",
        help="Prefix for shard filenames (default: pretrain)",
    )
    parser.add_argument(
        "--sample-class-name",
        type=str,
        default="PackedCaptioningSample",
        help="Sample class name for Energon (default: PackedCaptioningSample)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of workers for indexing (default: 32)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip sample validation",
    )

    args = parser.parse_args()
    args.validate_samples = not args.no_validate
    return args


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = WebDatasetConfig.from_yaml(args.config)

        # Override with command line arguments if provided
        if args.input_dir:
            config.input_dir = args.input_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.mode != "bmr_pack":
            config.mode = args.mode
    else:
        config = WebDatasetConfig.from_args(args)

    # Check for Energon availability
    if not ENERGON_AVAILABLE:
        print("Warning: Megatron-Energon not available.")
        print("Basic configuration will be generated without full indexing.")

    # Run converter
    converter = WebDatasetConverter(config)
    converter.run()

    print("\n✅ WebDataset conversion completed successfully!")


if __name__ == "__main__":
    main()
