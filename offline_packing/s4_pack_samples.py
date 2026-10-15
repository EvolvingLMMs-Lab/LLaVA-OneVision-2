#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack Samples - Combine Packed Bins into Final Training Data.

This script is Step 4 of the offline packing pipeline. It takes the bin
packing results from Step 3 and generates the final packed training samples
by combining multiple original samples into single training units.

The output format is optimized for multi-image VLM training with:
- Combined image references for packed samples
- Merged prompts and captions/responses
- Support for pretrain, SFT, and BMR (multi-round dialog) task types

Pipeline Overview:
    Step 1: s1_split_json_to_samples.py - Split source JSON/JSONL files
    Step 2: s2_compute_token_lengths.py - Calculate token lengths
    Step 3: s3_bin_packing.py          - Pack samples into bins
    Step 4: s4_pack_samples.py         - Generate final packed data (this script)

Usage:
    python s4_pack_samples.py --config config.yaml

    # Or with command line arguments:
    python s4_pack_samples.py \\
        --bins-file ./s2_ckpt/bins.pkl \\
        --source-dir /path/to/source/samples \\
        --output-dir /path/to/packed/output \\
        --task-type bmr \\
        --workers 32

Config File Format (YAML):
    bins_file: ./s2_ckpt/bins.pkl
    source_dir: /path/to/source/samples
    output_dir: /path/to/packed/output
    task_type: bmr  # pretrain, sft, or bmr
    workers: 32
    image_extension: jpg
    json_extension: json
    image_base_dir: null  # Optional, defaults to source_dir

Task Types:
    - pretrain: Caption-only data with random prompts
    - sft: Single-round QA (question-answer pairs)
    - bmr: Multi-round dialog with multi-image support

Output Format (packed sample JSON):
    {
        "images": [["img0_sub0.jpg", "img0_sub1.jpg"], ["img1_sub0.jpg"], ...],
        "prompts": [["prompt1_turn1", "prompt1_turn2"], ["prompt2_turn1"], ...],
        "captions": [["response1_turn1", "response1_turn2"], ["response2_turn1"], ...],
        "timestamp_decimal": [2, 1, ...]
    }

Author: LLaVA-OneVision Team
License: Apache-2.0
"""

import argparse
import json
import os
import pickle
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

# Default prompts for caption-style pretrain data
DEFAULT_CAPTION_PROMPTS = [
    "Describe this image in detail.",
    "What do you see in this image?",
    "Please describe the contents of this picture.",
    "Can you tell me what's happening in this image?",
    "Provide a detailed description of this image.",
    "What is shown in this photograph?",
    "Explain what you observe in this image.",
    "Give me a comprehensive description of this picture.",
    "What elements can you identify in this image?",
    "Describe everything you notice in this visual.",
    "Please analyze and describe this image.",
    "What story does this image tell?",
    "Walk me through what you see in this picture.",
    "Describe the scene depicted in this image.",
    "What are the main features of this image?",
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PackSamplesConfig:
    """Configuration for sample packing."""

    # Input/Output paths
    bins_file: str = ""
    source_dir: str = ""
    output_dir: str = ""

    # Optional: multiple source directories for cross-directory packing
    # When provided, samples are searched across all directories
    source_dirs: list[str] = field(default_factory=list)

    # Optional: separate directory for images (defaults to source_dir)
    image_base_dir: Optional[str] = None

    # Task type: pretrain, sft, or bmr
    task_type: str = "bmr"

    # File extensions
    image_extension: str = "jpg"
    json_extension: str = "json"

    # Processing options
    workers: int = 32
    clear_output_dir: bool = True

    # Testing options
    test_mode: bool = False
    test_samples: int = 100

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PackSamplesConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PackSamplesConfig":
        """Create configuration from command line arguments."""
        source_dirs = []
        if hasattr(args, 'source_dirs') and args.source_dirs:
            source_dirs = [d.strip() for d in args.source_dirs.split(",") if d.strip()]
        return cls(
            bins_file=args.bins_file,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            source_dirs=source_dirs,
            image_base_dir=args.image_base_dir,
            task_type=args.task_type,
            image_extension=args.image_extension,
            json_extension=args.json_extension,
            workers=args.workers,
            clear_output_dir=args.clear_output_dir,
            test_mode=args.test_mode,
            test_samples=args.test_samples,
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.bins_file:
            raise ValueError("bins_file is required")
        if not self.source_dir and not self.source_dirs:
            raise ValueError("source_dir or source_dirs is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.task_type not in ("pretrain", "sft", "bmr"):
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be pretrain, sft, or bmr")
        if not os.path.exists(self.bins_file):
            raise FileNotFoundError(f"Bins file not found: {self.bins_file}")
        if self.source_dirs:
            for d in self.source_dirs:
                if not os.path.isdir(d):
                    raise NotADirectoryError(f"Source directory not found: {d}")
        elif self.source_dir and not os.path.isdir(self.source_dir):
            raise NotADirectoryError(f"Source directory not found: {self.source_dir}")


# =============================================================================
# Statistics Tracking
# =============================================================================


@dataclass
class PackingStats:
    """Thread-safe statistics for packing process."""

    total_samples: int = 0
    total_images: int = 0
    samples_with_no_images: int = 0
    samples_with_single_image: int = 0
    samples_with_multiple_images: int = 0
    max_images_per_sample: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_sample(self, image_count: int) -> None:
        """Record a processed sample."""
        with self._lock:
            self.total_samples += 1
            self.total_images += image_count

            if image_count == 0:
                self.samples_with_no_images += 1
            elif image_count == 1:
                self.samples_with_single_image += 1
            else:
                self.samples_with_multiple_images += 1

            if image_count > self.max_images_per_sample:
                self.max_images_per_sample = image_count

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            avg_images = self.total_images / self.total_samples if self.total_samples > 0 else 0
            multi_ratio = self.samples_with_multiple_images / self.total_samples * 100 if self.total_samples > 0 else 0
            return {
                "total_samples": self.total_samples,
                "total_images": self.total_images,
                "avg_images_per_sample": avg_images,
                "max_images_per_sample": self.max_images_per_sample,
                "samples_with_no_images": self.samples_with_no_images,
                "samples_with_single_image": self.samples_with_single_image,
                "samples_with_multiple_images": self.samples_with_multiple_images,
                "multi_image_ratio": multi_ratio,
            }


# =============================================================================
# Sample Packer
# =============================================================================



def check_turn_pairing(prompts: list[str], responses: list[str], sample_name: str) -> None:
    """Fail loudly if a flattened conversation cannot be paired by index.

    Prompts and responses are paired positionally downstream; unequal counts mean
    the source conversation was not strictly alternating user/assistant and every
    later turn would be packed with the wrong answer. s1_build_manifest drops such
    samples; this guards manifests/sample dirs built without that check.
    """
    if len(prompts) != len(responses):
        raise ValueError(
            f"prompt/response turn mismatch ({len(prompts)} vs {len(responses)}) for sample "
            f"{sample_name}: conversation is not strictly alternating user/assistant. "
            "Fix the source data (s1_build_manifest drops these samples)."
        )


class SamplePacker:
    """
    Pack samples into final training format.

    This class handles the conversion of bin packing results into
    actual packed training samples with combined images and text.
    """

    def __init__(self, config: PackSamplesConfig) -> None:
        """
        Initialize the sample packer.

        Args:
            config: Packing configuration.
        """
        self.config = config
        self.stats = PackingStats()

        # Support multiple source directories for cross-directory packing
        if config.source_dirs:
            self.source_dirs = config.source_dirs
        elif config.source_dir:
            self.source_dirs = [config.source_dir]
        else:
            self.source_dirs = []

        # Set default image base directory (used when source_dirs is not set)
        self.image_base_dir = config.image_base_dir or config.source_dir

        # Cache for sample -> source_dir mapping (thread-safe via GIL for dict reads)
        self._sample_dir_cache: dict[str, str] = {}

        # Load caption prompts for pretrain task
        self.prompts = DEFAULT_CAPTION_PROMPTS

    @staticmethod
    def _parse_sample_name(raw_name: str) -> tuple[str, Optional[int]]:
        """
        Parse a potentially prefixed sample name.

        Sample names may carry a 'srcN::' prefix added by merge_pack.sh
        to disambiguate samples that share the same name across different
        source directories.  For example:
            'src0::sample_name'  ->  ('sample_name', 0)
            'src12::image.jpg'   ->  ('image.jpg', 12)
            'plain_name'         ->  ('plain_name', None)   # no prefix

        Args:
            raw_name: Raw sample name, possibly prefixed.

        Returns:
            Tuple of (actual_name, source_dir_index) where index is
            None when no valid prefix is found.
        """
        if raw_name.startswith("src") and "::" in raw_name:
            prefix_end = raw_name.index("::")
            try:
                dir_idx = int(raw_name[3:prefix_end])
                actual_name = raw_name[prefix_end + 2:]
                return actual_name, dir_idx
            except ValueError:
                pass
        return raw_name, None

    def _find_sample_source_dir(self, raw_name: str) -> tuple[Optional[str], str]:
        """
        Find which source directory contains the given sample.

        If the name carries a 'srcN::' prefix (from merge_pack.sh), the
        directory is resolved by direct index.  Otherwise, falls back to
        searching all source directories.

        Args:
            raw_name: Raw sample name (may contain srcN:: prefix).

        Returns:
            Tuple of (source_dir_or_None, actual_sample_name).
        """
        actual_name, dir_idx = self._parse_sample_name(raw_name)

        # Fast path: prefix tells us the exact directory
        if dir_idx is not None and dir_idx < len(self.source_dirs):
            return self.source_dirs[dir_idx], actual_name

        # Slow path: search all directories (backwards compat with non-prefixed names)
        if actual_name in self._sample_dir_cache:
            return self._sample_dir_cache[actual_name], actual_name

        json_ext = self.config.json_extension
        for source_dir in self.source_dirs:
            json_path = os.path.join(source_dir, f"{actual_name}.{json_ext}")
            if os.path.exists(json_path):
                self._sample_dir_cache[actual_name] = source_dir
                return source_dir, actual_name

        return None, actual_name

    def load_bins(self) -> list[Any]:
        """
        Load bin packing results from pickle file.

        Returns:
            List of bins, each containing sample information.
        """
        print(f"Loading bins from: {self.config.bins_file}")
        with open(self.config.bins_file, "rb") as f:
            bins = pickle.load(f)
        print(f"Loaded {len(bins)} bins")
        return bins

    def prepare_output_directory(self) -> None:
        """Prepare the output directory."""
        import subprocess
        
        output_dir = self.config.output_dir

        if os.path.exists(output_dir):
            if self.config.clear_output_dir:
                print(f"Clearing existing output directory: {output_dir}")
                print("  (This may take a while for large directories...)")
                
                # Use rm -rf for fast deletion of large directories
                # This is much faster than Python's shutil.rmtree for millions of files
                try:
                    # First try using system rm command (fastest)
                    result = subprocess.run(
                        ["rm", "-rf", output_dir],
                        capture_output=True,
                        text=True,
                        timeout=3600,  # 1 hour timeout
                    )
                    if result.returncode != 0:
                        print(f"  Warning: rm command failed: {result.stderr}")
                        # Fallback to shutil
                        shutil.rmtree(output_dir, ignore_errors=True)
                except subprocess.TimeoutExpired:
                    print("  Warning: rm command timed out, trying shutil...")
                    shutil.rmtree(output_dir, ignore_errors=True)
                except FileNotFoundError:
                    # rm command not available, use shutil
                    shutil.rmtree(output_dir, ignore_errors=True)
                except Exception as e:
                    print(f"  Warning: Fast delete failed ({e}), using shutil...")
                    shutil.rmtree(output_dir, ignore_errors=True)
                
                # Recreate the directory
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output directory cleared and recreated: {output_dir}")
            else:
                print(f"Warning: Output directory exists and will not be cleared: {output_dir}")
        else:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    def _extract_pretrain_content(self, json_path: str) -> Optional[str]:
        """
        Extract caption content from pretrain-style JSON.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Caption content string, or None if extraction fails.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            captions = data.get("captions", [])
            if captions and len(captions) > 0:
                return captions[0].get("content", "")
            return ""
        except Exception as e:
            print(f"Error extracting pretrain content from {json_path}: {e}")
            return None

    def _extract_sft_content(self, json_path: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract prompt and response from SFT-style JSON.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Tuple of (prompt, response), or (None, None) if extraction fails.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])

            prompt = None
            response = None

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user" and prompt is None:
                    prompt = content
                elif role == "assistant" and response is None:
                    response = content

            return prompt, response
        except Exception as e:
            print(f"Error extracting SFT content from {json_path}: {e}")
            return None, None

    def _extract_bmr_content(
        self, json_path: str, image_base_dir: Optional[str] = None
    ) -> tuple[list[str], list[str], list[str], list[str], Any, int, Optional[str]]:
        """
        Extract images, prompts, responses, patch_positions, fps, and timestamp precision from BMR JSON.

        Args:
            json_path: Path to the JSON file.
            image_base_dir: Override for image base directory. If None, uses self.image_base_dir.

        Returns:
            Tuple of (images, prompts, responses, patch_positions, fps, timestamp_decimal, data_type).
            fps is the raw value from the JSON (e.g., [30]) or None if not present.
            timestamp_decimal defaults to 1 when the source sample does not define it.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            base_dir = image_base_dir if image_base_dir is not None else self.image_base_dir

            # Extract images
            raw_images = data.get("images", [])
            images = []
            for img in raw_images:
                if img:
                    full_path = os.path.join(base_dir, img)
                    images.append(full_path)

            # Extract patch_positions
            raw_patch_positions = data.get("patch_positions", [])
            patch_positions = []
            for pp in raw_patch_positions:
                if pp:
                    full_path = os.path.join(base_dir, pp)
                    patch_positions.append(full_path)
                else:
                    patch_positions.append("")  # Empty string for missing .npy

            # Extract fps (optional field, e.g., [30])
            fps = data.get("fps", None)

            # Extract timestamp precision, defaulting to 1 for legacy samples.
            timestamp_decimal = data.get("timestamp_decimal", 1)
            if timestamp_decimal is None:
                timestamp_decimal = 1

            # Sample-level modality marker; absent on every pre-existing sample.
            data_type = data.get("data_type", None)

            # Handle field mapping: conversations -> messages
            if "conversations" in data and "messages" not in data:
                data["messages"] = data["conversations"]

            messages = data.get("messages", [])

            # Map role names: human -> user, gpt -> assistant
            prompts = []
            responses = []

            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    # Role mapping
                    if role in ("human", "user"):
                        prompts.append(content)
                    elif role in ("gpt", "assistant"):
                        responses.append(content)

            return images, prompts, responses, patch_positions, fps, timestamp_decimal, data_type

        except Exception as e:
            print(f"Error extracting BMR content from {json_path}: {e}")
            return [], [], [], [], None, 1, None

    def _get_random_prompts(self, count: int) -> list[str]:
        """Get random prompts for pretrain data."""
        return random.choices(self.prompts, k=count)

    def process_bin(
        self,
        bin_index: int,
        bin_data: list[Any],
    ) -> int:
        """
        Process a single bin and generate packed output.

        Args:
            bin_index: Index of the bin being processed.
            bin_data: List of samples in this bin.

        Returns:
            The bin index (for tracking).
        """
        output_dir = self.config.output_dir
        task_type = self.config.task_type
        json_ext = self.config.json_extension
        img_ext = self.config.image_extension

        packed_images = []
        packed_prompts = []
        packed_captions = []
        packed_patch_positions = []
        packed_fps = []
        packed_timestamp_decimals = []
        packed_data_types = []

        # Get sample names from bin data
        sample_names = [item["name"] for item in bin_data]

        for sample_idx, raw_sample_name in enumerate(sample_names):
            # Find the source directory for this sample (supports multi-dir packing)
            # raw_sample_name may have 'srcN::' prefix from merge_pack.sh
            sample_source_dir, sample_name = self._find_sample_source_dir(raw_sample_name)
            if sample_source_dir is None:
                print(f"Warning: Sample {raw_sample_name} not found in any source directory, skipping")
                continue
            json_path = os.path.join(sample_source_dir, f"{sample_name}.{json_ext}")
            sample_image_base = sample_source_dir

            if task_type == "pretrain":
                # Pretrain: caption-only with image
                img_src = os.path.join(sample_image_base, f"{sample_name}.{img_ext}")
                caption = self._extract_pretrain_content(json_path)

                if caption is not None:
                    # Copy image
                    img_name_dst = f"ps_{bin_index:08d}.img{sample_idx:03d}.{img_ext}"
                    img_dst = os.path.join(output_dir, img_name_dst)
                    shutil.copyfile(img_src, img_dst)

                    packed_images.append(f"img{sample_idx:03d}.{img_ext}")
                    packed_captions.append(caption)
                    self.stats.add_sample(1)

            elif task_type == "sft":
                # SFT: single-round QA
                img_src = os.path.join(sample_image_base, f"{sample_name}.{img_ext}")
                prompt, response = self._extract_sft_content(json_path)

                if prompt is not None and response is not None:
                    # Copy image
                    img_name_dst = f"ps_{bin_index:08d}.img{sample_idx:03d}.{img_ext}"
                    img_dst = os.path.join(output_dir, img_name_dst)
                    shutil.copyfile(img_src, img_dst)

                    packed_images.append(f"img{sample_idx:03d}.{img_ext}")
                    packed_prompts.append(prompt)
                    packed_captions.append(response)
                    self.stats.add_sample(1)

            elif task_type == "bmr":
                # BMR: multi-round dialog with multi-image support
                images, prompts, responses, patch_positions, fps, timestamp_decimal, data_type = self._extract_bmr_content(
                    json_path, image_base_dir=sample_image_base
                )
                check_turn_pairing(prompts, responses, json_path)

                if not images:
                    # Text-only sample
                    packed_images.append([])
                    packed_patch_positions.append([])
                    self.stats.add_sample(0)
                else:
                    # Process all images for this sample
                    sample_image_names = []
                    sample_patch_position_names = []
                    for img_idx, img_src in enumerate(images):
                        _, ext = os.path.splitext(img_src)
                        if not ext:
                            ext = f".{img_ext}"

                        img_name_dst = f"ps_{bin_index:08d}.img{sample_idx:03d}_sub{img_idx:03d}{ext}"
                        img_dst = os.path.join(output_dir, img_name_dst)

                        try:
                            shutil.copyfile(img_src, img_dst)
                            sample_image_names.append(f"img{sample_idx:03d}_sub{img_idx:03d}{ext}")
                        except Exception as e:
                            print(f"Warning: Failed to copy image {img_src}: {e}")

                        # Copy corresponding .npy file if it exists
                        if img_idx < len(patch_positions) and patch_positions[img_idx]:
                            pp_src = patch_positions[img_idx]
                            pp_name_dst = f"ps_{bin_index:08d}.img{sample_idx:03d}_sub{img_idx:03d}.npy"
                            pp_dst = os.path.join(output_dir, pp_name_dst)

                            try:
                                shutil.copyfile(pp_src, pp_dst)
                                sample_patch_position_names.append(f"img{sample_idx:03d}_sub{img_idx:03d}.npy")
                            except Exception as e:
                                print(f"Warning: Failed to copy patch_position {pp_src}: {e}")
                                sample_patch_position_names.append("")
                        else:
                            sample_patch_position_names.append("")

                    packed_images.append(sample_image_names)
                    packed_patch_positions.append(sample_patch_position_names)
                    self.stats.add_sample(len(sample_image_names))

                # Add prompts and responses
                packed_prompts.append(prompts if prompts else [""])
                packed_captions.append(responses if responses else [""])

                # Add fps (per-sample, e.g., [30] or None)
                packed_fps.append(fps)
                packed_timestamp_decimals.append(timestamp_decimal)
                packed_data_types.append(data_type)

        # Generate random prompts for pretrain task
        if task_type == "pretrain":
            packed_prompts = self._get_random_prompts(len(packed_images))

        # Write output JSON
        json_dst = os.path.join(output_dir, f"ps_{bin_index:08d}.{json_ext}")
        output_data = {
            "images": packed_images,
            "prompts": packed_prompts,
            "captions": packed_captions,
        }
        
        # Add patch_positions for BMR task (includes empty lists for text-only samples)
        if task_type == "bmr":
            output_data["patch_positions"] = packed_patch_positions

        # Add fps if any sample has fps info (includes None for samples without fps)
        if task_type == "bmr" and any(f is not None for f in packed_fps):
            output_data["fps"] = packed_fps

        if task_type == "bmr":
            output_data["timestamp_decimal"] = packed_timestamp_decimals

        # Positional, one entry per sub-sample. Omitted entirely when no sub-sample is
        # marked, so bins of ordinary data keep their existing payload unchanged.
        if task_type == "bmr" and any(dt is not None for dt in packed_data_types):
            output_data["data_type"] = packed_data_types

        try:
            with open(json_dst, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing output JSON {json_dst}: {e}")

        return bin_index

    def run(self) -> None:
        """Execute the packing process."""
        # Validate configuration
        self.config.validate()

        # Prepare output directory
        print("\n" + "=" * 60)
        print("Step 1: Preparing output directory")
        print("=" * 60)
        self.prepare_output_directory()

        # Load bins
        print("\n" + "=" * 60)
        print("Step 2: Loading bin packing results")
        print("=" * 60)
        bins = self.load_bins()

        # Apply test mode limit
        if self.config.test_mode:
            bins = bins[: self.config.test_samples]
            print(f"Test mode: Processing only {len(bins)} bins")

        total_bins = len(bins)

        # Process bins
        print("\n" + "=" * 60)
        print("Step 3: Processing packed samples")
        print("=" * 60)
        print(f"Processing {total_bins} bins with {self.config.workers} workers")

        completed = 0
        failed = 0

        with ThreadPoolExecutor(
            max_workers=self.config.workers,
            thread_name_prefix="PackWorker",
        ) as executor:
            futures = {executor.submit(self.process_bin, idx, bin_data): idx for idx, bin_data in enumerate(bins)}

            with tqdm(total=len(futures), desc="Packing progress", unit="bin") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                    except Exception as e:
                        bin_idx = futures[future]
                        print(f"\nError processing bin {bin_idx}: {e}")
                        failed += 1

                    pbar.update(1)

        # Print final statistics
        print("\n" + "=" * 60)
        print("Packing Complete")
        print("=" * 60)

        stats = self.stats.get_summary()
        print(f"  Total bins processed: {completed}")
        print(f"  Failed bins: {failed}")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Total images: {stats['total_images']:,}")
        print(f"  Average images per sample: {stats['avg_images_per_sample']:.2f}")
        print(f"  Max images per sample: {stats['max_images_per_sample']}")
        print(f"  Samples with no images: {stats['samples_with_no_images']:,}")
        print(f"  Samples with single image: {stats['samples_with_single_image']:,}")
        print(f"  Samples with multiple images: {stats['samples_with_multiple_images']:,}")
        print(f"  Multi-image sample ratio: {stats['multi_image_ratio']:.1f}%")
        print(f"\nOutput directory: {self.config.output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pack samples into final training format (Step 4 of packing pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file:
    python s4_pack_samples.py --config s4_config.yaml

    # Using command line arguments:
    python s4_pack_samples.py \\
        --bins-file ./s2_ckpt/bins.pkl \\
        --source-dir /path/to/samples \\
        --output-dir /path/to/output \\
        --task-type bmr \\
        --workers 32

    # Test mode (process only 100 bins):
    python s4_pack_samples.py --config s4_config.yaml --test-mode --test-samples 100
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--bins-file",
        type=str,
        default="",
        help="Path to the bin packing results (.pkl file)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="",
        help="Directory containing source sample JSON files",
    )
    parser.add_argument(
        "--source-dirs",
        type=str,
        default="",
        help="Comma-separated list of source directories for cross-directory packing "
             "(alternative to --source-dir, searches each dir for sample files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for packed samples",
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default=None,
        help="Base directory for images (defaults to source-dir)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["pretrain", "sft", "bmr"],
        default="bmr",
        help="Task type: pretrain, sft, or bmr (default: bmr)",
    )
    parser.add_argument(
        "--image-extension",
        type=str,
        default="jpg",
        help="Image file extension (default: jpg)",
    )
    parser.add_argument(
        "--json-extension",
        type=str,
        default="json",
        help="JSON file extension (default: json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker threads (default: 32)",
    )
    parser.add_argument(
        "--clear-output-dir",
        action="store_true",
        default=True,
        help="Clear output directory before processing (default: True)",
    )
    parser.add_argument(
        "--no-clear-output-dir",
        action="store_false",
        dest="clear_output_dir",
        help="Do not clear output directory before processing",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Enable test mode (process limited bins)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100,
        help="Number of bins to process in test mode (default: 100)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = PackSamplesConfig.from_yaml(args.config)

        # Override with command line arguments if provided
        if args.bins_file:
            config.bins_file = args.bins_file
        if args.source_dir:
            config.source_dir = args.source_dir
        if hasattr(args, 'source_dirs') and args.source_dirs:
            config.source_dirs = [d.strip() for d in args.source_dirs.split(",") if d.strip()]
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.image_base_dir:
            config.image_base_dir = args.image_base_dir
        if args.task_type != "bmr":
            config.task_type = args.task_type
        if args.test_mode:
            config.test_mode = True
            config.test_samples = args.test_samples
    else:
        config = PackSamplesConfig.from_args(args)

    # Print configuration
    print("\n" + "=" * 60)
    print("Sample Packer Configuration")
    print("=" * 60)
    print(f"  Bins file: {config.bins_file}")
    if config.source_dirs:
        print(f"  Source directories ({len(config.source_dirs)}):")
        for i, d in enumerate(config.source_dirs):
            print(f"    [{i}] {d}")
    else:
        print(f"  Source directory: {config.source_dir}")
    print(f"  Output directory: {config.output_dir}")
    img_base = config.image_base_dir or config.source_dir or "(from source_dirs)"
    print(f"  Image base directory: {img_base}")
    print(f"  Task type: {config.task_type}")
    print(f"  Workers: {config.workers}")
    print(f"  Test mode: {config.test_mode}")
    if config.test_mode:
        print(f"  Test samples: {config.test_samples}")

    # Run packer
    packer = SamplePacker(config)
    packer.run()

    print("\n✅ Packing completed successfully!")


if __name__ == "__main__":
    main()
