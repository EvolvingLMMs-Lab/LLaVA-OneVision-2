#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Token Lengths for Offline Packing Pipeline.

This script computes token lengths for samples in a dataset using a Qwen-VL
processor. It supports multi-image samples and uses a multi-process + multi-thread
architecture for efficient parallel processing.

The processing pipeline consists of three stages:
1. Stage0: Process chunks of samples in parallel processes, each using thread pools
2. Stage1: Merge stage0 files in batches (e.g., every 10 files)
3. Final: Merge all stage1 files into a single sorted output file

Usage:
    python s1_compute_token_lengths.py --config ./configs/s1_config.yaml

Author: LLaVA-OneVision Team
License: Apache-2.0
"""

import argparse
import gc
import json
import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from contextlib import contextmanager
from heapq import merge
from multiprocessing import Manager, Pool, Value
from pathlib import Path
from queue import Empty
from typing import Any, Optional

import numpy as np
import psutil
import yaml


# ============================================================================
# Resource Management Utilities
# ============================================================================

class TimeoutException(Exception):
    """Custom exception for timeout handling."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


@contextmanager
def time_limit(seconds: int):
    """
    Context manager for setting a time limit on operations.
    Only works on Unix-like systems (uses SIGALRM).
    
    Args:
        seconds: Maximum time allowed in seconds.
    """
    if sys.platform == 'win32':
        # Windows doesn't support SIGALRM, just yield without timeout
        yield
        return
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class ResourceMonitor:
    """
    Monitor system resources and provide adaptive scaling recommendations.
    """
    
    # Thresholds for resource management (relaxed for processing workloads)
    CPU_HIGH = 95       # High threshold - processing naturally uses CPU
    CPU_CRITICAL = 99   # Critical only when truly maxed out
    MEM_HIGH = 90       # Memory high threshold
    MEM_CRITICAL = 95   # Memory critical threshold
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._last_check = 0
        self._check_interval = 30  # Increased to 30 seconds to reduce overhead
        self._cached_status = None
        self._consecutive_high_count = 0  # Require consecutive high readings
    
    def get_status(self, force: bool = False) -> dict:
        """
        Get current system resource status.
        Uses caching to avoid frequent system calls.
        """
        now = time.time()
        if not force and self._cached_status and (now - self._last_check) < self._check_interval:
            return self._cached_status
        
        try:
            # Use interval=None for non-blocking call (returns value since last call)
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            
            is_high = cpu_percent > self.CPU_HIGH or mem.percent > self.MEM_HIGH
            is_critical = cpu_percent > self.CPU_CRITICAL or mem.percent > self.MEM_CRITICAL
            
            # Require consecutive high readings to avoid false positives
            if is_high:
                self._consecutive_high_count += 1
            else:
                self._consecutive_high_count = 0
            
            # Only consider high if we've seen 3+ consecutive high readings
            actual_is_high = self._consecutive_high_count >= 3
            actual_is_critical = self._consecutive_high_count >= 5 and is_critical
            
            self._cached_status = {
                'cpu_percent': cpu_percent,
                'mem_percent': mem.percent,
                'mem_available_gb': mem.available / (1024**3),
                'load_avg': os.getloadavg()[0] if hasattr(os, 'getloadavg') else cpu_percent / 100,
                'is_critical': actual_is_critical,
                'is_high': actual_is_high,
            }
            self._last_check = now
        except Exception as e:
            self.logger.warning(f"Failed to get system status: {e}")
            self._cached_status = {
                'cpu_percent': 50,
                'mem_percent': 50,
                'mem_available_gb': 8,
                'load_avg': 1.0,
                'is_critical': False,
                'is_high': False,
            }
        
        return self._cached_status
    
    def get_recommended_workers(self, min_workers: int, max_workers: int) -> int:
        """
        Get recommended number of workers based on current system load.
        """
        status = self.get_status()
        
        if status['is_critical']:
            # Critical load: use minimum workers
            recommended = min_workers
            self.logger.warning(
                f"Critical system load (CPU: {status['cpu_percent']:.1f}%, "
                f"MEM: {status['mem_percent']:.1f}%), reducing to {recommended} workers"
            )
        elif status['is_high']:
            # High load: reduce workers
            recommended = max(min_workers, max_workers // 2)
            self.logger.info(
                f"High system load (CPU: {status['cpu_percent']:.1f}%, "
                f"MEM: {status['mem_percent']:.1f}%), using {recommended} workers"
            )
        else:
            # Normal load: use max workers
            recommended = max_workers
        
        return recommended
    
    def should_pause(self) -> bool:
        """
        Check if processing should pause due to resource constraints.
        """
        status = self.get_status(force=True)
        return status['is_critical']
    
    def wait_for_resources(self, max_wait: int = 60) -> bool:
        """
        Wait for system resources to become available.
        
        Returns:
            True if resources are available, False if timed out.
        """
        waited = 0
        while self.should_pause() and waited < max_wait:
            self.logger.info(f"Waiting for system resources... ({waited}s/{max_wait}s)")
            time.sleep(5)
            waited += 5
            gc.collect()  # Try to free memory
        
        return not self.should_pause()


# Global resource monitor for subprocess access
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create the global resource monitor."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor
from jinja2 import Template

from transformers import AutoProcessor
from PIL import Image, ImageOps


def _load_image_pil(path: str) -> "Image.Image":
    """Open + EXIF-transpose + RGB-convert. Caller passes PIL to HF processor,
    so smart_resize alignment (patch_size * merge_size) is auto-applied."""
    img = Image.open(path)
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ============================================================================
# Fast image-metadata path: read W/H from header without decoding pixels
# ============================================================================
# Speedup: 5-50× per image vs full PIL decode (no need to materialize RGB
# array). Combined with smart_resize replica below, lets us compute exact
# token counts byte-equal to the full-decode path.

import math as _math


def _read_image_size_fast(path: str) -> tuple[int, int]:
    """Return (visual_height, visual_width) WITHOUT decoding pixel data.

    Applies EXIF Orientation tag so the result matches what
    ``ImageOps.exif_transpose`` would visually produce. Orientations
    5/6/7/8 (rotate 90/270 with optional mirror) swap W and H; 1/2/3/4
    keep dims unchanged.

    Reads only the file header + EXIF block — no IDCT, no RGB buffer.

    Raises:
        Any PIL exception if the header cannot be parsed (HEIC without
        plugin, corrupt header, etc.). Caller should fall back to full
        decode on failure.
    """
    with Image.open(path) as img:
        w, h = img.size  # PIL convention: (W, H)
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid header size {w}x{h}")
        orient = 1
        try:
            exif = img.getexif()
            if exif is not None:
                orient = int(exif.get(0x0112, 1))  # 0x0112 = Orientation
        except Exception:
            pass
        if orient in (5, 6, 7, 8):
            w, h = h, w
        return h, w


def _qwen2vl_smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Replica of transformers Qwen2-VL ``smart_resize`` (also used by
    LLaVA-OneVision-2 image processor).

    Aligns dimensions to ``factor`` (= patch_size * merge_size), then scales
    to fit within ``[min_pixels, max_pixels]`` budget while keeping aspect
    ratio. Returns (resized_height, resized_width) — both multiples of
    ``factor`` so ``grid_h = rH // patch_size`` and ``grid_w = rW //
    patch_size`` are integer multiples of ``merge_size``.
    """
    if max(height, width) / max(min(height, width), 1) > 200:
        # Extreme aspect ratio: refuse fast path so caller falls back to
        # full decode (where the processor will raise the same way).
        raise ValueError(f"aspect ratio too extreme: {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = _math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _math.floor(height / beta / factor) * factor)
        w_bar = max(factor, _math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = _math.sqrt(min_pixels / max(height * width, 1))
        h_bar = _math.ceil(height * beta / factor) * factor
        w_bar = _math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# Global cross-process counter. Under 'fork' it is inherited by workers; under
# 'spawn' each worker starts with None and must call `_init_worker` (Pool's
# initializer hook) to receive a handle to the same shared mp.Value.
global_total_counter: Optional[Value] = None


def _init_worker(counter):
    """Pool initializer: receive shared counter handle in spawned worker."""
    global global_total_counter
    global_total_counter = counter


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute token lengths for dataset samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file:
    python s2_compute_token_lengths.py --config ./configs/s2_config.yaml

    # Using command line arguments (no config file):
    python s2_compute_token_lengths.py \\
        --data-dir /path/to/samples \\
        --output /path/to/token_info.txt \\
        --model-path /path/to/qwen2-vl \\
        --max-len 16000 \\
        --task-type sft

    # With custom processing settings:
    python s2_compute_token_lengths.py \\
        --data-dir /path/to/samples \\
        --output /path/to/token_info.txt \\
        --model-path /path/to/qwen2-vl \\
        --max-len 8000 \\
        --chunk-size 500 \\
        --max-workers 64
        """,
    )

    # Config file (optional)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (if provided, other args override config)",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="[legacy] Directory containing sample JSON files (used when --manifest is not given)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Path to a manifest.jsonl produced by s1_build_manifest.py. "
            "When given, --data-dir is ignored and absolute image/npy paths from the manifest are used."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for token lengths (format: name:length)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Temporary file for base names (default: <output_dir>/base_names.txt)",
    )

    # Model settings
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Qwen-VL processor checkpoint",
    )

    # Sample settings
    parser.add_argument(
        "--max-len",
        type=int,
        default=16000,
        help="Maximum token length threshold (default: 16000)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["sft", "pretrain"],
        default="sft",
        help="Task type: sft or pretrain (default: sft)",
    )
    parser.add_argument(
        "--del-one-token",
        action="store_true",
        default=False,
        help="Add 1 to token count for tokenizer compatibility",
    )

    # Image settings
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=3136,
        help="Minimum pixels for image resizing (default: 3136)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum pixels for image resizing (default: None, will use processor's config.json value)",
    )

    # Processing settings
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of samples per processing chunk (default: 1000)",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=20,
        help="Minimum worker threads per process (default: 20)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=96,
        help="Maximum worker threads per process (default: 96)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=0,
        help="Number of worker processes (default: 0 = auto via calculate_optimal_concurrency). "
        "When >0, overrides auto-detection. Total IO concurrency = num_processes * threads_per_process.",
    )
    parser.add_argument(
        "--stage1-chunk",
        type=int,
        default=10,
        help="Number of stage0 files to merge per batch (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Processing timeout in seconds (default: 60)",
    )

    # Logging settings
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="s2_compute_token_lengths.log",
        help="Log file path (default: s2_compute_token_lengths.log)",
    )
    parser.add_argument(
        "--use-shm",
        action="store_true",
        default=False,
        help="Use /dev/shm for temporary files (faster on Linux)",
    )
    parser.add_argument(
        "--no-npy",
        action="store_true",
        default=False,
        help="Disable NPY mode, always use image processing even if .npy files exist",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build configuration dictionary from command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Configuration dictionary compatible with TokenLengthProcessor.
    """
    # Determine output paths
    if args.output:
        output_token = args.output
        output_dir = os.path.dirname(os.path.abspath(args.output))
        output_base = args.output_base or os.path.join(output_dir, "base_names.txt")
    else:
        output_token = "token_info.txt"
        output_base = args.output_base or "base_names.txt"

    return {
        "data": {
            "directory": args.data_dir,
            "manifest": args.manifest,
            "output_base": output_base,
            "output_token": output_token,
        },
        "model": {
            "checkpoint": args.model_path,
        },
        "sample": {
            "max_len": args.max_len,
            "task_type": args.task_type,
            "del_one_token": args.del_one_token,
        },
        "image": {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "processing": {
            "chunk_size": args.chunk_size,
            "min_workers": args.min_workers,
            "max_workers": args.max_workers,
            "num_processes": args.num_processes,
            "stage1_merge_chunk": args.stage1_chunk,
            "time_out": args.timeout,
        },
        "logging": {
            "level": args.log_level,
            "file": args.log_file,
            "use_shm": args.use_shm,
        },
        "processing_mode": {
            "no_npy": args.no_npy,
        },
    }


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(
    log_level: str,
    log_file: str,
) -> logging.Logger:
    """
    Configure logging with file and console handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to the log file.

    Returns:
        Configured logger instance.
    """
    file_handler = logging.FileHandler(log_file, delay=True, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, stream_handler],
    )
    return logging.getLogger(__name__)


def count_lines(file_path: str) -> int:
    """
    Count valid lines in a file (non-empty lines containing ':').

    Args:
        file_path: Path to the file to count.

    Returns:
        Number of valid lines.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip() and ":" in line.strip())
    except Exception as e:
        logging.error(f"Failed to count lines in {file_path}: {e}")
        return 0


def find_valid_json_files(directory: Path) -> set[str]:
    """
    Find all valid JSON files in a directory.

    Args:
        directory: Directory to search.

    Returns:
        Set of base names (without extension) of JSON files.
    """
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith(".json")}
    logging.info(f"Found {len(json_set)} JSON files")
    return json_set


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """
    Load a manifest.jsonl file into a dict keyed by ``name``.

    Each manifest line is expected to have the schema produced by
    ``s1_build_manifest.py``::

        {"name": ..., "raw": {...}, "images_abs": [...], "patch_pos_abs": [...], ...}

    Args:
        manifest_path: Path to manifest.jsonl.

    Returns:
        Dict mapping sample name to its manifest entry.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    entries: dict[str, dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning(f"manifest line {ln_no} is not valid JSON: {exc}")
                continue
            name = rec.get("name")
            if not name:
                logging.warning(f"manifest line {ln_no} missing 'name'; skipping")
                continue
            entries[name] = rec
    logging.info(f"Loaded {len(entries)} entries from manifest: {manifest_path}")
    return entries



def find_paired_files(directory: Path) -> set[str]:
    """
    Find files that have both JSON and image counterparts.

    Args:
        directory: Directory to search.

    Returns:
        Set of base names that have both JSON and image files.
    """
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith(".json")}
    img_set = {f[:-4] for f in files if f.lower().endswith((".jpg", ".jpeg"))}
    paired = json_set & img_set
    logging.info(f"Found {len(paired)} paired files")
    return paired


def write_base_names_to_file(base_names: set[str], output_file: Path) -> None:
    """
    Write sorted base names to a file.

    Args:
        base_names: Set of base names to write.
        output_file: Path to output file.
    """
    try:
        content = "\n".join(sorted(base_names)) + "\n"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Wrote {len(base_names)} base names to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write to {output_file}: {e}")
        raise


def read_lines_in_chunks(
    file_path: Path,
    chunk_size: int,
) -> Generator[list[str], None, None]:
    """
    Read file contents in chunks.

    Args:
        file_path: Path to the file to read.
        chunk_size: Number of lines per chunk.

    Yields:
        List of lines for each chunk.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = [line.strip() for _, line in zip(range(chunk_size), f) if line.strip()]
            if not chunk:
                break
            logging.info(f"Read chunk with {len(chunk)} samples")
            yield chunk


def get_adaptive_workers(
    min_workers: int = 20,
    max_workers: int = 96,
) -> int:
    """
    Adjust thread count based on system load using ResourceMonitor.

    Args:
        min_workers: Minimum number of workers.
        max_workers: Maximum number of workers.

    Returns:
        Adjusted number of workers.
    """
    return get_resource_monitor().get_recommended_workers(min_workers, max_workers)


def calculate_optimal_concurrency(total_samples: int, chunk_size: int) -> tuple[int, int]:
    """
    Calculate optimal number of processes and threads based on system resources.
    
    Args:
        total_samples: Total number of samples to process.
        chunk_size: Number of samples per chunk.
    
    Returns:
        Tuple of (n_processes, n_threads_per_process)
    """
    cpu_count = multiprocessing.cpu_count()
    mem_gb = psutil.virtual_memory().total / (1024**3)

    # Estimate memory per process (processor + PIL decoded images + tensor cache).
    # Empirically: HF processor with longest_edge=3072 -> ~10-12 GB RSS per worker
    # under sustained load. Earlier value of 3 GB caused 60x worker fork on 256 GB
    # machines, OOM and stall.
    mem_per_process_gb = int(os.environ.get("S2_MEM_PER_PROC_GB", "12"))
    max_processes_by_mem = max(1, int(mem_gb * 0.7 / mem_per_process_gb))

    # Hard upper cap to avoid fork storms regardless of CPU count.
    hard_cap = int(os.environ.get("S2_MAX_PROCESSES", "16"))

    # Limit processes to CPU count, memory constraint, and hard cap.
    n_processes = min(cpu_count, max_processes_by_mem, hard_cap)

    # Calculate threads per process
    # Rule: processes * threads should not exceed 2 * cpu_count for I/O bound tasks
    total_threads_budget = cpu_count * 2
    n_threads = max(4, total_threads_budget // n_processes)

    # Cap threads to avoid excessive context switching and tensor memory peak.
    n_threads = min(n_threads, int(os.environ.get("S2_MAX_THREADS_PER_PROC", "8")))

    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    n_processes = min(n_processes, total_chunks)
    
    logging.info(
        f"Optimal concurrency: {n_processes} processes x {n_threads} threads "
        f"(CPU: {cpu_count}, RAM: {mem_gb:.1f}GB)"
    )
    
    return n_processes, n_threads


class TokenLengthProcessor:
    """Processor for computing token lengths of samples."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger):
        """
        Initialize the token length processor.

        Args:
            config: Configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        self.processor = None

        # Extract configuration values
        self.max_token_len = config["sample"]["max_len"]
        self.task_type = config["sample"]["task_type"]
        self.del_one_token = config["sample"]["del_one_token"]

        # Manifest mode (preferred) or legacy data-dir mode
        manifest_path = config["data"].get("manifest")
        self.manifest_path: Optional[Path] = Path(manifest_path) if manifest_path else None
        self.use_manifest: bool = self.manifest_path is not None

        if self.use_manifest:
            # In manifest mode, data_directory is unused. Set to manifest's parent for compat.
            self.data_directory = self.manifest_path.parent
        else:
            data_dir = config["data"].get("directory")
            if not data_dir:
                raise ValueError("Either 'manifest' or 'directory' must be provided in config.data")
            self.data_directory = Path(data_dir)

        self.output_file = Path(config["data"]["output_base"])
        self.token_info_file = Path(config["data"]["output_token"])

        self.ckpt_dir = config["model"]["checkpoint"]
        self.min_pixels = config["image"]["min_pixels"]
        self.max_pixels = config["image"]["max_pixels"]

        self.timeout = config["processing"]["time_out"]
        self.stage1_chunk = config["processing"]["stage1_merge_chunk"]
        self.chunk_size = config["processing"]["chunk_size"]
        self.min_workers = config["processing"]["min_workers"]
        self.max_workers = config["processing"]["max_workers"]
        self.num_processes = config["processing"].get("num_processes", 0)

        self.use_shm = config["logging"]["use_shm"]
        self.temp_dir = "/dev/shm" if self.use_shm else None

        # NPY mode control
        self.no_npy = config.get("processing_mode", {}).get("no_npy", False)

        # Persistent stage0 directory for resume-on-restart.
        # Lives next to token_info.txt so the same output dir owns all artifacts.
        # Worker writes `stage0_chunk_{idx:05d}.txt` atomically (via tmp+rename),
        # main() scans existing ones on startup and skips them.
        self.resume_dir = self.token_info_file.parent / "stage0_persistent"
        self.resume_dir.mkdir(parents=True, exist_ok=True)

        # Initialize chat template based on task type
        self._init_template()

    def _init_template(self) -> None:
        """Initialize Jinja2 template based on task type."""
        if self.task_type == "pretrain":
            self.template = Template("<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>")
        elif self.task_type == "sft":
            chat_template = (
                "{% set image_count = namespace(value=0) %}"
                "{% set video_count = namespace(value=0) %}"
                "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] | replace('<image>', "
                "'<|vision_start|><|image_pad|><|vision_end|>') }}"
                "<|im_end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
            self.template = Template(chat_template)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def process_sample(
        self,
        json_path: str,
        img_path: str,
        processor: AutoProcessor,
    ) -> tuple[Optional[int], str]:
        """
        Process a single sample and compute token length.

        Args:
            json_path: Path to the JSON file.
            img_path: Path to the image file.
            processor: Hugging Face processor instance.

        Returns:
            Tuple of (token_length, base_name) or (None, error_message).
        """
        try:
            if not Path(json_path).exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            # Load JSON data
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Render text input based on task type
            if self.task_type == "pretrain":
                txt_input = self.template.render(captions=json_data["captions"])
            elif self.task_type == "sft":
                # Ensure template can find messages field
                if "conversations" in json_data and "messages" not in json_data:
                    json_data["messages"] = json_data["conversations"]
                txt_input = self.template.render(
                    json_data,
                    tokenize=False,
                    add_generation_prompt=False,
                )

            # Process image(s)
            if img_path == "_____.jpg":
                img_input = None
            else:
                if len(json_data.get("images", [])) > 1:
                    # Multi-image sample
                    img_input = []
                    for cur_img in json_data["images"]:
                        cur_img_path = str(self.data_directory / cur_img)
                        img_input.append(_load_image_pil(cur_img_path))
                else:
                    # Single image sample
                    img_input = _load_image_pil(img_path)

            # Compute token length
            base_name = Path(json_path).stem
            inputs = processor(
                text=[txt_input],
                images=img_input,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            return (inputs["input_ids"].shape[1], base_name)

        except Exception as e:
            return (None, f"Processing failed [{Path(json_path).stem}]: {e}")

    def merge_files_by_token(
        self,
        input_files: list[str],
        output_file: str,
    ) -> tuple[Optional[str], int]:
        """
        Merge multiple sorted files by token length in ascending order.

        Filters out samples with token length exceeding max_token_len.

        Args:
            input_files: List of input file paths.
            output_file: Path to output file.

        Returns:
            Tuple of (output_path, line_count) or (None, 0) on failure.
        """
        if not input_files:
            self.logger.warning("No files to merge")
            return (None, 0)

        # Validate input files and count total lines
        valid_files = []
        total_lines = 0
        for f in input_files:
            line_count = count_lines(f)
            if line_count > 0:
                valid_files.append(f)
                total_lines += line_count
                self.logger.debug(f"File {os.path.basename(f)} contains {line_count} lines")
            else:
                self.logger.warning(f"File {os.path.basename(f)} is empty or invalid, skipping")

        if not valid_files:
            return (None, 0)

        def sort_key(line: str) -> int:
            """Extract token length for sorting. Line format: name:token_len[:patch_count]."""
            parts = line.strip().split(":")
            return int(parts[1])

        try:
            filtered_count = 0
            with open(output_file, "w", encoding="utf-8") as out_f:
                # Create iterators for all files
                iterators = []
                file_handles = []

                for fpath in valid_files:
                    try:
                        fh = open(fpath, "r", encoding="utf-8")
                        file_handles.append(fh)
                        iterators.append((sort_key(line), line) for line in fh)
                    except Exception as e:
                        self.logger.error(f"Failed to open {os.path.basename(fpath)}: {e}")

                # Merge sort and filter by max token length
                for _, line in merge(*iterators, key=lambda x: x[0]):
                    parts = line.strip().split(":")
                    token_val = int(parts[1])
                    if token_val <= self.max_token_len:
                        out_f.write(line)
                    else:
                        self.logger.debug(f"Token length {token_val} > {self.max_token_len}: filtered")
                        filtered_count += 1

                # Close all file handles
                for fh in file_handles:
                    try:
                        fh.close()
                    except Exception as e:
                        self.logger.warning(f"Failed to close {fh.name}: {e}")

            # Verify output file integrity
            output_lines = count_lines(output_file) + filtered_count
            if output_lines != total_lines:
                self.logger.error(f"Data loss during merge! Input: {total_lines}, Output: {output_lines}")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return (None, 0)

            final_count = count_lines(output_file)
            self.logger.info(
                f"Merge successful: {total_lines} input, {final_count} output (token <= {self.max_token_len})"
            )
            return (output_file, final_count)

        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except Exception:
                    pass
            return (None, 0)


def stage1_merger(
    input_queue: multiprocessing.Queue,
    chunk_size: int,
    stage1_files: list[str],
    stop_event: multiprocessing.Event,
    processor: TokenLengthProcessor,
) -> None:
    """
    Stage1 merger thread that batches and merges stage0 files.

    Args:
        input_queue: Queue containing stage0 file paths.
        chunk_size: Number of stage0 files to merge in each batch.
        stage1_files: List to append merged file paths to.
        stop_event: Event to signal thread termination.
        processor: TokenLengthProcessor instance for merging.
    """
    buffer = []
    batch_counter = 0
    logger = logging.getLogger(__name__)
    logger.info(f"Stage1 merger thread started, merging every {chunk_size} files")

    try:
        while (not input_queue.empty()) or buffer or (not stop_event.is_set()):
            if not input_queue.empty():
                try:
                    file_path = input_queue.get(timeout=1)
                    buffer.append(file_path)
                    input_queue.task_done()
                    logger.debug(f"Stage1 received {os.path.basename(file_path)}, buffer: {len(buffer)}/{chunk_size}")

                    # Merge when buffer is full
                    if len(buffer) >= chunk_size:
                        batch_counter += 1
                        merged_file = tempfile.NamedTemporaryFile(
                            mode="w",
                            delete=False,
                            prefix=f"stage1_batch{batch_counter:03d}_",
                            encoding="utf-8",
                            dir=processor.temp_dir,
                        ).name

                        merged_path, line_count = processor.merge_files_by_token(buffer, merged_file)
                        if merged_path and line_count > 0:
                            stage1_files.append(merged_path)
                            logger.info(
                                f"Stage1 batch {batch_counter} complete: "
                                f"{os.path.basename(merged_path)}, {line_count} lines "
                                f"(merged {len(buffer)} files)"
                            )
                        else:
                            logger.warning(f"Stage1 batch {batch_counter} merge failed")
                        buffer = []

                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Stage1 processing error: {e}", exc_info=True)
            else:
                # Force merge remaining files when stop signal received
                if buffer and stop_event.is_set():
                    batch_counter += 1
                    merged_file = tempfile.NamedTemporaryFile(
                        mode="w",
                        delete=False,
                        prefix=f"stage1_remaining_batch{batch_counter:03d}_",
                        encoding="utf-8",
                        dir=processor.temp_dir,
                    ).name

                    merged_path, line_count = processor.merge_files_by_token(buffer, merged_file)
                    if merged_path and line_count > 0:
                        stage1_files.append(merged_path)
                        logger.info(
                            f"Stage1 remaining files merged: {os.path.basename(merged_path)}, {line_count} lines"
                        )
                    else:
                        logger.warning("Stage1 remaining files merge failed")
                    buffer = []
                else:
                    threading.Event().wait(0.5)

        if buffer:
            logger.error(f"Stage1 thread exiting with {len(buffer)} unprocessed files!")

    except Exception as e:
        logger.error(f"Stage1 thread exception: {e}", exc_info=True)
    finally:
        logger.info(f"Stage1 thread exiting, generated {len(stage1_files)} files")


def process_chunk(args: tuple) -> Optional[str]:
    """
    Process a single chunk of samples in a subprocess.

    Uses a thread pool within the subprocess for parallel processing.

    Args:
        args: Tuple containing chunk data, processor config, and queue.

    Returns:
        Path to generated stage0 file, or None on failure.
    """
    global global_total_counter

    (
        chunk_idx,
        chunk,
        ckpt_dir,
        min_pixels,
        max_pixels,
        stage0_queue,
        data_directory,
        task_type,
        del_one_token,
        min_workers,
        max_workers,
        temp_dir,
        sample_timeout,  # Timeout per sample in seconds
        no_npy,  # Disable NPY mode flag
        resume_dir,  # Persistent stage0 directory (str path) for resume-on-restart
    ) = args

    processor = None
    processed_count = 0
    logger = logging.getLogger(__name__)

    # libc handle for malloc_trim — forces glibc to return free pages to OS,
    # avoiding monotonic RSS growth in long-lived workers.
    try:
        import ctypes as _ctypes
        _libc = _ctypes.CDLL("libc.so.6")
    except Exception:
        _libc = None

    try:
        # FAST_TOKENIZER=1 (default 0 for safety): skip AutoProcessor entirely.
        # AutoProcessor pulls image_processor + tokenizer + chat_template +
        # builds tensors per call, costing ~1.5-2 GB baseline RSS per worker.
        # With fast tokenizer we only need:
        #   * tokenizer.encode(text)        → text token count
        #   * _read_image_size_fast(path)   → header-only W/H
        #   * _qwen2vl_smart_resize(...)    → grid_h/w
        # Cuts worker baseline to ~700 MB, enables 2-3x more parallelism.
        use_fast_tokenizer = os.environ.get("FAST_TOKENIZER", "0") == "1"

        # Initialize processor (each subprocess needs its own instance).
        # Pass min/max_pixels ONLY when explicitly set; otherwise honor processor_config.json.
        ap_kwargs = dict(trust_remote_code=True, use_fast=False)
        if min_pixels is not None:
            ap_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            ap_kwargs["max_pixels"] = max_pixels

        tokenizer = None
        image_processor = None
        if use_fast_tokenizer:
            # Need image_processor to read patch_size / merge_size / min/max pixels.
            # AutoImageProcessor is lightweight (no tokenizer, no tensors).
            from transformers import AutoTokenizer, AutoImageProcessor
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
            image_processor = AutoImageProcessor.from_pretrained(
                ckpt_dir, trust_remote_code=True, use_fast=False,
            )
            processor = None  # not used in this path
        else:
            processor = AutoProcessor.from_pretrained(ckpt_dir, **ap_kwargs)
            image_processor = getattr(processor, "image_processor", None)
            tokenizer = getattr(processor, "tokenizer", None)
        merge_size = getattr(image_processor, "merge_size", None) if image_processor is not None else None
        patch_size = getattr(image_processor, "patch_size", None) if image_processor is not None else None
        if merge_size is None:
            merge_size = 2
            logger.warning(
                f"merge_size not found on processor.image_processor; defaulting to {merge_size}. "
                f"This will give WRONG NPY token counts for models with merge_size != 2."
            )
        merge_factor = int(merge_size) ** 2

        # Fast-header-path parameters (read from processor; fall back gracefully)
        ip_size = getattr(image_processor, "size", None) or {}
        ip_min_pixels = (
            getattr(image_processor, "min_pixels", None)
            or (ip_size.get("shortest_edge") if isinstance(ip_size, dict) else None)
            or 3136
        )
        ip_max_pixels = (
            getattr(image_processor, "max_pixels", None)
            or (ip_size.get("longest_edge") if isinstance(ip_size, dict) else None)
            or 12845056
        )
        _factor = int(patch_size or 16) * int(merge_size)
        # FAST_IMG_META=1 (default) → read W/H from header, skip pixel decode.
        # Set FAST_IMG_META=0 to force legacy full-decode path (for A/B testing).
        use_fast_img_meta = os.environ.get("FAST_IMG_META", "1") == "1"

        # Lightweight text token counter. Uses tokenizer.encode (just an int list
        # of token IDs) when FAST_TOKENIZER=1; falls back to processor(text=[...])
        # which materializes input_ids tensor + attention_mask + padding.
        def _count_text_tokens(txt: str) -> int:
            if use_fast_tokenizer:
                return len(tokenizer.encode(txt, add_special_tokens=False))
            out = processor(
                text=[txt], images=None, videos=None,
                padding=True, return_tensors="pt",
            )
            n = out["input_ids"].shape[1]
            del out
            return n

        logger.info(
            f"Processor loaded: patch_size={patch_size}, merge_size={merge_size}, "
            f"factor={(patch_size or 0) * int(merge_size)}, merge_factor={merge_factor}, "
            f"effective min_pixels={getattr(image_processor, 'min_pixels', None)}, "
            f"max_pixels={getattr(image_processor, 'max_pixels', None)}, "
            f"fast_img_meta={use_fast_img_meta}, fast_tokenizer={use_fast_tokenizer}"
        )

        # Initialize template based on task type
        if task_type == "pretrain":
            template = Template("<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>")
        else:
            chat_template = (
                "{% set image_count = namespace(value=0) %}"
                "{% set video_count = namespace(value=0) %}"
                "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] | replace('<image>', "
                "'<|vision_start|><|image_pad|><|vision_end|>') }}"
                "<|im_end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
            template = Template(chat_template)

        data_directory = Path(data_directory)

        # Generate file paths for current chunk.
        # `chunk` is either:
        #   * a list of base-name strings (legacy --data-dir mode), or
        #   * a list of manifest-entry dicts (new --manifest mode).
        # Each entry is normalized to (json_path_or_None, raw_data_or_None,
        # img_path, npy_path, base_name).
        full_paths: list[tuple[Optional[str], Optional[dict], str, str, str]] = []
        manifest_mode = bool(chunk) and isinstance(chunk[0], dict)
        for entry in chunk:
            if manifest_mode:
                base_name = entry["name"]
                images_abs = entry.get("images_abs") or []
                patch_pos_abs = entry.get("patch_pos_abs") or []
                raw_data = dict(entry.get("raw", {}))  # shallow copy; we annotate it

                # Stash the absolute image list on the raw dict so the worker
                # can fetch multi-image samples without needing data_directory.
                if images_abs:
                    raw_data["__manifest_images_abs__"] = images_abs

                # Pick the npy path of the first image if present
                first_npy = patch_pos_abs[0] if patch_pos_abs and patch_pos_abs[0] else None

                if not no_npy and first_npy and os.path.exists(first_npy):
                    cur_img = "_____.jpg"
                    cur_npy = first_npy
                elif images_abs:
                    cur_img = images_abs[0]
                    cur_npy = "_____.npy"
                else:
                    cur_img = "_____.jpg"
                    cur_npy = "_____.npy"

                full_paths.append((None, raw_data, cur_img, cur_npy, base_name))
                continue

            # Legacy data-dir mode
            fn = entry
            cur_json = str(data_directory / f"{fn}.json")
            cur_npy = str(data_directory / f"{fn}.npy")

            # Check if npy file exists (and NPY mode is enabled)
            if not no_npy and os.path.exists(cur_npy):
                # NPY mode: has npy file
                cur_img = "_____.jpg"
                full_paths.append((cur_json, None, cur_img, cur_npy, fn))
            else:
                # Standard mode: process image
                if f"{fn}.json".startswith("__img--output_"):
                    cur_img = "_____.jpg"
                else:
                    with open(cur_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        images = data.get("images", [])
                        if images and len(images) > 0:
                            cur_img = images[0]
                            cur_img = str(data_directory / cur_img)
                        else:
                            # Text-only sample without images
                            cur_img = "_____.jpg"
                full_paths.append((cur_json, None, cur_img, "_____.npy", fn))

        n_samples = len(chunk)
        proc_name = multiprocessing.current_process().name
        logger.info(f"Process {proc_name} starting chunk {chunk_idx} with {n_samples} samples")

        # Create thread pool within subprocess
        n_workers = get_adaptive_workers(min_workers=min_workers, max_workers=max_workers)
        chunk_results = []

        def process_single_sample(
            json_path: Optional[str],
            raw_data: Optional[dict],
            img_path: str,
            npy_path: Optional[str] = None,
            base_name: str = "",
            sample_timeout: int = 120,
        ) -> tuple[Optional[int], str]:
            """
            Process a single sample within thread pool with timeout protection.

            Supports two modes:
            1. Standard mode: Process image and text using processor
            2. NPY mode: If npy_path is provided, calculate patch tokens directly
               from npy file's first dimension (patches = npy_dim0 // merge_size**2)

            Args:
                json_path: Path to JSON file. May be None when ``raw_data`` is
                    supplied (manifest mode).
                raw_data: Pre-loaded sample dict (manifest mode). Mutually
                    exclusive with ``json_path``: at least one must be set.
                img_path: Path to image file.
                npy_path: Path to NPY file (optional). If provided, uses NPY mode.
                base_name: The sample's globally unique name (used as the result key).
                sample_timeout: Timeout in seconds for processing this sample.
            """
            if not base_name:
                base_name = Path(json_path).stem if json_path else "unknown"

            try:
                if raw_data is not None:
                    json_data = raw_data
                else:
                    if not Path(json_path).exists():
                        raise FileNotFoundError(f"JSON file not found: {json_path}")
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                if task_type == "pretrain":
                    txt_input = template.render(captions=json_data["captions"])
                else:
                    if "conversations" in json_data and "messages" not in json_data:
                        json_data["messages"] = json_data["conversations"]
                    txt_input = template.render(
                        json_data,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                # NPY mode: Calculate token length from npy file directly
                if npy_path and npy_path != "_____.npy" and os.path.exists(npy_path):
                    try:
                        npy_data = np.load(npy_path, allow_pickle=False)
                        npy_first_dim = npy_data.shape[0]

                        patch_tokens = npy_first_dim // merge_factor

                        # Count image placeholders in text
                        n_image_placeholders = txt_input.count("<|image_pad|>")

                        # Compute text tokens (lightweight path when FAST_TOKENIZER=1)
                        text_tokens = _count_text_tokens(txt_input)

                        token_len = text_tokens - n_image_placeholders + patch_tokens
                        # --- TS-FIX: training (_rewrap_vision_by_frame) rewrites video to per-frame
                        # "<t seconds> VS PAD*k VE \n"; s2's per-canvas template misses the per-frame
                        # timestamp + vision_start + vision_end + newline. Add that so the pack budget
                        # matches real training tokens (else long-video bins overflow the CAP ~10%).
                        # measured token cost of "<D.d seconds>" = 5 + int_digits(D); +VS+VE+\n = +3;
                        # minus the 1 VS+VE the template already counted for the whole visual.
                        if npy_data.ndim == 2 and npy_data.shape[1] >= 3:
                            _fidx = np.unique(npy_data[:, 0])
                            if _fidx.size > 1:
                                # NATIVE-fps timestamps: npy col0 = native frame numbers; training uses
                                # timestamp_sec = frame_idx / native_fps, native_fps = max(frame_idx)/dur_s
                                # => timestamp_sec = col0 * dur_s / max(col0). (dur_s from dyn.) The /fpsN/
                                # path value is the master SAMPLING fps (not native) -> do NOT use it.
                                _dur = None
                                _dyn = json_data.get("dyn")
                                if isinstance(_dyn, list) and _dyn:
                                    _dur = _dyn[0].get("dur_s")
                                elif isinstance(_dyn, dict):
                                    _dur = _dyn.get("dur_s")
                                _maxc = float(_fidx.max())
                                _extra = 0
                                for _fi in _fidx.tolist():
                                    if _dur and _maxc > 0:
                                        _sec = float(_fi) * float(_dur) / _maxc
                                        _ts_tok = 5 + len(str(int(_sec)))
                                    else:
                                        _ts_tok = 8  # fallback if dur_s missing (typical mid-length video)
                                    _extra += _ts_tok + 3
                                token_len += _extra - 2
                        # patch_count for ViT load balancing = raw patches (before spatial merge)
                        patch_count = int(npy_first_dim)

                        # Clean up
                        del npy_data

                        return (token_len, base_name, patch_count)
                    except Exception as e:
                        raise RuntimeError(f"NPY processing failed: {e}")

                # Standard mode: Process image(s) and text
                if img_path == "_____.jpg":
                    img_input = None
                    # Text-only sample: no images to process.
                    # Under FAST_TOKENIZER=1 we never call processor(), so return now.
                    if use_fast_tokenizer:
                        token_len = _count_text_tokens(txt_input)
                        return (token_len, base_name, 0)
                else:
                    # Manifest mode supplies absolute paths via ``images_abs``;
                    # legacy mode stores relative names in json_data["images"].
                    manifest_imgs_abs = json_data.get("__manifest_images_abs__")
                    if len(json_data.get("images", [])) > 1 or (
                        manifest_imgs_abs and len(manifest_imgs_abs) > 1
                    ):
                        if manifest_imgs_abs:
                            img_paths_list = manifest_imgs_abs
                        else:
                            img_paths_list = [
                                str(data_directory / p) for p in json_data["images"]
                            ]
                    else:
                        img_paths_list = [img_path]

                    # --- Fast header path -------------------------------------
                    # Try to compute patch_count from JPEG/PNG headers without
                    # decoding pixels. Falls back to full PIL decode below on
                    # any error (unsupported format, corrupt header, etc.).
                    if use_fast_img_meta:
                        try:
                            n_image_placeholders = txt_input.count("<|image_pad|>")
                            text_tokens = _count_text_tokens(txt_input)

                            patch_count_total = 0
                            grid_tokens_total = 0  # in merge-units (= LLM image tokens)
                            for p in img_paths_list:
                                h, w = _read_image_size_fast(p)
                                rh, rw = _qwen2vl_smart_resize(
                                    h, w, _factor, ip_min_pixels, ip_max_pixels,
                                )
                                gh = rh // int(patch_size)
                                gw = rw // int(patch_size)
                                # raw patches before spatial merge:
                                patch_count_total += gh * gw
                                # merge-unit tokens (what LLM sees):
                                grid_tokens_total += (gh * gw) // merge_factor

                            token_len = text_tokens - n_image_placeholders + grid_tokens_total
                            return (token_len, base_name, patch_count_total)
                        except Exception:
                            # Fall through to legacy full-decode path.
                            pass

                    # --- Legacy full-decode path ------------------------------
                    # NOTE: Not supported when FAST_TOKENIZER=1 (no processor).
                    # That path requires processor() which we deliberately skip
                    # for memory efficiency. The fast_img_meta path above MUST
                    # succeed for these samples; if it raises we re-raise so
                    # the sample is counted as failed instead of silently using
                    # a wrong code branch.
                    if use_fast_tokenizer:
                        raise RuntimeError(
                            f"FAST_TOKENIZER=1 but fast_img_meta path failed for "
                            f"{img_paths_list}; processor() unavailable"
                        )
                    if len(img_paths_list) > 1:
                        img_input = [_load_image_pil(p) for p in img_paths_list]
                    else:
                        img_input = _load_image_pil(img_paths_list[0])

                inputs = processor(
                    text=[txt_input],
                    images=img_input,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                token_len = inputs["input_ids"].shape[1]

                # patch_count = sum of raw image patches across all images in this sample.
                # image_grid_thw is (n_images, 3) — already in merge-units, so multiply by merge_factor.
                patch_count = 0
                if img_input is not None and "image_grid_thw" in inputs:
                    igt = inputs["image_grid_thw"]
                    try:
                        patch_count = int(igt.prod(dim=1).sum().item()) * merge_factor
                    except Exception:
                        patch_count = 0

                # Clean up to free memory
                del inputs
                if img_input is not None:
                    del img_input

                return (token_len, base_name, patch_count)

            except TimeoutException:
                return (None, f"Timeout [{base_name}]: exceeded {sample_timeout}s", 0)
            except MemoryError:
                gc.collect()
                return (None, f"MemoryError [{base_name}]: insufficient memory", 0)
            except Exception as e:
                return (None, f"Processing failed [{base_name}]: {e}", 0)

        # Adaptive timeout based on sample complexity (use config value)
        # sample_timeout is passed from main process via args
        
        # Batch processing for memory efficiency
        batch_size = min(n_workers * 2, n_samples)  # Process in smaller batches
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            
            # Calculate reasonable timeout for this batch
            batch_timeout = sample_timeout * current_batch_size / n_workers + 120
            
            with ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix=f"proc-{multiprocessing.current_process().pid}-thread",
            ) as executor:
                tasks = {
                    executor.submit(
                        process_single_sample,
                        full_paths[idx][0],  # json_path or None
                        full_paths[idx][1],  # raw_data or None
                        full_paths[idx][2],  # img_path
                        full_paths[idx][3],  # npy_path
                        full_paths[idx][4],  # base_name
                        sample_timeout,
                    ): idx
                    for idx in range(batch_start, batch_end)
                }

                # Use timeout for as_completed to prevent hanging
                completed_in_batch = 0
                try:
                    for future in as_completed(tasks, timeout=batch_timeout):
                        try:
                            token_len, name, patch_count = future.result(timeout=sample_timeout)
                            if del_one_token:
                                token_len = token_len + 1 if token_len else None
                            if token_len is not None:
                                chunk_results.append((token_len, name, patch_count))
                                processed_count += 1
                            else:
                                logger.warning(name)
                            completed_in_batch += 1
                        except TimeoutError:
                            idx = tasks[future]
                            logger.error(f"Sample {idx} timed out after {sample_timeout}s")
                        except Exception as e:
                            logger.error(f"Thread task error: {e}")
                except TimeoutError:
                    logger.error(
                        f"Batch {batch_start//batch_size + 1} timeout after {batch_timeout:.0f}s, "
                        f"completed {completed_in_batch}/{current_batch_size}"
                    )
                    # Cancel remaining futures
                    for future in tasks:
                        if not future.done():
                            future.cancel()
            
            # Periodic garbage collection + malloc_trim to release pages to OS.
            # Without malloc_trim, glibc holds onto freed memory in the arena
            # and worker RSS grows monotonically (12+ GB observed).
            gc.collect()
            if _libc is not None:
                try:
                    _libc.malloc_trim(0)
                except Exception:
                    pass

        # Write stage0 file and add to queue.
        # Persistent path: {resume_dir}/stage0_chunk_{idx:05d}.txt
        # Written atomically (tmp + rename) so a partial file from a crashed
        # worker is never picked up by the main process on resume.
        if chunk_results:
            chunk_results_sorted = sorted(chunk_results, key=lambda x: x[0])
            final_name = f"stage0_chunk_{chunk_idx:05d}.txt"
            stage0_file = os.path.join(resume_dir, final_name)
            tmp_file = stage0_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                for token_len, name, patch_count in chunk_results_sorted:
                    f.write(f"{name}:{token_len}:{patch_count}\n")
            os.replace(tmp_file, stage0_file)

            stage0_queue.put(stage0_file)

            status = "🟢" if processed_count == n_samples else "🟡"
            logger.info(
                f"{status} Process {proc_name} completed chunk {chunk_idx}, "
                f"valid samples: {processed_count}/{n_samples}"
            )

            # Atomic increment of global counter
            with global_total_counter.get_lock():
                global_total_counter.value += processed_count

            return stage0_file
        else:
            logger.warning(f"Process {proc_name} chunk {chunk_idx}: no valid results!")
            return None

    except Exception as e:
        import traceback

        error_msg = (
            f"Process {multiprocessing.current_process().name} chunk {chunk_idx} "
            f"failed with error: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        # Also print to stderr for visibility
        print(error_msg, file=sys.stderr, flush=True)
        return None
    finally:
        if processor:
            del processor

    return None


def main() -> None:
    """Main entry point for token length computation."""
    global global_total_counter

    # Parse arguments
    args = parse_arguments()

    # Load configuration from file or build from args
    if args.config:
        config_path = Path(args.config)
        config = load_config(config_path)
        # Override with command line arguments if provided
        if args.log_level:
            config["logging"]["level"] = args.log_level.upper()
        if args.data_dir:
            config["data"]["directory"] = args.data_dir
        if args.manifest:
            config["data"]["manifest"] = args.manifest
        if args.output:
            config["data"]["output_token"] = args.output
        if args.model_path:
            config["model"]["checkpoint"] = args.model_path
        if args.max_len != 16000:
            config["sample"]["max_len"] = args.max_len
        if args.task_type != "sft":
            config["sample"]["task_type"] = args.task_type
    else:
        # Build config entirely from command line arguments
        if not args.data_dir and not args.manifest:
            print("Error: either --manifest or --data-dir is required when not using --config")
            sys.exit(1)
        if not args.output:
            print("Error: --output is required when not using --config")
            sys.exit(1)
        if not args.model_path:
            print("Error: --model-path is required when not using --config")
            sys.exit(1)
        config = build_config_from_args(args)

    log_level = config["logging"]["level"]

    # Setup logging
    logger = setup_logging(log_level, config["logging"]["file"])

    # Initialize processor configuration
    processor_config = TokenLengthProcessor(config, logger)

    stage0_files: list[str] = []
    stage1_files: list[str] = []

    try:
        logger.info("=" * 60)
        logger.info("Starting token length computation pipeline")
        logger.info("=" * 60)

        # Step 1: Discover samples (manifest mode preferred, legacy data-dir fallback)
        manifest_entries: dict[str, dict[str, Any]] = {}
        if processor_config.use_manifest:
            manifest_entries = load_manifest(processor_config.manifest_path)
            base_names = set(manifest_entries.keys())
        else:
            base_names = find_valid_json_files(processor_config.data_directory)
        total_original = len(base_names)
        logger.info(f"Found {total_original} original sample files")

        if total_original == 0:
            logger.warning("No samples found, exiting")
            return

        # Write base names for chunked reading (kept for legacy compatibility / debugging)
        write_base_names_to_file(base_names, processor_config.output_file)

        # Step 2: Initialize cross-process queue and event.
        # Use 'spawn' context (vs default 'fork'): each worker starts from a
        # fresh Python interpreter, importing transformers + tokenizer from
        # scratch (~700 MB baseline). With fork, every worker inherits parent's
        # ~10 GB RSS via copy-on-write, but as soon as Python allocates new
        # objects the pages become private (~5-6 GB private dirty per worker).
        # Spawn caps each worker's RSS at ~3 GB sustained, enabling ~24+
        # workers on a 256 GB box without OOM.
        use_spawn = os.environ.get("S2_USE_SPAWN", "1") == "1"
        if use_spawn:
            mp_ctx = multiprocessing.get_context("spawn")
            logger.info("Multiprocessing context: spawn (memory-efficient)")
        else:
            mp_ctx = multiprocessing.get_context("fork")
            logger.info("Multiprocessing context: fork (legacy)")
        manager = mp_ctx.Manager()
        stage0_queue = manager.Queue()
        stop_event = manager.Event()

        # Initialize global counter (must be ctx-bound for spawn)
        global_total_counter = mp_ctx.Value("i", 0)

        # Step 3: Start stage1 merger thread
        stage1_thread = threading.Thread(
            target=stage1_merger,
            args=(
                stage0_queue,
                processor_config.stage1_chunk,
                stage1_files,
                stop_event,
                processor_config,
            ),
            daemon=True,
        )
        stage1_thread.start()
        logger.info("Stage1 merger thread started")

        # Step 4: Prepare chunks for multiprocessing.
        # In manifest mode, each chunk is a list of manifest-entry dicts.
        # In legacy mode, each chunk is a list of base-name strings.
        if processor_config.use_manifest:
            entries_list = list(manifest_entries.values())
            cs = processor_config.chunk_size
            all_chunks = [entries_list[i:i + cs] for i in range(0, len(entries_list), cs)]
        else:
            all_chunks = list(
                read_lines_in_chunks(
                    processor_config.output_file,
                    processor_config.chunk_size,
                )
            )
        total_chunks = len(all_chunks)
        logger.info(f"Divided into {total_chunks} chunks")

        # Resume-on-restart: scan persistent stage0 directory and identify
        # chunks already completed in a previous run. Their files are pushed
        # straight into stage0_queue (so stage1 merger picks them up), and the
        # process_args list filters them out so workers don't redo them.
        resume_dir_path = processor_config.resume_dir
        done_idxs: set[int] = set()
        try:
            for fname in os.listdir(resume_dir_path):
                if not fname.startswith("stage0_chunk_") or not fname.endswith(".txt"):
                    continue
                try:
                    idx = int(fname[len("stage0_chunk_"):-len(".txt")])
                except ValueError:
                    continue
                # Sanity: file must be non-empty and look like name:tok:patch lines.
                fpath = os.path.join(resume_dir_path, fname)
                if os.path.getsize(fpath) == 0:
                    continue
                done_idxs.add(idx)
        except FileNotFoundError:
            pass

        if done_idxs:
            logger.info(
                f"Resume: found {len(done_idxs)} pre-existing stage0 files in "
                f"{resume_dir_path}; skipping those chunks"
            )
            # Pre-load completed stage0 files into the merger queue and counter.
            resumed_samples = 0
            for idx in sorted(done_idxs):
                fpath = os.path.join(resume_dir_path, f"stage0_chunk_{idx:05d}.txt")
                stage0_queue.put(fpath)
                resumed_samples += count_lines(fpath)
            with global_total_counter.get_lock():
                global_total_counter.value += resumed_samples
            logger.info(f"Resume: pre-queued {resumed_samples} samples from {len(done_idxs)} files")

        # Step 5: Process chunks with multiprocessing pool
        # Calculate optimal concurrency based on system resources
        n_processes, recommended_threads = calculate_optimal_concurrency(
            total_original, processor_config.chunk_size
        )

        if processor_config.num_processes > 0:
            n_processes_user = min(processor_config.num_processes, total_chunks)
            logger.info(
                f"Overriding n_processes: auto={n_processes} -> user={n_processes_user} "
                f"(--num-processes {processor_config.num_processes})"
            )
            n_processes = n_processes_user
            threads_cap = processor_config.max_workers
        else:
            threads_cap = min(processor_config.max_workers, recommended_threads)
        
        # Update process args with recommended thread count
        process_args = [
            (
                idx + 1,
                chunk,
                processor_config.ckpt_dir,
                processor_config.min_pixels,
                processor_config.max_pixels,
                stage0_queue,
                str(processor_config.data_directory),
                processor_config.task_type,
                processor_config.del_one_token,
                processor_config.min_workers,
                threads_cap,
                processor_config.temp_dir,
                processor_config.timeout,  # Timeout per sample
                processor_config.no_npy,  # Disable NPY mode flag
                str(processor_config.resume_dir),  # Persistent stage0 dir
            )
            for idx, chunk in enumerate(all_chunks)
            if (idx + 1) not in done_idxs
        ]
        remaining_chunks = len(process_args)
        logger.info(
            f"Chunks: total={total_chunks}, resumed={len(done_idxs)}, "
            f"remaining={remaining_chunks}"
        )
        
        # Use maxtasksperchild to prevent memory leaks in long-running processes.
        # Each worker accumulates ~10 GB RSS due to HF processor + PIL caches;
        # recycle aggressively (every 3 chunks = 1500 samples by default).
        maxtasksperchild = int(os.environ.get("S2_MAXTASKS_PER_CHILD", "3"))
        
        logger.info(
            f"Starting pool with {n_processes} processes, "
            f"maxtasksperchild={maxtasksperchild}, "
            f"max_threads={threads_cap}"
        )
        
        resource_monitor = get_resource_monitor()

        # Short-circuit: if every chunk was resumed, skip pool entirely.
        stage0_files = []
        if remaining_chunks == 0:
            logger.info("All chunks already completed in a previous run; skipping pool")
            n_processes = 0  # for the log below
        with mp_ctx.Pool(
            processes=max(1, n_processes),
            maxtasksperchild=maxtasksperchild,
            initializer=_init_worker,
            initargs=(global_total_counter,),
        ) as process_pool:
            failed_chunks = []
            stalled_check_interval = 180  # Check for stalled progress every 3 minutes
            last_progress_time = time.time()
            last_completed = 0
            start_time = time.time()

            try:
                # Use imap_unordered for better progress tracking
                result_iter = process_pool.imap_unordered(process_chunk, process_args)
                completed = 0
                
                while completed < remaining_chunks:
                    try:
                        # Use next() with timeout to detect stalls
                        result = result_iter.__next__()
                        completed += 1
                        
                        if result is not None:
                            stage0_files.append(result)
                        else:
                            failed_chunks.append(completed)
                        
                        # Reset stall detection
                        last_progress_time = time.time()
                        last_completed = completed

                        # Progress logging with ETA
                        if completed % 5 == 0 or completed == remaining_chunks:
                            elapsed = time.time() - start_time
                            eta = (elapsed / completed) * (remaining_chunks - completed) if completed > 0 else 0
                            status = resource_monitor.get_status()
                            logger.info(
                                f"Progress: {completed}/{remaining_chunks} chunks "
                                f"(resumed {len(done_idxs)}, total {total_chunks}) | "
                                f"{len(stage0_files)} fresh stage0 files | "
                                f"ETA: {eta/60:.1f}min | "
                                f"CPU: {status['cpu_percent']:.0f}% | "
                                f"MEM: {status['mem_percent']:.0f}%"
                            )
                        
                        # Periodic resource check and potential pause
                        if completed % 20 == 0 and resource_monitor.should_pause():
                            logger.warning("System under high load, pausing briefly...")
                            resource_monitor.wait_for_resources(max_wait=30)
                            
                    except StopIteration:
                        break
                    except Exception as e:
                        logger.error(f"Error getting result: {e}")
                        completed += 1
                        failed_chunks.append(completed)
                    
                    # Check for stalled progress
                    current_time = time.time()
                    if current_time - last_progress_time > stalled_check_interval:
                        if completed == last_completed:
                            logger.warning(
                                f"Processing appears stalled at {completed}/{total_chunks} "
                                f"for {stalled_check_interval}s. "
                                f"Consider checking for problematic samples or reducing concurrency."
                            )
                            # Try to get resource status for debugging
                            status = resource_monitor.get_status(force=True)
                            logger.warning(
                                f"Current system status - CPU: {status['cpu_percent']:.1f}%, "
                                f"MEM: {status['mem_percent']:.1f}%, "
                                f"Available: {status['mem_available_gb']:.1f}GB"
                            )
                            last_progress_time = current_time  # Reset to avoid spam

            except KeyboardInterrupt:
                logger.warning("Received interrupt signal, terminating gracefully...")
                process_pool.terminate()
                process_pool.join()
                raise
            except Exception as e:
                logger.error(f"Process pool error: {e}", exc_info=True)
                process_pool.terminate()
                process_pool.join()

            if failed_chunks:
                logger.warning(f"Failed chunks ({len(failed_chunks)}): {failed_chunks[:20]}{'...' if len(failed_chunks) > 20 else ''}")

        logger.info(f"All processes complete, generated {len(stage0_files)} stage0 files")

        # Verify data integrity
        total_processed = global_total_counter.value
        logger.info(f"Original samples: {total_original}, Processed: {total_processed}")

        if total_processed != total_original:
            logger.warning(
                f"Data incomplete! Original: {total_original}, "
                f"Processed: {total_processed}, "
                f"Difference: {total_original - total_processed}"
            )
        else:
            logger.info("Data integrity verified, all samples processed successfully")

        # Step 6: Wait for stage0 queue to be processed
        logger.info("Waiting for stage0 queue to complete...")
        stage0_queue.join()
        logger.info("Stage0 queue processing complete")

        # Signal stage1 thread to stop and process remaining files
        logger.info("Signaling stage1 thread to stop...")
        stop_event.set()

        # Wait for stage1 thread with timeout
        timeout_counter = 0
        while stage1_thread.is_alive() and timeout_counter < 60:
            logger.debug(f"Waiting for stage1 thread ({timeout_counter}/60s)")
            threading.Event().wait(1)
            timeout_counter += 1

        if stage1_thread.is_alive():
            logger.warning("Stage1 thread timed out")
        else:
            logger.info("Stage1 thread exited normally")

        # Verify stage1 file count.
        # Expected count must include resumed stage0 files (pre-queued to merger).
        total_stage0_in_merge = len(stage0_files) + len(done_idxs)
        expected_stage1_count = (
            total_stage0_in_merge + processor_config.stage1_chunk - 1
        ) // processor_config.stage1_chunk
        if len(stage1_files) != expected_stage1_count:
            logger.warning(
                f"Stage1 file count mismatch! Expected: {expected_stage1_count}, Actual: {len(stage1_files)}"
            )
        else:
            logger.info(f"Stage1 file count verified: {len(stage1_files)} files")

        # Step 7: Final merge of all stage1 files
        if not stage1_files:
            logger.warning("No stage1 files generated, check processing errors")
            return

        stage1_total = sum(count_lines(f) for f in stage1_files)
        logger.info(f"Starting final merge: {len(stage1_files)} stage1 files, total lines: {stage1_total}")

        final_path, final_lines = processor_config.merge_files_by_token(
            stage1_files,
            str(processor_config.token_info_file),
        )

        if final_path and final_lines > 0:
            logger.info(f"Final output generated: {processor_config.token_info_file}, containing {final_lines} lines")
            if final_lines != total_processed:
                logger.error(f"Line count mismatch! Processed: {total_processed}, Final file: {final_lines}")
            else:
                logger.info("Final file integrity verified")
        else:
            logger.error("Final merge failed")

    except Exception as e:
        logger.error(f"Main process error: {e}", exc_info=True)

    finally:
        # Cleanup
        stop_event.set()

        if "stage1_thread" in dir() and stage1_thread.is_alive():
            stage1_thread.join(timeout=2)

        threading.Event().wait(2)

        # Clean up temporary files.
        # NOTE: stage0_files now live in resume_dir (persistent). We deliberately
        # do NOT delete them so the next run can resume cheaply if final merge
        # crashes or the user wants to re-run downstream. Only stage1 (tmp) is
        # garbage-collected here.
        for fpath in stage1_files:
            if fpath != str(processor_config.token_info_file) and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug(f"Cleaned up temporary file: {os.path.basename(fpath)}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {os.path.basename(fpath)}: {e}")

        logger.info("=" * 60)
        logger.info("Token length computation pipeline complete")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
