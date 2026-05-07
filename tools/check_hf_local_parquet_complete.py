#!/usr/bin/env python3
"""Compare local HF parquet files with the Hugging Face dataset file listing."""

from __future__ import annotations

import argparse
import fnmatch
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _remote_files(repo_id: str, data_files: list[str]) -> dict[str, FileInfo]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefixes = sorted({pattern.split("*", 1)[0].rsplit("/", 1)[0] for pattern in data_files})
    files: dict[str, FileInfo] = {}

    for prefix in prefixes:
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=prefix,
            recursive=True,
        ):
            path = getattr(entry, "path", "")
            size = getattr(entry, "size", None)
            if not path.endswith(".parquet") or size is None:
                continue
            if any(fnmatch.fnmatch(path, pattern) for pattern in data_files):
                files[path] = FileInfo(path=path, size=int(size))
    return files


def _local_files(local_data_root: Path, data_files: list[str]) -> dict[str, FileInfo]:
    files: dict[str, FileInfo] = {}
    for pattern in data_files:
        for path in local_data_root.glob(pattern):
            if path.is_file():
                rel = path.relative_to(local_data_root).as_posix()
                files[rel] = FileInfo(path=rel, size=path.stat().st_size)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=["imagenet/EN/*/*.parquet"],
        help="HF repo-relative parquet globs to verify.",
    )
    parser.add_argument(
        "--local-data-root",
        type=Path,
        required=True,
        help="Local root mirroring the HF dataset repo layout.",
    )
    parser.add_argument("--max-print", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote = _remote_files(args.repo_id, args.data_files)
    local = _local_files(args.local_data_root, args.data_files)

    remote_size = sum(item.size for item in remote.values())
    local_size = sum(item.size for item in local.values())
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    size_mismatch = sorted(
        path for path in set(remote) & set(local) if remote[path].size != local[path].size
    )

    print(f"Repo          : {args.repo_id}")
    print(f"Local root    : {args.local_data_root}")
    print(f"Patterns      : {' '.join(args.data_files)}")
    print(f"Remote files  : {len(remote)} ({_human_size(remote_size)})")
    print(f"Local files   : {len(local)} ({_human_size(local_size)})")
    print(f"Missing       : {len(missing)}")
    print(f"Extra         : {len(extra)}")
    print(f"Size mismatch : {len(size_mismatch)}")

    for title, paths in (
        ("Missing files", missing),
        ("Extra local files", extra),
        ("Files with different sizes", size_mismatch),
    ):
        if paths:
            print(f"\n{title}:")
            for path in paths[: args.max_print]:
                print(f"  {path}")
            if len(paths) > args.max_print:
                print(f"  ... {len(paths) - args.max_print} more")

    if missing or size_mismatch:
        raise SystemExit(1)

    print("\nLocal parquet files match the Hugging Face listing.")


if __name__ == "__main__":
    main()
