#!/usr/bin/env python3
"""Print one caption WebDataset sample for run-log sanity checks."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def preview(data_path: Path, sample_index: int, max_chars: int) -> None:
    tar_paths = sorted(data_path.glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar shards found in {data_path}")

    seen = 0
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tar:
            json_members = sorted(
                (member for member in tar.getmembers() if member.isfile() and member.name.endswith(".json")),
                key=lambda member: member.name,
            )
            for member in json_members:
                if seen != sample_index:
                    seen += 1
                    continue

                raw = tar.extractfile(member)
                if raw is None:
                    raise RuntimeError(f"Could not read {member.name} from {tar_path}")
                payload = json.load(raw)

                key = member.name[: -len(".json")]
                image_members = [
                    name for name in tar.getnames() if name.startswith(f"{key}.img")
                ]

                image_info = "none"
                if image_members:
                    image_member = tar.getmember(image_members[0])
                    image_info = f"{image_members[0]} ({image_member.size} bytes)"
                    try:
                        from PIL import Image

                        image_file = tar.extractfile(image_member)
                        if image_file is not None:
                            with Image.open(io.BytesIO(image_file.read())) as image:
                                image_info += f", size={image.size[0]}x{image.size[1]}"
                    except Exception:
                        pass

                prompts = payload.get("prompts") or []
                captions = payload.get("captions") or []

                print("")
                print("========== DATA SAMPLE PREVIEW ==========")
                print(f"DATA_PATH: {data_path}")
                print(f"SHARD    : {tar_path.name}")
                print(f"KEY      : {key}")
                print(f"IMAGE    : {image_info}")
                print(f"PROMPT   : {_truncate(str(prompts[0] if prompts else ''), max_chars)}")
                print(f"CAPTION  : {_truncate(str(captions[0] if captions else ''), max_chars)}")
                print("=========================================")
                print("")
                return

    raise IndexError(f"sample_index={sample_index} out of range; found {seen} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preview(args.data_path, args.sample_index, args.max_chars)


if __name__ == "__main__":
    main()
