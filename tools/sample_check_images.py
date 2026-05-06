#!/usr/bin/env python3
"""Randomly sample N image paths from JSONL files and check existence."""
import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Reservoir sampling over (jsonl_path, line_no, image_path)
    reservoir = []
    seen = 0
    for jsonl_path in args.jsonl:
        try:
            with open(jsonl_path) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    images = item.get("images") or item.get("images_source") or []
                    for img in images:
                        seen += 1
                        if len(reservoir) < args.sample:
                            reservoir.append((jsonl_path, line_no, img))
                        else:
                            j = random.randrange(seen)
                            if j < args.sample:
                                reservoir[j] = (jsonl_path, line_no, img)
        except Exception as e:
            print(f"WARN reading {jsonl_path}: {e}")

    # Check existence
    missing = []
    for jsonl_path, line_no, img in reservoir:
        if not os.path.exists(img):
            missing.append((jsonl_path, line_no, img))

    with open(args.report, "w") as f:
        f.write(f"total_image_refs_seen: {seen}\n")
        f.write(f"sampled: {len(reservoir)}\n")
        f.write(f"missing: {len(missing)}\n")
        if reservoir:
            f.write(f"missing_rate: {len(missing)/len(reservoir)*100:.2f}%\n")
        if missing:
            f.write("\n--- missing ---\n")
            for jp, ln, img in missing[:200]:
                f.write(f"{jp}:{ln} -> {img}\n")

    print(f"sampled={len(reservoir)} missing={len(missing)}")


if __name__ == "__main__":
    main()
