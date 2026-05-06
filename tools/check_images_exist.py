#!/usr/bin/env python3
"""Check whether image files referenced in JSONL samples actually exist on disk.

Usage:
    python check_images_exist.py --jsonl a.jsonl b.jsonl ... --report /path/to/report.txt
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--report", required=True, help="Output report file")
    args = parser.parse_args()

    total = 0
    missing_count = 0
    missing_samples = []

    for jsonl_path in args.jsonl:
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
                    total += 1
                    if not os.path.exists(img):
                        missing_count += 1
                        if len(missing_samples) < 1000:
                            missing_samples.append(f"{jsonl_path}:{line_no} -> {img}")

    with open(args.report, "w") as f:
        f.write(f"total_images: {total}\n")
        f.write(f"missing_images: {missing_count}\n")
        f.write(f"missing_rate: {missing_count/total*100:.4f}%\n" if total > 0 else "missing_rate: N/A\n")
        if missing_samples:
            f.write("\n--- missing samples (up to 1000) ---\n")
            for s in missing_samples:
                f.write(s + "\n")

    print(f"total={total} missing={missing_count} ({missing_count/total*100:.4f}%)" if total else "no images found")


if __name__ == "__main__":
    main()
