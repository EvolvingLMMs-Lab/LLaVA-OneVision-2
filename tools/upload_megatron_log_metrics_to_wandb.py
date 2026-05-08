#!/usr/bin/env python3
"""Upload Megatron console metrics from a completed training log to W&B.

This is useful on systems where TensorBoard is unavailable. In this codebase,
some W&B scalar logging is gated behind a TensorBoard writer, but the same
metrics are still present in the console training log.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


LINE_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<train_iters>\d+)\s+\|"
    r"\s+consumed samples:\s+(?P<consumed_samples>\d+)\s+\|"
    r"\s+elapsed time per iteration \(ms\):\s+(?P<iteration_time_ms>[0-9.Ee+-]+)\s+\|"
    r"\s+throughput \(token/sec/GPU\):\s+(?P<throughput>[0-9.Ee+-]+)\s+\|"
    r"\s+learning rate:\s+(?P<learning_rate>[0-9.Ee+-]+)\s+\|"
    r"\s+global batch size:\s+(?P<global_batch_size>\d+)\s+\|"
    r"\s+lm loss:\s+(?P<lm_loss>[0-9.Ee+-]+)\s+\|"
    r"\s+loss scale:\s+(?P<loss_scale>[0-9.Ee+-]+)\s+\|"
    r"\s+grad norm:\s+(?P<grad_norm>[0-9.Ee+-]+)\s+\|"
    r"\s+num zeros:\s+(?P<num_zeros>\d+)\s+\|"
    r"\s+number of skipped iterations:\s+(?P<skipped_iterations>\d+)\s+\|"
    r"\s+number of nan iterations:\s+(?P<nan_iterations>\d+)\s+\|"
)


def parse_log(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "iteration": int(item["iteration"]),
                "train_iters": int(item["train_iters"]),
                "consumed_samples": int(item["consumed_samples"]),
                "iteration_time_ms": float(item["iteration_time_ms"]),
                "iteration_time_sec": float(item["iteration_time_ms"]) / 1000.0,
                "token_throughput_per_gpu": float(item["throughput"]),
                "learning_rate": float(item["learning_rate"]),
                "global_batch_size": int(item["global_batch_size"]),
                "lm_loss": float(item["lm_loss"]),
                "loss_scale": float(item["loss_scale"]),
                "grad_norm": float(item["grad_norm"]),
                "num_zeros": int(item["num_zeros"]),
                "skipped_iterations": int(item["skipped_iterations"]),
                "nan_iterations": int(item["nan_iterations"]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def upload_wandb(args: argparse.Namespace, rows: list[dict[str, float | int]], csv_path: Path | None) -> str:
    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        job_type="posthoc-log-upload",
        config={
            "source_log": str(args.log),
            "parsed_points": len(rows),
            "checkpoint": args.checkpoint,
        },
    )

    assert run is not None
    for row in rows:
        step = int(row["iteration"])
        wandb.log(
            {
                "lm_loss": row["lm_loss"],
                "learning_rate": row["learning_rate"],
                "grad_norm": row["grad_norm"],
                "token_throughput_per_gpu": row["token_throughput_per_gpu"],
                "iteration_time_ms": row["iteration_time_ms"],
                "iteration_time_sec": row["iteration_time_sec"],
                "global_batch_size": row["global_batch_size"],
                "consumed_samples": row["consumed_samples"],
                "loss_scale": row["loss_scale"],
                "num_zeros": row["num_zeros"],
                "skipped_iterations": row["skipped_iterations"],
                "nan_iterations": row["nan_iterations"],
            },
            step=step,
        )

    if csv_path is not None and csv_path.exists():
        artifact = wandb.Artifact(f"{args.name}-metrics", type="metrics")
        artifact.add_file(str(csv_path))
        run.log_artifact(artifact)

    url = run.url
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Megatron training log to parse.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--upload-wandb", action="store_true", help="Upload parsed metrics to W&B.")
    parser.add_argument("--entity", default="rana-zayed-mbzuai")
    parser.add_argument("--project", default="llava-ov-1_5")
    parser.add_argument("--name", default="stage1_5_imagenet_en_1000_posthoc_metrics")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"No Megatron metric lines found in: {args.log}")

    if args.csv_out is not None:
        write_csv(rows, args.csv_out)
        print(f"Wrote {len(rows)} metric rows to {args.csv_out}")
    else:
        print(f"Parsed {len(rows)} metric rows from {args.log}")

    print(
        "Final parsed point: "
        f"iteration={rows[-1]['iteration']}, lm_loss={rows[-1]['lm_loss']:.6g}, "
        f"grad_norm={rows[-1]['grad_norm']:.6g}, skipped={rows[-1]['skipped_iterations']}, "
        f"nan={rows[-1]['nan_iterations']}"
    )

    if args.upload_wandb:
        url = upload_wandb(args, rows, args.csv_out)
        print(f"Uploaded W&B metrics run: {url}")


if __name__ == "__main__":
    main()
