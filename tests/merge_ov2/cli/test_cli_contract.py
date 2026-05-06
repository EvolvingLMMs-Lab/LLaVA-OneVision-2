"""Phase 5 CLI contract: --help flags + missing-arg / missing-variant exit codes + dry-run no-save."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "tiny-dense"
BUILDER = REPO_ROOT / "tests" / "fixtures" / "build_tiny_triplet.py"


_MERGE_FLAGS = [
    "--variant",
    "--vit",
    "--adapter",
    "--llm",
    "--processor",
    "--out",
    "--spatial-merge-size",
    "--target-dtype",
    "--device",
    "--validate-skip",
    "--qwen-processor",
    "--img",
    "--sample-text",
    "--vit-validator-strategy",
    "--llm-validator-strategy",
]
_VALIDATE_FLAGS = [
    "--variant",
    "--ckpt",
    "--vit",
    "--llm",
    "--processor",
    "--device",
    "--validate-skip",
    "--qwen-processor",
    "--img",
    "--sample-text",
    "--vit-validator-strategy",
    "--llm-validator-strategy",
]
_DRY_RUN_FLAGS = ["--variant", "--vit", "--adapter", "--llm", "--processor", "--spatial-merge-size"]


def _help(subcmd: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "transformers_impl")}
    return subprocess.check_output(
        [sys.executable, "-m", "transformers_impl.merge_ov2", subcmd, "--help"],
        env=env,
        text=True,
    )


@pytest.mark.parametrize(
    "subcmd,flags", [("merge", _MERGE_FLAGS), ("validate", _VALIDATE_FLAGS), ("dry-run", _DRY_RUN_FLAGS)]
)
def test_help_lists_all_flags(subcmd, flags):
    out = _help(subcmd)
    missing = [f for f in flags if f not in out]
    assert not missing, f"{subcmd} --help missing flags: {missing}\n{out}"


def _ensure_fixture():
    if (FIXTURE_DIR / "llm" / "config.json").exists():
        return
    subprocess.run([sys.executable, str(BUILDER), "--variant", "dense"], check=True, cwd=REPO_ROOT)


def test_merge_missing_validator_args_fails(tmp_path):
    from transformers_impl.merge_ov2.cli import main

    _ensure_fixture()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "merge",
                "--variant",
                "dense",
                "--vit",
                str(FIXTURE_DIR / "vit"),
                "--adapter",
                str(FIXTURE_DIR / "adapter"),
                "--llm",
                str(FIXTURE_DIR / "llm"),
                "--processor",
                str(FIXTURE_DIR / "processor"),
                "--out",
                str(tmp_path / "x"),
            ]
        )
    assert exc.value.code == 2


def test_merge_missing_variant_fails(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "transformers_impl")}
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformers_impl.merge_ov2",
            "merge",
            "--vit",
            "/x",
            "--llm",
            "/x",
            "--processor",
            "/x",
            "--out",
            str(tmp_path / "x"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "--variant" in res.stderr


def test_dry_run_no_save(tmp_path):
    from transformers_impl.merge_ov2.cli import main

    _ensure_fixture()
    rc = main(
        [
            "dry-run",
            "--variant",
            "dense",
            "--vit",
            str(FIXTURE_DIR / "vit"),
            "--adapter",
            str(FIXTURE_DIR / "adapter"),
            "--llm",
            str(FIXTURE_DIR / "llm"),
            "--processor",
            str(FIXTURE_DIR / "processor"),
        ]
    )
    assert rc == 0
    assert list(tmp_path.iterdir()) == []
