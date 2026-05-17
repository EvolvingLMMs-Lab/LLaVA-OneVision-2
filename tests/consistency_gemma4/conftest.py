"""pytest fixtures for Gemma4-VL HF↔mcore consistency tests.

Mirrors ``tests/conftest.py`` (LlavaOnevision2) with the following deltas:

- HF model is ``Gemma4ForConditionalGeneration`` (native ``transformers``).
- Conversion script is
  ``examples/gemma4_vl/convert/convert_gemma4_26b_a4b_hf_to_mcore.sh`` and
  takes a 4th positional ``EP`` argument.
- Megatron CLI args set ``--model-name gemma4-26b-a4b-vl`` and add
  MoE-specific flags (``--num-experts 128 --moe-router-topk 8 ...``).
- ``mcore_model`` fixture imports ``model_provider`` from
  ``aiak_training_llm.train.pretrain.pretrain_gemma4_vl``.

This file is intentionally a clone (not a refactor of the OV2 conftest) per
P5(B) decision — keeps OV2 consistency tests insulated from any Gemma4 changes.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        pytest.fail(f"Missing required environment variable: {name}")
    if value is None:
        return ""
    return value


def _ensure_hf_weights(config_dir: str) -> str:
    """Return ``config_dir`` if it contains safetensors, else materialize a
    randomly-initialized Gemma4 checkpoint under
    ``<repo_root>/tmp_test_gemma4_random_weights/``.

    The 26B-A4B real checkpoint is ~49 GB; for unit-style smoke we may want a
    smaller random checkpoint. In practice ``HF_MODEL_PATH`` should point at
    the real Gemma4 directory which already contains safetensors, and this
    function is a no-op fast path.
    """
    if glob.glob(os.path.join(config_dir, "*.safetensors")):
        return config_dir

    repo_root = _repo_root()
    out_dir = repo_root / "tmp_test_gemma4_random_weights"

    if out_dir.exists() and glob.glob(str(out_dir / "*.safetensors")):
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for src_file in Path(config_dir).iterdir():
        if src_file.is_file() and not src_file.name.endswith(".safetensors"):
            shutil.copy2(src_file, out_dir / src_file.name)

    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    config = Gemma4Config.from_pretrained(config_dir)
    model = Gemma4ForConditionalGeneration(config)
    model = model.to(dtype=torch.bfloat16)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    del model
    torch.cuda.empty_cache()

    return str(out_dir)


def _build_megatron_cli_args(
    *,
    hf_model_path: str,
    mcore_checkpoint_path: str,
    tp: int,
    pp: int,
    ep: int,
    epp: int,
) -> list[str]:
    return [
        "pytest-megatron-init",
        "--model-name",
        "gemma4-26b-a4b-vl",
        "--tokenizer-type",
        "HFTokenizer",
        "--hf-tokenizer-path",
        hf_model_path,
        "--dataloader-type",
        "external",
        "--split",
        "100,0,0",
        "--num-workers",
        "4",
        "--chat-template",
        "qwen2-vl",
        "--seq-length",
        "4096",
        "--max-position-embeddings",
        "4096",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--bf16",
        "--load",
        mcore_checkpoint_path,
        "--ckpt-format",
        "torch",
        "--attention-backend",
        "flash",
        "--pipeline-model-parallel-size",
        str(pp),
        "--tensor-model-parallel-size",
        str(tp),
        "--expert-model-parallel-size",
        str(ep),
        "--encoder-pipeline-model-parallel-size",
        str(epp),
        "--num-experts",
        "128",
        "--moe-router-topk",
        "8",
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-router-dtype",
        "fp32",
        "--moe-aux-loss-coeff",
        "1e-3",
        "--distributed-backend",
        "nccl",
    ]


@pytest.fixture(scope="session")
def hf_model_path() -> str:
    config_path = _env_path(
        "HF_MODEL_PATH",
        "/ov2/pretrain_models/google/gemma-4-26B-A4B-it",
        required=True,
    )
    if not Path(config_path).exists():
        pytest.fail(f"HF model path does not exist: {config_path}")
    return _ensure_hf_weights(config_path)


@pytest.fixture(scope="session")
def converted_mcore_path(hf_model_path: str) -> str:
    provided = os.environ.get("MCORE_CHECKPOINT_PATH", "").strip()
    if provided:
        if not Path(provided).exists():
            pytest.fail(f"Provided MCORE_CHECKPOINT_PATH does not exist: {provided}")
        return provided

    repo_root = _repo_root()
    tp = int(os.environ.get("CONSISTENCY_TEST_TP", "1"))
    pp = int(os.environ.get("CONSISTENCY_TEST_PP", "1"))
    ep = int(os.environ.get("CONSISTENCY_TEST_EP", "1"))
    out_dir = repo_root / f"tmp_test_gemma4_mcore_ckpt_tp{tp}_pp{pp}_ep{ep}"

    env = os.environ.copy()
    env.setdefault("AIAK_TRAINING_PATH", str(repo_root))

    script = (
        repo_root
        / "examples"
        / "gemma4_vl"
        / "convert"
        / "convert_gemma4_26b_a4b_hf_to_mcore.sh"
    )
    result = subprocess.run(
        ["bash", str(script), hf_model_path, str(out_dir), str(tp), str(pp), str(ep)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"HF->mcore conversion failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    if not out_dir.exists():
        pytest.fail(f"Converted checkpoint path not created: {out_dir}")

    os.environ["MCORE_CHECKPOINT_PATH"] = str(out_dir)
    return str(out_dir)


@pytest.fixture(scope="session")
def preprocessor_path(hf_model_path: str) -> str:
    return _env_path("PREPROCESSOR_PATH", hf_model_path)


@pytest.fixture(scope="session")
def test_image_path() -> str:
    default_path = str(_repo_root() / "asset" / "performance.png")
    path = _env_path("TEST_IMAGE_PATH", default_path, required=True)
    if path.startswith("http://") or path.startswith("https://"):
        pytest.fail("TEST_IMAGE_PATH must be a local file path, not remote URL")
    if not Path(path).exists():
        pytest.fail(f"TEST_IMAGE_PATH does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def megatron_init(hf_model_path: str, converted_mcore_path: str):
    from aiak_training_llm.train.arguments import (
        aiak_extra_train_args_provider,
        parse_arguments,
        validate_aiak_extra_args,
    )
    from aiak_training_llm.utils import initialize_aiak_megatron

    tp = int(os.environ.get("CONSISTENCY_TEST_TP", "1"))
    pp = int(os.environ.get("CONSISTENCY_TEST_PP", "1"))
    ep = int(os.environ.get("CONSISTENCY_TEST_EP", "1"))
    epp = int(os.environ.get("CONSISTENCY_TEST_EPP", "0"))

    original_argv = sys.argv
    sys.argv = _build_megatron_cli_args(
        hf_model_path=hf_model_path,
        mcore_checkpoint_path=converted_mcore_path,
        tp=tp,
        pp=pp,
        ep=ep,
        epp=epp,
    )
    try:
        args = parse_arguments(
            extra_args_provider=aiak_extra_train_args_provider,
            validate_extra_args_provider=validate_aiak_extra_args,
            args_defaults={},
        )
        initialize_aiak_megatron(args=args)
    finally:
        sys.argv = original_argv

    return args


@pytest.fixture(scope="session")
def hf_config(hf_model_path: str):
    from transformers import Gemma4Config

    return Gemma4Config.from_pretrained(hf_model_path)


@pytest.fixture(scope="session")
def hf_vision_model(hf_model_path: str):
    """Load HF Gemma4 full model and return ``.model.vision_tower``.

    Mirrors OV2's ``hf_vision_model`` fixture which returns ``.model.visual``.
    Gemma4 names the submodule ``vision_tower`` (HF convention).
    """
    from transformers import Gemma4ForConditionalGeneration

    full_model = Gemma4ForConditionalGeneration.from_pretrained(
        hf_model_path, low_cpu_mem_usage=True
    )
    vision_model = full_model.model.vision_tower.to(dtype=torch.bfloat16, device="cuda").eval()
    del full_model
    return vision_model


@pytest.fixture(scope="session")
def hf_cond_gen_model(hf_model_path: str):
    from transformers import Gemma4ForConditionalGeneration

    model = Gemma4ForConditionalGeneration.from_pretrained(hf_model_path, low_cpu_mem_usage=True)
    return model.to(dtype=torch.bfloat16, device="cuda").eval()


@pytest.fixture(scope="session")
def mcore_model(megatron_init):
    from megatron.core.enums import ModelType
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.training import get_model, unwrap_model

    from aiak_training_llm.train.pretrain.pretrain_gemma4_vl import model_provider

    model_type = (
        ModelType.encoder_and_decoder
        if megatron_init.encoder_pipeline_model_parallel_size not in [0, None]
        else ModelType.encoder_or_decoder
    )

    # Consistency tests are inference-only. Avoid DDP wrapping here: the full
    # Gemma4-26B-A4B model cannot fit the extra fp32 training grad buffers on
    # 80GB GPUs. ModelType follows the selected encoder pipeline size so the
    # checkpoint layout and runtime pipeline layout stay aligned.
    model = get_model(
        model_provider,
        model_type,
        wrap_with_ddp=False,
    )
    load_checkpoint(model, None, None)
    return unwrap_model(model)[0].to("cuda").eval()


@pytest.fixture(scope="session")
def hf_processor(preprocessor_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(preprocessor_path, trust_remote_code=True)
