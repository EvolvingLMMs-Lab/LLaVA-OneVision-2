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
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        pytest.fail(f"Missing required environment variable: {name}")
    if value is None:
        return ""
    return value


def _ensure_hf_weights(config_dir: str) -> str:
    """If *config_dir* already contains safetensors, return it as-is.

    Otherwise, create a random-initialised HF checkpoint under
    ``<repo_root>/tmp_test_hf_random_weights/`` so that the conversion
    script (and every fixture that calls ``from_pretrained``) has real
    weight files to work with.
    """
    if glob.glob(os.path.join(config_dir, "*.safetensors")):
        return config_dir

    repo_root = _repo_root()
    out_dir = repo_root / "tmp_test_hf_random_weights"

    # Re-use a previously generated checkpoint.
    if out_dir.exists() and glob.glob(str(out_dir / "*.safetensors")):
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy every non-weight file from the config directory.
    for src_file in Path(config_dir).iterdir():
        if src_file.is_file() and not src_file.name.endswith(".safetensors"):
            shutil.copy2(src_file, out_dir / src_file.name)

    # Instantiate from config → random weights, then save.
    from transformers_impl.llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2Config
    from transformers_impl.llavaonevision2.modeling_llava_onevision2 import (
        LlavaOnevision2ForConditionalGeneration,
    )

    config = LlavaOnevision2Config.from_pretrained(config_dir, trust_remote_code=True)
    model = LlavaOnevision2ForConditionalGeneration(config)
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
) -> list[str]:
    model_name = os.environ.get("CONSISTENCY_TEST_MODEL_NAME", "llava-onevision2-4b")
    return [
        "pytest-megatron-init",
        "--model-name",
        model_name,
        "--tokenizer-type",
        "HFTokenizer",
        "--hf-tokenizer-path",
        hf_model_path,
        "--dataloader-type",
        "external",
        "--split",
        "100,0,0",
        "--num-workers",
        "16",
        "--chat-template",
        "qwen2-vl",
        "--seq-length",
        "4096",
        "--max-position-embeddings",
        "32768",
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
        "--distributed-backend",
        "nccl",
    ]


@pytest.fixture(scope="session")
def hf_model_path() -> str:
    config_path = _env_path(
        "HF_MODEL_PATH",
        "/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b/auto-model",
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
    variant = os.environ.get("CONSISTENCY_TEST_MODEL_VARIANT", "4b")
    out_dir = repo_root / f"tmp_test_mcore_ckpt_{variant}_tp{tp}_pp{pp}"

    env = os.environ.copy()
    env.setdefault("AIAK_TRAINING_PATH", str(repo_root))

    script = (
        repo_root
        / "examples"
        / "llava_onevision2"
        / "convert"
        / f"convert_{variant}_hf_to_mcore.sh"
    )
    result = subprocess.run(
        ["bash", str(script), hf_model_path, str(out_dir), str(tp), str(pp)],
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

    original_argv = sys.argv
    sys.argv = _build_megatron_cli_args(
        hf_model_path=hf_model_path,
        mcore_checkpoint_path=converted_mcore_path,
        tp=tp,
        pp=pp,
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
    from transformers_impl.llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2Config

    return LlavaOnevision2Config.from_pretrained(hf_model_path, trust_remote_code=True)


@pytest.fixture(scope="session")
def hf_vision_model(hf_model_path: str):
    # Must load via the Conditional class: ``LlavaOnevision2Model.from_pretrained``
    # has ``base_model_prefix = ""`` and silently matches 0/N weights against the
    # ``model.*`` prefix in transformers >=5.x checkpoints (no exception, just
    # random init).
    from transformers_impl.llavaonevision2.modeling_llava_onevision2 import (
        LlavaOnevision2ForConditionalGeneration,
    )

    full_model = LlavaOnevision2ForConditionalGeneration.from_pretrained(
        hf_model_path, low_cpu_mem_usage=True
    )
    vision_model = full_model.model.visual.to(dtype=torch.bfloat16, device="cuda").eval()
    del full_model
    return vision_model


@pytest.fixture(scope="session")
def hf_cond_gen_model(hf_model_path: str):
    from transformers_impl.llavaonevision2.modeling_llava_onevision2 import LlavaOnevision2ForConditionalGeneration

    model = LlavaOnevision2ForConditionalGeneration.from_pretrained(hf_model_path, low_cpu_mem_usage=True)
    return model.to(dtype=torch.bfloat16, device="cuda").eval()


@pytest.fixture(scope="session")
def mcore_model(megatron_init):
    del megatron_init
    from megatron.core.enums import ModelType
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.training import get_model, unwrap_model

    from aiak_training_llm.train.pretrain.pretrain_llava_onevision2 import model_provider

    model = get_model(model_provider, ModelType.encoder_or_decoder)
    load_checkpoint(model, None, None)
    return unwrap_model(model)[0].to("cuda").eval()


@pytest.fixture(scope="session")
def hf_processor(preprocessor_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(preprocessor_path, trust_remote_code=True)
