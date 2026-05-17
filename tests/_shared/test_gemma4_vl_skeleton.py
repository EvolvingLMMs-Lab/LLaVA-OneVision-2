"""GPU smoke test for Gemma4-VL skeleton (P1.14).

Boots Megatron in-process on a single GPU with shrunken LLM dims, instantiates
the registered ``gemma4_vl`` provider with the Gemma4 vision tower + adapter
monkey-patched out (their dims are real Gemma4-26B-A4B-it production sizes —
27 layers / hidden=1152 / intermediate=4304 — and would dwarf the
shrunken-LLM smoke test), runs one forward pass on dummy text tokens, and
asserts no crash + correct logits shape.

Stubbing the vision tower / adapter keeps this smoke test focused on the
Gemma4 LLM stack (Gemma4Model + Gemma4TransformerLayer + Gemma4ParallelDenseMoE)
which is the moving part in P1.

Skip rules:
- requires CUDA (skipped on CPU-only hosts)
- requires single-GPU only (TP=1 PP=1 EP=1)

Run inside Docker:
    pytest tests/_shared/test_gemma4_vl_skeleton.py -v -s
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


_MICRO_ARGV = [
    "pytest-gemma4-skeleton",
    "--model-name", "gemma4_vl",
    "--tokenizer-type", "NullTokenizer",
    "--vocab-size", "1024",
    "--micro-batch-size", "1",
    "--global-batch-size", "1",
    "--seq-length", "16",
    "--max-position-embeddings", "16",
    "--num-layers", "2",
    "--hidden-size", "128",
    "--num-attention-heads", "4",
    "--num-query-groups", "2",
    "--kv-channels", "32",
    "--ffn-hidden-size", "64",
    "--num-experts", "4",
    "--moe-router-topk", "2",
    "--bf16",
    "--disable-bias-linear",
    "--attention-backend", "flash",
    "--pipeline-model-parallel-size", "1",
    "--tensor-model-parallel-size", "1",
    "--expert-model-parallel-size", "1",
    "--distributed-backend", "nccl",
    "--position-embedding-type", "rope",
    "--no-rope-fusion",
    "--use-mcore-models",
    "--transformer-impl", "transformer_engine",
]


@pytest.fixture(scope="module")
def megatron_micro_init():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from aiak_training_llm.train.arguments import (
        aiak_extra_train_args_provider,
        parse_arguments,
        validate_aiak_extra_args,
    )
    from aiak_training_llm.utils import initialize_aiak_megatron

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12377")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    saved_argv = sys.argv
    sys.argv = list(_MICRO_ARGV)
    try:
        args = parse_arguments(
            extra_args_provider=aiak_extra_train_args_provider,
            validate_extra_args_provider=validate_aiak_extra_args,
        )
        initialize_aiak_megatron(args=args)
    finally:
        sys.argv = saved_argv

    yield args


def test_provider_lookup_resolves_gemma4_vl():
    from aiak_training_llm.models import get_model_family, get_model_provider

    fam = get_model_family("gemma4-26b-a4b-vl")
    assert fam == "gemma4_vl", f"family lookup wrong: {fam}"

    provider = get_model_provider(fam)
    assert provider is not None, "gemma4_vl provider not registered"
    assert provider.__name__ == "gemma4_vl_model_provider"
    assert "gemma4_vl_provider" in provider.__module__


def test_gemma4_vl_micro_forward(megatron_micro_init, monkeypatch):
    args = megatron_micro_init
    from aiak_training_llm.models.gemma4_vl import gemma4_vl_model as gvm
    from aiak_training_llm.models.gemma4_vl.gemma4_vl_provider import (
        gemma4_vl_model_provider,
    )

    class _StubVision(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError("vision tower stubbed in skeleton smoke")

    class _StubAdapter(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError("adapter stubbed in skeleton smoke")

    monkeypatch.setattr(gvm, "Gemma4VisionTower", _StubVision)
    monkeypatch.setattr(gvm, "Gemma4Adapter", _StubAdapter)

    model = gemma4_vl_model_provider(
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        parallel_output=False,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    micro_image_token_id = 1
    model.config.image_token_id = micro_image_token_id

    seq_len = 8
    input_ids = torch.randint(
        low=0, high=args.padded_vocab_size, size=(1, seq_len),
        dtype=torch.long, device="cuda",
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    attention_mask = None

    with torch.no_grad():
        logits = model(
            images=None,
            image_grid_thw=None,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    assert logits is not None, "model returned None"
    assert isinstance(logits, torch.Tensor), f"expected Tensor, got {type(logits)}"
    assert logits.shape[0] == 1, f"batch dim wrong: {logits.shape}"
    assert args.padded_vocab_size in logits.shape, (
        f"vocab dim {args.padded_vocab_size} not in logits shape {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"


def test_gemma4_vl_vision_path_requires_grid(megatron_micro_init, monkeypatch):
    args = megatron_micro_init
    from aiak_training_llm.models.gemma4_vl import gemma4_vl_model as gvm
    from aiak_training_llm.models.gemma4_vl.gemma4_vl_provider import (
        gemma4_vl_model_provider,
    )

    class _StubVision(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError("should not be reached: forward must short-circuit")

    class _StubAdapter(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError("should not be reached: forward must short-circuit")

    monkeypatch.setattr(gvm, "Gemma4VisionTower", _StubVision)
    monkeypatch.setattr(gvm, "Gemma4Adapter", _StubAdapter)

    model = gemma4_vl_model_provider(
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        parallel_output=False,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    dummy_images = torch.zeros(1, 1, 768, dtype=torch.bfloat16, device="cuda")
    input_ids = torch.randint(
        low=0, high=args.padded_vocab_size, size=(1, 4),
        dtype=torch.long, device="cuda",
    )
    position_ids = torch.arange(4, dtype=torch.long, device="cuda").unsqueeze(0)

    with pytest.raises(ValueError, match="image_grid_thw is required"):
        with torch.no_grad():
            model(
                images=dummy_images,
                image_grid_thw=None,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )


def test_gemma4_vl_vision_path_micro_forward(megatron_micro_init, monkeypatch):
    args = megatron_micro_init
    from aiak_training_llm.models.gemma4_vl import gemma4_vl_model as gvm
    from aiak_training_llm.models.gemma4_vl.gemma4_vl_provider import (
        gemma4_vl_model_provider,
    )

    class _StubVision(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, pixel_values, pixel_position_ids):
            assert pixel_values.shape == (1, 4, 768)
            assert pixel_position_ids.shape == (1, 4, 2)
            return torch.ones(4, 32, dtype=pixel_values.dtype, device=pixel_values.device)

    class _StubAdapter(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            return torch.ones(
                hidden_states.shape[0],
                args.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

    monkeypatch.setattr(gvm, "Gemma4VisionTower", _StubVision)
    monkeypatch.setattr(gvm, "Gemma4Adapter", _StubAdapter)

    model = gemma4_vl_model_provider(
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        parallel_output=False,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    micro_image_token_id = 1
    model.config.image_token_id = micro_image_token_id

    dummy_images = torch.zeros(4, 768, dtype=torch.bfloat16, device="cuda")
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device="cuda")
    patch_positions = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
        dtype=torch.long,
        device="cuda",
    )
    input_ids = torch.randint(
        low=0, high=args.padded_vocab_size, size=(1, 8),
        dtype=torch.long, device="cuda",
    )
    input_ids[0, :4] = micro_image_token_id
    position_ids = torch.arange(8, dtype=torch.long, device="cuda").unsqueeze(0)

    with torch.no_grad():
        logits = model(
            images=dummy_images,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            patch_positions=patch_positions,
        )

    assert logits is not None
    assert isinstance(logits, torch.Tensor)
    assert torch.isfinite(logits).all()
