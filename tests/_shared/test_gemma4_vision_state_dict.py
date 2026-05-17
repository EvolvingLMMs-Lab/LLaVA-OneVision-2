"""Narrow unit tests for Gemma4 vision tower + adapter state-dict fidelity.

Verifies the HF-verbatim port preserves exactly the state-dict keys that the
Phase C HF↔Megatron checkpoint converter will round-trip:

- ``Gemma4Adapter`` exposes exactly one trainable tensor:
  ``embedding_projection.weight`` of shape ``[text_hidden, vision_hidden]``.
  The ``embedding_pre_projection_norm`` (RMSNorm with ``with_scale=False``)
  contributes NO key — HF stores no ``embedding_pre_projection_norm.weight``
  for this checkpoint, so registering one would break ckpt loading.

- ``Gemma4VisionTower`` registers ``std_bias`` and ``std_scale`` ONLY when
  ``standardize=True`` (persistent buffers visible in state_dict). With
  ``standardize=False`` they must NOT appear.

These tests are CPU-only and do not require Megatron initialisation; they
guard the converter contract independently of the GPU smoke test.
"""

from __future__ import annotations

import torch

from aiak_training_llm.models.gemma4_vl.gemma4_adapter import Gemma4Adapter
from aiak_training_llm.models.gemma4_vl.gemma4_vision_tower import Gemma4VisionTower


class _DummyTransformerConfig:
    def __init__(self) -> None:
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.sequence_parallel = False
        self.params_dtype = torch.float32
        self.timers = None
        self.perform_initialization = False
        self.deterministic_mode = False
        self.use_cpu_initialization = True


def _build_tower(*, standardize: bool) -> Gemma4VisionTower:
    return Gemma4VisionTower(
        transformer_config=_DummyTransformerConfig(),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=64,
        patch_size=4,
        in_channels=3,
        position_embedding_size=8,
        pooling_kernel_size=2,
        rope_theta=100.0,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        use_clipped_linears=False,
        standardize=standardize,
    )


def test_adapter_state_dict_has_only_embedding_projection_weight():
    adapter = Gemma4Adapter(
        vision_hidden_size=1152,
        text_hidden_size=2816,
        rms_norm_eps=1e-6,
    )
    keys = sorted(adapter.state_dict().keys())
    assert keys == ["embedding_projection.weight"], (
        f"Gemma4Adapter must expose exactly one ckpt key "
        f"(embedding_projection.weight); got {keys}"
    )
    w = adapter.state_dict()["embedding_projection.weight"]
    assert w.shape == (2816, 1152), f"projection weight shape wrong: {w.shape}"

    pre_norm = adapter.embedding_pre_projection_norm
    assert not any(True for _ in pre_norm.parameters()), (
        "embedding_pre_projection_norm must be parameter-free "
        "(with_scale=False); registering a weight would inject an "
        "unexpected ckpt key and break HF loading."
    )


def test_tower_registers_std_buffers_iff_standardize():
    tower_with_std = _build_tower(standardize=True)
    keys = set(tower_with_std.state_dict().keys())
    assert "std_bias" in keys, "std_bias must be in state_dict when standardize=True"
    assert "std_scale" in keys, "std_scale must be in state_dict when standardize=True"

    tower_no_std = _build_tower(standardize=False)
    keys = set(tower_no_std.state_dict().keys())
    assert "std_bias" not in keys, (
        "std_bias must NOT be in state_dict when standardize=False"
    )
    assert "std_scale" not in keys, (
        "std_scale must NOT be in state_dict when standardize=False"
    )
