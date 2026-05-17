from __future__ import annotations

import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from aiak_training_llm.models.gemma4_vl.gemma4_attention import (
    _resolve_per_layer_overrides,
)
from aiak_training_llm.models.gemma4_vl.gemma4_vl_config import gemma4_26b_a4b_vl


def _build_tc() -> TransformerConfig:
    cfg = gemma4_26b_a4b_vl()
    tc_fields = {f.name for f in TransformerConfig.__dataclass_fields__.values()}
    tc_kwargs = {k: v for k, v in cfg.__dict__.items() if k in tc_fields}
    tc_kwargs.update(
        {
            "bf16": True,
            "params_dtype": torch.bfloat16,
            "pipeline_dtype": torch.bfloat16,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": False,
            "add_bias_linear": False,
            "gated_linear_unit": True,
            "activation_func": torch.nn.functional.gelu,
        }
    )
    for k in (
        "layer_pattern",
        "per_layer_kv_channels",
        "per_layer_num_query_groups",
        "kv_tied_layers",
        "sliding_window",
        "rotary_base_sliding",
    ):
        if hasattr(cfg, k):
            tc_kwargs[k] = getattr(cfg, k)
    return TransformerConfig(**tc_kwargs)


def test_per_layer_window_size_matches_layer_pattern() -> None:
    tc = _build_tc()
    sliding_w = tc.sliding_window
    assert sliding_w == 1024

    expected_te_window = (sliding_w - 1, 0)

    for layer_number in range(1, len(tc.layer_pattern) + 1):
        eff = _resolve_per_layer_overrides(tc, layer_number)
        layer_type = tc.layer_pattern[layer_number - 1]
        if layer_type == "sliding":
            assert eff.window_size == expected_te_window, (
                f"layer {layer_number} (sliding): window_size={eff.window_size}, expected {expected_te_window}"
            )
            assert eff.kv_channels == 256
            assert eff.num_query_groups == 8
        elif layer_type == "global":
            assert eff.window_size is None, (
                f"layer {layer_number} (global): window_size={eff.window_size}, expected None"
            )
            assert eff.kv_channels == 512
            assert eff.num_query_groups == 2
        else:
            raise AssertionError(f"unknown layer_type: {layer_type}")


def test_global_layers_match_hf_full_attention_positions() -> None:
    tc = _build_tc()
    actual_global_1based = [i + 1 for i, t in enumerate(tc.layer_pattern) if t == "global"]
    assert actual_global_1based == [6, 12, 18, 24, 30]


def test_resolve_returns_unchanged_for_empty_layer_pattern() -> None:
    tc = _build_tc()
    tc.layer_pattern = []
    out = _resolve_per_layer_overrides(tc, 1)
    assert out is tc


def test_resolve_clones_config_when_pattern_present() -> None:
    tc = _build_tc()
    out = _resolve_per_layer_overrides(tc, 1)
    assert out is not tc
    assert out.window_size == (tc.sliding_window - 1, 0)
    assert tc.window_size != out.window_size or tc.window_size is None
