from __future__ import annotations

import os

import pytest


def _has_full_consistency_env() -> bool:
    has_hf = bool(os.environ.get("HF_MODEL_PATH"))
    has_mcore = bool(os.environ.get("MCORE_CHECKPOINT_PATH"))
    allow_convert = os.environ.get("CONSISTENCY_RUN_CONVERT", "0") == "1"
    return has_hf and (has_mcore or allow_convert)


def test_collection_smoke():
    assert True


def test_full_gemma4_consistency_fixtures_lazy(request):
    if not _has_full_consistency_env():
        pytest.skip(
            "Set HF_MODEL_PATH and either MCORE_CHECKPOINT_PATH or "
            "CONSISTENCY_RUN_CONVERT=1 to run Gemma4 full consistency."
        )

    hf_config = request.getfixturevalue("hf_config")
    hf_vision_model = request.getfixturevalue("hf_vision_model")
    mcore_model = request.getfixturevalue("mcore_model")

    assert getattr(hf_config, "model_type", None) == "gemma4"
    assert hasattr(hf_vision_model, "forward")
    assert hasattr(mcore_model, "forward")
