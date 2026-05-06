from __future__ import annotations

from contextlib import nullcontext
from glob import glob
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torchvision import transforms

from .test_consistency_utils import (
    align_encoder_debug_tensors,
    align_rotary_debug_tensors,
    compare_arrays,
    convert_hf_qkv_to_mcore_layout,
    cosine_similarity,
    load_and_resize_image,
    summarize_layer_results,
    summarize_named_results,
)


def _autocast_context() -> Any:
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _generate_patch_positions(
    grid_thw: torch.Tensor,
    device: torch.device,
    spatial_merge_size: int = 1,
) -> torch.Tensor:
    patch_positions = []
    for i in range(grid_thw.shape[0]):
        t, h, w = grid_thw[i].tolist()
        t_idx = torch.arange(t, device=device, dtype=torch.float32)
        h_idx = torch.arange(h, device=device, dtype=torch.float32)
        w_idx = torch.arange(w, device=device, dtype=torch.float32)
        mesh_t, mesh_h, mesh_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        sample = torch.stack([mesh_t, mesh_h, mesh_w], dim=-1).reshape(-1, 3)
        sms = spatial_merge_size
        if sms > 1:
            total = t * h * w
            indices = torch.arange(total, device=device).view(t, h, w)
            h_m, w_m = h // sms, w // sms
            indices = indices.view(t, h_m, sms, w_m, sms).permute(0, 1, 3, 2, 4).contiguous().view(total)
            sample = sample[indices]
        patch_positions.append(sample)
    return torch.cat(patch_positions, dim=0)


def _maybe_gather_tp_weight(weight: torch.Tensor, mcore_key: str) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import _gather_along_first_dim, _gather_along_last_dim

    if mcore_key.endswith("self_attention.linear_qkv.weight") or mcore_key.endswith("self_attention.linear_qkv.bias"):
        return _gather_along_first_dim(weight)
    if mcore_key.endswith("mlp.linear_fc1.weight") or mcore_key.endswith("mlp.linear_fc1.bias"):
        return _gather_along_first_dim(weight)
    if mcore_key.endswith("self_attention.linear_proj.weight"):
        return _gather_along_last_dim(weight)
    if mcore_key.endswith("mlp.linear_fc2.weight"):
        return _gather_along_last_dim(weight)
    return weight


def test_weight_consistency(hf_vision_model, hf_config, mcore_model):
    hf_state_dict = hf_vision_model.state_dict()
    mcore_state_dict = mcore_model.vision_model.state_dict()

    weight_mappings = [
        ("embeddings.patch_embedding.weight", "patch_embed.proj.weight", "Patch Embedding Conv Weight"),
        ("embeddings.patch_embedding.bias", "patch_embed.proj.bias", "Patch Embedding Conv Bias"),
        ("embeddings.class_embedding", "class_embedding", "Class Embedding"),
        ("layernorm_pre.weight", "pre_layernorm.weight", "Pre-LayerNorm Weight"),
        ("layernorm_pre.bias", "pre_layernorm.bias", "Pre-LayerNorm Bias"),
        ("layernorm_post.weight", "post_layernorm.weight", "Post-LayerNorm Weight"),
        ("layernorm_post.bias", "post_layernorm.bias", "Post-LayerNorm Bias"),
    ]

    num_layers = hf_config.vision_config.num_hidden_layers
    for layer_idx in range(num_layers):
        weight_mappings.extend(
            [
                (
                    f"encoder.layers.{layer_idx}.layer_norm1.weight",
                    f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight",
                    f"Layer {layer_idx} Input LayerNorm Weight",
                ),
                (
                    f"encoder.layers.{layer_idx}.layer_norm1.bias",
                    f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_bias",
                    f"Layer {layer_idx} Input LayerNorm Bias",
                ),
                (
                    f"encoder.layers.{layer_idx}.self_attn.qkv.weight",
                    f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight",
                    f"Layer {layer_idx} QKV Weight",
                ),
                (
                    f"encoder.layers.{layer_idx}.self_attn.qkv.bias",
                    f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias",
                    f"Layer {layer_idx} QKV Bias",
                ),
                (
                    f"encoder.layers.{layer_idx}.self_attn.proj.weight",
                    f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight",
                    f"Layer {layer_idx} Proj Weight",
                ),
                (
                    f"encoder.layers.{layer_idx}.self_attn.proj.bias",
                    f"decoder.layers.{layer_idx}.self_attention.linear_proj.bias",
                    f"Layer {layer_idx} Proj Bias",
                ),
                (
                    f"encoder.layers.{layer_idx}.layer_norm2.weight",
                    f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight",
                    f"Layer {layer_idx} Post-Attn LayerNorm Weight",
                ),
                (
                    f"encoder.layers.{layer_idx}.layer_norm2.bias",
                    f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_bias",
                    f"Layer {layer_idx} Post-Attn LayerNorm Bias",
                ),
                (
                    f"encoder.layers.{layer_idx}.mlp.fc1.weight",
                    f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight",
                    f"Layer {layer_idx} MLP FC1 Weight",
                ),
                (
                    f"encoder.layers.{layer_idx}.mlp.fc1.bias",
                    f"decoder.layers.{layer_idx}.mlp.linear_fc1.bias",
                    f"Layer {layer_idx} MLP FC1 Bias",
                ),
                (
                    f"encoder.layers.{layer_idx}.mlp.fc2.weight",
                    f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight",
                    f"Layer {layer_idx} MLP FC2 Weight",
                ),
                (
                    f"encoder.layers.{layer_idx}.mlp.fc2.bias",
                    f"decoder.layers.{layer_idx}.mlp.linear_fc2.bias",
                    f"Layer {layer_idx} MLP FC2 Bias",
                ),
            ]
        )

    num_heads = hf_config.vision_config.num_attention_heads
    weight_comparisons: dict[str, dict[str, Any]] = {}

    for hf_key, mcore_key, description in weight_mappings:
        if hf_key not in hf_state_dict or mcore_key not in mcore_state_dict:
            continue

        hf_weight = hf_state_dict[hf_key].float().cpu().numpy()
        mcore_weight = _maybe_gather_tp_weight(mcore_state_dict[mcore_key], mcore_key).float().cpu().numpy()

        if hf_weight.shape != mcore_weight.shape:
            if hf_weight.ndim == 4 and mcore_weight.ndim == 2:
                hf_weight = hf_weight.reshape(hf_weight.shape[0], -1)
            elif hf_weight.ndim == 2 and mcore_weight.ndim == 4:
                mcore_weight = mcore_weight.reshape(mcore_weight.shape[0], -1)

            if hf_weight.shape != mcore_weight.shape:
                weight_comparisons[description] = {
                    "status": "shape_mismatch",
                    "hf_shape": list(hf_weight.shape),
                    "mcore_shape": list(mcore_weight.shape),
                    "hf_key": hf_key,
                    "mcore_key": mcore_key,
                }
                continue

        is_qkv_weight = "QKV Weight" in description
        is_qkv_bias = "QKV Bias" in description
        if is_qkv_weight or is_qkv_bias:
            hf_weight = convert_hf_qkv_to_mcore_layout(hf_weight, num_heads, is_bias=is_qkv_bias)

        comparison = compare_arrays(hf_weight, mcore_weight, threshold=0.9999)
        mean_diff = float(np.mean(np.abs(hf_weight - mcore_weight)))
        weight_comparisons[description] = {
            **comparison,
            "mean_diff": mean_diff,
            "hf_key": hf_key,
            "mcore_key": mcore_key,
        }

    summary = summarize_named_results(weight_comparisons)
    assert summary["mismatched"] == 0, summary


def test_vision_encoder_consistency_336px(hf_vision_model, mcore_model, hf_processor, test_image_path):
    image = load_and_resize_image(test_image_path, image_size=336)
    processed = hf_processor.image_processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
    grid_thw = processed["image_grid_thw"].to(device="cuda")
    patch_positions = _generate_patch_positions(grid_thw, pixel_values.device)

    with torch.no_grad(), _autocast_context():
        hf_debug_outputs = hf_vision_model.forward_debug(pixel_values, grid_thw, patch_positions=patch_positions)
        mcore_debug_outputs = mcore_model.vision_model.forward_debug(
            pixel_values, grid_thw=grid_thw, patch_positions=patch_positions
        )

    layers_to_compare = ["after_patch_embed", "rotary_pos_emb", "after_pre_layernorm", "before_adapter"]
    comparisons: dict[str, dict[str, Any]] = {}

    for layer_key in layers_to_compare:
        assert layer_key in hf_debug_outputs, f"HF missing {layer_key}"
        assert layer_key in mcore_debug_outputs, f"mcore missing {layer_key}"

        hf_output = hf_debug_outputs[layer_key]
        mcore_output = mcore_debug_outputs[layer_key]

        if layer_key == "rotary_pos_emb":
            hf_tensor, mcore_tensor = align_rotary_debug_tensors(hf_output, mcore_output)
        else:
            hf_tensor = hf_output.float().cpu().numpy()
            mcore_tensor = mcore_output.float().cpu().numpy()

        threshold = 0.95 if layer_key == "before_adapter" else 0.99
        comparisons[layer_key] = compare_arrays(hf_tensor, mcore_tensor, threshold=threshold)

    summary = summarize_named_results(comparisons)
    assert summary["mismatched"] == 0, summary


def test_mllm_after_merger_336px(hf_vision_model, mcore_model, hf_processor, test_image_path):
    image = load_and_resize_image(test_image_path, image_size=336)
    processed = hf_processor.image_processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
    grid_thw = processed["image_grid_thw"].to(device="cuda")

    with torch.no_grad(), _autocast_context():
        patch_positions = _generate_patch_positions(grid_thw, pixel_values.device)
        hf_debug = hf_vision_model.forward_debug(pixel_values, grid_thw, patch_positions=patch_positions)
    hf_after_merger = hf_debug.get("after_merger")
    assert hf_after_merger is not None, "HF forward_debug missing after_merger"

    with torch.no_grad(), _autocast_context():
        mcore_vision_output = mcore_model.vision_model(pixel_values, grid_thw=grid_thw)
        mcore_after_merger = mcore_model.adapter(mcore_vision_output)

    hf_np = np.squeeze(hf_after_merger.float().cpu().numpy())
    mcore_np = np.squeeze(mcore_after_merger.float().cpu().numpy())

    assert hf_np.shape == mcore_np.shape, f"shape mismatch: hf={hf_np.shape}, mcore={mcore_np.shape}"
    similarity = cosine_similarity(hf_np, mcore_np)
    assert similarity > 0.99, f"after_merger cosine too low: {similarity}"


@pytest.mark.slow
def test_encoder_layer_wise_consistency(hf_vision_model, hf_config, mcore_model, hf_processor, test_image_path):
    image = load_and_resize_image(test_image_path, image_size=336)
    processed = hf_processor.image_processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
    grid_thw = processed["image_grid_thw"].to(device="cuda")
    patch_positions = _generate_patch_positions(grid_thw, pixel_values.device)

    with torch.no_grad(), _autocast_context():
        hf_debug = hf_vision_model.forward_debug(pixel_values, grid_thw, patch_positions=patch_positions)
        mcore_debug = mcore_model.vision_model.forward_debug(
            pixel_values, grid_thw=grid_thw, patch_positions=patch_positions
        )

    hf_layers = hf_debug.get("layer_outputs", {})
    mcore_layers = mcore_debug.get("layer_outputs", {})
    assert hf_layers, "HF layer_outputs empty"
    assert mcore_layers, "mcore layer_outputs empty"

    layer_comparisons: dict[str, dict[str, Any]] = {}
    num_layers = hf_config.vision_config.num_hidden_layers

    for i in range(num_layers):
        layer_input_key = f"layer_{i}_input"
        layer_output_key = f"layer_{i}_output"

        assert layer_input_key in hf_layers and layer_input_key in mcore_layers, f"missing {layer_input_key}"
        assert layer_output_key in hf_layers and layer_output_key in mcore_layers, f"missing {layer_output_key}"

        hf_input, mcore_input = align_encoder_debug_tensors(hf_layers[layer_input_key], mcore_layers[layer_input_key])
        hf_output, mcore_output = align_encoder_debug_tensors(
            hf_layers[layer_output_key], mcore_layers[layer_output_key]
        )

        layer_threshold = 0.95
        layer_comparisons[f"layer_{i}"] = {
            "input_comparison": compare_arrays(hf_input, mcore_input, threshold=layer_threshold),
            "output_comparison": compare_arrays(hf_output, mcore_output, threshold=layer_threshold),
        }

    assert "input_hidden_states" in hf_layers or "layer_0_input" in hf_layers
    assert "input_hidden_states" in mcore_layers or "layer_0_input" in mcore_layers
    assert "final_output" in hf_layers or f"layer_{num_layers - 1}_output" in hf_layers
    assert "final_output" in mcore_layers or f"layer_{num_layers - 1}_output" in mcore_layers

    hf_in_key = "input_hidden_states" if "input_hidden_states" in hf_layers else "layer_0_input"
    mcore_in_key = "input_hidden_states" if "input_hidden_states" in mcore_layers else "layer_0_input"
    hf_out_key = "final_output" if "final_output" in hf_layers else f"layer_{num_layers - 1}_output"
    mcore_out_key = "final_output" if "final_output" in mcore_layers else f"layer_{num_layers - 1}_output"

    hf_in, mcore_in = align_encoder_debug_tensors(
        hf_layers[hf_in_key], mcore_layers[mcore_in_key]
    )
    hf_out, mcore_out = align_encoder_debug_tensors(hf_layers[hf_out_key], mcore_layers[mcore_out_key])

    encoder_input_cmp = compare_arrays(hf_in, mcore_in, threshold=0.99)
    encoder_output_cmp = compare_arrays(hf_out, mcore_out, threshold=0.95)

    layer_summary = summarize_layer_results(layer_comparisons)
    assert layer_summary["mismatched_layers"] == 0, layer_summary
    assert encoder_input_cmp["status"] == "match", encoder_input_cmp
    assert encoder_output_cmp["status"] == "match", encoder_output_cmp


@pytest.mark.slow
def test_llm_output_consistency(hf_cond_gen_model, mcore_model, hf_processor, test_image_path):
    image = load_and_resize_image(test_image_path, image_size=336)
    prompt = "Describe this image."
    text = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>"

    processed = hf_processor(text=text, images=image, return_tensors="pt")
    processed = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in processed.items()}

    input_ids = processed["input_ids"]
    pixel_values = processed["pixel_values"].to(dtype=torch.bfloat16)
    image_grid_thw = processed["image_grid_thw"]
    attention_mask = processed["attention_mask"].logical_not()
    sms = getattr(hf_cond_gen_model.config.vision_config, "spatial_merge_size", 1)
    patch_positions = _generate_patch_positions(image_grid_thw, pixel_values.device, spatial_merge_size=sms)

    with torch.no_grad(), _autocast_context():
        hf_output = hf_cond_gen_model(
            input_ids=input_ids,
            attention_mask=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            return_dict=True,
        )
    hf_logits = hf_output.logits

    with torch.no_grad(), _autocast_context():
        mcore_logits = mcore_model(
            images=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=None,
            labels=None,
            patch_positions=patch_positions,
        ).contiguous()

    assert hf_logits.shape == mcore_logits.shape, f"shape mismatch: hf={hf_logits.shape}, mcore={mcore_logits.shape}"
    similarity = cosine_similarity(hf_logits.float().cpu().numpy(), mcore_logits.float().cpu().numpy())
    assert similarity > 0.99, f"llm logits cosine too low: {similarity}"


@pytest.mark.slow
def test_hf_loading_consistency(hf_model_path, hf_vision_model, hf_config, test_image_path):
    from transformers_impl.llavaonevision2.modeling_llava_onevision2 import (
        LlavaOnevision2Config,
        LlavaOnevision2ForConditionalGeneration,
    )

    from_pretrained_full = LlavaOnevision2ForConditionalGeneration.from_pretrained(
        hf_model_path, low_cpu_mem_usage=True
    )
    from_pretrained_vision = from_pretrained_full.model.visual.to(dtype=torch.bfloat16, device="cuda").eval()

    full_config = LlavaOnevision2Config.from_pretrained(hf_model_path)
    manual_full = LlavaOnevision2ForConditionalGeneration(full_config)
    safetensors_files = sorted(glob(f"{hf_model_path}/*.safetensors"))
    assert safetensors_files, "No safetensors files found for manual load"

    state_dict: dict[str, torch.Tensor] = {}
    for sf_file in safetensors_files:
        state_dict.update(load_file(sf_file))
    missing, unexpected = manual_full.load_state_dict(state_dict, strict=False)
    vision_unexpected = [k for k in unexpected if k.startswith("model.visual.")]
    assert not vision_unexpected, f"Unexpected vision keys in manual HF load: {vision_unexpected}"
    vision_missing = [k for k in missing if k.startswith("model.visual.")]
    assert not vision_missing, f"Missing vision keys in manual HF load: {vision_missing}"

    manual_vision = manual_full.model.visual.to(dtype=torch.bfloat16, device="cuda").eval()

    fp_state = from_pretrained_vision.state_dict()
    manual_state = manual_vision.state_dict()
    assert set(fp_state.keys()) == set(manual_state.keys())

    for key in fp_state:
        a = fp_state[key].float().cpu().numpy()
        b = manual_state[key].float().cpu().numpy()
        assert a.shape == b.shape, f"weight shape mismatch for {key}: {a.shape} vs {b.shape}"
        assert np.allclose(a, b, rtol=1e-5, atol=1e-5), f"weight mismatch for {key}"

    test_image = load_and_resize_image(test_image_path, 336)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    pixel_values = transform(test_image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    patch_size = hf_config.vision_config.patch_size
    grid_thw = torch.tensor([[1, 336 // patch_size, 336 // patch_size]], dtype=torch.long, device="cuda")
    patch_positions = _generate_patch_positions(grid_thw, torch.device("cuda"))

    with torch.no_grad(), _autocast_context():
        fp_debug = from_pretrained_vision.forward_debug(pixel_values, grid_thw, patch_positions=patch_positions)
        manual_debug = manual_vision.forward_debug(pixel_values, grid_thw, patch_positions=patch_positions)

    for key, fp_val in fp_debug.items():
        assert key in manual_debug, f"manual debug missing key: {key}"
        manual_val = manual_debug[key]

        if not isinstance(fp_val, torch.Tensor):
            continue

        a = fp_val.float().cpu().numpy()
        b = manual_val.float().cpu().numpy()
        similarity = cosine_similarity(a, b)
        assert similarity > 0.9999, f"forward_debug mismatch at {key}: cosine={similarity}"

    del from_pretrained_full, from_pretrained_vision, manual_full, manual_vision
    if missing:
        assert all(k.startswith("model.language_model.") for k in missing), missing
    torch.cuda.empty_cache()
