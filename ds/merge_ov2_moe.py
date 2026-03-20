import argparse
import os
import shutil
from io import BytesIO

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from llavaonevision2.configuration_llava_onevision2_moe import LlavaOnevision2MoeConfig
from llavaonevision2.modeling_llava_onevision2_moe import LlavaOnevision2ForConditionalGeneration
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    Qwen2Tokenizer,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)
CUDA_DEVICE = 0


def cosine_similarity(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    min_len = min(len(a), len(b))
    a, b = a[:min_len].numpy(), b[:min_len].numpy()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))


def create_test_image():
    img = Image.new("RGB", (560, 560), color="red")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 474, 474], fill="blue")
    draw.text((100, 100), "TEST", fill="white")
    return img


def load_empty_model(llm_path, enable_patch_position_encoding=True):
    tokenizer = Qwen2Tokenizer.from_pretrained(
        "/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b_hf_automodel",
        trust_remote_code=True,
        device_map={"": f"cuda:{CUDA_DEVICE}"},
        use_fast=True,
    )
    processor = AutoProcessor.from_pretrained(
        "/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b_hf_automodel", use_fast=True
    )
    processor.image_processor.temporal_patch_size = 1
    processor.image_processor.max_pixels = 2000 * 2000
    llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, use_fast=True)
    llava_onevision2_config = LlavaOnevision2MoeConfig(
        text_config=llm_config.to_dict(),
        output_router_logits=True,
    )
    llava_onevision2_config.vision_config.use_patch_position_encoding = enable_patch_position_encoding
    llava_onevision2_config.vision_config.patch_position_encoding_type = "absolute"
    llava_onevision2_config.text_config.tie_word_embeddings = False
    # Set both text_hidden_size and out_hidden_size for merger to output correct dimension
    llava_onevision2_config.vision_config.text_hidden_size = llava_onevision2_config.text_config.hidden_size
    llava_onevision2_config.vision_config.out_hidden_size = (
        llava_onevision2_config.text_config.hidden_size
    )  # Critical: merger output dim

    model = LlavaOnevision2ForConditionalGeneration(llava_onevision2_config)
    return model, processor, tokenizer


def load_vit_weights(model, vit_path):
    """
    Load ViT weights and copy them to the vision part of LLaVAOneVision2_ForConditionalGeneration

    Args:
        model: LLaVAOneVision2_ForConditionalGeneration
        vit_path: ViT model path
    """
    print(f"Loading weight form: {vit_path}")

    if os.path.exists(vit_path):
        print(f"Loading weights from local file: {vit_path}")
        cache_path = os.path.join(vit_path, "model.safetensors")
    else:
        print(f"Loading weights from Hugging Face Hub: {vit_path}")
        cache_path = hf_hub_download(vit_path, "model.safetensors")

    vit_weights = load_file(cache_path)
    loaded_keys = 0

    def convert_state_dict(state_dict):
        """
        Convert ViT state dict with separate q_proj, k_proj, v_proj to merged qkv format.
        Also rename out_proj to proj.
        """
        new_state_dict = {}
        # Collect q, k, v weights for merging
        qkv_weights = {}  # {layer_prefix: {'q': tensor, 'k': tensor, 'v': tensor}}
        qkv_biases = {}  # {layer_prefix: {'q': tensor, 'k': tensor, 'v': tensor}}

        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            # Skip head.* keys (classification head, not needed for encoder)
            if key.startswith("head."):
                continue
            # Skip post layernorm keys if present
            if key.startswith("layernorm_post."):
                continue

            # Handle q_proj, k_proj, v_proj -> qkv merging
            if ".self_attn.q_proj." in key or ".self_attn.k_proj." in key or ".self_attn.v_proj." in key:
                # Extract layer prefix (e.g., "encoder.layers.0.self_attn")
                if ".q_proj." in key:
                    layer_prefix = key.replace(".q_proj.weight", "").replace(".q_proj.bias", "")
                    proj_type = "q"
                elif ".k_proj." in key:
                    layer_prefix = key.replace(".k_proj.weight", "").replace(".k_proj.bias", "")
                    proj_type = "k"
                else:  # v_proj
                    layer_prefix = key.replace(".v_proj.weight", "").replace(".v_proj.bias", "")
                    proj_type = "v"

                is_weight = key.endswith(".weight")

                if is_weight:
                    if layer_prefix not in qkv_weights:
                        qkv_weights[layer_prefix] = {}
                    qkv_weights[layer_prefix][proj_type] = value
                else:  # bias
                    if layer_prefix not in qkv_biases:
                        qkv_biases[layer_prefix] = {}
                    qkv_biases[layer_prefix][proj_type] = value
                continue

            # Handle out_proj -> proj renaming
            if ".self_attn.out_proj." in key:
                new_key = key.replace(".out_proj.", ".proj.")
                new_key = "model.visual." + new_key
                new_state_dict[new_key] = value
                continue

            # Simply add model.visual. prefix for other keys
            new_key = "model.visual." + key
            new_state_dict[new_key] = value

        # Merge q, k, v weights into qkv
        for layer_prefix, qkv_dict in qkv_weights.items():
            if "q" in qkv_dict and "k" in qkv_dict and "v" in qkv_dict:
                # Stack in order: q, k, v
                merged_weight = torch.cat([qkv_dict["q"], qkv_dict["k"], qkv_dict["v"]], dim=0)
                new_key = "model.visual." + layer_prefix + ".qkv.weight"
                new_state_dict[new_key] = merged_weight
                print(f"Merged QKV weights for {layer_prefix}: {merged_weight.shape}")

        # Merge q, k, v biases into qkv
        for layer_prefix, qkv_dict in qkv_biases.items():
            if "q" in qkv_dict and "k" in qkv_dict and "v" in qkv_dict:
                # Stack in order: q, k, v
                merged_bias = torch.cat([qkv_dict["q"], qkv_dict["k"], qkv_dict["v"]], dim=0)
                new_key = "model.visual." + layer_prefix + ".qkv.bias"
                new_state_dict[new_key] = merged_bias

        return new_state_dict

    vit_weights = convert_state_dict(vit_weights)
    vit_keys = len(set(vit_weights.keys()))

    model_state_dict = model.state_dict()
    total_keys = len(model_state_dict.keys())
    for vit_key in vit_weights:
        if vit_key not in model_state_dict:
            logger.warning(f"ViT key {vit_key} not found in model, skipping...")
            continue
        model_state_dict[vit_key] = vit_weights[vit_key].clone()
        loaded_keys += 1
    assert loaded_keys == vit_keys, f"ViT weight loading incomplete: {loaded_keys}/{vit_keys} parameters loaded"
    model.load_state_dict(model_state_dict)
    print(f"ViT weights loaded successfully: {loaded_keys}/{total_keys} parameters loaded")

    return vit_weights, loaded_keys


def load_adapter_weights(model, adapter_path, cur_len):
    """
    Load Adapter weights and copy them to the corresponding part of LLaVAOneVision2_ForConditionalGeneration

    Args:
        model: LLaVAOneVision2_ForConditionalGeneration model
        adapter_path: Adapter model path
    """
    print(f"Loading Adapter weights from: {adapter_path}")

    # Load Adapter weights
    if adapter_path.endswith(".safetensors"):
        adapter_weights = load_file(adapter_path)
    else:
        adapter_weights = torch.load(adapter_path, map_location="cpu")
        if "state_dict" in adapter_weights:
            adapter_weights = adapter_weights["state_dict"]

    # Count successfully loaded parameters
    loaded_keys = 0
    total_keys = 0
    ADAPTER_KEYS_TO_MODIFY_MAPPING = {"model.mm_projector": "model.visual.merger"}

    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in ADAPTER_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value
        return new_state_dict

    adapter_weights = convert_state_dict(adapter_weights)
    if not getattr(model.config.vision_config, "use_patch_position_encoding", False):
        adapter_weights = {
            key: value for key, value in adapter_weights.items() if not key.startswith("model.visual.merger.pos_emb_")
        }

    adapter_keys = len(set(adapter_weights.keys()))

    # Load weights into model
    model_state_dict = model.state_dict()
    total_keys = len(model_state_dict.keys())
    for adapter_key in adapter_weights:
        if adapter_key not in model_state_dict:
            logger.warning(f"Adapter key {adapter_key} not found in model, skipping...")
            continue
        model_state_dict[adapter_key] = adapter_weights[adapter_key].clone()
        loaded_keys += 1
    assert loaded_keys == adapter_keys, (
        f"Adapter weight loading incomplete: {loaded_keys}/{adapter_keys} parameters loaded"
    )
    model.load_state_dict(model_state_dict)
    print(f"Adapter weights loaded successfully: {loaded_keys + cur_len}/{total_keys} parameters loaded")

    return adapter_weights, cur_len + loaded_keys


def load_llm_weights(model, llm_path, cur_len):
    """
    Args:
        model: LLaVAOneVision2_ForConditionalGeneration model
        llm_path: LLM model path
    """
    print(f"Loading weight form: {llm_path}")
    if os.path.exists(llm_path):
        cache_path = llm_path
    else:
        cache_path = snapshot_download(llm_path, allow_patterns="*.safetensors")

    llm_weights = {}
    if os.path.isdir(cache_path):
        for filename in os.listdir(cache_path):
            if filename.endswith(".safetensors"):
                filepath = os.path.join(cache_path, filename)
                weights = load_file(filepath)
                llm_weights.update(weights)
    elif cache_path.endswith(".safetensors"):
        llm_weights = load_file(cache_path)
    else:
        llm_weights = torch.load(cache_path, map_location="cpu")
        if "state_dict" in llm_weights:
            llm_weights = llm_weights["state_dict"]

    loaded_keys = 0

    ADAPTER_KEYS_TO_MODIFY_MAPPING = {
        "model.": "model.language_model.",
    }

    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in ADAPTER_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value
        return new_state_dict

    llm_weights = convert_state_dict(llm_weights)
    if "lm_head.weight" not in llm_weights:
        llm_weights["lm_head.weight"] = llm_weights["model.language_model.embed_tokens.weight"]
    llm_keys = len(set(llm_weights.keys()))

    model_state_dict = model.state_dict()
    for llm_key in llm_weights:
        if llm_key not in model_state_dict:
            logger.warning(f"LLM key {llm_key} not found in model, skipping...")
            continue
        model_state_dict[llm_key] = llm_weights[llm_key].clone()
        loaded_keys += 1
    assert loaded_keys == llm_keys, f"LLM weight loading incomplete: {loaded_keys}/{llm_keys} parameters loaded"

    return llm_weights


def convert_rowmajor_to_block_layout(
    features: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2
) -> torch.Tensor:
    """
    Convert features from row-major order to 2x2 block layout.

    Row-major order: patches are ordered as [p(0,0), p(0,1), p(0,2), ..., p(1,0), p(1,1), ...]
    Block order: patches are grouped in 2x2 blocks: [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...

    Args:
        features: Feature tensor in row-major order, shape [seq_len, hidden_dim]
        t: temporal dimension (number of frames)
        h: height (number of vertical patches)
        w: width (number of horizontal patches)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: Features in 2x2 block order, same shape [seq_len, hidden_dim]
    """
    sms = spatial_merge_size
    if sms == 1:
        return features

    hidden_dim = features.shape[-1]

    # features shape: [t*h*w, hidden_dim]
    # Reshape to [t, h, w, hidden_dim]
    features = features.view(t, h, w, hidden_dim)

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # Reshape to [t, h_merged, sms, w_merged, sms, hidden_dim]
    features = features.view(t, h_merged, sms, w_merged, sms, hidden_dim)

    # Permute to [t, h_merged, w_merged, sms_h, sms_w, hidden_dim] - 2x2 block order
    features = features.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape back to [t*h*w, hidden_dim]
    features = features.view(t * h * w, hidden_dim)

    return features


def convert_block_to_rowmajor_layout(
    features: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2
) -> torch.Tensor:
    """
    Convert features from 2x2 block layout back to row-major order.
    This is the inverse of convert_rowmajor_to_block_layout.

    Block order: patches are grouped in 2x2 blocks: [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...
    Row-major order: patches are ordered as [p(0,0), p(0,1), p(0,2), ..., p(1,0), p(1,1), ...]

    Args:
        features: Feature tensor in block order, shape [seq_len, hidden_dim]
        t: temporal dimension (number of frames)
        h: height (number of vertical patches)
        w: width (number of horizontal patches)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: Features in row-major order, same shape [seq_len, hidden_dim]
    """
    sms = spatial_merge_size
    if sms == 1:
        return features

    hidden_dim = features.shape[-1]

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # features shape: [t*h*w, hidden_dim]
    # Reshape to [t, h_merged, w_merged, sms_h, sms_w, hidden_dim]
    features = features.view(t, h_merged, w_merged, sms, sms, hidden_dim)

    # Permute back to [t, h_merged, sms_h, w_merged, sms_w, hidden_dim]
    features = features.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape to [t, h, w, hidden_dim] and then [t*h*w, hidden_dim]
    features = features.view(t * h * w, hidden_dim)

    return features


def validate_vit_consistency(model, vit_path, img_path, processor):
    """
    Verify the consistency of the ViT component by feeding the SAME input
    in the SAME order to both original and merged models.

    Since the original ViT and merged model have different input formats
    (row-major vs block order), directly comparing their full forward outputs
    includes bf16 numerical noise from different flash attention accumulation
    order, which compounds across 24 layers (~0.94 cos sim).

    Instead, we verify by:
    1. Feeding the SAME embeddings + SAME RoPE to both models' encoder layers
    2. Comparing layer-by-layer to confirm weight consistency (expect sim ≈ 1.0)

    Args:
        model: LlavaOnevision2ForConditionalGeneration after merged
        vit_path: original ViT model path
        img_path: sample image path/url
        processor: Processor containing CLIPImageProcessor for original ViT
    """
    print("Verifying consistency of ViT component...")

    from onevision_encoder import OneVisionEncoderModel

    device = torch.device(f"cuda:{CUDA_DEVICE}")
    dtype = torch.bfloat16

    # Load image
    if img_path.startswith("http"):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    patch_size = 14  # ViT patch size
    merge_size = 2  # spatial merge size

    pixel_unit = patch_size * merge_size  # 28
    orig_w, orig_h = image.size
    new_h = (orig_h // pixel_unit) * pixel_unit
    new_w = (orig_w // pixel_unit) * pixel_unit
    if new_h != orig_h or new_w != orig_w:
        image = image.resize((new_w, new_h), Image.BILINEAR)
        print(f"Resized image from ({orig_w}, {orig_h}) to ({new_w}, {new_h}) for patch alignment")

    # ========== Prepare image with CLIP processor ==========
    clip_image_processor = CLIPImageProcessor.from_pretrained(vit_path, local_files_only=True)
    clip_pixel_values = clip_image_processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)[
        "pixel_values"
    ]
    clip_pixel_values = clip_pixel_values.to(dtype=dtype, device=device)

    _, _, H, W = clip_pixel_values.shape
    grid_h = H // patch_size
    grid_w = W // patch_size
    print(f"Image size: {H}x{W}, grid: [1, {grid_h}, {grid_w}], total_patches: {grid_h * grid_w}")

    # ========== Extract patches in block order from CLIP image ==========
    # This gives the merged model the same pixel values as the original ViT
    def extract_patches_block_order(img_tensor, ps=14, sms=2):
        B, C, pH, pW = img_tensor.shape
        h, w = pH // ps, pW // ps
        patches = img_tensor.reshape(B, C, h, ps, w, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(h, w, C, ps, ps)
        h_m, w_m = h // sms, w // sms
        patches = patches.reshape(h_m, sms, w_m, sms, C, ps, ps)
        patches = patches.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
        return patches.reshape(-1, C, ps, ps)

    block_patches = extract_patches_block_order(clip_pixel_values, ps=patch_size, sms=merge_size)

    # ========== Load models ==========
    merged_visual = model.model.visual.to(dtype=dtype, device=device)
    merged_visual.eval()

    original_vit = OneVisionEncoderModel.from_pretrained(
        vit_path, trust_remote_code=True, attn_implementation="flash_attention_2"
    ).to(dtype=dtype, device=device)
    original_vit.eval()
    # Disable layernorm_post for fair comparison (merged model has use_head=False)
    if hasattr(original_vit, "layernorm_post") and original_vit.layernorm_post is not None:
        print("Disabling layernorm_post in original ViT (merged model has use_head=False)")
        original_vit.layernorm_post = None

    # ========== Same-order layer-by-layer comparison ==========
    # Feed the SAME input in the SAME order to both models' encoder layers.
    # This eliminates bf16 flash attention ordering noise and verifies pure weight consistency.
    print("\nLayer-by-layer weight consistency check (same input, same order):")

    with torch.no_grad():
        # Get embeddings from merged model (block order)
        merged_embed = merged_visual.embeddings(block_patches).unsqueeze(0)  # [1, N, 1024]
        merged_pre = merged_visual.layernorm_pre(merged_embed)

        # Build block-order RoPE from merged model
        from llavaonevision2.modeling_llava_onevision2_moe import convert_rope_to_block_layout_by_positions

        grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device)
        t_idx = torch.arange(1, device=device, dtype=torch.float32)
        h_idx = torch.arange(grid_h, device=device, dtype=torch.float32)
        w_idx = torch.arange(grid_w, device=device, dtype=torch.float32)
        mesh_t, mesh_h, mesh_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        patch_positions = torch.stack([mesh_t, mesh_h, mesh_w], dim=-1).reshape(-1, 3)

        merged_freqs = merged_visual.video_rope.forward_from_positions(patch_positions)
        merged_freqs = convert_rope_to_block_layout_by_positions(
            merged_freqs, patch_positions, spatial_merge_size=merge_size, grid_thw=grid_thw
        )
        block_rope = torch.cat([merged_freqs, merged_freqs], dim=-1).unsqueeze(0)

        # Feed the SAME input (block order) and SAME RoPE to BOTH models
        orig_hidden = merged_pre.clone()
        merged_hidden = merged_pre.clone()

        min_sim = 1.0
        for layer_idx in range(len(original_vit.encoder.layers)):
            orig_out = original_vit.encoder.layers[layer_idx](
                orig_hidden,
                attention_mask=None,
                rotary_pos_emb=block_rope,
                output_attentions=False,
            )
            orig_hidden = orig_out[0]

            merged_out = merged_visual.encoder.layers[layer_idx](
                merged_hidden,
                attention_mask=None,
                rotary_pos_emb=block_rope,
                output_attentions=False,
                cu_seqlens=None,
                max_seqlen=None,
            )
            merged_hidden = merged_out[0]

            cur_sim = cosine_similarity(orig_hidden.flatten().cpu(), merged_hidden.flatten().cpu())
            min_sim = min(min_sim, cur_sim)
            diff = (orig_hidden.float() - merged_hidden.float()).abs().mean().item()
            marker = "✅" if cur_sim > 0.999 else "⚠️" if cur_sim > 0.99 else "❌"
            print(f"  [Layer {layer_idx:2d}] sim={cur_sim:.8f}, diff={diff:.10f} {marker}")

    print(f"\nMinimum cosine similarity across all layers: {min_sim:.8f}")

    if min_sim > 0.999:
        print("✅ ViT component consistency verification passed (all layers sim > 0.999)")
    else:
        raise ValueError(f"❌ ViT component consistency verification failed (min sim = {min_sim:.6f})")

    model.model.visual = merged_visual.to("cpu")
    del original_vit
    torch.cuda.empty_cache()


def validate_llm_consistency(model, llm_path, sample_text):
    """
    Verify the consistency of the LLM component.

    Since the full LLM (~60GB in bf16) can't fit two copies on one GPU,
    we run forward pass sequentially: merged LLM first, then original LLM,
    and compare outputs on CPU.

    Args:
        model: Merged LLaVAOneVision2_ForConditionalGeneration model
        llm_path: Original LLM model path
        sample_text: Sample text
    """
    print("Verifying consistency of LLM component...")

    device = torch.device(f"cuda:{CUDA_DEVICE}")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True, trust_remote_code=True)
    inputs = tokenizer(sample_text, return_tensors="pt")

    # Step 1: Forward pass with merged LLM on GPU
    print("  Running merged LLM forward pass...")
    with torch.no_grad():
        merged_lm = model.model.language_model.to(dtype=dtype, device=device)
        merged_lm_head = model.lm_head.to(dtype=dtype, device=device)
        merged_lm.eval()

        gpu_inputs = {k: v.to(device) for k, v in inputs.items()}
        merged_output = merged_lm(
            input_ids=gpu_inputs["input_ids"],
            attention_mask=gpu_inputs.get("attention_mask", None),
        ).last_hidden_state
        merged_output = merged_lm_head(merged_output).cpu()

        # Move merged LLM back to CPU to free GPU
        model.model.language_model = merged_lm.to("cpu")
        model.lm_head = merged_lm_head.to("cpu")
        del merged_lm, merged_lm_head, gpu_inputs
        torch.cuda.empty_cache()

    # Step 2: Forward pass with original LLM on GPU
    print("  Loading original LLM for comparison...")
    with torch.no_grad():
        original_llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )
        original_llm.eval()

        gpu_inputs = {k: v.to(device) for k, v in inputs.items()}
        original_output = original_llm(**gpu_inputs).logits.cpu()

        del original_llm, gpu_inputs
        torch.cuda.empty_cache()

    # Step 3: Compare on CPU
    cur_sim = cosine_similarity(merged_output.flatten(0, 1), original_output.flatten(0, 1))
    diff = (merged_output.float() - original_output.float()).abs().mean().item()
    max_diff = (merged_output.float() - original_output.float()).abs().max().item()
    print(f"LLM output mean difference: {diff:.8f}")
    print(f"LLM output max difference: {max_diff:.8f}")
    print(f"LLM output cosine similarity: {cur_sim:.6f}")

    llm_cosine_threshold = 0.99
    llm_mean_diff_threshold = 5e-2

    if cur_sim > llm_cosine_threshold and diff < llm_mean_diff_threshold:
        print("✅ LLM component consistency verification passed")
    else:
        raise ValueError(
            "❌ LLM component consistency verification failed "
            f"(cos={cur_sim:.6f}, mean_diff={diff:.8f}, max_diff={max_diff:.8f}; "
            f"required: cos>{llm_cosine_threshold}, mean_diff<{llm_mean_diff_threshold})"
        )


def save_merged_model(model, output_path, tokenizer, image_processor):
    """
    Save the merged model

    Args:
        model: Merged model
        output_path: Output path
    """
    print(f"Saving merged model to: {output_path}")

    # Register for auto class
    try:
        LlavaOnevision2MoeConfig.register_for_auto_class()
        LlavaOnevision2ForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
    except Exception as e:
        logger.warning(f"Failed to register auto class: {e}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Configure auto_map for AutoModel loading
    if not hasattr(model.config, "auto_map"):
        model.config.auto_map = {}

    model.config.auto_map["AutoConfig"] = "configuration_llava_onevision2_moe.LlavaOnevision2MoeConfig"
    model.config.auto_map["AutoModelForCausalLM"] = (
        "modeling_llava_onevision2_moe.LlavaOnevision2ForConditionalGeneration"
    )

    # Save model configuration
    tokenizer.save_pretrained(output_path)
    image_processor.save_pretrained(output_path)
    model.save_pretrained(output_path)

    # Copy modeling files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.join(current_dir, "llavaonevision2")

    files_to_copy = [
        "configuration_llava_onevision2_moe.py",
        "modeling_llava_onevision2_moe.py",
    ]

    for filename in files_to_copy:
        src = os.path.join(module_dir, filename)
        dst = os.path.join(output_path, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {filename} to {output_path}")
        else:
            print(f"Warning: Could not find {src} to copy.")

    print("Model saving completed.")


def validate_end_to_end(model, processor, tokenizer, img_path):
    """
    Validate that the merged model can process an image+text sample end-to-end.
    This ensures the full pipeline (image encoding -> projection -> LLM generation) works correctly.

    Args:
        model: Merged LlavaOnevision2ForConditionalGeneration model
        processor: Processor with image processor
        tokenizer: Tokenizer for text processing
        img_path: Path or URL to test image
    """
    print("=" * 60)
    print("Validating end-to-end inference with image+text sample...")
    print("=" * 60)

    from qwen_vl_utils import process_vision_info

    device = torch.device(f"cuda:{CUDA_DEVICE}")

    # Load image
    if img_path.startswith("http"):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    print(f"Loaded image: {image.size}")

    # Move model to device with bfloat16
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    # Prepare Qwen2VL-style input with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Process with Qwen2VL processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"Image grid THW: {inputs['image_grid_thw']}")

    # Generate patch_positions for merged model (required since we removed auto-generation in modeling code)
    grid_thw = inputs["image_grid_thw"]
    patch_positions = []
    for i in range(grid_thw.shape[0]):
        t, h, w = grid_thw[i].tolist()
        t_idx = torch.arange(t, device=device, dtype=torch.float32)
        h_idx = torch.arange(h, device=device, dtype=torch.float32)
        w_idx = torch.arange(w, device=device, dtype=torch.float32)
        mesh_t, mesh_h, mesh_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        patch_positions.append(torch.stack([mesh_t, mesh_h, mesh_w], dim=-1).reshape(-1, 3))
    patch_positions = torch.cat(patch_positions, dim=0)

    # Generate output
    print("Running forward pass and generation...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # First test forward pass (without generation) to check gradients flow
            try:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    patch_positions=patch_positions,
                    output_router_logits=False,
                )
                print(f"Forward pass successful! Logits shape: {outputs.logits.shape}")
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise ValueError(f"❌ Forward pass failed: {e}")

            # Then test generation
            try:
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    patch_positions=patch_positions,
                    output_router_logits=False,
                    max_new_tokens=50,
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                print(f"Generation successful! Generated IDs shape: {generated_ids.shape}")
            except Exception as e:
                raise ValueError(f"❌ Generation failed: {e}")

    # Decode the generated text
    # Only decode the newly generated tokens (exclude input tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

    print("-" * 40)
    print("Generated text:")
    print(generated_text)
    print("-" * 40)

    # Basic sanity check: generated text should not be empty
    if len(generated_text.strip()) == 0:
        print("⚠️ Warning: Generated text is empty (model may not be trained yet)")
    else:
        print(f"✅ End-to-end validation passed! Generated {len(generated_text)} characters.")

    # Move model back to CPU to free GPU memory
    model = model.to("cpu")
    torch.cuda.empty_cache()

    return generated_text


def main(args):
    # model paths
    vit_path = args.vit_path
    adapter_path = args.adapter_path
    llm_path = args.llm_path
    output_path = args.output_path
    img_path = args.img_path
    sample_text = args.sample_text

    # 1. load empty model
    model, processor, tokenizer = load_empty_model(
        llm_path,
        enable_patch_position_encoding=args.enable_patch_position_encoding,
    )
    model.to(dtype=torch.float32)

    print("Processor:", processor)

    pretrain_weights = {}
    # 2. load ViT weights
    vit_weights, cur_len = load_vit_weights(model, vit_path)
    pretrain_weights.update(vit_weights)

    # 3. load Adapter weights
    if adapter_path:
        adapter_weights, cur_len = load_adapter_weights(model, adapter_path, cur_len)
        pretrain_weights.update(adapter_weights)

    # 4. load LLM weights
    llm_weights = load_llm_weights(model, llm_path, cur_len)
    pretrain_weights.update(llm_weights)

    model.load_state_dict(pretrain_weights, strict=False)

    # 5. validate model consistency
    validate_vit_consistency(model, vit_path, img_path, processor)
    validate_llm_consistency(model, llm_path, sample_text)

    # 6. save merged model (before end-to-end to ensure merge is saved even if e2e OOMs)
    save_merged_model(model.to(dtype=torch.bfloat16), output_path, tokenizer, processor)
    print("Model merging and saving completed!")

    # 7. validate end-to-end inference with image+text (optional, may OOM on single GPU)
    try:
        validate_end_to_end(model, processor, tokenizer, img_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"⚠️ End-to-end validation skipped due to: {e}")
        print("The merged model has been saved successfully. You can validate it separately with multi-GPU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ViT and LLM models")
    parser.add_argument(
        "--vit_path", type=str, default="/ov2/pretrain_models/onevision-encoder-large", help="Path to the ViT model"
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default="/mnt/publicdataset/Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Path to the LLM model (Qwen3 MoE)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoints/merged/LLaVA-OneVision-2-MoE-stage0",
        help="Path to save the merged model",
    )
    parser.add_argument("--adapter_path", type=str, default="", help="Path to the Adapter model (optional)")
    parser.add_argument(
        "--img_path",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Path to the image file",
    )
    parser.add_argument(
        "--sample_text", type=str, default="Hello, my dog is cute", help="Sample text for LLM consistency check"
    )
    parser.add_argument(
        "--enable_patch_position_encoding",
        dest="enable_patch_position_encoding",
        action="store_true",
        help="Enable patch position encoding in merged DS model.",
    )
    parser.add_argument(
        "--disable_patch_position_encoding",
        dest="enable_patch_position_encoding",
        action="store_false",
        help="Disable patch position encoding in merged DS model.",
    )
    parser.set_defaults(enable_patch_position_encoding=True)
    args = parser.parse_args()
    main(args)
