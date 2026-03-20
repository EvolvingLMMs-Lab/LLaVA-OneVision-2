import argparse
import json
import os
from contextlib import nullcontext
from datetime import datetime
from glob import glob
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training import print_rank_0
from megatron.training.checkpointing import _load_base_checkpoint, fix_query_key_value_ordering
from megatron.training.training import get_model, unwrap_model
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

import transformers
from aiak_training_llm.train.arguments import aiak_extra_train_args_provider, parse_arguments, validate_aiak_extra_args
from aiak_training_llm.train.pretrain.pretrain_llava_onevision2 import model_provider
from aiak_training_llm.utils import get_args, initialize_aiak_megatron
from ds.llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2Config
from ds.llavaonevision2.configuration_llava_onevision2_moe import LlavaOnevision2MoeConfig
from ds.llavaonevision2.modeling_llava_onevision2 import LlavaOnevision2ForConditionalGeneration, LlavaOnevision2Model
from ds.llavaonevision2.modeling_llava_onevision2_moe import (
    LlavaOnevision2ForConditionalGeneration as LlavaOnevision2MoeForConditionalGeneration,
)
from ds.llavaonevision2.modeling_llava_onevision2_moe import LlavaOnevision2Model as LlavaOnevision2MoeModel
from transformers import AutoProcessor


# Suppress transformers warnings
transformers.logging.set_verbosity_error()


def log(level: str, msg: str):
    """Log message with timestamp."""
    print_rank_0(f"[{level}] {datetime.now():%Y-%m-%d %H:%M:%S} - {msg}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two arrays."""
    a, b = a.flatten(), b.flatten()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))


def convert_hf_qkv_to_mcore_layout(hf_weight: np.ndarray, num_heads: int, is_bias: bool = False) -> np.ndarray:
    """
    Convert HuggingFace QKV weight/bias layout to Megatron-Core layout.

    HuggingFace layout (concatenated Q, K, V):
        Weight: [3 * hidden_size, hidden_size] where rows are [Q_all; K_all; V_all]
        Bias: [3 * hidden_size] where elements are [Q_all; K_all; V_all]

    Megatron-Core layout (interleaved per head):
        Weight: [3 * hidden_size, hidden_size] where rows are interleaved per head
                [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ..., Q_hn-1, K_hn-1, V_hn-1]
        Bias: [3 * hidden_size] with same interleaved pattern

    Args:
        hf_weight: HuggingFace QKV weight [3*hidden_size, hidden_size] or bias [3*hidden_size]
        num_heads: Number of attention heads
        is_bias: Whether this is a bias tensor (1D) or weight tensor (2D)

    Returns:
        np.ndarray: Weight/bias converted to Megatron-Core layout
    """
    # Input validation
    if hf_weight is None:
        raise ValueError("hf_weight cannot be None")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    if is_bias:
        # Bias: [3 * hidden_size]
        if hf_weight.ndim != 1:
            raise ValueError(f"Expected 1D tensor for bias, got shape {hf_weight.shape}")

        total_size = hf_weight.shape[0]
        if total_size % 3 != 0:
            raise ValueError(f"Bias size {total_size} is not divisible by 3")

        hidden_size = total_size // 3
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")

        head_dim = hidden_size // num_heads

        # Split into Q, K, V
        q = hf_weight[:hidden_size]  # [hidden_size]
        k = hf_weight[hidden_size : 2 * hidden_size]  # [hidden_size]
        v = hf_weight[2 * hidden_size :]  # [hidden_size]

        # Reshape to per-head: [num_heads, head_dim]
        q = q.reshape(num_heads, head_dim)
        k = k.reshape(num_heads, head_dim)
        v = v.reshape(num_heads, head_dim)

        # Stack as [num_heads, 3, head_dim] - interleaved QKV per head
        mcore_bias = np.stack([q, k, v], axis=1)  # [num_heads, 3, head_dim]

        # Reshape back to [3 * hidden_size]
        mcore_bias = mcore_bias.reshape(-1)

        return mcore_bias
    else:
        # Weight: [3 * hidden_size, hidden_size]
        if hf_weight.ndim != 2:
            raise ValueError(f"Expected 2D tensor for weight, got shape {hf_weight.shape}")

        out_features, in_features = hf_weight.shape
        if out_features % 3 != 0:
            raise ValueError(f"out_features {out_features} is not divisible by 3")

        hidden_size = out_features // 3
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")

        head_dim = hidden_size // num_heads

        # Split into Q, K, V weights
        q_weight = hf_weight[:hidden_size, :]  # [hidden_size, hidden_size]
        k_weight = hf_weight[hidden_size : 2 * hidden_size, :]  # [hidden_size, hidden_size]
        v_weight = hf_weight[2 * hidden_size :, :]  # [hidden_size, hidden_size]

        # Reshape to per-head: [num_heads, head_dim, hidden_size]
        q_weight = q_weight.reshape(num_heads, head_dim, in_features)
        k_weight = k_weight.reshape(num_heads, head_dim, in_features)
        v_weight = v_weight.reshape(num_heads, head_dim, in_features)

        # Stack as [num_heads, 3, head_dim, hidden_size] - interleaved QKV per head
        mcore_weight = np.stack([q_weight, k_weight, v_weight], axis=1)  # [num_heads, 3, head_dim, in_features]

        # Reshape back to [3 * hidden_size, hidden_size]
        mcore_weight = mcore_weight.reshape(out_features, in_features)

        return mcore_weight


def load_and_resize_image(image_path: str, image_size: int = 336) -> Image.Image:
    """Load image from path and resize to specified size."""
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
        img = img.resize((image_size, image_size)).convert("RGB")
        log("INFO", f"Successfully loaded and resized image to {image_size}x{image_size}")
        return img
    except Exception as e:
        log("ERROR", f"Error loading image: {e}")
        # Fallback to creating a simple test image
        img = Image.new("RGB", (image_size, image_size), color="red")
        return img


def convert_mcore_pixel_values_to_hf_format(
    pixel_values_mcore: torch.Tensor,
    image_grid_thw: torch.Tensor,
    patch_size: int = 14,
    temporal_patch_size: int = 1,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """
    Convert Megatron-Core pixel_values format (from Qwen2VLImageProcessor) back to HuggingFace format.

    The Qwen2VLImageProcessor outputs pixel_values in a flattened format with spatial_merge_size=2:
    - Mcore format: (num_patches, patch_dim) where patch_dim = C * temporal_patch_size * patch_size * patch_size
    - The patches are arranged in 2x2 blocks (spatial_merge_size=2) and flattened
    - HF format: (batch_size, C, H, W) with patches in 1x1 (row-major) order

    We need to convert from spatial_merge_size=2 arrangement back to spatial_merge_size=1 (row-major) order.

    For spatial_merge_size=2, patches are grouped in 2x2 blocks:
    - Original row-major order (spatial_merge_size=1): 0, 1, 2, 3, 4, 5, 6, 7, ...
    - With spatial_merge_size=2, patches are grouped: [0,1,w,w+1], [2,3,w+2,w+3], ...

    Args:
        pixel_values_mcore: Megatron input tensor of shape [num_patches, patch_dim]
        image_grid_thw: Tensor of shape [num_images, 3] containing (t, h_patches, w_patches) for each image
                        Note: h_patches and w_patches are UNMERGED original patch counts
                        num_patches == t * h_patches * w_patches
        patch_size: Size of each square patch (e.g., 14)
        temporal_patch_size: Temporal patch size (default 1 for images)
        spatial_merge_size: Spatial merge factor used by the processor (default 2)

    Returns:
        torch.Tensor: HuggingFace input tensor of shape [B, C, H, W]
    """
    C = 3
    num_patches, patch_dim = pixel_values_mcore.shape
    t, h_patches, w_patches = image_grid_thw[0].tolist()

    # Each patch contains C * temporal_patch_size * patch_size * patch_size values
    expected_patch_dim = C * temporal_patch_size * patch_size * patch_size
    assert patch_dim == expected_patch_dim, f"Expected patch_dim={expected_patch_dim}, got {patch_dim}"

    # grid_thw contains the UNMERGED patch dimensions
    # num_patches == t * h_patches * w_patches
    expected_num_patches = t * h_patches * w_patches
    assert num_patches == expected_num_patches, (
        f"Expected {expected_num_patches} patches (t={t}, h_patches={h_patches}, w_patches={w_patches}), got {num_patches}"
    )

    # Calculate merged dimensions
    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size

    # Reshape patches: (num_patches, C * temporal_patch_size * patch_size * patch_size)
    # -> (num_patches, C, temporal_patch_size, patch_size, patch_size)
    patches = pixel_values_mcore.view(num_patches, C, temporal_patch_size, patch_size, patch_size)

    # For temporal_patch_size=1, squeeze the temporal dimension
    if temporal_patch_size == 1:
        patches = patches.squeeze(2)  # (num_patches, C, patch_size, patch_size)

    # The mcore patches are arranged with spatial_merge_size=2:
    # They are in order of merged blocks: for each merged block position (i, j),
    # the 4 patches within the 2x2 block are consecutive.
    # Shape: (h_merged * w_merged * sms * sms, C, patch_size, patch_size)

    # Reshape to: (h_merged, w_merged, spatial_merge_size, spatial_merge_size, C, patch_size, patch_size)
    patches = patches.view(h_merged, w_merged, spatial_merge_size, spatial_merge_size, C, patch_size, patch_size)

    # Permute to get row-major order:
    # From: (h_merged, w_merged, sms_h, sms_w, C, patch_size, patch_size)
    # To:   (h_merged, sms_h, w_merged, sms_w, C, patch_size, patch_size)
    # This interleaves the spatial merge dimensions properly
    patches = patches.permute(0, 2, 1, 3, 4, 5, 6)

    # Now reshape to: (h_patches, w_patches, C, patch_size, patch_size)
    patches = patches.contiguous().view(h_patches, w_patches, C, patch_size, patch_size)

    # Permute to: (C, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4)

    # Reshape to final image: (C, h_patches * patch_size, w_patches * patch_size)
    H = h_patches * patch_size
    W = w_patches * patch_size
    image = patches.contiguous().view(C, H, W)

    # Add batch dimension: (1, C, H, W)
    image = image.unsqueeze(0)

    return image


def convert_hf_output_to_mcore_format(
    hf_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """
    Convert HuggingFace forward_debug output format to Megatron-Core format.

    HF output is in row-major order (spatial_merge_size=1):
    - Patches ordered: p(0,0), p(0,1), p(0,2), ..., p(1,0), p(1,1), ...

    Mcore output is in 2x2 block order (spatial_merge_size=2):
    - Patches grouped: [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...

    Args:
        hf_output: HF output tensor of shape [num_patches, hidden_dim]
                   where num_patches = h_patches * w_patches
        image_grid_thw: Tensor of shape [num_images, 3] containing (t, h_patches, w_patches)
                        Note: h_patches and w_patches are UNMERGED original patch counts
        spatial_merge_size: Spatial merge factor (default 2)

    Returns:
        torch.Tensor: Output rearranged to mcore format
    """
    t, h_patches, w_patches = image_grid_thw[0].tolist()

    # Handle different tensor shapes
    # Could be 2D: (num_patches, hidden_dim)
    # Could be 3D: (seq_len, batch, hidden_dim) or (batch, seq_len, hidden_dim)
    shape = hf_output.shape

    if len(shape) == 2:
        num_patches, hidden_dim = shape
        original_shape = "2D"
    elif len(shape) == 3:
        # Assume (seq_len, batch, hidden_dim) format from transformer layers
        # or (batch, seq_len, hidden_dim) - we'll handle both
        if shape[1] == 1:
            # (seq_len, 1, hidden_dim) - typical HF encoder output
            num_patches = shape[0]
            hidden_dim = shape[2]
            hf_output = hf_output.squeeze(1)  # Remove batch dim: (seq_len, hidden_dim)
            original_shape = "3D_squeeze"
        elif shape[0] == 1:
            # (1, seq_len, hidden_dim) - typical batch=1 format
            num_patches = shape[1]
            hidden_dim = shape[2]
            hf_output = hf_output.squeeze(0)  # Remove batch dim: (seq_len, hidden_dim)
            original_shape = "3D_squeeze_0"
        else:
            # Cannot determine format - return as is
            return hf_output
    else:
        # Unknown shape - return as is
        return hf_output

    expected_num_patches = h_patches * w_patches
    if num_patches != expected_num_patches:
        # Shape mismatch - return as is (may be a different stage output)
        return hf_output

    # Calculate merged dimensions
    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size

    # Reshape HF output from row-major: (h_patches * w_patches, hidden_dim)
    # to: (h_patches, w_patches, hidden_dim)
    patches = hf_output.view(h_patches, w_patches, hidden_dim)

    # Now reshape to interleave for spatial_merge_size=2
    # From: (h_merged * sms, w_merged * sms, hidden_dim)
    # To: (h_merged, sms, w_merged, sms, hidden_dim)
    sms = spatial_merge_size
    patches = patches.view(h_merged, sms, w_merged, sms, hidden_dim)

    # Permute to group 2x2 blocks together:
    # From: (h_merged, sms_h, w_merged, sms_w, hidden_dim)
    # To: (h_merged, w_merged, sms_h, sms_w, hidden_dim)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()

    # Reshape to final mcore format: (num_patches, hidden_dim)
    patches = patches.view(num_patches, hidden_dim)

    return patches


class LlavaOnevision2ConsistencyTester:
    """Tester for HuggingFace vs Megatron-LM LlavaOnevision2 vision encoder consistency."""

    def __init__(
        self,
        hf_model_path: str,
        preprocessor_path: str,
        test_image_path: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
        test_profile: str = "full",
    ):
        self.hf_model_path = hf_model_path
        self.preprocessor_path = preprocessor_path
        self.test_image_path = test_image_path
        self.test_profile = test_profile
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_cond_gen_model = None

        self.hf_processor = AutoProcessor.from_pretrained(self.preprocessor_path, trust_remote_code=True)
        self.image_processor = self.hf_processor.image_processor

        # Load models
        self.hf_model, self.hf_config = self._load_hf_model()
        self.megatron_model = self._load_megatron_model()

    def _load_hf_config_dict(self) -> dict[str, Any]:
        config_path = os.path.join(self.hf_model_path, "config.json")
        with open(config_path) as f:
            return json.load(f)

    def _resolve_hf_classes(self, config_dict: dict[str, Any]):
        if config_dict.get("model_type") == "llava_onevision2_moe":
            return (
                LlavaOnevision2MoeConfig.from_dict(config_dict),
                LlavaOnevision2MoeModel,
                LlavaOnevision2MoeForConditionalGeneration,
            )
        return (
            LlavaOnevision2Config.from_dict(config_dict),
            LlavaOnevision2Model,
            LlavaOnevision2ForConditionalGeneration,
        )

    def _generate_patch_positions(self, grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
        patch_positions = []
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            t_idx = torch.arange(t, device=device, dtype=torch.float32)
            h_idx = torch.arange(h, device=device, dtype=torch.float32)
            w_idx = torch.arange(w, device=device, dtype=torch.float32)
            mesh_t, mesh_h, mesh_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
            patch_positions.append(torch.stack([mesh_t, mesh_h, mesh_w], dim=-1).reshape(-1, 3))
        return torch.cat(patch_positions, dim=0)

    def _tokenize_and_preprocess(self, image, text):
        processed = self.hf_processor(text=text, images=image, return_tensors="pt")
        processed = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in processed.items()}
        input_ids = processed["input_ids"][0]  # [seq_len]
        image_grid_thw = processed["image_grid_thw"]  # [num_images, 3] or similar
        pixel_values = processed["pixel_values"]  # wrap in list for consistency
        attention_mask_neg = processed["attention_mask"][0].logical_not()  # inverted mask

        return input_ids, pixel_values, image_grid_thw, attention_mask_neg

    def _autocast_context(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _process_image_for_mcore(self, image: Image.Image) -> tuple:
        """
        Process image using the Qwen2VLImageProcessor for Megatron model.

        Returns:
            tuple: (pixel_values, image_grid_thw) for Megatron model
        """
        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
        image_grid_thw = processed["image_grid_thw"].to(device=self.device)
        return pixel_values, image_grid_thw

    def _process_image_for_hf(self, pixel_values_mcore: torch.Tensor, image_grid_thw: torch.Tensor) -> tuple:
        """
        Return pixel values for HuggingFace model.

        Since the new HF model now uses the same 2x2 memory layout as Megatron-Core,
        no conversion is needed - we can use the mcore pixel values directly.

        Args:
            pixel_values_mcore: Tensor from Qwen2VLImageProcessor, shape (num_patches, patch_dim)
            image_grid_thw: Tensor of shape [num_images, 3] with (t, h, w)

        Returns:
            tuple: (pixel_values, grid_thw) for HuggingFace model (same as input)
        """
        # No conversion needed - HF model now uses the same 2x2 memory layout as mcore
        return pixel_values_mcore, image_grid_thw

    def _load_hf_model(self):
        """Load HuggingFace LlavaOnevision2 model."""
        log("INFO", f"Loading HuggingFace model from: {self.hf_model_path}")

        config_dict = self._load_hf_config_dict()
        config, model_cls, cond_gen_model_cls = self._resolve_hf_classes(config_dict)
        self._hf_cond_gen_model_cls = cond_gen_model_cls
        full_model = model_cls.from_pretrained(self.hf_model_path, low_cpu_mem_usage=True)

        vision_model = full_model.visual
        del full_model

        # Convert to bfloat16 and move to cuda to match Megatron model
        vision_model = vision_model.to(dtype=torch.bfloat16, device=self.device)
        vision_model = vision_model.eval()
        log(
            "INFO",
            f"✓ HuggingFace vision model loaded with {sum(p.numel() for p in vision_model.parameters())} parameters",
        )

        return vision_model, config

    def _get_hf_cond_gen_model(self):
        if self.hf_cond_gen_model is None:
            log("INFO", "Loading HuggingFace ForConditionalGeneration model for logits comparison...")
            self.hf_cond_gen_model = self._hf_cond_gen_model_cls.from_pretrained(
                self.hf_model_path,
                low_cpu_mem_usage=True,
            )
            self.hf_cond_gen_model = self.hf_cond_gen_model.to(dtype=torch.bfloat16, device=self.device).eval()
            log(
                "INFO",
                "✓ HuggingFace ForConditionalGeneration model loaded with "
                f"{sum(p.numel() for p in self.hf_cond_gen_model.parameters())} parameters",
            )
        return self.hf_cond_gen_model

    def _load_megatron_model(self):
        """Load Megatron-LM model."""
        log("INFO", "Loading Megatron-LM model")

        args = get_args()

        model_type = ModelType.encoder_or_decoder
        model = get_model(model_provider, model_type)

        # Load checkpoint
        state_dict, _, _, _ = _load_base_checkpoint(load_dir=args.load, args=args, rank0=False)
        model = unwrap_model(model)

        if len(model) == 1:
            model[0].load_state_dict(state_dict["model"], strict=True)

        checkpoint_version = state_dict.get("checkpoint_version", 0)
        fix_query_key_value_ordering(model, checkpoint_version)

        megatron_model = model[0].to(self.device).eval()
        log(
            "INFO", f"✓ Megatron-LM model loaded with {sum(p.numel() for p in megatron_model.parameters())} parameters"
        )

        return megatron_model

    def test_vision_encoder_consistency(self, resolutions: list[int] = [336, 448, 672]) -> list[dict[str, Any]]:
        """
        Test vision encoder consistency between HuggingFace and Megatron-LM
        for multiple image resolutions.

        Args:
            resolutions: List of image resolutions to test.

        Returns:
            List of test results for each resolution.
        """
        all_results = []

        layers_to_compare = [
            "after_patch_embed",
            "rotary_pos_emb",
            "after_pre_layernorm",
            "before_adapter",
        ]

        log("INFO", f"Starting vision encoder consistency tests for resolutions: {resolutions}")

        for image_size in resolutions:
            log("INFO", f"--- Testing for resolution: {image_size}x{image_size} ---")

            # Load and resize test image
            test_image = load_and_resize_image(self.test_image_path, image_size)

            # Process image using Qwen2VLImageProcessor for Megatron model
            # This produces pixel_values in (num_patches, patch_dim) format
            pixel_values_mcore, image_grid_thw = self._process_image_for_mcore(test_image)

            # Convert Megatron format back to HF format: (B, C, H, W)
            pixel_values_hf, grid_thw = self._process_image_for_hf(pixel_values_mcore, image_grid_thw)

            log("INFO", f"HF pixel values shape: {pixel_values_hf.shape}")
            log("INFO", f"Megatron pixel values shape: {pixel_values_mcore.shape}")
            log("INFO", f"Grid THW: {grid_thw}")

            # Verify input consistency - log input statistics
            log("INFO", "=" * 40)
            log("INFO", "Input Consistency Check:")
            log("INFO", f"  HF input shape: {pixel_values_hf.shape} (num_patches, patch_dim)")
            log("INFO", f"  Megatron input shape: {pixel_values_mcore.shape} (num_patches, patch_dim)")
            log("INFO", f"  HF pixel_values dtype: {pixel_values_hf.dtype}")
            log("INFO", f"  HF pixel_values device: {pixel_values_hf.device}")
            log("INFO", f"  HF pixel_values min: {pixel_values_hf.min().item():.6f}")
            log("INFO", f"  HF pixel_values max: {pixel_values_hf.max().item():.6f}")
            log("INFO", f"  HF pixel_values mean: {pixel_values_hf.mean().item():.6f}")
            log("INFO", f"  HF pixel_values std: {pixel_values_hf.std().item():.6f}")
            log("INFO", f"  Mcore pixel_values min: {pixel_values_mcore.min().item():.6f}")
            log("INFO", f"  Mcore pixel_values max: {pixel_values_mcore.max().item():.6f}")
            log("INFO", f"  grid_thw: {grid_thw.tolist()}")
            log("INFO", "=" * 40)

            # Generate patch_positions for merged model
            patch_positions = self._generate_patch_positions(grid_thw, pixel_values_hf.device)

            # Get outputs from both models using forward_debug
            # Both HF and Megatron models now receive (num_patches, patch_dim) format
            with torch.no_grad(), self._autocast_context():
                hf_debug_outputs = self.hf_model.forward_debug(
                    pixel_values_hf, grid_thw, patch_positions=patch_positions
                )
                megatron_debug_outputs = self.megatron_model.vision_model.forward_debug(
                    pixel_values_mcore, grid_thw=grid_thw, patch_positions=patch_positions
                )

            # Compare layers
            layer_similarities = {}
            is_overall_success = True

            # Log input shapes from debug outputs (they will be different due to different formats)
            if "input_pixel_values" in hf_debug_outputs and "input_pixel_values" in megatron_debug_outputs:
                log("INFO", f"HF recorded input shape: {hf_debug_outputs['input_pixel_values'].shape}")
                log("INFO", f"Megatron recorded input shape: {megatron_debug_outputs['input_pixel_values'].shape}")

            for layer_key in layers_to_compare:
                if layer_key not in hf_debug_outputs or layer_key not in megatron_debug_outputs:
                    log("WARNING", f"Layer '{layer_key}' not found in debug output. Skipping.")
                    continue

                hf_output = hf_debug_outputs[layer_key]
                megatron_output = megatron_debug_outputs[layer_key]

                # No conversion needed - HF model now uses the same 2x2 memory layout as mcore

                hf_tensor = hf_output.float().cpu().numpy()
                megatron_tensor = megatron_output.float().cpu().numpy()

                # Calculate cosine similarity
                similarity = cosine_similarity(hf_tensor, megatron_tensor)

                metric_key = f"similarity_{layer_key}"
                layer_similarities[metric_key] = float(similarity)

                log("INFO", f"Similarity for '{layer_key}': {similarity:.6f}")

                if similarity <= 0.99:
                    is_overall_success = False

            test_status = "success" if is_overall_success else "failed"
            log("INFO", f"Overall test status for {image_size}x{image_size}: {test_status}")

            result = {
                "test_type": "vision_encoder_layerwise_consistency",
                "resolution": f"{image_size}x{image_size}",
                "timestamp": datetime.now().isoformat(),
                "metrics": layer_similarities,
                "status": test_status,
            }
            all_results.append(result)

        return all_results

    def test_multisize_vision_encoder(self) -> dict[str, Any]:
        """
        Test vision encoder with multiple image sizes in a single batch.
        Verifies that batched processing produces consistent results.
        """
        log("INFO", "Testing Vision Encoder with multiple image sizes (224x224, 336x336, 448x448)")

        # Create test images of different sizes
        img_224 = load_and_resize_image(self.test_image_path, 224)
        img_336 = load_and_resize_image(self.test_image_path, 336)
        img_448 = load_and_resize_image(self.test_image_path, 448)

        # Process images using Qwen2VLImageProcessor for Megatron model
        pixel_224_mcore, grid_thw_224 = self._process_image_for_mcore(img_224)
        pixel_336_mcore, grid_thw_336 = self._process_image_for_mcore(img_336)
        pixel_448_mcore, grid_thw_448 = self._process_image_for_mcore(img_448)

        # HF model now uses the same 2x2 memory layout as mcore - no conversion needed
        pixel_224_hf, _ = self._process_image_for_hf(pixel_224_mcore, grid_thw_224)
        pixel_336_hf, _ = self._process_image_for_hf(pixel_336_mcore, grid_thw_336)
        pixel_448_hf, _ = self._process_image_for_hf(pixel_448_mcore, grid_thw_448)

        log("INFO", f"HF 224 input shape: {pixel_224_hf.shape}, Megatron 224 input shape: {pixel_224_mcore.shape}")
        log("INFO", f"HF 336 input shape: {pixel_336_hf.shape}, Megatron 336 input shape: {pixel_336_mcore.shape}")
        log("INFO", f"HF 448 input shape: {pixel_448_hf.shape}, Megatron 448 input shape: {pixel_448_mcore.shape}")

        # Get HF outputs for individual images
        with torch.no_grad(), self._autocast_context():
            patch_pos_224 = self._generate_patch_positions(grid_thw_224, pixel_224_hf.device)
            patch_pos_336 = self._generate_patch_positions(grid_thw_336, pixel_336_hf.device)
            patch_pos_448 = self._generate_patch_positions(grid_thw_448, pixel_448_hf.device)

            hf_out_224 = self.hf_model.forward_debug(pixel_224_hf, grid_thw_224, patch_positions=patch_pos_224)[
                "before_adapter"
            ]
            hf_out_336 = self.hf_model.forward_debug(pixel_336_hf, grid_thw_336, patch_positions=patch_pos_336)[
                "before_adapter"
            ]
            hf_out_448 = self.hf_model.forward_debug(pixel_448_hf, grid_thw_448, patch_positions=patch_pos_448)[
                "before_adapter"
            ]

        # No conversion needed - HF model now uses the same 2x2 memory layout as mcore

        hf_features_224 = hf_out_224.float().cpu().numpy()
        hf_features_336 = hf_out_336.float().cpu().numpy()
        hf_features_448 = hf_out_448.float().cpu().numpy()

        log("INFO", f"HF 224 features shape: {hf_features_224.shape}")
        log("INFO", f"HF 336 features shape: {hf_features_336.shape}")
        log("INFO", f"HF 448 features shape: {hf_features_448.shape}")

        # Get Megatron outputs for individual images (using Megatron input format)
        with torch.no_grad(), self._autocast_context():
            # Regenerate patch positions for mcore just in case
            patch_pos_224 = self._generate_patch_positions(grid_thw_224, pixel_224_mcore.device)
            patch_pos_336 = self._generate_patch_positions(grid_thw_336, pixel_336_mcore.device)
            patch_pos_448 = self._generate_patch_positions(grid_thw_448, pixel_448_mcore.device)

            mcore_out_224 = self.megatron_model.vision_model.forward_debug(
                pixel_224_mcore, grid_thw=grid_thw_224, patch_positions=patch_pos_224
            )["before_adapter"]
            mcore_out_336 = self.megatron_model.vision_model.forward_debug(
                pixel_336_mcore, grid_thw=grid_thw_336, patch_positions=patch_pos_336
            )["before_adapter"]
            mcore_out_448 = self.megatron_model.vision_model.forward_debug(
                pixel_448_mcore, grid_thw=grid_thw_448, patch_positions=patch_pos_448
            )["before_adapter"]

        mcore_features_224 = mcore_out_224.float().cpu().numpy()
        mcore_features_336 = mcore_out_336.float().cpu().numpy()
        mcore_features_448 = mcore_out_448.float().cpu().numpy()

        log("INFO", f"Megatron 224 features shape: {mcore_features_224.shape}")
        log("INFO", f"Megatron 336 features shape: {mcore_features_336.shape}")
        log("INFO", f"Megatron 448 features shape: {mcore_features_448.shape}")

        # Compare results
        sim_224 = cosine_similarity(hf_features_224, mcore_features_224)
        sim_336 = cosine_similarity(hf_features_336, mcore_features_336)
        sim_448 = cosine_similarity(hf_features_448, mcore_features_448)

        log("INFO", f"HF vs Megatron - 224x224 similarity: {sim_224:.6f}")
        log("INFO", f"HF vs Megatron - 336x336 similarity: {sim_336:.6f}")
        log("INFO", f"HF vs Megatron - 448x448 similarity: {sim_448:.6f}")

        all_sims = [sim_224, sim_336, sim_448]

        return {
            "test_type": "multisize_vision_encoder",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "224_similarity": float(sim_224),
                "336_similarity": float(sim_336),
                "448_similarity": float(sim_448),
            },
            "status": "success" if all(s > 0.99 for s in all_sims) else "failed",
        }

    def test_weight_consistency(self) -> dict[str, Any]:
        """
        Test weight consistency between HuggingFace and Megatron-LM models.
        Compares key layers to verify weights are correctly loaded/converted.

        Name mapping reference:
            "huggingface": {
                "transformer": "visual",
                "layer_prefix": "encoder.layers",
                "input_layernorm": "layer_norm1",
                "attention.query_key_value": "attn.qkv",
                "attention.dense": "self_attn.proj",
                "post_attention_layernorm": "layer_norm2",
                "mlp.dense_h_to_4h": "mlp.fc1",
                "mlp.dense_4h_to_h": "mlp.fc2"
            },
            "mcore": {
                "transformer": "model",
                "layer_prefix": "vision_model.decoder.layers",
                "input_layernorm": "self_attention.linear_qkv.layer_norm",
                "attention.query_key_value": "self_attention.linear_qkv",
                "attention.dense": "self_attention.linear_proj",
                "post_attention_layernorm": "mlp.linear_fc1.layer_norm",
                "mlp.dense_h_to_4h": "mlp.linear_fc1",
                "mlp.dense_4h_to_h": "mlp.linear_fc2"
            }

        Returns:
            Dictionary containing weight comparison results.
        """
        log("INFO", "=" * 60)
        log("INFO", "Testing Weight Consistency")
        log("INFO", "=" * 60)

        weight_comparisons = {}

        # Get state dicts
        hf_state_dict = self.hf_model.state_dict()
        mcore_state_dict = self.megatron_model.vision_model.state_dict()

        log("INFO", f"HF model has {len(hf_state_dict)} parameters")
        log("INFO", f"Megatron model has {len(mcore_state_dict)} parameters")

        # Define weight mapping between HF and Megatron
        # Based on the name_map provided:
        # HF: visual -> encoder.layers.{i} -> layer_norm1, attn.qkv, self_attn.proj, layer_norm2, mlp.fc1, mlp.fc2
        # Megatron: vision_model.decoder.layers.{i} -> self_attention.linear_qkv.layer_norm, self_attention.linear_qkv,
        #           self_attention.linear_proj, mlp.linear_fc1.layer_norm, mlp.linear_fc1, mlp.linear_fc2

        # Format: (hf_key, mcore_key, description)
        weight_mappings = [
            # Patch embedding
            ("embeddings.patch_embedding.weight", "patch_embed.proj.weight", "Patch Embedding Conv Weight"),
            ("embeddings.patch_embedding.bias", "patch_embed.proj.bias", "Patch Embedding Conv Bias"),
            # Class embedding
            ("embeddings.class_embedding", "class_embedding", "Class Embedding"),
            # Pre-layernorm
            ("layernorm_pre.weight", "pre_layernorm.weight", "Pre-LayerNorm Weight"),
            ("layernorm_pre.bias", "pre_layernorm.bias", "Pre-LayerNorm Bias"),
            # Post-layernorm
            ("layernorm_post.weight", "post_layernorm.weight", "Post-LayerNorm Weight"),
            ("layernorm_post.bias", "post_layernorm.bias", "Post-LayerNorm Bias"),
        ]

        # Add transformer layer mappings for ALL layers
        # Using the name_map:
        # HF layer_prefix: encoder.layers
        # Megatron layer_prefix: decoder.layers
        num_layers = self.hf_config.vision_config.num_hidden_layers
        log("INFO", f"Checking all {num_layers} transformer layers...")
        for layer_idx in range(num_layers):
            layer_mappings = [
                # Input LayerNorm (before attention)
                # HF: encoder.layers.{i}.layer_norm1 -> Megatron: decoder.layers.{i}.self_attention.linear_qkv.layer_norm
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
                # Self attention QKV
                # HF: encoder.layers.{i}.self_attn.qkv -> Megatron: decoder.layers.{i}.self_attention.linear_qkv
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
                # Self attention projection
                # HF: encoder.layers.{i}.self_attn.proj -> Megatron: decoder.layers.{i}.self_attention.linear_proj
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
                # Post-attention LayerNorm (before MLP)
                # HF: encoder.layers.{i}.layer_norm2 -> Megatron: decoder.layers.{i}.mlp.linear_fc1.layer_norm
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
                # MLP FC1
                # HF: encoder.layers.{i}.mlp.fc1 -> Megatron: decoder.layers.{i}.mlp.linear_fc1
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
                # MLP FC2
                # HF: encoder.layers.{i}.mlp.fc2 -> Megatron: decoder.layers.{i}.mlp.linear_fc2
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
            weight_mappings.extend(layer_mappings)

        # Get number of attention heads for QKV conversion
        num_heads = self.hf_config.vision_config.num_attention_heads

        # Compare weights
        all_passed = True
        for hf_key, mcore_key, description in weight_mappings:
            if hf_key not in hf_state_dict:
                log("WARNING", f"HF key not found: {hf_key}")
                weight_comparisons[description] = {
                    "status": "hf_key_not_found",
                    "hf_key": hf_key,
                    "mcore_key": mcore_key,
                }
                continue

            if mcore_key not in mcore_state_dict:
                log("WARNING", f"Megatron key not found: {mcore_key}")
                weight_comparisons[description] = {
                    "status": "mcore_key_not_found",
                    "hf_key": hf_key,
                    "mcore_key": mcore_key,
                }
                continue

            hf_weight = hf_state_dict[hf_key].float().cpu().numpy()
            mcore_weight = mcore_state_dict[mcore_key].float().cpu().numpy()

            # Check shape
            if hf_weight.shape != mcore_weight.shape:
                log(
                    "WARNING", f"{description}: Shape mismatch - HF: {hf_weight.shape}, Megatron: {mcore_weight.shape}"
                )
                weight_comparisons[description] = {
                    "status": "shape_mismatch",
                    "hf_shape": list(hf_weight.shape),
                    "mcore_shape": list(mcore_weight.shape),
                    "hf_key": hf_key,
                    "mcore_key": mcore_key,
                }
                all_passed = False
                continue

            # Special handling for QKV weights/biases - convert HF layout to Megatron layout
            # Megatron uses interleaved QKV per head: [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
            # HuggingFace uses concatenated: [Q_all; K_all; V_all]
            is_qkv_weight = "QKV Weight" in description
            is_qkv_bias = "QKV Bias" in description
            if is_qkv_weight or is_qkv_bias:
                log("INFO", f"Converting HF QKV layout to Megatron layout for: {description}")
                hf_weight = convert_hf_qkv_to_mcore_layout(hf_weight, num_heads, is_bias=is_qkv_bias)

            # Calculate metrics
            similarity = cosine_similarity(hf_weight, mcore_weight)
            max_diff = float(np.max(np.abs(hf_weight - mcore_weight)))
            mean_diff = float(np.mean(np.abs(hf_weight - mcore_weight)))

            status = "match" if similarity > 0.9999 else "mismatch"
            if status == "mismatch":
                all_passed = False

            log(
                "INFO",
                f"{description}: similarity={similarity:.6f}, max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}",
            )

            weight_comparisons[description] = {
                "status": status,
                "similarity": float(similarity),
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "hf_shape": list(hf_weight.shape),
                "mcore_shape": list(mcore_weight.shape),
                "hf_key": hf_key,
                "mcore_key": mcore_key,
            }

        # List all available keys for debugging
        log("INFO", "=" * 60)
        log("INFO", "Available HF keys (first 20):")
        for i, key in enumerate(sorted(hf_state_dict.keys())[:20]):
            log("INFO", f"  {key}")

        log("INFO", "Available Megatron keys (first 20):")
        for i, key in enumerate(sorted(mcore_state_dict.keys())[:20]):
            log("INFO", f"  {key}")

        return {
            "test_type": "weight_consistency",
            "timestamp": datetime.now().isoformat(),
            "weight_comparisons": weight_comparisons,
            "hf_total_params": len(hf_state_dict),
            "mcore_total_params": len(mcore_state_dict),
            "status": "success" if all_passed else "failed",
        }

    def test_encoder_layer_wise_consistency(self, resolution: int = 336) -> dict[str, Any]:
        """
        Test encoder layer-by-layer consistency between HuggingFace and Megatron-LM.
        Uses the forward_debug methods of OneVisionEncoderEncoder (HF) and TransformerBlock (Megatron)
        to compare each layer's input and output.

        Args:
            resolution: Image resolution to test.

        Returns:
            Dictionary containing layer-by-layer comparison results.
        """
        log("INFO", "=" * 60)
        log("INFO", f"Testing Encoder Layer-by-Layer Consistency ({resolution}x{resolution})")
        log("INFO", "=" * 60)

        # Load and resize test image
        test_image = load_and_resize_image(self.test_image_path, resolution)

        # Process image using Qwen2VLImageProcessor for Megatron model
        pixel_values_mcore, image_grid_thw = self._process_image_for_mcore(test_image)

        # HF model now uses the same 2x2 memory layout as mcore - no conversion needed
        pixel_values_hf, grid_thw = self._process_image_for_hf(pixel_values_mcore, image_grid_thw)

        log("INFO", f"HF pixel values shape: {pixel_values_hf.shape}")
        log("INFO", f"Megatron pixel values shape: {pixel_values_mcore.shape}")
        log("INFO", f"Grid THW: {grid_thw}")

        # Generate patch_positions for merged model
        patch_positions = self._generate_patch_positions(grid_thw, pixel_values_hf.device)

        # Get vision model debug outputs - this gives us encoder-level debug info
        with torch.no_grad(), self._autocast_context():
            hf_vision_debug = self.hf_model.forward_debug(pixel_values_hf, grid_thw, patch_positions=patch_positions)
            mcore_vision_debug = self.megatron_model.vision_model.forward_debug(
                pixel_values_mcore, grid_thw=grid_thw, patch_positions=patch_positions
            )

        layer_comparisons = {}
        all_passed = True

        # Check if layer_outputs exist in both debug outputs
        hf_layer_outputs = hf_vision_debug.get("layer_outputs", {})
        mcore_layer_outputs = mcore_vision_debug.get("layer_outputs", {})

        if not hf_layer_outputs:
            log("WARNING", "HF model forward_debug did not return layer_outputs")
        if not mcore_layer_outputs:
            log("WARNING", "Megatron model forward_debug did not return layer_outputs")

        # Get number of layers
        num_layers = self.hf_config.vision_config.num_hidden_layers
        log("INFO", f"Comparing {num_layers} transformer layers...")

        # Compare each layer's input and output
        for i in range(num_layers):
            layer_input_key = f"layer_{i}_input"
            layer_output_key = f"layer_{i}_output"

            layer_result = {
                "layer_index": i,
                "input_comparison": {},
                "output_comparison": {},
            }

            # Compare layer inputs
            if layer_input_key in hf_layer_outputs and layer_input_key in mcore_layer_outputs:
                hf_input_tensor = hf_layer_outputs[layer_input_key]
                mcore_input_tensor = mcore_layer_outputs[layer_input_key]

                # No conversion needed - HF model now uses the same 2x2 memory layout as mcore

                hf_input = hf_input_tensor.float().cpu().numpy()
                mcore_input = mcore_input_tensor.float().cpu().numpy()

                input_sim = cosine_similarity(hf_input, mcore_input)
                input_max_diff = float(
                    np.max(
                        np.abs(
                            hf_input.flatten()[: min(len(hf_input.flatten()), len(mcore_input.flatten()))]
                            - mcore_input.flatten()[: min(len(hf_input.flatten()), len(mcore_input.flatten()))]
                        )
                    )
                )

                layer_result["input_comparison"] = {
                    "similarity": float(input_sim),
                    "max_diff": input_max_diff,
                    "hf_shape": list(hf_input.shape),
                    "mcore_shape": list(mcore_input.shape),
                    "status": "match" if input_sim > 0.99 else "mismatch",
                }

                log("INFO", f"Layer {i} Input: similarity={input_sim:.6f}, max_diff={input_max_diff:.6e}")

                if input_sim <= 0.99:
                    all_passed = False
            else:
                layer_result["input_comparison"] = {"status": "key_not_found"}
                log("WARNING", f"Layer {i} input key not found")

            # Compare layer outputs
            if layer_output_key in hf_layer_outputs and layer_output_key in mcore_layer_outputs:
                hf_output_tensor = hf_layer_outputs[layer_output_key]
                mcore_output_tensor = mcore_layer_outputs[layer_output_key]

                # No conversion needed - HF model now uses the same 2x2 memory layout as mcore

                hf_output = hf_output_tensor.float().cpu().numpy()
                mcore_output = mcore_output_tensor.float().cpu().numpy()

                output_sim = cosine_similarity(hf_output, mcore_output)
                output_max_diff = float(
                    np.max(
                        np.abs(
                            hf_output.flatten()[: min(len(hf_output.flatten()), len(mcore_output.flatten()))]
                            - mcore_output.flatten()[: min(len(hf_output.flatten()), len(mcore_output.flatten()))]
                        )
                    )
                )

                layer_result["output_comparison"] = {
                    "similarity": float(output_sim),
                    "max_diff": output_max_diff,
                    "hf_shape": list(hf_output.shape),
                    "mcore_shape": list(mcore_output.shape),
                    "status": "match" if output_sim > 0.99 else "mismatch",
                }

                log("INFO", f"Layer {i} Output: similarity={output_sim:.6f}, max_diff={output_max_diff:.6e}")

                if output_sim <= 0.99:
                    all_passed = False
            else:
                layer_result["output_comparison"] = {"status": "key_not_found"}
                log("WARNING", f"Layer {i} output key not found")

            layer_comparisons[f"layer_{i}"] = layer_result

        # Compare encoder inputs
        encoder_input_comparison = {}
        if "input_hidden_states" in hf_layer_outputs and "input_hidden_states" in mcore_layer_outputs:
            hf_enc_input_tensor = hf_layer_outputs["input_hidden_states"]
            mcore_enc_input_tensor = mcore_layer_outputs["input_hidden_states"]

            # No conversion needed - HF model now uses the same 2x2 memory layout as mcore

            hf_enc_input = hf_enc_input_tensor.float().cpu().numpy()
            mcore_enc_input = mcore_enc_input_tensor.float().cpu().numpy()

            enc_input_sim = cosine_similarity(hf_enc_input, mcore_enc_input)
            encoder_input_comparison = {
                "similarity": float(enc_input_sim),
                "hf_shape": list(hf_enc_input.shape),
                "mcore_shape": list(mcore_enc_input.shape),
            }
            log("INFO", f"Encoder Input: similarity={enc_input_sim:.6f}")

        # Compare encoder final outputs
        encoder_output_comparison = {}
        if "final_output" in hf_layer_outputs and "final_output" in mcore_layer_outputs:
            hf_enc_output_tensor = hf_layer_outputs["final_output"]
            mcore_enc_output_tensor = mcore_layer_outputs["final_output"]

            # No conversion needed - HF model now uses the same 2x2 memory layout as mcore

            hf_enc_output = hf_enc_output_tensor.float().cpu().numpy()
            mcore_enc_output = mcore_enc_output_tensor.float().cpu().numpy()

            enc_output_sim = cosine_similarity(hf_enc_output, mcore_enc_output)
            encoder_output_comparison = {
                "similarity": float(enc_output_sim),
                "hf_shape": list(hf_enc_output.shape),
                "mcore_shape": list(mcore_enc_output.shape),
            }
            log("INFO", f"Encoder Final Output: similarity={enc_output_sim:.6f}")

        return {
            "test_type": "encoder_layer_wise_consistency",
            "resolution": f"{resolution}x{resolution}",
            "timestamp": datetime.now().isoformat(),
            "num_layers": num_layers,
            "encoder_input_comparison": encoder_input_comparison,
            "layer_comparisons": layer_comparisons,
            "encoder_output_comparison": encoder_output_comparison,
            "status": "success" if all_passed else "failed",
        }

    def test_mllm_after_merger_consistency(self, resolution: int = 336) -> dict[str, Any]:
        """
        Test MLLM after-merger output consistency between HuggingFace and Megatron-LM.

        This test compares the outputs after the vision encoder + merger/adapter
        to verify the vision pipeline consistency before language model processing.

        Args:
            resolution: Image resolution to test.

        Returns:
            Dictionary containing after-merger comparison results.
        """
        log("INFO", "=" * 60)
        log("INFO", f"Testing After-Merger Consistency ({resolution}x{resolution})")
        log("INFO", "=" * 60)

        # Load and resize test image
        test_image = load_and_resize_image(self.test_image_path, resolution)

        # Process image using Qwen2VLImageProcessor for Megatron model
        pixel_values_mcore, image_grid_thw = self._process_image_for_mcore(test_image)

        # HF model now uses the same 2x2 memory layout as mcore - no conversion needed
        pixel_values_hf, grid_thw = self._process_image_for_hf(pixel_values_mcore, image_grid_thw)

        log("INFO", f"HF pixel values shape: {pixel_values_hf.shape}")
        log("INFO", f"Megatron pixel values shape: {pixel_values_mcore.shape}")
        log("INFO", f"Grid THW: {grid_thw}")

        # Get HF model output using forward_debug (now includes after_merger)
        with torch.no_grad(), self._autocast_context():
            patch_positions = self._generate_patch_positions(grid_thw, pixel_values_hf.device)
            hf_debug_output = self.hf_model.forward_debug(pixel_values_hf, grid_thw, patch_positions=patch_positions)

        hf_after_merger = hf_debug_output.get("after_merger")
        if hf_after_merger is None:
            log("ERROR", "HF model forward_debug did not return 'after_merger' key")
            return {
                "test_type": "mllm_after_merger_consistency",
                "resolution": f"{resolution}x{resolution}",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": "HF model did not return 'after_merger' key",
            }

        log("INFO", f"HF after_merger shape: {hf_after_merger.shape}")

        # Get Megatron model output by running vision model + adapter separately
        # This is more reliable than forward_debug
        with torch.no_grad(), self._autocast_context():
            # Step 1: Get vision encoder output
            mcore_vision_output = self.megatron_model.vision_model(pixel_values_mcore, grid_thw=image_grid_thw)
            log("INFO", f"Megatron vision output shape: {mcore_vision_output.shape}")

            # Step 2: Run through adapter
            mcore_after_merger = self.megatron_model.adapter(mcore_vision_output)
            log("INFO", f"Megatron after_merger shape: {mcore_after_merger.shape}")

        if mcore_after_merger is None:
            log("ERROR", "Megatron model vision_model + adapter did not produce output")
            return {
                "test_type": "mllm_after_merger_consistency",
                "resolution": f"{resolution}x{resolution}",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": "Megatron model vision_model + adapter did not produce output",
            }

        # Convert to numpy for comparison
        hf_after_merger_np = hf_after_merger.float().cpu().numpy()
        mcore_after_merger_np = mcore_after_merger.float().cpu().numpy()

        results = {
            "test_type": "mllm_after_merger_consistency",
            "resolution": f"{resolution}x{resolution}",
            "timestamp": datetime.now().isoformat(),
            "hf_after_merger_shape": list(hf_after_merger.shape),
            "mcore_after_merger_shape": list(mcore_after_merger.shape),
            "hf_after_merger_stats": {
                "mean": float(hf_after_merger_np.mean()),
                "std": float(hf_after_merger_np.std()),
                "min": float(hf_after_merger_np.min()),
                "max": float(hf_after_merger_np.max()),
            },
            "mcore_after_merger_stats": {
                "mean": float(mcore_after_merger_np.mean()),
                "std": float(mcore_after_merger_np.std()),
                "min": float(mcore_after_merger_np.min()),
                "max": float(mcore_after_merger_np.max()),
            },
        }
        hf_after_merger_np = np.squeeze(hf_after_merger_np)
        mcore_after_merger_np = np.squeeze(mcore_after_merger_np)

        # Compare shapes first
        if hf_after_merger_np.shape != mcore_after_merger_np.shape:
            log("WARNING", f"Shape mismatch: HF {hf_after_merger_np.shape} vs Megatron {mcore_after_merger_np.shape}")

            # Try to flatten and compare min(lengths)
            hf_flat = hf_after_merger_np.flatten()
            mcore_flat = mcore_after_merger_np.flatten()
            min_len = min(len(hf_flat), len(mcore_flat))

            if min_len > 0:
                similarity = cosine_similarity(hf_flat[:min_len], mcore_flat[:min_len])
                max_diff = float(np.max(np.abs(hf_flat[:min_len] - mcore_flat[:min_len])))
                results["similarity_partial"] = float(similarity)
                results["max_diff_partial"] = max_diff
                log("INFO", f"Partial comparison (first {min_len} elements): similarity={similarity:.6f}")

            results["status"] = "shape_mismatch"
            results["note"] = (
                f"Shape mismatch: HF {list(hf_after_merger_np.shape)} vs Megatron {list(mcore_after_merger_np.shape)}"
            )
        else:
            # Compute full comparison
            similarity = cosine_similarity(hf_after_merger_np, mcore_after_merger_np)
            max_diff = float(np.max(np.abs(hf_after_merger_np - mcore_after_merger_np)))

            results["similarity"] = float(similarity)
            results["max_diff"] = max_diff
            results["status"] = "success" if similarity > 0.99 else "failed"

            log("INFO", f"After-Merger Similarity: {similarity:.6f}")
            log("INFO", f"After-Merger Max Diff: {max_diff:.6e}")

        return results

    def test_llm_output_consistency(self, resolution: int = 336) -> dict[str, Any]:
        """
        Test LLM output consistency between HuggingFace and Megatron-LM.

        This test compares the full model outputs (after vision encoder + adapter + language model)
        to verify end-to-end consistency.

        Args:
            resolution: Image resolution to test.

        Returns:
            Dictionary containing LLM output comparison results.
        """
        log("INFO", "=" * 60)
        log("INFO", f"Testing LLM Output Consistency ({resolution}x{resolution})")
        log("INFO", "=" * 60)

        # Load and resize test image
        test_image = load_and_resize_image(self.test_image_path, resolution)
        prompt = "Describe this image."
        text = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>"

        input_ids, pixel_values, image_grid_thw, attention_mask_neg = self._tokenize_and_preprocess(test_image, text)
        # HF model now uses the same 2x2 memory layout as mcore - no conversion needed
        pixel_values_mcore = pixel_values
        pixel_values_hf = pixel_values  # Same format now

        log("INFO", f"HF pixel values shape: {pixel_values_hf.shape}")
        log("INFO", f"Megatron pixel values shape: {pixel_values_mcore.shape}")
        log("INFO", f"Grid THW: {image_grid_thw}")

        num_image_tokens = pixel_values_mcore.size(0)
        log("INFO", f"Number of image tokens: {num_image_tokens}")

        batch_input_id = input_ids.unsqueeze(0)
        batch_attention_mask_neg = attention_mask_neg.unsqueeze(0)

        # Generate patch_positions for merged model
        patch_positions = []
        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            t_idx = torch.arange(t, device=pixel_values.device, dtype=torch.float32)
            h_idx = torch.arange(h, device=pixel_values.device, dtype=torch.float32)
            w_idx = torch.arange(w, device=pixel_values.device, dtype=torch.float32)
            mesh_t, mesh_h, mesh_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
            patch_positions.append(torch.stack([mesh_t, mesh_h, mesh_w], dim=-1).reshape(-1, 3))
        patch_positions = torch.cat(patch_positions, dim=0)

        # Get HF model output using ForConditionalGeneration model (returns logits)
        hf_cond_gen_model = self._get_hf_cond_gen_model()

        with torch.no_grad(), self._autocast_context():
            hf_output = hf_cond_gen_model(
                input_ids=batch_input_id,
                attention_mask=None,
                pixel_values=pixel_values_hf,
                image_grid_thw=image_grid_thw,
                patch_positions=patch_positions,
                return_dict=True,
            )

        hf_logits = hf_output.logits
        log("INFO", f"HF logits shape: {hf_logits.shape}")

        with torch.no_grad(), self._autocast_context():
            # The Megatron model forward returns loss if labels are provided, or logits otherwise
            mcore_output = self.megatron_model(
                images=pixel_values_mcore,
                image_grid_thw=image_grid_thw,
                input_ids=batch_input_id,
                position_ids=None,
                attention_mask=batch_attention_mask_neg,
                attn_mask_type=None,
                labels=None,
            )

        mcore_logits = mcore_output.contiguous()
        log("INFO", f"Megatron logits shape: {mcore_logits.shape}")

        # Convert to numpy for comparison
        hf_logits_np = hf_logits.float().cpu().numpy()
        mcore_logits_np = mcore_logits.float().cpu().numpy()

        # Get logits statistics
        results = {
            "test_type": "llm_output_consistency",
            "resolution": f"{resolution}x{resolution}",
            "timestamp": datetime.now().isoformat(),
            "hf_logits_shape": list(hf_logits.shape),
            "mcore_logits_shape": list(mcore_logits.shape),
            "hf_logits_stats": {
                "mean": float(hf_logits_np.mean()),
                "std": float(hf_logits_np.std()),
                "min": float(hf_logits_np.min()),
                "max": float(hf_logits_np.max()),
            },
            "mcore_logits_stats": {
                "mean": float(mcore_logits_np.mean()),
                "std": float(mcore_logits_np.std()),
                "min": float(mcore_logits_np.min()),
                "max": float(mcore_logits_np.max()),
            },
        }

        # Compare logits if shapes match
        if hf_logits_np.shape == mcore_logits_np.shape:
            similarity = cosine_similarity(hf_logits_np, mcore_logits_np)
            max_diff = float(np.max(np.abs(hf_logits_np - mcore_logits_np)))

            results["similarity"] = float(similarity)
            results["max_diff"] = max_diff
            results["status"] = "success" if similarity > 0.99 else "failed"

            log("INFO", f"LLM Logits Similarity: {similarity:.6f}")
            log("INFO", f"LLM Logits Max Diff: {max_diff:.6e}")
        else:
            # Shapes don't match - this shouldn't happen now since both return logits
            log("WARNING", f"Shape mismatch: HF {hf_logits_np.shape} vs Megatron {mcore_logits_np.shape}")
            results["status"] = "shape_mismatch"
            results["note"] = (
                f"Shape mismatch: HF {list(hf_logits_np.shape)} vs Megatron {list(mcore_logits_np.shape)}"
            )

        return results

    def test_hf_loading_consistency(self) -> dict[str, Any]:
        """
        Test consistency between HuggingFace model loaded via `load_file` (safetensors)
        vs `from_pretrained`. This verifies that manual weight loading produces identical results.

        Returns:
            Dictionary containing comparison results.
        """
        log("INFO", "=" * 60)
        log("INFO", "Testing HF Loading Consistency: load_file vs from_pretrained")
        log("INFO", "=" * 60)

        # Load model via from_pretrained (already loaded as self.hf_model)
        # We need to load again to get the full model
        log("INFO", "Loading HF model via from_pretrained...")
        config_dict = self._load_hf_config_dict()
        config, model_cls, _ = self._resolve_hf_classes(config_dict)
        hf_from_pretrained = model_cls.from_pretrained(self.hf_model_path, low_cpu_mem_usage=True)
        hf_from_pretrained_vision = hf_from_pretrained.visual
        hf_from_pretrained_vision = hf_from_pretrained_vision.to(dtype=torch.bfloat16, device=self.device).eval()

        # Load model via load_file (safetensors)
        log("INFO", "Loading HF model via load_file (safetensors)...")

        # Load config
        hf_load_file_model = model_cls(config)

        # Find and load safetensors files
        safetensors_files = glob(os.path.join(self.hf_model_path, "*.safetensors"))
        if safetensors_files:
            log("INFO", f"Found {len(safetensors_files)} safetensors files")
            state_dict = {}
            for sf_file in sorted(safetensors_files):
                log("INFO", f"Loading: {os.path.basename(sf_file)}")
                state_dict.update(load_file(sf_file))
            hf_load_file_model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to pytorch_model.bin
            log("INFO", "No safetensors found, trying pytorch_model.bin...")
            pytorch_model_path = os.path.join(self.hf_model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
                hf_load_file_model.load_state_dict(state_dict, strict=False)
            else:
                log("ERROR", "No model weights found!")
                return {
                    "test_type": "hf_loading_consistency",
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": "No model weights found",
                }

        hf_load_file_vision = hf_load_file_model.visual
        hf_load_file_vision = hf_load_file_vision.to(dtype=torch.bfloat16, device=self.device).eval()

        # Compare state dicts
        log("INFO", "Comparing state dicts...")
        pretrained_state = hf_from_pretrained_vision.state_dict()
        loadfile_state = hf_load_file_vision.state_dict()

        weight_comparisons = {}
        all_match = True

        # Compare all weights
        for key in pretrained_state.keys():
            if key not in loadfile_state:
                log("WARNING", f"Key not found in load_file model: {key}")
                weight_comparisons[key] = {
                    "status": "key_not_found",
                }
                all_match = False
                continue

            pretrained_weight = pretrained_state[key].float().cpu().numpy()
            loadfile_weight = loadfile_state[key].float().cpu().numpy()

            if pretrained_weight.shape != loadfile_weight.shape:
                log("WARNING", f"{key}: Shape mismatch")
                weight_comparisons[key] = {
                    "status": "shape_mismatch",
                    "pretrained_shape": list(pretrained_weight.shape),
                    "loadfile_shape": list(loadfile_weight.shape),
                }
                all_match = False
                continue

            # Calculate metrics
            similarity = cosine_similarity(pretrained_weight, loadfile_weight)
            max_diff = float(np.max(np.abs(pretrained_weight - loadfile_weight)))
            is_exact = np.allclose(pretrained_weight, loadfile_weight, rtol=1e-5, atol=1e-5)

            if not is_exact:
                all_match = False
                log("INFO", f"{key}: similarity={similarity:.6f}, max_diff={max_diff:.6e}, exact={is_exact}")

            weight_comparisons[key] = {
                "status": "match" if is_exact else "mismatch",
                "similarity": float(similarity),
                "max_diff": max_diff,
                "is_exact": is_exact,
            }

        # Also compare forward outputs
        log("INFO", "Comparing forward outputs...")
        test_image = load_and_resize_image(self.test_image_path, 336)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        pixel_values = transform(test_image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        patch_size = self.hf_config.vision_config.patch_size
        grid_thw = torch.tensor([[1, 336 // patch_size, 336 // patch_size]], dtype=torch.long, device=self.device)

        with torch.no_grad(), self._autocast_context():
            patch_positions = self._generate_patch_positions(grid_thw, self.device)
            pretrained_output = hf_from_pretrained_vision.forward_debug(
                pixel_values, grid_thw, patch_positions=patch_positions
            )
            loadfile_output = hf_load_file_vision.forward_debug(
                pixel_values, grid_thw, patch_positions=patch_positions
            )

        output_comparisons = {}
        for key in pretrained_output.keys():
            if key not in loadfile_output:
                output_comparisons[key] = {"status": "key_not_found"}
                continue

            pretrained_val = pretrained_output[key]
            loadfile_val = loadfile_output[key]

            # Skip if not a tensor (e.g., layer_outputs is a dict)
            if not isinstance(pretrained_val, torch.Tensor):
                log("INFO", f"Output '{key}': skipping (not a tensor, type={type(pretrained_val).__name__})")
                output_comparisons[key] = {"status": "skipped", "reason": "not_a_tensor"}
                continue

            pretrained_tensor = pretrained_val.float().cpu().numpy()
            loadfile_tensor = loadfile_val.float().cpu().numpy()

            similarity = cosine_similarity(pretrained_tensor, loadfile_tensor)
            max_diff = float(np.max(np.abs(pretrained_tensor - loadfile_tensor)))

            output_comparisons[key] = {
                "similarity": float(similarity),
                "max_diff": max_diff,
            }
            log("INFO", f"Output '{key}': similarity={similarity:.6f}, max_diff={max_diff:.6e}")

        # Cleanup
        del hf_from_pretrained, hf_from_pretrained_vision, hf_load_file_model, hf_load_file_vision
        torch.cuda.empty_cache()

        return {
            "test_type": "hf_loading_consistency",
            "timestamp": datetime.now().isoformat(),
            "weight_comparisons_summary": {
                "total_weights": len(pretrained_state),
                "all_match": all_match,
                "mismatches": [k for k, v in weight_comparisons.items() if v.get("status") != "match"],
            },
            "output_comparisons": output_comparisons,
            "status": "success" if all_match else "failed",
        }

    def run_all_tests(self) -> dict[str, Any]:
        """Run all consistency tests and return results."""
        log("INFO", "=" * 60)
        log("INFO", "LLAVA-ONEVISION2 VIT CONSISTENCY TEST")
        log("INFO", "=" * 60)

        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hf_model_path": self.hf_model_path,
                "device": str(self.device),
                "test_profile": self.test_profile,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "test_image_path": self.test_image_path,
            },
            "tests": {},
        }

        # Run weight consistency test
        results["tests"]["weight_consistency"] = self.test_weight_consistency()

        # Run encoder layer-by-layer consistency test
        results["tests"]["encoder_layer_wise"] = self.test_encoder_layer_wise_consistency(336)

        if self.test_profile == "low_vram":
            results["tests"]["vision_encoder_layerwise"] = self.test_vision_encoder_consistency([336])
            results["tests"]["mllm_after_merger"] = self.test_mllm_after_merger_consistency(336)
        else:
            # Run layer-wise consistency tests
            results["tests"]["vision_encoder_layerwise"] = self.test_vision_encoder_consistency([336, 448])

            # Run multi-size test
            results["tests"]["multisize_vision_encoder"] = self.test_multisize_vision_encoder()

            # Run after-merger consistency test (vision encoder + adapter/merger)
            results["tests"]["mllm_after_merger"] = self.test_mllm_after_merger_consistency(336)

            # Run LLM output consistency test
            results["tests"]["llm_output_336"] = self.test_llm_output_consistency(336)
            results["tests"]["llm_output_448"] = self.test_llm_output_consistency(448)
            results["tests"]["llm_output_1120"] = self.test_llm_output_consistency(1120)

        # Print summary
        log("INFO", "=" * 60)
        log("INFO", "TEST SUMMARY")
        log("INFO", "=" * 60)

        all_passed = True
        for test_name, test_results in results["tests"].items():
            if isinstance(test_results, list):
                for result in test_results:
                    status = result.get("status", "unknown")
                    resolution = result.get("resolution", "N/A")
                    log("INFO", f"  {test_name} ({resolution}): {status}")
                    if status != "success":
                        all_passed = False
            else:
                status = test_results.get("status", "unknown")
                log("INFO", f"  {test_name}: {status}")
                if status != "success":
                    all_passed = False

        results["overall_status"] = "success" if all_passed else "failed"
        log("INFO", f"Overall Status: {results['overall_status']}")
        log("INFO", "=" * 60)

        return results


def _add_extra_check_args(parser: argparse.ArgumentParser):
    """Add extra arguments for consistency check."""
    group = parser.add_argument_group(title="consistency_check")
    group.add_argument("--hf-model-path", type=str, required=True, help="Path to HuggingFace LlavaOnevision2 model")
    group.add_argument(
        "--preprocessor-path",
        type=str,
        default="/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5",
        help="Path to image preprocessor (Qwen2VL-style processor)",
    )
    group.add_argument(
        "--output-path", type=str, default="vit_consistency_results.json", help="Output file for results"
    )
    group.add_argument(
        "--test-image-path",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Path to test image (URL or local path)",
    )
    group.add_argument(
        "--test-profile",
        type=str,
        choices=["full", "low_vram"],
        default="full",
        help="Consistency test profile. 'low_vram' skips the heaviest full-model checks.",
    )
    return parser


def main():
    """Main entry point."""

    def _wrapper(parser: argparse.ArgumentParser):
        parser = aiak_extra_train_args_provider(parser)
        parser = _add_extra_check_args(parser)
        return parser

    args = parse_arguments(
        extra_args_provider=_wrapper, validate_extra_args_provider=validate_aiak_extra_args, args_defaults={}
    )

    # Initialize Megatron
    initialize_aiak_megatron(args=args)

    # Create tester and run tests
    tester = LlavaOnevision2ConsistencyTester(
        hf_model_path=args.hf_model_path,
        preprocessor_path=args.preprocessor_path,
        test_image_path=args.test_image_path,
        test_profile=args.test_profile,
    )

    results = tester.run_all_tests()

    # Save results
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    log("INFO", f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
