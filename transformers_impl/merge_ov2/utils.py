import time
from contextlib import contextmanager
from io import BytesIO
from typing import Optional

import requests
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import logging


logger = logging.get_logger(__name__)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def load_image(path: str) -> Image.Image:
    if path.startswith("http"):
        return Image.open(BytesIO(requests.get(path).content)).convert("RGB")
    return Image.open(path).convert("RGB")


def build_patch_positions(grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts = []
    for t, h, w in grid_thw.tolist():
        ti = torch.arange(t, device=device, dtype=torch.float32)
        hi = torch.arange(h, device=device, dtype=torch.float32)
        wi = torch.arange(w, device=device, dtype=torch.float32)
        mt, mh, mw = torch.meshgrid(ti, hi, wi, indexing="ij")
        parts.append(torch.stack([mt, mh, mw], dim=-1).reshape(-1, 3))
    return torch.cat(parts, dim=0)


def rowmajor_to_block(features: torch.Tensor, t: int, h: int, w: int, sms: int) -> torch.Tensor:
    """Reorder [t*h*w, d] features from row-major to sms x sms block layout."""
    if sms == 1:
        return features
    d = features.shape[-1]
    assert h % sms == 0 and w % sms == 0, f"({h},{w}) not divisible by sms={sms}"
    return features.view(t, h // sms, sms, w // sms, sms, d).permute(0, 1, 3, 2, 4, 5).contiguous().view(t * h * w, d)


def _infer_hw_from_positions(group_positions: torch.Tensor, spatial_merge_size: int = 2) -> tuple[int, int]:
    """Infer (H, W) for a single frame from its (t, h, w) patch positions."""
    h = torch.unique(group_positions[:, 1]).shape[0]
    w = torch.unique(group_positions[:, 2]).shape[0]
    assert h % spatial_merge_size == 0, f"Height {h} not divisible by {spatial_merge_size}"
    assert w % spatial_merge_size == 0, f"Width {w} not divisible by {spatial_merge_size}"
    return h, w


def convert_rope_to_block_layout(
    freqs: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2
) -> torch.Tensor:
    """Reorder [t*h*w, half] RoPE freqs from row-major to sms x sms block layout.

    Mirrors :func:`rowmajor_to_block` for RoPE frequency tensors. Used by the
    vit_layerwise validator so merged-side block-layout encoder layers receive
    RoPE matching their patch ordering.
    """
    sms = spatial_merge_size
    if sms == 1:
        return freqs
    half = freqs.shape[-1]
    return (
        freqs.view(t, h // sms, sms, w // sms, sms, half).permute(0, 1, 3, 2, 4, 5).contiguous().view(t * h * w, half)
    )


def convert_rope_to_block_layout_by_positions(
    freqs: torch.Tensor,
    patch_positions: torch.Tensor,
    spatial_merge_size: int = 2,
    grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Block-layout RoPE freqs across one or more samples, deducing T/H/W per sample.

    Prefers ``grid_thw`` when provided (single-sample fast path, all-same-HW fast
    path, or per-sample loop). Otherwise groups by frame_id from ``patch_positions``.

    Canonical reference: ``aiak_training_llm/models/llava_onevision2/onevision_encoder_model.py``
    (Megatron side). Re-implemented here because ``transformers_impl`` cannot
    reverse-import the Megatron training package.
    """
    sms = spatial_merge_size
    if sms == 1:
        return freqs

    if grid_thw is not None:
        num_samples = grid_thw.shape[0]
        if num_samples == 1:
            t, h, w = (grid_thw[0, i].item() for i in range(3))
            return convert_rope_to_block_layout(freqs, t=t, h=h, w=w, spatial_merge_size=sms)

        all_same_hw = (
            torch.all(grid_thw[:, 1] == grid_thw[0, 1]).item() and torch.all(grid_thw[:, 2] == grid_thw[0, 2]).item()
        )
        if all_same_hw:
            total_t = grid_thw[:, 0].sum().item()
            h = grid_thw[0, 1].item()
            w = grid_thw[0, 2].item()
            return convert_rope_to_block_layout(freqs, t=total_t, h=h, w=w, spatial_merge_size=sms)

        result = torch.empty_like(freqs)
        offset = 0
        for i in range(num_samples):
            t, h, w = (grid_thw[i, j].item() for j in range(3))
            n = int(t * h * w)
            result[offset : offset + n] = convert_rope_to_block_layout(
                freqs[offset : offset + n], t=t, h=h, w=w, spatial_merge_size=sms
            )
            offset += n
        return result

    seq_len = freqs.shape[0]
    t_indices = patch_positions[:, 0]
    unique_t, _, counts = torch.unique_consecutive(t_indices, return_inverse=True, return_counts=True)
    num_groups = unique_t.shape[0]

    if num_groups == 1:
        hw = int(seq_len**0.5)
        if hw * hw == seq_len:
            return convert_rope_to_block_layout(freqs, t=1, h=hw, w=hw, spatial_merge_size=sms)

    first_count = counts[0].item()
    all_same_size = torch.all(counts == first_count).item()
    if all_same_size:
        hw = int(first_count**0.5)
        if hw * hw == first_count:
            return convert_rope_to_block_layout(freqs, t=num_groups, h=hw, w=hw, spatial_merge_size=sms)

    cum_counts = torch.cumsum(counts, dim=0)
    start_indices = torch.cat([torch.tensor([0], device=counts.device), cum_counts[:-1]])
    result_freqs = torch.empty_like(freqs)
    for group_idx in range(num_groups):
        start_idx = start_indices[group_idx].item()
        group_size = counts[group_idx].item()
        end_idx = start_idx + group_size
        hw = int(group_size**0.5)
        if hw * hw == group_size:
            h, w = hw, hw
        else:
            h, w = _infer_hw_from_positions(patch_positions[start_idx:end_idx], sms)
        result_freqs[start_idx:end_idx] = convert_rope_to_block_layout(
            freqs[start_idx:end_idx], t=1, h=h, w=w, spatial_merge_size=sms
        )
    return result_freqs


@contextmanager
def log_stage(name: str):
    bar = "=" * 6
    logger.info(f"{bar} {name} {bar}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"{bar} {name} done in {dt:.2f}s {bar}")
