import os

import torch
from PIL import Image

from transformers import AutoModel, AutoProcessor, CLIPImageProcessor, logging

from ..utils import build_patch_positions, cosine_similarity, load_image, rowmajor_to_block


logger = logging.get_logger(__name__)


def _load_orig_vit(vit_path: str, device: torch.device):
    m = AutoModel.from_pretrained(
        vit_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    return m.to(device).eval()


def run(model, vit_path: str, qwen_processor_path: str, img_path: str, device: torch.device):
    from qwen_vl_utils import process_vision_info

    sms = model.config.vision_config.spatial_merge_size
    patch_size = model.config.vision_config.patch_size
    pixel_unit = patch_size * sms

    image = load_image(img_path)
    w0, h0 = image.size
    h, w = (h0 // pixel_unit) * pixel_unit, (w0 // pixel_unit) * pixel_unit
    if (w, h) != (w0, h0):
        image = image.resize((w, h), Image.BILINEAR)
        logger.info(f"resized {(w0, h0)} -> {(w, h)} for sms={sms} alignment")

    clip_proc = CLIPImageProcessor.from_pretrained(vit_path, local_files_only=os.path.isdir(vit_path))
    clip_px = clip_proc(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
    clip_px = clip_px.to(dtype=torch.bfloat16, device=device)
    grid_h, grid_w = h // patch_size, w // patch_size

    qwen_proc = AutoProcessor.from_pretrained(qwen_processor_path, use_fast=True, trust_remote_code=True)
    ip = qwen_proc.image_processor
    ip.do_resize = False
    ip.max_pixels = ip.min_pixels = h * w
    ip.temporal_patch_size = 1
    ip.patch_size = patch_size
    if hasattr(ip, "merge_size"):
        ip.merge_size = sms

    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "x"}]}]
    text = qwen_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    qwen_inputs = qwen_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    qwen_px = qwen_inputs["pixel_values"].to(dtype=torch.bfloat16, device=device)
    qwen_grid = qwen_inputs["image_grid_thw"].to(device)

    qt, qh, qw = qwen_grid[0].tolist()
    assert (1, grid_h, grid_w) == (qt, qh, qw), f"grid mismatch CLIP=[1,{grid_h},{grid_w}] Qwen=[{qt},{qh},{qw}]"

    visual = model.model.visual.to(dtype=torch.bfloat16, device=device).eval()
    orig_vit = _load_orig_vit(vit_path, device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        orig_out = orig_vit(clip_px).last_hidden_state[0]
        patch_pos = build_patch_positions(qwen_grid, device)
        merged_out = visual(qwen_px, qwen_grid, patch_positions=patch_pos, skip_merger=True).last_hidden_state[0]
        orig_block = rowmajor_to_block(orig_out, 1, grid_h, grid_w, sms)

    diff = (merged_out.float() - orig_block.float()).abs().mean().item()
    sim = cosine_similarity(merged_out.cpu(), orig_block.cpu())
    logger.info(f"ViT diff={diff:.6f} cos={sim:.6f}")
    if not (diff < 5e-1 and sim > 0.9):
        raise ValueError(f"ViT mismatch (diff={diff}, cos={sim})")
    logger.info("ViT consistency OK")

    model.model.visual = visual.to("cpu")
    del orig_vit
    torch.cuda.empty_cache()
