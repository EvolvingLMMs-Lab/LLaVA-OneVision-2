import torch
from PIL import Image

from transformers import CLIPImageProcessor

from ..utils import (
    convert_rope_to_block_layout_by_positions,
    cosine_similarity,
    load_image,
    rowmajor_to_block,
)
from .vit_blockorder import _load_orig_vit


def _extract_block_patches(img_tensor, ps: int, sms: int):
    b, c, ph, pw = img_tensor.shape
    h, w = ph // ps, pw // ps
    patches = img_tensor.reshape(b, c, h, ps, w, ps).permute(0, 2, 4, 1, 3, 5).reshape(h, w, c, ps, ps)
    h_m, w_m = h // sms, w // sms
    patches = patches.reshape(h_m, sms, w_m, sms, c, ps, ps).permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    return patches.reshape(-1, c, ps, ps)


def run(model, vit_path: str, qwen_processor_path: str, img_path: str, device: torch.device):
    sms = model.config.vision_config.spatial_merge_size
    patch_size = model.config.vision_config.patch_size
    pixel_unit = patch_size * sms
    dtype = torch.bfloat16

    image = load_image(img_path)
    w0, h0 = image.size
    h, w = (h0 // pixel_unit) * pixel_unit, (w0 // pixel_unit) * pixel_unit
    if (w, h) != (w0, h0):
        image = image.resize((w, h), Image.BILINEAR)

    clip_proc = CLIPImageProcessor.from_pretrained(vit_path)
    clip_px = clip_proc(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
    clip_px = clip_px.to(dtype=dtype, device=device)
    grid_h, grid_w = h // patch_size, w // patch_size
    block_patches = _extract_block_patches(clip_px, ps=patch_size, sms=sms)

    merged_visual = model.model.visual.to(dtype=dtype, device=device).eval()
    merged_visual.config._attn_implementation = "flash_attention_2"
    orig_vit = _load_orig_vit(vit_path, device)
    if hasattr(orig_vit, "layernorm_post") and orig_vit.layernorm_post is not None:
        orig_vit.layernorm_post = None

    with torch.no_grad():
        merged_emb = merged_visual.embeddings(block_patches)
        orig_emb_rowmajor = orig_vit.embeddings(clip_px)
        orig_emb_block = rowmajor_to_block(orig_emb_rowmajor[0], 1, grid_h, grid_w, sms)
        emb_sim = cosine_similarity(merged_emb.flatten().cpu(), orig_emb_block.flatten().cpu())
        print(
            f"[vit_layerwise] embedding sim={emb_sim:.8f}  "
            f"merged={tuple(merged_emb.shape)}  orig_block={tuple(orig_emb_block.shape)}",
            flush=True,
        )

        merged_pre = merged_visual.layernorm_pre(merged_emb.unsqueeze(0))
        orig_pre = orig_vit.layernorm_pre(orig_emb_block.unsqueeze(0))
        pre_sim = cosine_similarity(merged_pre.flatten().cpu(), orig_pre.flatten().cpu())
        print(f"[vit_layerwise] layernorm_pre sim={pre_sim:.8f}  shape={tuple(merged_pre.shape)}", flush=True)

        grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device)
        t_idx = torch.arange(1, device=device, dtype=torch.float32)
        h_idx = torch.arange(grid_h, device=device, dtype=torch.float32)
        w_idx = torch.arange(grid_w, device=device, dtype=torch.float32)
        mt, mh, mw = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        patch_positions = torch.stack([mt, mh, mw], dim=-1).reshape(-1, 3)

        merged_freqs = merged_visual.video_rope.forward_from_positions(patch_positions)
        merged_freqs = convert_rope_to_block_layout_by_positions(
            merged_freqs, patch_positions, spatial_merge_size=sms, grid_thw=grid_thw
        )
        block_rope = torch.cat([merged_freqs, merged_freqs], dim=-1).unsqueeze(0)
        print(
            f"[vit_layerwise] block_rope shape={tuple(block_rope.shape)}  dtype={block_rope.dtype}",
            flush=True,
        )

        # Use shared starting state (orig_pre) so layer-0 input is identical.
        orig_h, merged_h = orig_pre.clone(), orig_pre.clone()
        min_sim = 1.0
        for i in range(len(orig_vit.encoder.layers)):
            orig_h = orig_vit.encoder.layers[i](
                orig_h, attention_mask=None, rotary_pos_emb=block_rope, output_attentions=False
            )[0]
            merged_h = merged_visual.encoder.layers[i](
                merged_h,
                attention_mask=None,
                rotary_pos_emb=block_rope,
                output_attentions=False,
                cu_seqlens=None,
                max_seqlen=None,
            )[0]
            sim = cosine_similarity(orig_h.flatten().cpu(), merged_h.flatten().cpu())
            min_sim = min(min_sim, sim)
            print(f"[vit_layerwise] [Layer {i:2d}] sim={sim:.8f}", flush=True)

    print(f"[vit_layerwise] min layer sim={min_sim:.8f}", flush=True)
    if not (min_sim > 0.98):
        raise ValueError(f"ViT layerwise mismatch (min sim={min_sim:.6f})")
    print("[vit_layerwise] ViT layerwise consistency OK", flush=True)

    model.model.visual = merged_visual.to("cpu")
    del orig_vit
    torch.cuda.empty_cache()
