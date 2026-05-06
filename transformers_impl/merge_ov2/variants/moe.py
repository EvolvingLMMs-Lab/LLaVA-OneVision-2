import os

import torch

from llavaonevision2.configuration_llava_onevision2_moe import LlavaOnevision2MoeConfig
from llavaonevision2.modeling_llava_onevision2_moe import LlavaOnevision2ForConditionalGeneration

from transformers import AutoConfig, AutoProcessor, Qwen2Tokenizer


class MoeVariant:
    name = "moe"

    def build_empty(
        self,
        llm_path: str,
        processor_path: str,
        *,
        spatial_merge_size: int,
        target_dtype: torch.dtype = torch.float32,
        vit_path: str | None = None,
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(processor_path, trust_remote_code=True, use_fast=True)
        processor = AutoProcessor.from_pretrained(processor_path, use_fast=True, trust_remote_code=True)
        processor.image_processor.temporal_patch_size = 1
        processor.image_processor.max_pixels = 2000 * 2000
        if hasattr(processor.image_processor, "merge_size"):
            processor.image_processor.merge_size = spatial_merge_size

        llm_cfg = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
        cfg = LlavaOnevision2MoeConfig(text_config=llm_cfg.to_dict(), output_router_logits=True)
        # See dense.py: AutoConfig falls back to stock ViTConfig without a config.json,
        # silently corrupting model shape. Gate on file existence.
        if vit_path is not None and os.path.isfile(os.path.join(vit_path, "config.json")):
            vit_cfg = AutoConfig.from_pretrained(vit_path, trust_remote_code=True)
            cfg.vision_config.patch_size = getattr(vit_cfg, "patch_size", cfg.vision_config.patch_size)
            cfg.vision_config.image_size = getattr(vit_cfg, "image_size", cfg.vision_config.image_size)
            cfg.vision_config.hidden_size = vit_cfg.hidden_size
            cfg.vision_config.num_hidden_layers = vit_cfg.num_hidden_layers
            cfg.vision_config.num_attention_heads = vit_cfg.num_attention_heads
            cfg.vision_config.intermediate_size = vit_cfg.intermediate_size
            if hasattr(processor.image_processor, "patch_size"):
                processor.image_processor.patch_size = cfg.vision_config.patch_size
        cfg.vision_config.spatial_merge_size = spatial_merge_size
        cfg.text_config.tie_word_embeddings = False
        cfg.vision_config.text_hidden_size = cfg.text_config.hidden_size
        cfg.vision_config.out_hidden_size = cfg.text_config.hidden_size

        model = LlavaOnevision2ForConditionalGeneration(cfg)
        model.to(dtype=target_dtype)
        return model, processor, tokenizer
