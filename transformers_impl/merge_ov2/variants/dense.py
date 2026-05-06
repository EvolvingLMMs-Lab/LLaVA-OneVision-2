import os
import json

import torch

from llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2Config
from llavaonevision2.modeling_llava_onevision2 import LlavaOnevision2ForConditionalGeneration

from transformers import AutoConfig, AutoProcessor, Qwen2Tokenizer


class DenseVariant:
    name = "dense"

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

        cfg = LlavaOnevision2Config()
        # Only honor vit_path when it carries an explicit config.json. AutoConfig otherwise
        # silently falls back to a stock ViTConfig (768/12 layers/patch16), which would
        # corrupt the empty model shape for fixtures that ship weights without a config.
        if vit_path is not None and os.path.isfile(os.path.join(vit_path, "config.json")):
            try:
                vit_cfg = AutoConfig.from_pretrained(vit_path, trust_remote_code=True)
                cfg.vision_config.patch_size = getattr(vit_cfg, "patch_size", cfg.vision_config.patch_size)
                cfg.vision_config.image_size = getattr(vit_cfg, "image_size", cfg.vision_config.image_size)
                cfg.vision_config.hidden_size = vit_cfg.hidden_size
                cfg.vision_config.num_hidden_layers = vit_cfg.num_hidden_layers
                cfg.vision_config.num_attention_heads = vit_cfg.num_attention_heads
                cfg.vision_config.intermediate_size = vit_cfg.intermediate_size
            except ValueError:
                with open(os.path.join(vit_path, "config.json"), "r", encoding="utf-8") as f:
                    vit_cfg = json.load(f)
                cfg.vision_config.patch_size = vit_cfg.get("patch_size", cfg.vision_config.patch_size)
                cfg.vision_config.image_size = vit_cfg.get("image_size", cfg.vision_config.image_size)
                cfg.vision_config.hidden_size = vit_cfg["hidden_size"]
                cfg.vision_config.num_hidden_layers = vit_cfg["num_hidden_layers"]
                cfg.vision_config.num_attention_heads = vit_cfg["num_attention_heads"]
                cfg.vision_config.intermediate_size = vit_cfg["intermediate_size"]
            if hasattr(processor.image_processor, "patch_size"):
                processor.image_processor.patch_size = cfg.vision_config.patch_size
        cfg.vision_config.spatial_merge_size = spatial_merge_size
        cfg.text_config.update(AutoConfig.from_pretrained(llm_path, trust_remote_code=True).to_dict())
        cfg.text_config.tie_word_embeddings = False
        cfg.vision_config.text_hidden_size = cfg.text_config.hidden_size
        cfg.vision_config.out_hidden_size = cfg.text_config.hidden_size

        model = LlavaOnevision2ForConditionalGeneration(cfg)
        model.to(dtype=target_dtype)
        return model, processor, tokenizer
