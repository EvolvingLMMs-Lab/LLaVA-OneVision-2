import os
import json
import struct

import torch

from llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2Config
from llavaonevision2.modeling_llava_onevision2 import LlavaOnevision2ForConditionalGeneration

from transformers import AutoConfig, AutoProcessor, Qwen2Tokenizer


# Vision placeholder tokens whose ids must match the LLM tokenizer (they differ
# across tokenizers, e.g. 151655 for Qwen3 vs 248056 for Qwen3.5). Derived from
# the tokenizer so the merged config stays consistent with whatever LLM is used.
_VISION_TOKEN_MAP = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}


def _source_float32_param_names(llm_path):
    """Merged-model param names that the source LLM stores in float32.

    Qwen3.5 keeps SSM params (linear_attn.A_log, linear-attn RMSNorm) in float32
    under mamba_ssm_dtype. We read safetensors headers (no tensor load) and remap
    source keys (model.X) to the merged layout (model.language_model.X), matching
    remap_llm, so build_empty can keep those params fp32 instead of bf16.
    """
    if not os.path.isdir(llm_path):
        return set()
    idx = os.path.join(llm_path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as fh:
            files = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        files = [x for x in os.listdir(llm_path) if x.endswith(".safetensors")]
    names = set()
    for fn in files:
        with open(os.path.join(llm_path, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            if v.get("dtype") == "F32":
                mk = "model.language_model." + k[len("model."):] if k.startswith("model.") else k
                names.add(mk)
    return names


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

        # Build text_config from the LLM up front so LlavaOnevision2Config.__post_init__
        # resolves the CONCRETE config class via CONFIG_MAPPING[model_type] (e.g.
        # Qwen3_5TextConfig for qwen3_5_text). The default ctor + .update() keeps the
        # placeholder Qwen3Config class, so AutoModel.from_config would build a plain
        # Qwen3 model and silently drop linear-attention layers.
        llm_cfg = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
        cfg = LlavaOnevision2Config(text_config=llm_cfg.to_dict())
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
        cfg.text_config.tie_word_embeddings = False
        cfg.vision_config.text_hidden_size = cfg.text_config.hidden_size
        cfg.vision_config.out_hidden_size = cfg.text_config.hidden_size

        # Align vision placeholder token ids with the actual LLM tokenizer.
        unk = getattr(tokenizer, "unk_token_id", None)
        for attr, tok in _VISION_TOKEN_MAP.items():
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != unk:
                setattr(cfg, attr, tid)

        model = LlavaOnevision2ForConditionalGeneration(cfg)
        model.to(dtype=target_dtype)

        # Preserve params the source LLM stores in float32 (e.g. Qwen3.5 SSM A_log /
        # linear-attn norm). model.to() above cast everything to target_dtype; restore
        # fp32 so the merge does not quantize these down to bf16. Paired with loader
        # not force-casting, apply_weights then copies the fp32 source in as fp32.
        if target_dtype != torch.float32:
            f32_names = _source_float32_param_names(llm_path)
            if f32_names:
                restored = 0
                for name, p in model.named_parameters():
                    if name in f32_names:
                        p.data = p.data.float()
                        restored += 1
                print(f"[build_empty] kept {restored} source-fp32 params in fp32")
        return model, processor, tokenizer
