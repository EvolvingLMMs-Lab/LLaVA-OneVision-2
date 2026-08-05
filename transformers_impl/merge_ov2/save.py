import os
import shutil

from transformers import logging


logger = logging.get_logger(__name__)


_VARIANT_SPEC = {
    "dense": {
        "config_cls_path": ("llavaonevision2.configuration_llava_onevision2", "LlavaOnevision2Config"),
        "model_cls_path": ("llavaonevision2.modeling_llava_onevision2", "LlavaOnevision2ForConditionalGeneration"),
        "auto_config": "configuration_llava_onevision2.LlavaOnevision2Config",
        "auto_model": "modeling_llava_onevision2.LlavaOnevision2ForConditionalGeneration",
        "files": ("configuration_llava_onevision2.py", "modeling_llava_onevision2.py",
                  "processing_llava_onevision2.py", "video_processing_llava_onevision2.py"),
    },
    "moe": {
        "config_cls_path": ("llavaonevision2.configuration_llava_onevision2_moe", "LlavaOnevision2MoeConfig"),
        "model_cls_path": ("llavaonevision2.modeling_llava_onevision2_moe", "LlavaOnevision2ForConditionalGeneration"),
        "auto_config": "configuration_llava_onevision2_moe.LlavaOnevision2MoeConfig",
        "auto_model": "modeling_llava_onevision2_moe.LlavaOnevision2ForConditionalGeneration",
        "files": ("configuration_llava_onevision2_moe.py", "modeling_llava_onevision2_moe.py"),
    },
}


def _import_class(module_path: str, class_name: str):
    import importlib

    return getattr(importlib.import_module(module_path), class_name)


def save_merged(model, output_path: str, tokenizer, processor, variant: str = "dense"):
    spec = _VARIANT_SPEC[variant]
    config_cls = _import_class(*spec["config_cls_path"])
    model_cls = _import_class(*spec["model_cls_path"])
    try:
        config_cls.register_for_auto_class()
        model_cls.register_for_auto_class("AutoModelForCausalLM")
    except Exception as e:
        logger.warning(f"register_for_auto_class failed [{type(e).__name__}]: {e}", exc_info=True)

    os.makedirs(output_path, exist_ok=True)
    if not hasattr(model.config, "auto_map"):
        model.config.auto_map = {}
    # AutoModel intentionally maps to ForConditionalGeneration: the bare
    # LlavaOnevision2Model class has a different param layout and silently
    # re-initializes the vision tower when loading these checkpoints.
    model.config.auto_map.update({
        "AutoConfig": spec["auto_config"],
        "AutoModel": spec["auto_model"],
        "AutoModelForCausalLM": spec["auto_model"],
        "AutoModelForImageTextToText": spec["auto_model"],
    })

    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    model.save_pretrained(output_path)

    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llavaonevision2")
    for fn in spec["files"]:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, fn))
    logger.info(f"Saved merged model -> {output_path}")
