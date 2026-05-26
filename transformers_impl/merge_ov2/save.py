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
        "files": ("configuration_llava_onevision2.py", "modeling_llava_onevision2.py"),
    },
    "moe": {
        "config_cls_path": ("llavaonevision2.configuration_llava_onevision2_moe", "LlavaOnevision2MoeConfig"),
        "model_cls_path": ("llavaonevision2.modeling_llava_onevision2_moe", "LlavaOnevision2ForConditionalGeneration"),
        "auto_config": "configuration_llava_onevision2_moe.LlavaOnevision2MoeConfig",
        "auto_model": "modeling_llava_onevision2_moe.LlavaOnevision2ForConditionalGeneration",
        "files": ("configuration_llava_onevision2_moe.py", "modeling_llava_onevision2_moe.py"),
    },
}


_PROCESSOR_AUX_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "video_preprocessor_config.json",
)


def _import_class(module_path: str, class_name: str):
    import importlib

    return getattr(importlib.import_module(module_path), class_name)


def _save_processor(processor, output_path: str, processor_src: str | None) -> None:
    """Save ``processor`` to ``output_path``, supporting custom processors that
    don't inherit ``ProcessorMixin`` (and therefore lack ``save_pretrained``).

    For such processors we save each sub-component (image / tokenizer / video)
    independently and copy custom code + auxiliary configs from
    ``processor_src`` so the saved checkpoint stays self-contained.
    """
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_path)
        return

    cls_name = type(processor).__name__
    if processor_src is None:
        raise RuntimeError(
            f"{cls_name} has no save_pretrained() and no processor source path "
            f"was recorded; cannot persist auxiliary processor files."
        )

    for attr in ("image_processor", "tokenizer", "video_processor"):
        sub = getattr(processor, attr, None)
        if sub is None or not hasattr(sub, "save_pretrained"):
            continue
        try:
            sub.save_pretrained(output_path)
        except Exception as e:
            logger.warning(f"{cls_name}.{attr}.save_pretrained failed: {e}")

    if not os.path.isdir(processor_src):
        return
    for fn in os.listdir(processor_src):
        if fn.endswith((".py", ".jinja")) or fn in _PROCESSOR_AUX_FILES:
            src = os.path.join(processor_src, fn)
            dst = os.path.join(output_path, fn)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy(src, dst)


def save_merged(
    model,
    output_path: str,
    tokenizer,
    processor,
    variant: str = "dense",
    processor_src: str | None = None,
):
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
    model.config.auto_map.update({"AutoConfig": spec["auto_config"], "AutoModelForCausalLM": spec["auto_model"]})

    tokenizer.save_pretrained(output_path)
    _save_processor(processor, output_path, processor_src)
    model.save_pretrained(output_path)

    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llavaonevision2")
    for fn in spec["files"]:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, fn))
    logger.info(f"Saved merged model -> {output_path}")
