from transformers import CONFIG_MAPPING, AutoConfig
from transformers.configuration_utils import PretrainedConfig


class LlavaOnevision2VisionConfig(PretrainedConfig):
    model_type = "llava_onevision2"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        use_head=False,
        out_hidden_size=1024,
        spatial_merge_size=2,
        tokens_per_second=1,
        temporal_patch_size=1,
        frame_windows_size=4,
        use_patch_position_encoding=False,
        patch_position_encoding_type="absolute",
        max_position_embeddings=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.use_head = use_head
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.tokens_per_second = tokens_per_second
        self.temporal_patch_size = temporal_patch_size
        self.frame_windows_size = frame_windows_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.patch_position_encoding_type = patch_position_encoding_type
        self.max_position_embeddings = max_position_embeddings


class LlavaOnevision2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaOnevision2Model`]. It is used to instantiate a
    LlavaOnevision2Model model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Llava-Onevision 1.5 [lmms-lab/LLaVA-OneVision-1.5-8B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `LlavaOnevision2VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.

    ```python
    >>> from transformers import LlavaOnevision2Model, LlavaOnevision2Config

    >>> # Initializing a LlavaOnevision2 style configuration
    >>> configuration = LlavaOnevision2Config()

    >>> # Initializing a model from the Llava-Onevision-1.5-8B style configuration
    >>> model = LlavaOnevision2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_onevision2"
    sub_configs = {"vision_config": LlavaOnevision2VisionConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are same as for `LlavaOnevision2TextConfig`
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen3")
            self.sub_configs["text_config"] = CONFIG_MAPPING[text_config["model_type"]]
        elif text_config is None:
            self.sub_configs["text_config"] = CONFIG_MAPPING["qwen3"]

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # Attention implementation to use. It sets it recursively on sub-configs so we call it again in the end
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


__all__ = ["LlavaOnevision2Config"]
