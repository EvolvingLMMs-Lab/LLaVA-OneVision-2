from dataclasses import dataclass

import torch
from torch.nn.functional import gelu

from aiak_training_llm.models.factory import register_model_config
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies


@dataclass
class AdapterConfig:
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """

    normalization: str
    activation_func: torch.nn.Module = gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06



@dataclass
class LlavaOnevision2Config:
    """config for llava one vision 2 model"""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = None
    make_vocab_size_divisible_by: int = 128
    norm_epsilon: float = 1e-06
    rotary_base: int = 1000000
    kv_channels: int = None
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    # --- Qwen3.5 hybrid-attention (M-RoPE + gated delta-net) extras ---
    # Defaults keep every existing arch (Qwen3 path) unaffected: language_model_type
    # stays None -> provider/QwenModel take the standard rope path.
    language_model_type: str = None
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    mrope_section: list = None
    mrope_interleaved: bool = False
    partial_rotary_factor: float = 1.0

@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-layer1")
def llava_onevision2_layer1():
    """llava-onevision2-layer1"""
    return LlavaOnevision2Config(
        num_layers=1,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=1000000,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-2b")
def llava_onevision2_2b():
    """llava-onevision2-2b"""
    return LlavaOnevision2Config(
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=1000000,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-3b")
def llava_onevision2_3b():
    """llava-onevision2-3b"""
    return LlavaOnevision2Config(
        num_layers=36,
        hidden_size=2048,
        ffn_hidden_size=11008,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=2,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        untie_embeddings_and_output_weights=False,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-4b")
def llava_onevision2_4b():
    """llava-onevision2-4b"""
    return LlavaOnevision2Config(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=5000000,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-4b-p16m3"
)
def llava_onevision2_4b_p16m3():
    """llava-onevision2-4b with patch_size=16 and spatial_merge_size=3 (rope 3x3 layout).

    LLM portion is identical to llava-onevision2-4b; only the ViT differs
    (configured separately in get_vision_config via the 'p16m3' suffix).
    """
    return LlavaOnevision2Config(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=5000000,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-4b-p14m3"
)
def llava_onevision2_4b_p14m3():
    """llava-onevision2-4b with patch_size=14 and spatial_merge_size=3 (rope 3x3 layout).

    LLM portion is identical to llava-onevision2-4b; only the ViT differs
    (configured separately in get_vision_config via the 'p14m3' suffix).
    """
    return LlavaOnevision2Config(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=5000000,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-4b-p16m2"
)
def llava_onevision2_4b_p16m2():
    """llava-onevision2-4b with patch_size=16 and spatial_merge_size=2."""
    return LlavaOnevision2Config(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=5000000,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-30b-a3b"
)
def llava_onevision2_30b_a3b():
    """llava-onevision2-30b-a3b"""
    return LlavaOnevision2Config(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        num_experts=128,
        moe_ffn_hidden_size=768,
        rotary_base=10000000,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-8b")
def llava_onevision2_8b():
    """llava-onevision2-8b"""
    return LlavaOnevision2Config(
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=8000000,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-32b")
def llava_onevision2_32b():
    """llava-onevision2-32b"""
    return LlavaOnevision2Config(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=25600,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=1000000,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-14b")
def llava_onevision2_14b():
    """llava-onevision2-14b"""
    return LlavaOnevision2Config(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=17408,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        rotary_base=1000000,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-4b-qwen35"
)
def llava_onevision2_4b_qwen35():
    """llava-onevision2-4b with the Qwen3.5 hybrid-attention LLM backbone.

    LLM = ported+validated Qwen3.5-4B (32 hybrid layers: linear gated-delta-net +
    periodic full attention every full_attention_interval, interleaved 3D M-RoPE,
    partial rotary 0.25). ViT/adapter reuse the p16/merge3 encoder (see
    get_vision_config 'qwen35' branch). language_model_type='qwen3_5' triggers the
    provider + QwenModel Qwen3.5 branches.
    """
    return LlavaOnevision2Config(
        num_layers=32,
        hidden_size=2560,
        ffn_hidden_size=9216,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=256,
        vocab_size_in_config_file=248320,  # Qwen3.5 padded vocab (248320 % 128 == 0); matches merged HF text_config.vocab_size
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        add_qkv_bias=False,
        swiglu=True,
        rotary_base=10000000,
        untie_embeddings_and_output_weights=False,  # tie_word_embeddings=True
        norm_epsilon=1e-06,
        # Qwen3.5 hybrid-attention extras
        language_model_type="qwen3_5",
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        partial_rotary_factor=0.25,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-2b-qwen35"
)
def llava_onevision2_2b_qwen35():
    """llava-onevision2-2b with the Qwen3.5-2B hybrid-attention LLM backbone.

    Same architecture family as llava_onevision2_4b_qwen35 but with the
    Qwen3.5-2B dims (from its HF text_config): 24 hybrid layers, hidden 2048,
    ffn 6144, full-attn 8 heads / 2 KV groups (head_dim 256), and
    linear_num_value_heads=16 (4B uses 32). rope/mrope identical to 4B
    (theta 1e7, mrope [11,11,10] interleaved, partial rotary 0.25).
    ViT/adapter reuse the p16/merge3 encoder ("qwen35" in model_name).
    """
    return LlavaOnevision2Config(
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=8,
        group_query_attention=True,
        num_query_groups=2,
        kv_channels=256,
        vocab_size_in_config_file=248320,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        add_qkv_bias=False,
        swiglu=True,
        rotary_base=10000000,
        untie_embeddings_and_output_weights=False,  # tie_word_embeddings=True
        norm_epsilon=1e-06,
        # Qwen3.5 hybrid-attention extras
        language_model_type="qwen3_5",
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        partial_rotary_factor=0.25,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-9b-qwen35"
)
def llava_onevision2_9b_qwen35():
    """llava-onevision2-9b with the Qwen3.5-9B hybrid-attention LLM backbone.

    Dims from the Qwen3.5-9B HF text_config: 32 hybrid layers (same count as
    4B, widened), hidden 4096, ffn 12288, 16 heads / 4 KV groups (head_dim
    256). GDN dims identical to 4B (16 key / 32 value heads, head_dim 128,
    conv 4, full attention every 4th layer). Embeddings are UNTIED: the source
    checkpoint ships a separate lm_head.weight that differs bitwise from
    embed_tokens. rope/mrope identical to 2B/4B (theta 1e7, mrope [11,11,10]
    interleaved, partial rotary 0.25).
    """
    return LlavaOnevision2Config(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=256,
        vocab_size_in_config_file=248320,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        add_qkv_bias=False,
        swiglu=True,
        rotary_base=10000000,
        untie_embeddings_and_output_weights=True,  # tie_word_embeddings=False (9B ships separate lm_head)
        norm_epsilon=1e-06,
        # Qwen3.5 hybrid-attention extras
        language_model_type="qwen3_5",
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        partial_rotary_factor=0.25,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_ONEVISION2, model_arch="llava-onevision2-27b-qwen35"
)
def llava_onevision2_27b_qwen35():
    """llava-onevision2-27b with the Qwen3.8-27B hybrid-attention LLM backbone.

    Dims from the Qwen3.8-27B HF text_config (model_type qwen3_5_text, dense):
    64 hybrid layers, hidden 5120, ffn 17408, 24 heads / 4 KV groups (head_dim
    256). GDN: 16 key / 48 value heads (head_dim 128), conv 4, full attention
    every 4th layer. Embeddings UNTIED (separate lm_head, verified at
    extraction). rope/mrope identical to 2B/4B/9B. TP must divide 4 (KV groups).
    """
    return LlavaOnevision2Config(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=17408,
        num_attention_heads=24,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=256,
        vocab_size_in_config_file=248320,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        add_qkv_bias=False,
        swiglu=True,
        rotary_base=10000000,
        untie_embeddings_and_output_weights=True,  # tie_word_embeddings=False
        norm_epsilon=1e-06,
        # Qwen3.5 hybrid-attention extras
        language_model_type="qwen3_5",
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        partial_rotary_factor=0.25,
    )


@dataclass
class VisionConfig:
    """configuration for vision model

    The fields need to be consistent with the definitions in args
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: tuple[int]
    image_size: tuple[int]
    kv_channels: int
    normalization: str
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    layernorm_epsilon: float = 1e-05
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "none"
    frame_windows_size: int = 4
    spatial_merge_size: int = 2


def get_vision_config(model_family, model_name):
    """get vision config"""
    config = VisionConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        patch_size=14,
        image_size=(1344, 1344),
        kv_channels=64,
        normalization="LayerNorm",
        swiglu=False,
        class_token_len=0,
        group_query_attention=False,
        attention_dropout=0,
        hidden_dropout=0,
        layernorm_epsilon=1e-5,
        activation_func=torch.nn.functional.gelu,
        bias_activation_fusion=False,
        gated_linear_unit=False,
        in_channels=3,
        num_query_groups=16,
        add_bias_linear=True,
        add_qkv_bias=True,
        position_embedding_type="rope",
    )
    if "vision-2b" in model_name:
        config.num_layers = 48
        config.hidden_size = 1664
        config.ffn_hidden_size = 8192
        config.kv_channels = 104
    elif "llava-onevision2-layer1" == model_name:
        config.num_layers = 1
    if "p16m2" in model_name:
        config.patch_size = 16
        config.image_size = (224, 224)
        config.spatial_merge_size = 2
    elif "p16m3" in model_name:
        config.patch_size = 16
        config.image_size = (384, 384)
        config.spatial_merge_size = 3
    elif "p14m3" in model_name:
        config.patch_size = 14
        config.image_size = (336, 336)
        config.spatial_merge_size = 3
    elif "qwen35" in model_name:
        # Qwen3.5 VLM reuses the p16/merge3 encoder (patch16, 384px, 3x3 spatial merge).
        config.patch_size = 16
        config.image_size = (384, 384)
        config.spatial_merge_size = 3
        # Qwen3.5 ViT uses per-frame self-attention (no 4-frame temporal merge);
        # frame_windows_size=1 => each frame is its own attention segment.
        config.frame_windows_size = 1
    return config


def get_adapeter_config(model_family, model_name=None):
    """get adapeter config"""
    config = AdapterConfig(
        normalization="LayerNorm",
        add_bias_linear=True,
    )
    return config
