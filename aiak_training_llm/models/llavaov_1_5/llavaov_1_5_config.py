import torch
from dataclasses import dataclass

from megatron.training.activations import quick_gelu
from torch.nn.functional import gelu
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies
from aiak_training_llm.models.factory import register_model_config


@dataclass
class AdapterConfig:
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """
    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06


@dataclass
class LlavaOnevision1_5Config:
    """config for llava one vision 1.5 model"""
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


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5, model_arch="llava-ov-1.5-3b")
def llava_one_vision_1_5_3b():
    """llava-ov-1.5-3b"""
    return LlavaOnevision1_5Config(
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

@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5, model_arch="llava-ov-1.5-4b")
def llava_one_vision_1_5_4b():
    """llava-ov-1.5-4b"""
    return LlavaOnevision1_5Config(
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

@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5, model_arch="llava-ov-1.5-30b-a3b")
def llava_one_vision_1_5_30b_a3b():
    """llava-ov-1.5-30b-a3b"""
    return LlavaOnevision1_5Config(
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

@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5, model_arch="llava-ov-1.5-8b")
def llava_one_vision_1_5_8b():
    """llava-ov-1.5-8b"""
    return LlavaOnevision1_5Config(
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
        rotary_base=1000000,
    )   


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5, model_arch="llava-ov-1.5-32b")
def llava_one_vision_1_5_32b():
    """llava-ov-1.5-32b"""
    return LlavaOnevision1_5Config(
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


@register_model_config(model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5, model_arch="llava-ov-1.5-14b")
def llava_one_vision_1_5_14b():
    """llava-ov-1.5-14b"""
    return LlavaOnevision1_5Config(
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


def get_vision_config(model_family, model_name):
    """ get vision config """
    config = VisionConfig(
        num_layers=24,
        hidden_size=3072,
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
        position_embedding_type="rope"
    )
    if "vision-2b" in model_name:
        config.num_layers = 48
        config.hidden_size = 1664
        config.ffn_hidden_size = 8192
        config.kv_channels = 104

    return config


def get_adapeter_config(model_family):
    """ get adapeter config """
    config = AdapterConfig(
        normalization="LayerNorm",
        add_bias_linear=True,
    )
    return config
