"""
LLaVA-OneVision-1.5 with MobileLLM backbone configuration

This module configures LLaVA-OneVision-1.5 to use MobileLLM-R1-140M as the language model
instead of Qwen. It maintains the vision encoder (SigLIP or FastViT) and adapter layers
while replacing the language model architecture.
"""

import torch
from dataclasses import dataclass
from aiak_training_llm.models.factory import register_model_config
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies


@dataclass
class LlavaOvMobileLLMConfig:
    """Unified config for LLaVA-OV with MobileLLM backbone (matching LlavaOnevision1_5Config structure)"""
    # Core architecture
    num_layers: int = 15
    hidden_size: int = 576
    ffn_hidden_size: int = 2048
    num_attention_heads: int = 9
    
    # GQA configuration
    group_query_attention: bool = True
    num_query_groups: int = 3
    
    # Position embeddings
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    rotary_base: int = 8000000
    
    # Normalization
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-05
    
    # Activation
    swiglu: bool = True
    
    # Dropout
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Bias settings
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = False
    
    # Embeddings
    untie_embeddings_and_output_weights: bool = False
    
    # Vocabulary
    vocab_size_in_config_file: int = 128256
    make_vocab_size_divisible_by: int = 128
    
    # Optional
    kv_channels: int = None
    num_experts: int = None
    moe_ffn_hidden_size: int = None


@dataclass
class MobileLLMLanguageConfig:
    """Language model configuration for MobileLLM-R1-140M"""
    # Architecture from facebook/MobileLLM-R1-140M
    num_layers: int = 15
    hidden_size: int = 576
    num_attention_heads: int = 9
    num_query_groups: int = 3  # GQA: 3 KV heads
    ffn_hidden_size: int = 2048
    
    # Vocabulary and sequence
    vocab_size: int = 128256
    max_position_embeddings: int = 32768
    
    # Normalization
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-05
    layernorm_zero_centered_gamma: bool = False
    
    # Activation
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    activation_func: str = "swiglu"
    bias_activation_fusion: bool = True
    
    # Embeddings
    untie_embeddings_and_output_weights: bool = False  # True means tied (shared)
    
    # Position embeddings
    position_embedding_type: str = "rope"
    rotary_base: int = 8000000
    rotary_percent: float = 1.0
    rotary_interleaved: bool = False
    
    # Attention
    attention_dropout: float = 0.0
    
    # Initialization
    init_method_std: float = 0.02
    apply_query_key_layer_scaling: bool = False
    attention_softmax_in_fp32: bool = True


@dataclass
class VisionConfig:
    """Vision encoder configuration (can use SigLIP or FastViT)"""
    num_layers: int = 27
    hidden_size: int = 1152
    num_attention_heads: int = 16
    ffn_hidden_size: int = 4304
    patch_size: int = 14
    image_resolution: int = 384


@dataclass
class AdapterConfig:
    """Adapter/Projection layer configuration"""
    adapter_dim: int = 2048  # Projection dimension
    adapter_act: str = "gelu"  # Activation function


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5,
    model_arch="llava-ov-mobilellm-140m"
)
def get_llava_ov_mobilellm_140m_config():
    """
    Configuration for LLaVA-OneVision-1.5 with MobileLLM-R1-140M backbone
    """
    return LlavaOvMobileLLMConfig()


@register_model_config(
    model_family=VisionLanguageModelFamilies.LLAVA_OV_1_5,
    model_arch="llava-ov-mobilellm-140m-fastvit"
)
def get_llava_ov_mobilellm_140m_fastvit_config():
    """
    Configuration for LLaVA-OneVision-1.5 with MobileLLM-R1-140M backbone and FastViT
    
    Uses FastViT (MobileCLIP) as the vision encoder for efficiency
    """
    # Use default MobileLLM config (FastViT vision config can be handled separately)
    return LlavaOvMobileLLMConfig()


def get_vision_config(model_family: str, model_name: str):
    """
    Get vision encoder configuration based on model name
    
    Args:
        model_family: Model family (llava_ov_1_5)
        model_name: Specific model architecture name
        
    Returns:
        VisionConfig dataclass instance
    """
    config = get_llava_ov_mobilellm_140m_config()
    
    if "fastvit" in model_name.lower() or "mobilellm" in model_name.lower():
        config = get_llava_ov_mobilellm_140m_fastvit_config()
    
    return config["vision"]


def get_adapter_config(model_family: str):
    """
    Get adapter/projection configuration
    
    Args:
        model_family: Model family (llava_ov_1_5)
        
    Returns:
        AdapterConfig dataclass instance
    """
    config = get_llava_ov_mobilellm_140m_config()
    return config["adapter"]


def get_language_config(model_name: str):
    """
    Get language model configuration for MobileLLM
    
    Args:
        model_name: Specific model architecture name
        
    Returns:
        MobileLLMLanguageConfig dataclass instance
    """
    config = get_llava_ov_mobilellm_140m_config()
    return config["language"]
