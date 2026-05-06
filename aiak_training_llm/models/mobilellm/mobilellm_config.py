"""
MobileLLM Configuration

Loads configuration from facebook/MobileLLM-R1-140M HuggingFace checkpoint.
Reference: https://github.com/facebookresearch/MobileLLM-R1
"""

import json
import os
from megatron.core.transformer import TransformerConfig


def load_mobilellm_hf_config(checkpoint_dir):
    """Load MobileLLM config from HuggingFace config.json"""
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'r') as f:
        hf_config = json.load(f)
    return hf_config


def get_mobilellm_config(args):
    """
    Create TransformerConfig for MobileLLM-R1-140M from HuggingFace checkpoint.
    
    Reads the actual config.json from the downloaded checkpoint to ensure
    we match the exact architecture of the pretrained model.
    """
    # Load HuggingFace config
    checkpoint_dir = getattr(args, 'mobilellm_checkpoint_dir', 
                            'aiak_training_llm/models/mobilellm/hf_checkpoint')
    hf_config = load_mobilellm_hf_config(checkpoint_dir)
    
    # Extract architecture from HuggingFace config
    num_layers = hf_config['num_hidden_layers']  # 15
    hidden_size = hf_config['hidden_size']  # 576
    num_attention_heads = hf_config['num_attention_heads']  # 9
    num_key_value_heads = hf_config['num_key_value_heads']  # 3 (GQA)
    ffn_hidden_size = hf_config['intermediate_size_mlp']  # 2048
    vocab_size = hf_config['vocab_size']  # 128256
    max_position_embeddings = hf_config['max_position_embeddings']  # 32768
    rope_theta = hf_config['rope_theta']  # 8000000.0
    rms_norm_eps = hf_config['rms_norm_eps']  # 1e-05
    tie_word_embeddings = hf_config['tie_word_embeddings']  # True
    
    print(f"\n[MobileLLM Config] Loaded from: {checkpoint_dir}/config.json")
    print(f"  Layers: {num_layers}, Hidden: {hidden_size}, Heads: {num_attention_heads}, KV Heads: {num_key_value_heads}")
    print(f"  FFN: {ffn_hidden_size}, Vocab: {vocab_size}, Max Seq: {max_position_embeddings}")
    print(f"  RoPE Theta: {rope_theta}, RMS Epsilon: {rms_norm_eps}, Tied Embeddings: {tie_word_embeddings}\n")
    
    return TransformerConfig(
        # Model architecture from HF config
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,  # GQA
        ffn_hidden_size=ffn_hidden_size,
        
        # Vocabulary and sequence
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        
        # Normalization from HF config
        normalization="RMSNorm",
        norm_epsilon=rms_norm_eps,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=rms_norm_eps,
        
        # Activation (SwiGLU from HF config hidden_act: "silu")
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func="swiglu",
        bias_activation_fusion=True,
        
        # Embeddings from HF config
        untie_embeddings_and_output_weights=not tie_word_embeddings,
        
        # Position embeddings:
        # HF Llama4TextAttention treats no_rope_layers[layer_idx] as the
        # use_rope flag. MobileLLM-R1-140M sets it to 1 for all layers, so this
        # Megatron integration should use RoPE with rope_theta=8000000.
        position_embedding_type="rope",
        rotary_base=rope_theta,
        rotary_percent=1.0,
        rotary_interleaved=False,
        
        # Attention
        kv_channels=hidden_size // num_attention_heads,
        attention_dropout=0.0,
        
        # Initialization
        init_method_std=0.02,
        apply_query_key_layer_scaling=False,
        attention_softmax_in_fp32=True,
        
        # Precision
        params_dtype=getattr(args, 'params_dtype', 'bfloat16'),
        bf16=getattr(args, 'bf16', True),
        fp16=getattr(args, 'fp16', False),
        
        # Initialization
        use_cpu_initialization=True,
        perform_initialization=True,
        
        # Fusion and optimization
        gradient_accumulation_fusion=False,
        async_tensor_model_parallel_allreduce=False,
        
        # Parallelism
        sequence_parallel=getattr(args, 'sequence_parallel', False),
        
        # Memory optimization
        use_distributed_optimizer=getattr(args, 'use_distributed_optimizer', True),
    )


def print_mobilellm_config(config):
    """Print MobileLLM configuration for verification."""
    print("\n" + "="*50)
    print("MobileLLM-R1-140M Configuration")
    print("="*50)
    print(f"Layers:                {config.num_layers}")
    print(f"Hidden Size:           {config.hidden_size}")
    print(f"Attention Heads:       {config.num_attention_heads}")
    print(f"KV Heads (GQA):        {config.num_query_groups}")
    print(f"FFN Hidden:            {config.ffn_hidden_size}")
    print(f"Vocab Size:            {config.vocab_size}")
    print(f"Max Seq Length:        {config.max_position_embeddings}")
    print(f"RoPE Base:             {config.rotary_base}")
    print(f"Normalization:         {config.normalization}")
    print(f"Activation:            swiglu (gated_linear_unit={config.gated_linear_unit})")
    print(f"Shared Embeddings:     {not config.untie_embeddings_and_output_weights}")
    print(f"Precision:             {config.params_dtype}")
    print("="*50 + "\n")
