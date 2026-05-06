"""LLaVA-OneVision-1.5 model provider - supports Qwen and MobileLLM backbones"""

from copy import deepcopy
from dataclasses import asdict

from aiak_training_llm.models.factory import register_model_provider
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_layer_spec import (
    get_adapeter_layer_with_spec, get_qwen_layer_with_te_spec,
    get_mobilellm_layer_with_te_spec, get_vision_layer_with_spec)
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_config import (
    get_adapeter_config, get_vision_config)
# MobileLLM config is now in llavaov_1_5_config.py and applied via args
from aiak_training_llm.utils import (build_transformer_config, get_args,
                                     print_rank_0, get_tokenizer)
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies
from megatron.core import mpu
from megatron.core.transformer.spec_utils import import_module

from .llavaov_1_5_model import LlavaOnevision1_5

# model provider registration
@register_model_provider(model_family=[VisionLanguageModelFamilies.LLAVA_OV_1_5])
def rice_vl_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    parallel_output: bool = True,

) -> LlavaOnevision1_5:
    """Builds the llava-ov-1.5 model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        RiceVLModel: The returned model
    """
    args = get_args()

    print_rank_0(f'building {args.model_name} model ...')
    print_rank_0(f'[DEBUG PROVIDER] Model name: {args.model_name}')
    print_rank_0(f'[DEBUG PROVIDER] Args num_layers: {args.num_layers}')
    print_rank_0(f'[DEBUG PROVIDER] Args hidden_size: {args.hidden_size}')
    print_rank_0(f'[DEBUG PROVIDER] Args vocab_size: {args.vocab_size_in_config_file}')

    config = build_transformer_config(args) #base transformer config with all hyperparams
    print_rank_0(f'[DEBUG PROVIDER] TransformerConfig built with num_layers: {config.num_layers}, hidden_size: {config.hidden_size}')

    language_config = deepcopy(config) # For Qwen2.5 language model
    vision_config = deepcopy(config) # For vision encoder (SigLIP)
    adapter_config = deepcopy(config) ## For adapter (projection)
    print_rank_0(f'[DEBUG PROVIDER] Initial language_config: layers={language_config.num_layers}, hidden={language_config.hidden_size}')

        #     Vision Encoder → Adapter → Language Model
        #    (SigLIP)    (Projection)   (Qwen2.5)

    from aiak_training_llm.models import get_model_family
    model_family = get_model_family(args.model_name)
    print_rank_0(f'[DEBUG PROVIDER] Model family: {model_family}')
    
    # Detect if using MobileLLM backbone
    use_mobilellm = "mobilellm" in args.model_name.lower()
    print_rank_0(f'[DEBUG PROVIDER] use_mobilellm flag: {use_mobilellm}')
    
    if use_mobilellm:
        print_rank_0(f'[DEBUG PROVIDER] ✓ Using MobileLLM-R1-140M as language backbone')
        print_rank_0(f'[DEBUG PROVIDER] Language config BEFORE override: layers={language_config.num_layers}, '
                     f'hidden={language_config.hidden_size}, heads={language_config.num_attention_heads}')
        # MobileLLM params are already in language_config from args!
        # No need to load separately - they were applied in _validate_extra_model_args
        print_rank_0(f'[DEBUG PROVIDER] Language config AFTER (should be same): layers={language_config.num_layers}, '
                     f'hidden={language_config.hidden_size}, heads={language_config.num_attention_heads}, '
                     f'query_groups={language_config.num_query_groups}')
    else:
        print_rank_0(f'[DEBUG PROVIDER] Using Qwen2.5 as language backbone')
    
    print_rank_0(f'[DEBUG PROVIDER] ========== LANGUAGE CONFIG ==========')
    print_rank_0(f'{language_config}')
    print_rank_0(f'[DEBUG PROVIDER] ======================================')
    
    # get vision specific config : no. of layers, hidden size, Patch size, Image resolution
    print_rank_0(f'[DEBUG PROVIDER] Loading vision config...')
    
    # Check if using FastViT - it has its own config system
    if getattr(args, 'use_fastvit', False):
        # FastViT loads config from mobileclip_l.json - set TransformerConfig to match
        import json
        import os
        
        vision_tower_name = getattr(args, 'vision_tower_name', 'mobileclip_l_1024')
        setattr(vision_config, 'vision_tower_name', vision_tower_name)
        
        # Parse resolution from vision_tower_name (e.g., "mobileclip_l_1024" -> 1024)
        resolution = int(vision_tower_name.split('_')[-1]) if '_' in vision_tower_name else 1024
        
        # Load the actual mobileclip JSON config
        json_config_path = os.path.join(
            os.path.dirname(__file__), 
            '../fastvit/mobileclip/configs/mobileclip_l.json'
        )
        with open(json_config_path, 'r') as f:
            mobileclip_config = json.load(f)
        
        # Extract image_cfg from the JSON
        image_cfg = mobileclip_config['image_cfg']
        
        # Set vision_config values from mobileclip_l.json
        setattr(vision_config, 'num_layers', 24)  # RepMixer layers in main stage
        setattr(vision_config, 'hidden_size', image_cfg['embed_dim'])  # 3072
        setattr(vision_config, 'patch_size', image_cfg['patch_size'])  # 64
        setattr(vision_config, 'image_size', resolution)  # from vision_tower_name (1024)
        setattr(vision_config, 'num_attention_heads', image_cfg['embed_dim'] // 64)  # 48
        train_vision_model = (
            args.trainable_modules == ['all']
            or "vision_model" in args.trainable_modules
        )
        setattr(vision_config, 'unfreeze_mm_vision_tower', train_vision_model)
        
        print_rank_0(f'[DEBUG PROVIDER] ✓ Using FastViT with vision_tower_name: {vision_tower_name}')
        print_rank_0(f'[DEBUG PROVIDER] FastViT train vision tower: {train_vision_model}')
        print_rank_0(f'[DEBUG PROVIDER] FastViT config loaded from {json_config_path}')
        print_rank_0(f'[DEBUG PROVIDER] image_cfg: embed_dim={image_cfg["embed_dim"]}, patch_size={image_cfg["patch_size"]}, model={image_cfg["model_name"]}')
        print_rank_0(f'[DEBUG PROVIDER] ========== VISION CONFIG (FastViT) ==========')
        print_rank_0(f'{vision_config}')
        print_rank_0(f'[DEBUG PROVIDER] ================================================')
    else:
        # For SigLIP/Rice models, use generic vision config
        for k, v in asdict(get_vision_config(model_family, args.model_name)).items():
            setattr(vision_config, k, v)
        print_rank_0(f'[DEBUG PROVIDER] Vision config (SigLIP/Rice): layers={vision_config.num_layers}, hidden={vision_config.hidden_size}')
        print_rank_0(f'[DEBUG PROVIDER] ========== VISION CONFIG (SigLIP/Rice) ==========')
        print_rank_0(f'{vision_config}')
        print_rank_0(f'[DEBUG PROVIDER] ====================================================')
    
    # get adapter specific config : Projection dimension, Activation function
    print_rank_0(f'[DEBUG PROVIDER] Loading adapter config...')
    for k, v in asdict(get_adapeter_config(model_family)).items():
        setattr(adapter_config, k, v)
    print_rank_0(f'[DEBUG PROVIDER] ========== ADAPTER CONFIG ==========')
    print_rank_0(f'{adapter_config}')
    print_rank_0(f'[DEBUG PROVIDER] ======================================')

    # set special token ids for language model using the shared runtime tokenizer
    image_token_id = 151655
    video_token_id = 151656
    try:
        tokenizer = get_tokenizer()
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")

        if image_token_id is None or video_token_id is None:
            vocab = getattr(tokenizer, "vocab", None)
            if isinstance(vocab, dict):
                if image_token_id is None:
                    image_token_id = vocab.get("<|image_pad|>")
                if video_token_id is None:
                    video_token_id = vocab.get("<|video_pad|>")

        if image_token_id is None:
            image_token_id = 151655
        if video_token_id is None:
            video_token_id = 151656

        print_rank_0(
            f"[DEBUG PROVIDER] Resolved vision token ids from tokenizer: "
            f"image_token_id={image_token_id}, video_token_id={video_token_id}"
        )
    except Exception as e:
        print_rank_0(
            f"[WARN PROVIDER] Failed to resolve vision token ids from tokenizer "
            f"({e}); fallback to defaults image_token_id={image_token_id}, video_token_id={video_token_id}"
        )

    setattr(language_config, "image_token_id", int(image_token_id))
    setattr(language_config, "video_token_id", int(video_token_id))

    #Handle pipeline parallelism 

    # FIXME: fix this if model_type is encoder_and_decoder
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        #UNIFIED MODEL (TP=1, PP=1)
        vision_config.pipeline_model_parallel_size = 1
        vision_config.tensor_model_parallel_size = 1
        vision_config.sequence_parallel = False
        vision_config.tp_comm_overlap = False
        vision_config.context_parallel_size = 1
        vision_config.context_parallel_ulysses_degree = 1

        add_encoder = mpu.is_pipeline_first_stage() #True on first stage 
        add_decoder = True #Always add language model decoder
    else:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "vision model and projection can only live on 1 pipeline stage."
        vision_config.pipeline_model_parallel_size = args.encoder_pipeline_model_parallel_size
        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = args.encoder_tensor_model_parallel_size

        # Make sure the vision model does not inherit first and last pipeline num layers from the language model.
        vision_config.first_pipeline_num_layers = vision_config.last_pipeline_num_layers = None

        # TODO: Vision model and projection do not use SP/CP yet.
        vision_config.sequence_parallel = False
        vision_config.context_parallel_size = 1
        vision_config.tp_comm_overlap = False

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        language_layer_spec = import_module(args.spec)
        print_rank_0(f'[DEBUG PROVIDER] Using custom spec: {args.spec}')
    else:
        print_rank_0(f'[DEBUG PROVIDER] Building layer specs...')
        adapter_layer_spec = get_adapeter_layer_with_spec()
        vision_layer_spec = get_vision_layer_with_spec()
        
        # Choose language layer spec based on backbone
        if use_mobilellm:
            print_rank_0(f'[DEBUG PROVIDER] ✓ Using MobileLLM layer specification')
            print_rank_0(f'[DEBUG PROVIDER] MobileLLM layer config: layers={language_config.num_layers}, '
                         f'hidden={language_config.hidden_size}, heads={language_config.num_attention_heads}')
            language_layer_spec = get_mobilellm_layer_with_te_spec(language_config)
            print_rank_0(f'[DEBUG PROVIDER] MobileLLM layer spec created successfully')
        else:
            print_rank_0(f'[DEBUG PROVIDER] Using Qwen layer specification')
            language_layer_spec = get_qwen_layer_with_te_spec(language_config)

#     # Vision layer spec (Transformer block)
# - MultiheadAttention
# - LayerNorm
# - MLP (feedforward)
# - Residual connections

# # Language layer spec (Qwen2.5 block)
# - MultiQueryAttention or GroupedQueryAttention
# - RMSNorm
# - SwiGLU MLP
# - RoPE (Rotary Position Embedding)

# # Adapter spec (projection)
# - Linear layer(s)
# - Optional activation

#create the model 
    print_rank_0(f'[DEBUG PROVIDER] Creating LlavaOnevision1_5 model...')
    print_rank_0(f'[DEBUG PROVIDER] Final language_config: layers={language_config.num_layers}, '
                 f'hidden={language_config.hidden_size}, vocab={args.padded_vocab_size}, '
                 f'rotary_base={args.rotary_base}')
    model = LlavaOnevision1_5(
        language_config=language_config,
        vision_config=vision_config,
        adapter_config=adapter_config,
        language_layer_spec=language_layer_spec,
        vision_layer_spec=vision_layer_spec,
        adapter_layer_spec=adapter_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process, #compute embeddings?
        post_process=post_process, #compute output logits/loss?
        add_encoder=add_encoder, #add vision encoder?
        add_decoder=add_decoder, #add language model decoder?
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type, #"rope"
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        # When using FastViT, adapter dimensions change, so allow missing adapter weights
        allow_missing_adapter_checkpoint=getattr(args, 'use_fastvit', False),
    )
    print_rank_0(f'[DEBUG PROVIDER] ✓ LlavaOnevision1_5 model created successfully!')
    # Vision encoder: SigLIP with 27 layers
    # Adapter: Projection network
    # Language model: Qwen2.5 with 32 layers
    # All distributed training wrappers (TP, PP, DP)

    #freeze components if needed
    if args.trainable_modules != ['all']:
        train_language_model = "language_model" in args.trainable_modules
        train_vision_model = "vision_model" in args.trainable_modules
        train_adapter = "adapter" in args.trainable_modules
        model.freeze(freeze_language_model=not train_language_model,
                    freeze_vision_model=not train_vision_model,
                    freeze_adapter=not train_adapter)
    # Stage 0 (Pre-training): Only train adapter
    # trainable_modules = ["adapter"]
    # → Freeze vision encoder ✓
    # → Freeze language model ✓
    # → Train adapter ✓

    return model
