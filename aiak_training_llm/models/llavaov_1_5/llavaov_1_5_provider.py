"""qwen model provider"""

from copy import deepcopy
from dataclasses import asdict

from aiak_training_llm.models.factory import register_model_provider
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_layer_spec import (
    get_adapeter_layer_with_spec, get_qwen_layer_with_te_spec,
    get_vision_layer_with_spec)
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_config import (
    get_adapeter_config, get_vision_config)
from aiak_training_llm.utils import (build_transformer_config, get_args,
                                     print_rank_0)
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

    config = build_transformer_config(args) #base transformer config with all hyperparams

    language_config = deepcopy(config) # For Qwen2.5 language model
    vision_config = deepcopy(config) # For vision encoder (SigLIP)
    adapter_config = deepcopy(config) ## For adapter (projection)

        #     Vision Encoder → Adapter → Language Model
        #    (SigLIP)    (Projection)   (Qwen2.5)

    from aiak_training_llm.models import get_model_family
    model_family = get_model_family(args.model_name)
    # get vision specific config : no. of layers, hidden size, Patch size, Image resolution
    for k, v in asdict(get_vision_config(model_family, args.model_name)).items():
        setattr(vision_config, k, v)
    
    # Add FastViT-specific configuration if enabled
    if getattr(args, 'use_fastvit', False):
        vision_tower_name = getattr(args, 'vision_tower_name', 'mobileclip_l_384')
        setattr(vision_config, 'vision_tower_name', vision_tower_name)
        print_rank_0(f'Using FastViT with vision_tower_name: {vision_tower_name}')
    
    print_rank_0(f'Vision config: {vision_config}')
    # get adapter specific config : Projection dimension, Activation function
    for k, v in asdict(get_adapeter_config(model_family)).items():
        setattr(adapter_config, k, v)
    print_rank_0(f'Adapter config: {adapter_config}')

    # set special token ids for language model
    setattr(language_config, "image_token_id", 151655)
    setattr(language_config, "video_token_id", 151656)

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
    else:
        adapter_layer_spec = get_adapeter_layer_with_spec()
        vision_layer_spec = get_vision_layer_with_spec()
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
