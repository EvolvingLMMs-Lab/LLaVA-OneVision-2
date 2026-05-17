"""Provider for the Gemma4-VL model family.

Mirrors :mod:`llava_onevision2_provider` but switches:
  - ``register_model_provider`` family to ``VisionLanguageModelFamilies.GEMMA4_VL``
  - ``language_layer_spec`` builder to :func:`get_gemma4_layer_with_te_spec`
  - top-level VLM class to :class:`Gemma4VL`

All other knobs (TP/PP/SP gating, freeze gating, encoder PP) match the
LlavaOnevision2 provider verbatim so any base-provider improvements port
directly to this provider.
"""

from copy import deepcopy
from dataclasses import asdict

from megatron.core import mpu
from megatron.core.transformer.spec_utils import import_module

from aiak_training_llm.models.factory import register_model_provider
from aiak_training_llm.models.gemma4_vl.gemma4_vl_config import (
    get_adapeter_config,
    get_vision_config,
)
from aiak_training_llm.models.gemma4_vl.gemma4_vl_layer_spec import (
    get_adapter_layer_with_spec,
    get_gemma4_layer_with_te_spec,
    get_vision_layer_with_spec,
)
from aiak_training_llm.models.gemma4_vl.gemma4_vl_model import Gemma4VL
from aiak_training_llm.utils import build_transformer_config, get_args, print_rank_0
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies


@register_model_provider(model_family=[VisionLanguageModelFamilies.GEMMA4_VL])
def gemma4_vl_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    parallel_output: bool = True,
) -> Gemma4VL:
    """Build a Gemma4-VL model instance.

    Args:
        pre_process: include the embedding layer (pipeline-first stage).
        post_process: include the output layer / loss (pipeline-last stage).
        add_encoder / add_decoder: pipeline-encoder/decoder gating.
        parallel_output: keep outputs split across TP ranks.
    """
    args = get_args()

    print_rank_0(f"building {args.model_name} model ...")

    config = build_transformer_config(args)

    language_config = deepcopy(config)
    vision_config = deepcopy(config)
    adapter_config = deepcopy(config)

    from aiak_training_llm.models import get_model_family

    model_family = get_model_family(args.model_name)
    for k, v in asdict(get_vision_config(model_family, args.model_name)).items():
        setattr(vision_config, k, v)
    for k, v in asdict(get_adapeter_config(model_family, args.model_name)).items():
        setattr(adapter_config, k, v)

    setattr(language_config, "image_token_id", 258880)
    setattr(language_config, "video_token_id", 258884)

    if args.encoder_pipeline_model_parallel_size in [0, None]:
        vision_config.pipeline_model_parallel_size = 1
        vision_config.first_pipeline_num_layers = (
            vision_config.last_pipeline_num_layers
        ) = None
        vision_config.tp_comm_overlap = False
        vision_config.context_parallel_size = 1
        vision_config.context_parallel_ulysses_degree = 1
        add_encoder = mpu.is_pipeline_first_stage()
        add_decoder = True
    else:
        assert args.encoder_pipeline_model_parallel_size == 1, (
            "vision model and projection can only live on 1 pipeline stage."
        )
        vision_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )
        vision_config.first_pipeline_num_layers = (
            vision_config.last_pipeline_num_layers
        ) = None
        vision_config.context_parallel_size = 1
        vision_config.tp_comm_overlap = False

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        language_layer_spec = import_module(args.spec)
    else:
        adapter_layer_spec = get_adapter_layer_with_spec()
        vision_layer_spec = get_vision_layer_with_spec()
        language_layer_spec = get_gemma4_layer_with_te_spec(language_config)

    model = Gemma4VL(
        language_config=language_config,
        vision_config=vision_config,
        adapter_config=adapter_config,
        language_layer_spec=language_layer_spec,
        vision_layer_spec=vision_layer_spec,
        adapter_layer_spec=adapter_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rotary_base_sliding=getattr(args, "rotary_base_sliding", None),
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    if args.trainable_modules != ["all"]:
        train_language_model = "language_model" in args.trainable_modules
        train_vision_model = "vision_model" in args.trainable_modules
        train_adapter = "adapter" in args.trainable_modules
        model.freeze(
            freeze_language_model=not train_language_model,
            freeze_vision_model=not train_vision_model,
            freeze_adapter=not train_adapter,
        )

    return model
