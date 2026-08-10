from copy import deepcopy
from dataclasses import asdict

from megatron.core import mpu
from megatron.core.transformer.spec_utils import import_module

from aiak_training_llm.models.factory import register_model_provider
from aiak_training_llm.models.llava_onevision2.llava_onevision2_config import get_adapeter_config, get_vision_config
from aiak_training_llm.models.llava_onevision2.adapter import get_adapter_type
from aiak_training_llm.models.llava_onevision2.llava_onevision2_layer_spec import (
    get_adapeter_layer_with_spec,
    get_qwen_layer_with_te_spec,
    get_vision_layer_with_spec,
)
from aiak_training_llm.models.llava_onevision2.llava_onevision2_model import LlavaOnevision2
from aiak_training_llm.models.qwen35.qwen35_layer_spec import get_qwen3_5_transformer_layer_spec
from aiak_training_llm.utils import build_transformer_config, get_args, print_rank_0
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies


# Fields that the Qwen3.5 mcore modules read off the TransformerConfig via getattr.
# core_transformer_config_from_args only copies known TransformerConfig fields, so
# these must be threaded onto language_config explicitly.
_QWEN35_EXTRA_FIELDS = (
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_conv_kernel_dim",
    "full_attention_interval",
    "mrope_section",
    "mrope_interleaved",
    "partial_rotary_factor",
)


def is_qwen35(src) -> bool:
    """True if this arch uses the Qwen3.5 hybrid-attention LLM backbone."""
    return getattr(src, "language_model_type", None) in ("qwen3_5", "qwen35")


def apply_qwen35_language_config(language_config, src):
    """Thread the Qwen3.5 hybrid-attention fields + required rope flags from ``src``
    (the args namespace or a model-config object) onto ``language_config`` so the
    ported mcore modules and Qwen35RotaryEmbedding read them via getattr.

    Mirrors the proven full-model parity recipe: rotary_interleaved=False,
    apply_rope_fusion=False, multi_latent_attention=False.
    """
    for k in _QWEN35_EXTRA_FIELDS:
        v = getattr(src, k, None)
        if v is not None:
            setattr(language_config, k, v)
    language_config.language_model_type = "qwen3_5"
    language_config.rotary_interleaved = False
    language_config.apply_rope_fusion = False
    language_config.multi_latent_attention = False
    # num_moe_experts absent -> dense MLP path in the layer spec.
    if getattr(language_config, "num_moe_experts", None) is None:
        language_config.num_moe_experts = None
    return language_config


@register_model_provider(model_family=[VisionLanguageModelFamilies.LLAVA_ONEVISION2])
def llavaov_2_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    parallel_output: bool = True,
) -> LlavaOnevision2:
    """Builds the LLavaOneVision2 model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        LlavaOnevision2: The returned model
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
    # CLI override for the ViT temporal attention window (--vision-frame-windows-size);
    # None = keep the VisionConfig default.
    if getattr(args, "vision_frame_windows_size", None) is not None:
        vision_config.frame_windows_size = args.vision_frame_windows_size
    for k, v in asdict(get_adapeter_config(model_family, args.model_name)).items():
        setattr(adapter_config, k, v)
    # The adapter returns full replicated image embeddings for masked_scatter;
    # language-side sequence parallelism is applied after fusion.
    adapter_config.sequence_parallel = False
    print_rank_0(f"using {get_adapter_type()} as adapter type")
    # The adapter's linears use parallel_mode=None (non-tensor-parallel), so there is
    # no TP communication to overlap. Disable tp_comm_overlap for the adapter config
    # (mirrors vision_config above); otherwise TELinear asserts on a missing ub buffer
    # name when --tp-comm-overlap is enabled.
    adapter_config.tp_comm_overlap = False
    # print(vision_config)
    # Vision placeholder token ids MUST match the training tokenizer, else the
    # model never finds image tokens in input_ids -> the adapter is bypassed in
    # forward -> grad norm 0 (adapter never trains). Qwen3.5 uses a different
    # token space (248056/248057/248053/248054) than Qwen3/OV1.5 (151655/...).
    _is_qwen35 = (getattr(args, "language_model_type", None) in ("qwen3_5", "qwen35")
                  or "qwen35" in getattr(args, "model_name", ""))
    if _is_qwen35:
        setattr(language_config, "image_token_id", 248056)
        setattr(language_config, "video_token_id", 248057)
        setattr(language_config, "vision_start_token_id", 248053)
        setattr(language_config, "vision_end_token_id", 248054)
    else:
        setattr(language_config, "image_token_id", 151655)
        setattr(language_config, "video_token_id", 151656)
        # Fields needed by the multimodal 3D M-RoPE get_rope_index (Qwen2.5-VL token space).
        setattr(language_config, "vision_start_token_id", 151652)
    setattr(language_config, "spatial_merge_size", getattr(vision_config, "spatial_merge_size", 2))
    setattr(language_config, "tokens_per_second", 2)

    # FIXME: fix this if model_type is encoder_and_decoder
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        vision_config.pipeline_model_parallel_size = 1
        # The vision encoder shares the TP process group with the language model.
        # Attention head splitting in the encoder uses the actual TP world size
        # (parallel_state.get_tensor_model_parallel_world_size()), so the TP size
        # in the config must match the global TP size.  Do NOT override
        # tensor_model_parallel_size here; it already inherits the correct value
        # from the base config (build_transformer_config).  Sequence parallel is
        # likewise inherited: when TP=1 the value has no effect because TE's SP
        # logic is gated on tp_size > 1.
        vision_config.first_pipeline_num_layers = vision_config.last_pipeline_num_layers = None
        vision_config.tp_comm_overlap = False
        vision_config.context_parallel_size = 1
        vision_config.context_parallel_ulysses_degree = 1
        add_encoder = mpu.is_pipeline_first_stage()
        add_decoder = True
    else:
        assert args.encoder_pipeline_model_parallel_size == 1, (
            "vision model and projection can only live on 1 pipeline stage."
        )
        vision_config.pipeline_model_parallel_size = args.encoder_pipeline_model_parallel_size
        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = args.encoder_tensor_model_parallel_size

        # Make sure the vision model does not inherit first and last pipeline num layers from the language model.
        vision_config.first_pipeline_num_layers = vision_config.last_pipeline_num_layers = None

        # Context parallel is not yet supported for the vision encoder.
        # Sequence parallel is supported when the encoder TP size > 1; the
        # vision_config inherits sequence_parallel from the base config
        # (args.sequence_parallel) – ColumnParallelLinear will automatically
        # disable SP when TP world size == 1.
        vision_config.context_parallel_size = 1
        vision_config.tp_comm_overlap = False

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        language_layer_spec = import_module(args.spec)
    else:
        adapter_layer_spec = get_adapeter_layer_with_spec()
        vision_layer_spec = get_vision_layer_with_spec()
        if is_qwen35(args):
            # Qwen3.5 hybrid-attention backbone: thread extra fields + rope flags onto
            # language_config, then build the hybrid (linear + full attention) spec.
            apply_qwen35_language_config(language_config, args)
            language_layer_spec = get_qwen3_5_transformer_layer_spec(language_config)[0]
        else:
            language_layer_spec = get_qwen_layer_with_te_spec(language_config)

    model = LlavaOnevision2(
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
