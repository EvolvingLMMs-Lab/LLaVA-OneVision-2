"""qwen model provider"""
import inspect
from contextlib import nullcontext

from megatron.core.transformer.spec_utils import import_module

from aiak_training_llm.utils import get_args, build_transformer_config, print_rank_0
from aiak_training_llm.utils.constants import LanguageModelFamilies

from aiak_training_llm.models.factory import register_model_provider

from .qwen_model import QwenModel
from .qwen_layer_spec import get_qwen_layer_with_te_spec


@register_model_provider(model_family=[LanguageModelFamilies.QWEN,
                                       LanguageModelFamilies.QWEN1_5,
                                       LanguageModelFamilies.QWEN2,
                                       LanguageModelFamilies.QWEN2_5,
                                       LanguageModelFamilies.QWEN3])
def qwen_model_provider(
    pre_process: bool = True, post_process: bool = True, parallel_output: bool = True,
) -> QwenModel:
    """Builds the qwen model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        QwenModel: The returned model
    """
    args = get_args()

    print_rank_0('building Qwen model ...')

    config = build_transformer_config(args)

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_qwen_layer_with_te_spec(config)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except Exception:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine,but not found.")

    with build_model_context(**build_model_context_args):
        model = QwenModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            rope_scaling_factor=args.rope_scaling_factor,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        )

    return model
