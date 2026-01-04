"""AIAK arguments"""

import os
import argparse
import importlib
from dataclasses import fields

from megatron.core.transformer.enums import AttnBackend

from aiak_training_llm.models import (
    get_support_model_family_and_archs,
    get_model_config,
    get_model_family,
    get_support_model_archs,
)
from aiak_training_llm.tokenizer import get_default_tokenizer
from aiak_training_llm.data import get_support_templates

from aiak_training_llm.utils import constants, parse_arguments, print_rank_0
from aiak_training_llm.utils.utils import get_default_sft_dataset_config


def parse_train_args(args_defaults={}):
    """parse arguments for training"""
    args = parse_arguments(
        extra_args_provider=aiak_extra_train_args_provider,
        validate_extra_args_provider=validate_aiak_extra_args,
        args_defaults=args_defaults,
    )
    return args


def aiak_extra_train_args_provider(parser: argparse.ArgumentParser):
    """Add AIAK arguments to parser"""
    parser.conflict_handler = 'resolve'
    parser = _add_extra_model_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_extra_sft_args(parser)
    parser = _add_extra_video_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_extra_multimodal_args(parser)
    parser = _add_extra_parallel_args(parser)

    parser = _add_extra_training_rice_vl_args(parser)
    return parser


def validate_aiak_extra_args(args):
    """"Validate AIAK extra arguments"""
    args.model_family = get_model_family(args.model_name)
    _validate_extra_model_args(args)
    _validate_extra_tokenizer_args(args)
    _validate_extra_training_args(args)
    _validate_extra_sft_args(args)
    _validata_extra_multimodal_args(args)
    _validata_extra_video_args(args)
    _validata_extra_parallel_args(args)

    # megatron one_logger is not supported in aiak
    args.enable_one_logger = False


def _add_extra_training_rice_vl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Create a dedicated group for Rice-VL arguments for better organization in the help message."""
    group = parser.add_argument_group(
        title='Training Rice-VL',
        description='Arguments specific to the Rice-VL model training configuration.'
    )

    group.add_argument(
        '--training-rice-vl-max-image-area',
        type=int,
        default=int(1344*1344),  # Corresponds to a 1344x1344 image
        help=(
            "The maximum total arsea (width * height) for an image during training. "
            "Images with a larger area will be scaled down while preserving aspect ratio."
        )
    )
    group.add_argument(
        '--training-rice-vl-max-answer-length',
        type=int,
        default=512,  # Changed from 4096 to 512 to match seq_length
        help=(
            "The maximum number of characters allowed in an answer during training. "
            "Answers longer than this will be truncated."
        )
    )
    return parser


def _add_extra_model_args(parser: argparse.ArgumentParser):
    """Add model arguments"""
    group = parser.add_argument_group(title='extra-model')
    group.add_argument('--model-name',
                       type=str,
                       required=True,
                       choices=get_support_model_family_and_archs(),
                       help='The name of model to be trained, which can be either a model family name (e.g., llama2) '
                            'or a model architecture name (e.g., llama2-7b). '
                            'If specifies the model family name, you need to completely configure the hyperparameters '
                            'of the model architecture, such as num_layers, hidden_size, etc. '
                            'And if specifies the model architecture name, aiak system will automatically override the'
                            'model architecture hyperparameters to ensure consistency with the open source version. ')

    # use for cogvlm2
    group.add_argument('--no-rope-in-fp32',  action='store_false', dest="rope_in_fp32", help='Disable Rope in FP32')

    # use for baichuan2
    group.add_argument('--use-normhead',
                       action='store_true',
                       help='use NormHead. https://arxiv.org/pdf/2309.10305.pdf. '
                            'Note that this option is only valid for the model family baichuan2 now.')

    # use for deepseek v3

    group.add_argument('--mtp-loss-coef', type=float, default=0.1,
                       help='The coefficient of MTP loss.')

    # use for xpu
    group.add_argument('--separate-layernorm-and-collinear', action='store_true',
                       help='separate layernorm and attention/mlp column parallel linear')

    # use for mla
    group.add_argument('--enable-fa-within-mla', action='store_true',
                       help="Since qk_head_dim != v_head_dim in MLA, fa cannot be used by default. Enable "
                       "this option, the head dimensions will be aligned by padding, so that fa can be used."
                       "Deprecated: use --attention-backend=flash")

    return parser


def _add_extra_tokenizer_args(parser: argparse.ArgumentParser):
    """Add data arguments"""
    group = parser.add_argument_group(title='extra-tokenizer')
    group.add_argument('--tokenizer-type',
                       type=str,
                       default=None,
                       choices=['NullTokenizer',
                                'HFTokenizer'], # recommended
                       help='What type of tokenizer to use. Default: None, and aiak automatically determines '
                            'the type of tokenizer required')

    group.add_argument('--hf-tokenizer-path',
                       type=str,
                       default=None,
                       help='HuggingFace tokenizer path: '
                            '1) A string, the *model id* of a predefined tokenizer hosted inside a model repo '
                            'on huggingface.co'
                            '2) A path to a *directory* containing vocabulary files required by the tokenizer')

    group.add_argument('--use-fast-tokenizer',
                       action='store_true',
                       help='Whether or not to use the fast tokenizer when --tokenizer-type=HFTokenizer.'
                            'Default: False',
                       dest='use_fast_tokenizer')

    group.add_argument('--split-special-tokens',
                       action='store_true',
                       help="Whether or not the special tokens should be split during the tokenization process "
                            "when --tokenizer-type=HFTokenizer. Default: False")

    group.add_argument('--padding-side',
                       default="right",
                       choices=["left", "right"],
                       help=f"The side on which the padding should be applied when --tokenizer-type=HFTokenizer. "
                             "Default: right")

    group.add_argument("--additional-special-tokens",
                       type=str,
                       default=None,
                       help="Additional special tokens to add to the tokenizer. Use commas to separate multiple tokens")

    group.add_argument('--vocab-size-in-config-file', type=int, default=None,
                       help='Size of vocab from hf config file.')

    group.add_argument('--padded-vocab-size', type=int, default=None,
                       help='Specify padded vocab size.')

    return parser


def _add_extra_sft_args(parser: argparse.ArgumentParser):
    """Add SFT arguments"""
    group = parser.add_argument_group(title='extra-sft')
    group.add_argument('--chat-template',
                       type=str,
                       choices=get_support_templates(),
                       default=None,
                       help='The template to apply to instruction data.')

    group.add_argument('--sft-dataset-config',
                       type=str,
                       default=None,
                       help="A json file that contains the dataset configuration."
                            "default: configs/dataset_config.jsoin")

    group.add_argument('--sft-dataset',
                       nargs="*",
                       default=None,
                       help='The name list for a set of dataset according to --data-path. Note that:'
                            '(1) the dataset name should be defined in the dataset config file (--sft-dataset-config). '
                            '(2) the accepted formats are: a single name or a list of names e.g. dataset1 dataset2. '
                            '(3) if multiple dataset are required, the order of names should be consistent with'
                            '--data-path. '
                            'This argument is exclusive to the other independent --sft-*-dataset arguments.')

    group.add_argument('--sft-train-dataset',
                       nargs="*",
                       default=None,
                       help='The name list for a set of independent train dataset according to --train-data-path. '
                            'Follows the same pattern rules as --sft-dataset')

    group.add_argument('--sft-valid-dataset',
                       nargs="*",
                       default=None,
                       help='The name list for a set of independent valid dataset according to --valid-data-path. '
                            'Follows the same pattern rules as --sft-dataset')

    group.add_argument('--sft-test-dataset',
                       nargs="*",
                       default=None,
                       help='The name list for a set of independent test dataset according to --test-data-path. '
                            'Follows the same pattern rules as --sft-dataset')

    group.add_argument('--sft-sort-batch',
                       action='store_true',
                       help='Sort the entire dataset from smallest to largest; '
                            'if the --packing-sft-data option is enabled, sort the data after packing. Default: False')

    group.add_argument('--sft-data-streaming',
                       action='store_true',
                       help="enable data streaming. Default: False")

    group.add_argument("--streaming-buffer-size",
                       type=int,
                       default=16384,
                       help="The size of the buffer to randomly sample examples from in dataset streaming")

    group.add_argument("--sft-data-mix-strategy",
                       type=str,
                       choices=["concat", "interleave_under", "interleave_over"],
                       default="concat",
                       help="The strategy to mix the sft data. Default: concat")

    group.add_argument('--sft-num-preprocess-workers',
                       type=int,
                       default=None,
                       help='The number of workers to use for data preprocessing. Only support non-streaming mode.')

    group.add_argument('--train-on-prompt',
                       action='store_true',
                       help='Whether compute loss on prompt. Default: False')

    group.add_argument("--is-tokenized-data",
                       action='store_true',
                       help="Whether the data is already tokenized. Default: False.")

    group.add_argument("--packing-sft-data",
                       action='store_true',
                       help="Whether to pack multiple sft data into one.")

    group.add_argument("--enable-discard-sample",
                       action='store_true',
                       help="Whether to discard sample when its length is greater than seq-length.")

    group.add_argument("--packing-batch-size",
                       type=int,
                       default=10000,
                       help="Perform packing in batches, deciding how many samples each batch contains;"
                            "if the --sft-sort-batch option is enabled, the samples will be sorted after packing.")

    return parser


def _add_extra_video_args(parser):
    group = parser.add_argument_group(title="extra-video")

    # use for stdit models
    group.add_argument('--latent-in-channels', type=int,
                       help='Number of channels in input latent data')

    group.add_argument('--latent-out-channels', type=int,
                       help='Number of channels in output latent data')

    group.add_argument('--caption-channels', type=int,
                       help='Number of channels in caption data')

    group.add_argument('--latent-patch-size', type=tuple, default=(1, 1, 1),
                       help='Patch size for vision task')

    group.add_argument('--latent-space-scale', type=float, default=1.0,
                       help='Space scale for vision task')

    group.add_argument('--latent-time-scale', type=float, default=1.0,
                       help='Time scale for vision task')

    group.add_argument('--num-latent-frames', type=int,
                       help='Number of frames in video')

    group.add_argument('--max-latent-height', type=int,
                       help='Maximum height of video')

    group.add_argument('--max-latent-width', type=int,
                       help='Maximum width of video')

    group.add_argument('--latent-frame-interval', type=int, default=1,
                       help='Interval between frames')

    group.add_argument('--max-text-length', type=int,
                       help='Maximum text length')

    group.add_argument('--stdit-bucket-config', type=str,
                       help='bucket config file')

    group.add_argument('--num-bucket-build-workers', type=int, default=1,
                       help='Number of workers to build bucket')

    return parser


def _add_extra_training_args(parser: argparse.ArgumentParser):
    """Add training arguments"""
    group = parser.add_argument_group(title='extra-training')

    group.add_argument('--training-phase', type=str,
                       default=constants.TrainingPhase.PRETRAIN,
                       choices=[constants.TrainingPhase.PRETRAIN, constants.TrainingPhase.SFT],
                       help='Which phase to train. Default: pretrain')

    group.add_argument('--no-detail-log',
                       action='store_false',
                       help='If set, the detail-log-interval will no longer take effect.',
                       dest='log_detail')

    group.add_argument('--detail-log-interval', type=int, default=20,
                       help='Report timing interval.'
                            ' detail-log-interval will only take effect when the'
                            ' timing-log-level is set to 0')

    group.add_argument('--variable-seq-lengths',
                       action='store_true',
                       help='DEPRECATED. This flag is ignored.'
                            'Support for variable sequence lengths across microbatches.')

    group.add_argument('--enable-ema', action='store_true',
                        help='enable Model EMA (Exponential Moving Average)'
                            ' to maintain moving averages of the trained parameters')

    group.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay rate')

    group.add_argument('--save-ema', type=str, default=None,
                       help='Output directory to save ema checkpoints to, default to ${args.save}/ema')

    group.add_argument('--load-ema', type=str, default=None,
                       help='Directory containing a ema checkpoint, default to ${args.load}/ema')
    
    group.add_argument('--ckpt-format', default='torch',
                       choices=['torch', 'torch_dist', 'zarr'],
                       help='Checkpoint format to use. Default: torch')
    group.add_argument("--length-sort-pool-size", type=int, default=0,
                    help=">0 启用局部长度池化排序；推荐为 batch_size 的 10~50 倍")
    group.add_argument("--length-sort-desc", action="store_true",
                    help="按长度从长到短")

    return parser


def _add_extra_multimodal_args(parser):
    """Add multimodal arguments"""
    # FIXME: Currently, multimodal implementation is based on cogvlm, and whether the newly added parameters
    # are universally applicable needs to be determined subsequently;

    group = parser.add_argument_group(title='extra-multimodal')
    group.add_argument('--language-model-type',
                       type=str,
                       default=None,
                       choices=get_support_model_archs(constants.LanguageModelFamilies.names()))

    group.add_argument('--trainable-modules', default=['all'], nargs='*',
                       help='choices: all, language_model, adapter, vision_model, '
                            'language_expert_linear, vision_expert_linear'),
    
    group.add_argument("--dataloader-save", type=str, default=None,
                       help="Energon dataloader state save path")

    group.add_argument("--packing-pretrain-data", action='store_true',
                       help="Whether to pack multiple pretrain data into one.")
    
    group.add_argument("--add-question-in-pretrain", action="store_true",
                       help="Whether add question in pretrain VQASample")

    # use for qwen2vl now
    group.add_argument('--image-resolution', type=int,
                       help='Resolution of image inputs')
    group.add_argument('--min-pixels', type=int, default=4 * 28 * 28,
                       help='Minimum image pixels')

    group.add_argument('--max-pixels', type=int, default=16384 * 28 * 28,
                       help='Maximum image pixels')

    group.add_argument('--frame-min-pixels', type=int, default=128 * 28 * 28,
                       help='Minimum frame pixels')

    group.add_argument('--frame-max-pixels', type=int, default=768 * 28 * 28,
                       help='Maximum frame pixels')

    group.add_argument('--video-max-pixels', type=int, default=65536 * 28 * 28,
                       help='Maximum video pixels')

    group.add_argument('--fps', type=float, default=2.0,
                       help='The fps to extract frames for model inputs')

    group.add_argument('--fps-min-frames', type=int, default=4,
                       help='The minimum number of frames of the video')

    group.add_argument('--fps-max-frames', type=int, default=768,
                       help='The maximum number of frames of the video')
    return parser


def _add_extra_parallel_args(parser):
    """Add parallel arguments"""
    group = parser.add_argument_group(title='extra-parallel')
    
    # NOTE：In order to be compatible with the old version of AIAK,
    # --context-parallel-ulysses-degree temporarily retained.
    group.add_argument('--context-parallel-ulysses-degree', type=int, default=1,
                       help='Degree of context parallelism in ulysses attention.')

    return parser


def _validate_extra_model_args(args):
    """Setup model config based on the given model name."""
    model_config = get_model_config(args.model_name)
    if model_config is not None:
        # the structural configuration of model will be overwritten, such as num_layers, hidden_states..
        print_rank_0(f'-------------- Configure model to {args.model_name} --------------', args.rank)

        for field in fields(model_config.__class__):
            assert hasattr(args, field.name), f"The model config field ({field.name}) is not defined in args."
            key, value = field.name, getattr(model_config, field.name)
            setattr(args, key, value)
            print_rank_0(f"  {key} = {value} ", args.rank)

        print_rank_0('---------------- End of configuration ----------------', args.rank)

    if args.enable_fa_within_mla:
        args.attention_backend = AttnBackend.flash
        print_rank_0(f"--enable-fa-within-mla is enabled, setting attention backend to FlashAttention", args.rank)


def _validate_extra_tokenizer_args(args):
    """Setup tokenizer based on the given model name."""
    if args.tokenizer_type is None:
        args.tokenizer_type = get_default_tokenizer(args.model_family)
        assert args.tokenizer_type is not None, \
            'No default tokenizer found for the given model name, please set --tokenizer-type'

        print_rank_0(f'Configure tokenizer to {args.tokenizer_type}', args.rank)

    if args.additional_special_tokens is not None:
        args.additional_special_tokens = [token.strip() for token in args.additional_special_tokens.split(',')]


def _validate_extra_sft_args(args):
    """Validate SFT arguments"""
    if args.training_phase != constants.TrainingPhase.SFT:
        return

    if args.tokenizer_type != 'HFTokenizer':
        raise ValueError('--tokenizer-type should be HFTokenizer when training phase is sft')

    args.dataloader_type = 'external'
    print_rank_0(f"INFO: Set dataloader type to external since --training-phase=SFT", args.rank)

    if args.chat_template is None:
        raise ValueError('--chat-template is required when training phase is sft')

    if args.sft_dataset_config is None:
        # set default sft-dataset-config
        default_config = get_default_sft_dataset_config()
        if default_config is not None:
            args.sft_dataset_config = default_config
            print_rank_0(f"WARNING: --sft-dataset-config is not specified, setup to default config ({default_config})",
                         args.rank)
        else:
            raise ValueError('--sft-dataset-config is not specified, and '
                             'the default config does not exist, please setup it')
    if args.sft_data_streaming:
        assert args.sft_sort_batch is None or not args.sft_sort_batch, \
            '--sft-sort-batch" cannot be used together with --sft-data-streaming'

    # Defaults to True but enforced as fixed-length for specific features (e.g., tp-comm-overlap/ moe allgather)
    args.variable_seq_lengths = True
    if args.tp_comm_overlap:
        # tp_comm_overlap requires fixed-length
        args.variable_seq_lengths = False

    if (args.num_experts is not None and
        args.num_experts > 0 and
        args.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']):
        # allgather or alltoall_seq requires fixed-length
        args.variable_seq_lengths = False

    if args.packing_sft_data:
        if args.micro_batch_size > 1:
            args.micro_batch_size = 1
            print_rank_0('WARING: Setting args.micro_batch_size to 1 since packing_sft_data is enabled', args.rank)

        if args.context_parallel_size > 1:
            if args.context_parallel_ulysses_degree < args.context_parallel_size and args.cp_comm_type == 'allgather':
                args.cp_comm_type = 'p2p'
                print_rank_0("WARNING: Setting args.cp_comm_type to p2p since ring attention "
                             "does not support all gather while packing_sft_data is enabled",
                             args.rank)
                
        # check if the model is supported
        if args.multi_latent_attention:
            if not args.enable_fa_within_mla:
                args.enable_fa_within_mla = True
                args.attention_backend = AttnBackend.flash
                print_rank_0('WARING: Setting args.enable_fa_within_mla to true since enable sft-packing with mla',
                             args.rank)

    if args.padding_side == "left":
        args.padding_side = "right"
        print_rank_0('WARING: Setting args.padding_side to right when run sft.', args.rank)


def _validate_extra_training_args(args):
    """Validate training arguments"""

    # check ema
    if args.enable_ema:
        assert args.model_family in [
            constants.VideoLanguageModelFamilies.STDIT,
            constants.VideoLanguageModelFamilies.STDIT3,
        ], f'EMA only supports STDIT models.'

        if args.load_ema is None and args.load is not None:
            args.load_ema = os.path.join(args.load, 'ema')

        if args.save_ema is None and args.save is not None:
            args.save_ema = os.path.join(args.save, 'ema')


def _validata_extra_multimodal_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.VisionLanguageModelFamilies.names():
        return

    for module in args.trainable_modules:
        assert module in \
            ['all', 'language_model', 'adapter', 'vision_model', 'language_expert_linear', 'vision_expert_linear']

    args.variable_seq_lengths = True
    if not (args.packing_pretrain_data or args.packing_sft_data):
        args.packing_batch_size = None


def _validata_extra_video_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.VideoLanguageModelFamilies.names():
        return

    # make text length divisible by cp size
    if args.max_text_length is not None and args.context_parallel_size > 1:
        while (args.max_text_length % args.context_parallel_size) != 0:
            args.max_text_length += 1


def _validata_extra_parallel_args(args):
    """Validate parallel arguments"""
    # check cp, NOTE: maybe removed in the future
    if args.context_parallel_size > 1:
        if args.context_parallel_ulysses_degree is None or args.context_parallel_ulysses_degree < 1:
            # not set
            return

        assert args.hierarchical_context_parallel_sizes is None, \
            "ERROR: Cannot specify both hierarchical_context_parallel_sizes and context_parallel_ulysses_degree"

        assert (
            args.context_parallel_ulysses_degree <= args.context_parallel_size and
            args.context_parallel_size % args.context_parallel_ulysses_degree == 0
        ), "ERROR: context_parallel_ulysses_degree must less than context_parallel_size and divisible by it"

        # only cp
        if args.context_parallel_ulysses_degree == 1:
            # just use cp
            assert 'a2a' not in args.cp_comm_type, "p2p or allgather are allowed for non-ulysses context parallel"
        # only ulysses
        elif args.context_parallel_ulysses_degree == args.context_parallel_size:
            # just use all2all
            args.cp_comm_type = 'a2a'
            print_rank_0('Setting cp_comm_type to a2a because context_parallel_ulysses_degree equals '
                         'to context_parallel_size', args.rank)
        else:
            cp_degree = args.context_parallel_size // args.context_parallel_ulysses_degree
            args.cp_comm_type = 'a2a+p2p'
            args.hierarchical_context_parallel_sizes = [args.context_parallel_ulysses_degree, cp_degree]

    # check tp overlap
    if args.tp_comm_overlap:
        if importlib.util.find_spec("torch_xmlir") is None and args.fp16:
            args.tp_comm_overlap = False
            print_rank_0('Disabling tp comm overlap since fp16 is not supported on GPU', args.rank)
