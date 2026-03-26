"""AIAK Tokenizer"""

from typing import TYPE_CHECKING, Optional

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

from megatron.training.tokenizer import build_tokenizer as build_megatron_tokenizer

from aiak_training_llm.utils import constants, print_rank_0

from .tokenization_hf import AutoTokenizerFromHF
from .special_tokens import ensure_multimodal_special_tokens, MM_SPECIAL_TOKENS


if TYPE_CHECKING:
    from aiak_training_llm.data import ChatTemplate


def _update_tokenizer_with_template(args, tokenizer: AutoTokenizerFromHF, chat_template: "ChatTemplate"):
    """Update tokenizer with chat template."""
    stop_words = chat_template.stop_words
    if chat_template.replace_eos:
        assert stop_words, "EOS replacement requires stop words."

        _eos_token = tokenizer.hf_tokenizer().eos_token
        num_added = tokenizer.add_special_tokens({"eos_token": stop_words[0]})
        if _eos_token is None:
            print_rank_0(f"WARNING: tokenizer does not have an EOS token, setting to {stop_words[0]}, "
                         f"and will add {num_added} new tokens to tokenizer.", args.rank)
        else:
            print_rank_0(f"WARNING: tokenizer already has an EOS token, replace {_eos_token} with "
                         f"{stop_words[0]}, and will add {num_added} new tokens to tokenizer.", args.rank)

        stop_words = stop_words[1:]

    if tokenizer.eos is None:
        # set default eos token
        num_added = tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        print_rank_0(f"WARNING: tokenizer does not have an EOS token, setting to <|endoftext|>,"
                     f" and will add {num_added} new tokens to tokenizer.", args.rank)

    if tokenizer.pad is None:
        tokenizer.pad = tokenizer.eos
        print_rank_0(f"WARNING: tokenizer does not have a pad token, setting to eos token"
                     f"({tokenizer.hf_tokenizer().pad_token}) (token id: {tokenizer.pad}).", args.rank)

    if stop_words:
        num_added = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        print_rank_0(f"WARNING: add stop tokens ({','.join(stop_words)}) to tokenizer, "
                     f" and will add {num_added} new tokens", args.rank)


def build_tokenizer(args, chat_template: Optional["ChatTemplate"] = None) -> Optional[MegatronTokenizer]:
    """Build tokenizer and chat template if needed."""
    if args.tokenizer_type == 'HFTokenizer':
        print_rank_0(f'> AIAK building {args.tokenizer_type} tokenizer ...', args.rank)

        assert args.hf_tokenizer_path is not None, "HFTokenizer requires a tokenizer name or path."

        tokenizer = AutoTokenizerFromHF(name_or_path=args.hf_tokenizer_path,
                                        use_fast_tokenizer=args.use_fast_tokenizer,
                                        padding_side=args.padding_side,
                                        model_max_length=args.seq_length,
                                        split_special_tokens=args.split_special_tokens)

        if (
            args.model_family in constants.VisionLanguageModelFamilies.names()
            or args.model_family in constants.VideoLanguageModelFamilies.names()
        ):
            added_mm_tokens = ensure_multimodal_special_tokens(tokenizer.hf_tokenizer())
            print_rank_0(
                f"INFO: ensured multimodal special tokens {MM_SPECIAL_TOKENS}; added={added_mm_tokens}",
                args.rank,
            )
            current_tokenizer_vocab_size = tokenizer.vocab_size
            if getattr(args, "vocab_size_in_config_file", None) is not None and \
               current_tokenizer_vocab_size > args.vocab_size_in_config_file:
                print_rank_0(
                    f"WARNING: tokenizer vocab size increased from config size "
                    f"{args.vocab_size_in_config_file} to {current_tokenizer_vocab_size}; "
                    f"updating args.vocab_size_in_config_file to avoid embedding index OOB.",
                    args.rank,
                )
                args.vocab_size_in_config_file = current_tokenizer_vocab_size

        if args.additional_special_tokens is not None:
            added_tokens = tokenizer.add_special_tokens(
                dict(additional_special_tokens=args.additional_special_tokens),
                replace_additional_special_tokens=False,
            )
            print_rank_0(f"INFO: Added {added_tokens} additional special tokens, "
                         f"include {args.additional_special_tokens}.", args.rank)

        if args.training_phase == constants.TrainingPhase.PRETRAIN:
            if tokenizer.eos is None:
                if args.model_family == constants.LanguageModelFamilies.QWEN:
                    # tokenizer.eos = tokenizer.tokenizer.eod_id
                    tokenizer.eos = tokenizer.eod_id

        elif chat_template is not None:
            # update the tokenizer with chat template
            _update_tokenizer_with_template(args, tokenizer, chat_template)

    else:
        # megatron tokenizer already handles padding.
        tokenizer = build_megatron_tokenizer(args)
        return tokenizer

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        ori_vocab_size = tokenizer.vocab_size
        if getattr(args, "vocab_size_in_config_file", None) is not None:
            ori_vocab_size = getattr(args, "vocab_size_in_config_file", None)
        args.padded_vocab_size = _vocab_size_with_padding(ori_vocab_size, args)

    return tokenizer
