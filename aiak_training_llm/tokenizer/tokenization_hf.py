"""auto tokenizer"""

from typing import Dict, List, Union, Optional

from transformers import AutoTokenizer

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


class AutoTokenizerFromHF(MegatronTokenizer):
    """
    这里我们采用封装 HF Tokenizer 并继承 MegatronTokenizer 的方式实现, 这种做法的目的：
    1、使用体验和用户使用 HF 一致，方便使用 HuggingFace Tokenizer 提供的功能，后续如果有需要，可继续扩展接口；
    2、可满足 MegatronTokenizer 的接口定义, 避免出现与 Megatron 接口不兼容问题;
    """
    def __init__(self,
                 name_or_path: str,
                 use_fast_tokenizer: bool,
                 padding_side: str,
                 model_max_length: int,
                 split_special_tokens: bool,
                 **kwargs,
    ):
        super().__init__(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=use_fast_tokenizer,
            padding_side=padding_side,
            split_special_tokens=split_special_tokens,
            model_max_length=model_max_length,
            trust_remote_code=True,
            **kwargs,
        )

    def tokenize(self, text: str, **kwargs) -> List[int]:
        """tokenize text
        
        Args:
            text (`str`): The text to be tokenized.
            **kwargs: Additional keyword arguments passed along to the `encode` method
        
        Returns:
            `List[int]`: The token ids of the text.
        
        """
        return self.tokenizer.encode(text, **kwargs)

    def detokenize(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """Convert embedding ids to text
        
        Args:
            token_ids (`int` or `List[int]`): One or several token id(s) to convert to text.
            skip_special_tokens (`bool`, *optional*, default to `False`): Whether to remove all special tokens from the
                output string.
            clean_up_tokenization_spaces (`bool`, *optional*, default to `None`): Whether to clean up the tokenization
                spaces before decoding.
            **kwargs: Additional keyword arguments passed along to the `decode` method
        
        Returns:
            `str`: The decoded text.
        """
        return self.tokenizer.decode(token_ids=token_ids,
                                     skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                                     **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def add_special_tokens(
        self,
        special_tokens_dict,
        replace_additional_special_tokens: bool = True,
    ) -> int:
        """add special tokens
        
        Returns:
            `int`: Number of tokens added to the vocabulary.
        """
        return self.tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens)

    @property
    def vocab(self) -> Dict[str, int]:
        """get vocab"""
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self) -> Dict[int, str]:
        """get inverse vocab"""
        return {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """The vocabulary size"""
        return len(self.tokenizer)

    @property
    def cls(self) -> Optional[int]:
        """The CLS token id"""
        return self.tokenizer.cls_token_id

    @property
    def sep(self) -> Optional[int]:
        """The SEP token id"""
        return self.tokenizer.sep_token_id

    @property
    def pad(self) -> Optional[int]:
        """The PAD token id"""
        return self.tokenizer.pad_token_id

    @property
    def eod(self) -> Optional[int]:
        """The EOD token id"""
        return self.eos

    @property
    def bos(self) -> Optional[int]:
        """The BOS token id"""
        return self.tokenizer.bos_token_id

    @property
    def eos(self) -> Optional[int]:
        """The EOS token id"""
        return self.tokenizer.eos_token_id

    @property
    def mask(self) -> Optional[int]:
        """The MASK token id"""
        return self.tokenizer.mask_token_id

    @cls.setter
    def cls(self, value):
        """set CLS token id"""
        self.tokenizer.cls_token_id = value

    @sep.setter
    def sep(self, value):
        """set sep token id"""
        self.tokenizer.sep_token_id = value

    @pad.setter
    def pad(self, value):
        """set PAD token id"""
        self.tokenizer.pad_token_id = value

    @bos.setter
    def bos(self, value):
        """set BOS token id"""
        self.tokenizer.bos_token_id = value
        
    @eos.setter
    def eos(self, value):
        """set EOS token id"""
        self.tokenizer.eos_token_id = value
        
    @mask.setter
    def mask(self, value):
        """set MASK token id"""
        self.tokenizer.mask_token_id = value

    @property
    def padding_side(self) -> str:
        """padding side"""
        return self.tokenizer.padding_side

    def hf_tokenizer(self):
        """return low level hf tokenizer"""
        return self.tokenizer