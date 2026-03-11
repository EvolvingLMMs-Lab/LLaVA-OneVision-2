"""Shared multimodal special-token helpers."""

from typing import List, Optional

MM_SPECIAL_TOKENS: List[str] = [
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def ensure_multimodal_special_tokens(hf_tokenizer) -> int:
    """Ensure multimodal tokens exist in tokenizer; returns number of newly added tokens."""
    missing = [token for token in MM_SPECIAL_TOKENS if hf_tokenizer.convert_tokens_to_ids(token) is None]
    if not missing:
        return 0
    return hf_tokenizer.add_special_tokens(
        {"additional_special_tokens": missing},
        replace_additional_special_tokens=False,
    )


def get_mm_token_id(hf_tokenizer, token: str) -> Optional[int]:
    """Resolve token id robustly from convert() then vocab fallback."""
    token_id = hf_tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        vocab = hf_tokenizer.get_vocab()
        token_id = vocab.get(token)
    return token_id
