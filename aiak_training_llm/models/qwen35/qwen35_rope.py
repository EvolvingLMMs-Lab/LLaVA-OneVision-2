# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 Multimodal RoPE (M-RoPE) for AIAK-Megatron.

Ported from LoongForge ``Qwen35RotaryEmbedding`` (which extended mcore's
``MultimodalRotaryEmbedding``). Our mcore 0.12.0rc0 has no
``MultimodalRotaryEmbedding``, so this is implemented as a standalone
``torch.nn.Module`` modelled on our existing Qwen2-VL M-RoPE
(``aiak_training_llm/models/qwen_vl/qwen2_vl_model.py::Qwen2VLRotaryEmbedding``).

Qwen3.5 specifics added on top of the Qwen2-VL 3D-rope base:
  * ``mrope_interleaved=True`` : interleaved section application
    (``apply_interleaved_mrope``), reorganising the chunked [TTT..HHH..WWW]
    frequency layout into interleaved [THTHWH..TT].
  * ``partial_rotary_factor=0.25`` : only the first 25% of ``head_dim`` gets rope.
"""

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen35RotaryEmbedding(torch.nn.Module):
    """Qwen3.5 Rotary Position Embedding (interleaved M-RoPE, partial rotary)."""

    def __init__(self, config):
        """Initialize Qwen3.5 rotary position embedding.

        Args:
            config: model configuration. Reads (via getattr with defaults):
                kv_channels / head_dim, rotary_base, mrope_section,
                mrope_interleaved, partial_rotary_factor,
                rotary_seq_len_interpolation_factor.
        """
        super().__init__()
        self.config = config

        # head_dim: prefer explicit head_dim, else kv_channels, else hidden/heads.
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "kv_channels", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim

        # ADAPTATION: partial rotary -- only first 25% of head_dim gets rope.
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.25)
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        # rotary_dim must be even for the half-rotation.
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = rotary_dim

        self.rotary_base = getattr(config, "rotary_base", 1000000)
        self.seq_len_interpolation_factor = getattr(
            config, "rotary_seq_len_interpolation_factor", None
        )

        self.mrope_section = getattr(config, "mrope_section", [16, 24, 24])
        self.mrope_interleaved = getattr(config, "mrope_interleaved", True)
        assert self.mrope_interleaved, "For Qwen3.5, mrope_interleaved should be True."

        inv_freq = 1.0 / (
            self.rotary_base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).float() / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.

        Args:
            freqs: (3, bs, seq_len, rotary_dim // 2)
            mrope_section: (3,)

        Returns:
            freqs_t: (bs, seq_len, rotary_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, position_ids, packed_seq=False):
        """Forward pass of Qwen3.5 rotary position embedding.

        Args:
            position_ids: Tensor of shape (3, bs, seq_len) with T/H/W position ids.
            packed_seq: Whether the input is in packed (thd) format.

        Returns:
            Tensor containing the rotary position embedding, shape
            (seq_len, bs, 1, 2 * rotary_dim).
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq = seq * (1 / self.seq_len_interpolation_factor)

        # shape (3, bs, dim, 1)
        # .float() both operands (inv_freq buffer may be cast to bf16 by the fp16/bf16
        # module wrapper during training); matches Qwen2-VL Qwen2VLRotaryEmbedding.
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)

        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()

        if parallel_state.get_context_parallel_world_size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb
