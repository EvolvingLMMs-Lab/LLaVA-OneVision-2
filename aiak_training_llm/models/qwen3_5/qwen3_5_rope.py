# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 Multimodal RoPE (M-RoPE) for AIAK-Megatron.

Ported from LoongForge ``Qwen3_5RotaryEmbedding`` (which extended mcore's
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
from megatron.core.models.common.embeddings.rope_utils import (
    _rotate_half,
    get_pos_emb_on_this_cp_rank,
)


# Cos/sin cache for Qwen3.5 LLM RoPE.
#
# Mirrors the ViT pattern (commit 1b9d69b). The freqs tensor built by
# Qwen3_5RotaryEmbedding.forward is the same Python object across every
# full_attention LLM layer in a single forward, and (q, k) at each layer both
# receive the same freqs, so the cos/sin computation in the unfused bshd/thd
# path is duplicated 2 * num_full_attention_layers times per step. Caching
# keyed by the (sliced) freqs identity lets every call after the first hit
# the cache.
#
# Only triggers when apply_rope_fusion=False (the unfused path). When fused,
# the upstream FusedRoPEFunc.apply -> tex.fused_rope_thd_forward / backward
# kernel handles cos/sin on the CUDA side and Python-side caching is moot.
#
# Bounded by clearing at the model-forward entry (qwen_model.py::forward)
# so a freed-then-reallocated freqs tensor (same data_ptr, different content
# via the CUDA caching allocator) cannot return stale cos/sin on the next
# step. Within a single forward we keep all per-segment entries (thd path
# can produce several) instead of single-entry evict-on-insert so segments
# other than the last one still hit on subsequent layer calls.
_LLM_ROPE_CACHE = {}


def _cached_apply_rotary_pos_emb_bshd(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool,
    multi_latent_attention: bool,
    mscale: float,
) -> torch.Tensor:
    """Cached cos/sin + apply rotation, mirroring mcore's
    ``_apply_rotary_pos_emb_bshd`` so parity is preserved exactly.
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t_rot[..., 0::2]
        x2 = t_rot[..., 1::2]
        t_rot = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    key = (freqs.data_ptr(), tuple(freqs.shape), t.dtype)
    cs = _LLM_ROPE_CACHE.get(key)
    if cs is None:
        cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
        sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
        _LLM_ROPE_CACHE[key] = (cos_, sin_)
    else:
        cos_, sin_ = cs

    t_rot = (t_rot * cos_) + (_rotate_half(t_rot, rotary_interleaved) * sin_)
    return torch.cat((t_rot, t_pass), dim=-1)


def apply_rotary_pos_emb_qwen3_5_llm(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config,
    cu_seqlens=None,
    mscale: float = 1.0,
):
    """Qwen3.5 LLM RoPE wrapper that caches the unfused-path cos/sin.

    Signature matches ``megatron.core.models.common.embeddings.rope_utils.
    apply_rotary_pos_emb`` so it can be plugged into the Qwen3.5 LLM layer
    spec as ``submodules.self_attention.submodules.apply_rotary_fn``.

    Behaviour:

    * ``config.apply_rope_fusion=True``: delegates to the mcore fused
      dispatcher (the CUDA-side FusedRoPEFunc handles cos/sin itself, so
      Python-side caching is moot and we don't shadow it).
    * ``config.apply_rope_fusion=False``: re-implements the unfused thd /
      bshd split (CP-aware per-segment slicing + rotation) with the
      cos/sin computation replaced by a cache lookup keyed on the
      per-segment freqs identity.

    For Qwen3.5 the typical config is ``rotary_interleaved=False``,
    ``multi_latent_attention=False``, ``mscale=1.0``; those flow through
    the same code path as mcore's bshd/thd reference.
    """
    if config.apply_rope_fusion:
        from megatron.core.models.common.embeddings.rope_utils import (
            apply_rotary_pos_emb as _mcore_apply_rotary_pos_emb,
        )
        return _mcore_apply_rotary_pos_emb(
            t, freqs, config, cu_seqlens=cu_seqlens, mscale=mscale
        )

    rotary_interleaved = getattr(config, "rotary_interleaved", False)
    multi_latent_attention = getattr(config, "multi_latent_attention", False)

    if cu_seqlens is None:
        return _cached_apply_rotary_pos_emb_bshd(
            t, freqs, rotary_interleaved, multi_latent_attention, mscale
        )

    # thd path: split t by cu_seqlens, slice freqs per segment (mirrors
    # mcore _apply_rotary_pos_emb_thd for the CP=1 case; for CP>1 we
    # replicate the two-stripe slice pattern from
    # _get_thd_freqs_on_this_cp_rank to avoid touching that private helper).
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens_local = cu_seqlens // cp_size
    seqlens = (cu_seqlens_local[1:] - cu_seqlens_local[:-1]).tolist()

    segments_out = []
    for x in torch.split(t, seqlens):
        if cp_size > 1:
            cp_seg = x.size(0) // 2
            full_seqlen = cp_size * x.size(0)
            sliced_freqs = torch.cat(
                [
                    freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                    freqs[
                        full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg
                    ],
                ]
            )
        else:
            sliced_freqs = freqs[: x.size(0)]
        out = _cached_apply_rotary_pos_emb_bshd(
            x.unsqueeze(1),
            sliced_freqs,
            rotary_interleaved,
            multi_latent_attention,
            mscale,
        ).squeeze(1)
        segments_out.append(out)
    return torch.cat(segments_out)


def clear_llm_rope_cache():
    """Clear the cos/sin cache. Called at the start of every LLM forward so
    a CUDA-allocator-reused freqs data_ptr cannot return stale cos/sin on
    the next step.
    """
    _LLM_ROPE_CACHE.clear()


class Qwen3_5RotaryEmbedding(torch.nn.Module):
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
