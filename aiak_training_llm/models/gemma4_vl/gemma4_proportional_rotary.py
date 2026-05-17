"""Gemma4 proportional RoPE — global-layer rotary with HF zero-padded inv_freq.

HF Gemma4 global (full_attention) layers use ``rope_type='proportional'`` with
``partial_rotary_factor=0.25`` and ``rope_theta=1e6``. This is **not** the same
as Megatron's stock ``rotary_percent`` — the two differ in two ways that
together push logits off by O(1e-3) on a real micro-batch:

  1. Denominator. HF proportional uses the **full head_dim** as the denominator
     when computing inv_freq exponents:

         inv_freq_rotated[k] = 1 / base^( 2*k / head_dim )       for k in [0, rope_angles)

     where ``rope_angles = int(rope_proportion * head_dim // 2)``. Megatron's
     ``RotaryEmbedding`` instead uses ``dim = int(kv_channels * rotary_percent)``
     as the denominator: ``inv_freq[k] = 1 / base^(2*k / dim)``. With
     ``head_dim=512`` and ``partial_rotary_factor=0.25``, HF divides by 512 while
     stock Megatron would divide by 128 — a 4× larger exponent — producing a
     completely different frequency spectrum.

  2. Padding vs. truncation. HF returns an inv_freq of length ``head_dim // 2``
     where the first ``rope_angles`` entries are non-zero and the remainder is
     zero-padded; the cat-(freqs, freqs) in the forward then produces a
     ``head_dim``-long cos/sin pair whose tail is identity (cos=1, sin=0).
     Megatron's stock partial-RoPE truncates instead: it returns inv_freq of
     length ``dim // 2`` and the cos/sin pair is ``dim``-long, with the
     attention head splitting Q/K into rotated + un-rotated sub-vectors. The
     attention math is **incompatible** between the two without code changes
     elsewhere; we match HF by zero-padding inside ``__init__``.

This subclass therefore reimplements ``__init__``'s inv_freq computation:
construct a length-``head_dim // 2`` buffer whose first ``rope_angles`` entries
are HF-equivalent and the rest are zero. The parent ``forward`` is left intact
because (a) ``cat((freqs, freqs), dim=-1)`` correctly emits a length-``head_dim``
cos/sin pair, and (b) zeros in inv_freq produce cos=1, sin=0 — the desired
identity rotation on the un-rotated tail.

For sliding (``rope_type='default'`` with no ``partial_rotary_factor``) we
**don't need this class** — the stock ``RotaryEmbedding(kv_channels=256,
rotary_percent=1.0, rotary_base=1e4)`` is bit-for-bit equivalent to HF.

References:
  - HF ``_compute_proportional_rope_parameters`` in
    ``transformers/src/transformers/modeling_rope_utils.py:187-254`` (commit c9de109).
  - HF ``Gemma4TextRotaryEmbedding`` in
    ``transformers/src/transformers/models/gemma4/modeling_gemma4.py:1046-1130``.
"""

from __future__ import annotations

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma4 global-layer RoPE with HF-equivalent zero-padded inv_freq.

    Args:
        head_dim: Full attention head dimension (e.g. 512 for Gemma4 global).
            This is the **denominator** in the HF inv_freq exponent and the
            length of the emitted cos/sin (= 2 * len(inv_freq)).
        partial_rotary_factor: Fraction of the head_dim to actually rotate
            (e.g. 0.25 for Gemma4 global → rotate first 64 angle pairs of 256).
        rotary_base: RoPE base / theta (e.g. 1_000_000 for Gemma4 global).
        rotary_interleaved: Forwarded to parent. Gemma4 uses False (cat layout).
        seq_len_interpolation_factor: Forwarded to parent.
        use_cpu_initialization: Forwarded to parent.
    """

    def __init__(
        self,
        head_dim: int,
        partial_rotary_factor: float,
        rotary_base: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        use_cpu_initialization: bool = False,
    ) -> None:
        # Parent __init__ computes self.inv_freq using kv_channels + rotary_percent.
        # We pass kv_channels=head_dim, rotary_percent=1.0 so the parent emits an
        # inv_freq of length head_dim//2 with the **correct denominator** (head_dim).
        # Then we zero out the tail to match HF proportional padding.
        super().__init__(
            kv_channels=head_dim,
            rotary_percent=1.0,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            use_cpu_initialization=use_cpu_initialization,
        )

        rope_angles = int(partial_rotary_factor * head_dim // 2)
        if rope_angles < 0 or rope_angles > head_dim // 2:
            raise ValueError(
                f"Gemma4ProportionalRotaryEmbedding: rope_angles={rope_angles} out of "
                f"range [0, {head_dim // 2}] for head_dim={head_dim}, "
                f"partial_rotary_factor={partial_rotary_factor}."
            )

        # Zero-pad the inv_freq tail so cos[..., rope_angles:]=1 and sin[..., rope_angles:]=0
        # in the parent forward. Use in-place fill on the registered buffer / parameter view.
        # self.inv_freq is a Tensor (set by parent as a plain attribute, not a buffer).
        if rope_angles < head_dim // 2:
            with torch.no_grad():
                self.inv_freq[rope_angles:].zero_()


__all__ = ["Gemma4ProportionalRotaryEmbedding"]
