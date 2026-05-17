"""Gemma4-specific transformer layer subclass.

Wraps Megatron's :class:`TransformerLayer` to attach two Gemma4-only mechanisms
that have no slot in the upstream layer:

1. ``post_feedforward_layernorm`` (with weight) — applied to the merged
   ``dense + experts`` output **before** the residual add. In HF
   (``Gemma4TextDecoderLayer.forward``) this lives between the merged sum
   and the residual add::

       hidden_states = self.post_feedforward_layernorm(hidden_states)
       hidden_states = residual + hidden_states

   Megatron's stock layer goes ``mlp -> mlp_bda(mlp_out, residual)`` with no
   hookable position in between, so we override ``forward`` and inject the
   norm by wrapping ``self.mlp(...)`` instead of patching ``mlp_bda``.

2. ``layer_scalar`` — non-trainable buffer of shape ``(1,)`` initialised to
   ``1.0`` and multiplied into the layer output at the very end. Loaded
   from the HF checkpoint (per-layer distinct scalars after pretraining).
   Gated by ``config.use_layer_scalar`` so the buffer is not registered when
   the flag is off.

Both mechanisms are no-ops when their gates are off, which is required so
this subclass can be used by any Gemma4-VL spec without hard-coding Gemma4
into upstream Megatron.
"""

from dataclasses import dataclass
from typing import Union

import torch
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)


@dataclass
class Gemma4TransformerLayerSubmodules(TransformerLayerSubmodules):
    """Adds the ``post_feedforward_layernorm`` slot used by Gemma4.

    Default ``None`` means "no post-FFN norm"; ``Gemma4TransformerLayer`` only
    wraps ``self.mlp`` when this slot is populated, so other consumers of
    :class:`TransformerLayerSubmodules` are unaffected.
    """

    post_feedforward_layernorm: Union[ModuleSpec, type, None] = None


class Gemma4TransformerLayer(TransformerLayer):
    """``TransformerLayer`` plus Gemma4's post-merge norm and ``layer_scalar``.

    The base ``TransformerLayer.forward`` calls ``self.mlp(...)`` then
    ``mlp_bda(mlp_out, residual)`` (see ``transformer_layer.py:550-562``).
    To inject ``post_feedforward_layernorm`` between them without forking the
    full forward, we monkey-patch ``self.mlp`` at construction time so that
    its ``__call__`` returns ``(post_ffn_norm(mlp_out), bias)``. The base
    forward then sees the post-normed tensor and feeds it to ``mlp_bda``,
    which performs the residual add — exactly the HF order.

    ``layer_scalar`` is post-multiplied into the layer's final output and
    handles both the single-tensor and ``(output, context)`` return shapes
    of the base forward.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
        )

        post_ffn_spec = getattr(submodules, "post_feedforward_layernorm", None)
        if post_ffn_spec is not None:
            self.post_feedforward_layernorm = build_module(
                post_ffn_spec,
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )
            self._wrap_mlp_with_post_ffn_norm()

        if getattr(self.config, "use_layer_scalar", False):
            self.register_buffer("layer_scalar", torch.ones(1))

    def _wrap_mlp_with_post_ffn_norm(self):
        original_mlp = self.mlp
        post_ffn = self.post_feedforward_layernorm

        class _MlpWithPostFfnNorm(torch.nn.Module):
            def __init__(inner_self):
                super().__init__()
                inner_self._inner_mlp = original_mlp
                inner_self._post_ffn = post_ffn

            def __getattr__(inner_self, name):
                if name in ("_inner_mlp", "_post_ffn"):
                    return super().__getattr__(name)
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(inner_self._inner_mlp, name)

            def forward(inner_self, hidden_states, *args, **kwargs):
                output, bias = inner_self._inner_mlp(hidden_states, *args, **kwargs)
                output = inner_self._post_ffn(output)
                return output, bias

        self.mlp = _MlpWithPostFfnNorm()

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if not hasattr(self, "layer_scalar"):
            return out
        if isinstance(out, tuple):
            return (out[0] * self.layer_scalar, *out[1:])
        return out * self.layer_scalar
