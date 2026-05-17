"""Gemma4-VL router (real subclass of Megatron's :class:`Router`).

Faithfully reproduces HF ``Gemma4TextRouter`` semantics inside Megatron's MoE
plumbing. The HF router is a self-contained module — no algebraic fusion with
any neighbouring norm is possible (see plan §0 point 2 / §1.4 / Change Log v4
#1) — so we reimplement it verbatim here and override the ``Router`` interface
to return the ``(scores, routing_map)`` tuple Megatron's token dispatcher
expects.

HF reference (``transformers/models/gemma4/modeling_gemma4.py``,
transformers==5.7.0)::

    class Gemma4TextRouter(nn.Module):
        def __init__(self, config):
            self.norm = Gemma4RMSNorm(hidden_size, eps, with_scale=False)
            self.proj = nn.Linear(hidden_size, num_experts, bias=False)
            self.scale = nn.Parameter(torch.ones(hidden_size))
            self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

        def forward(self, hidden_states):
            h = self.norm(hidden_states)
            h = h * self.scale * (hidden_size ** -0.5)
            scores = self.proj(h)
            probs = F.softmax(scores, dim=-1)
            top_w, top_i = torch.topk(probs, k=top_k_experts, dim=-1)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
            top_w = top_w * self.per_expert_scale[top_i]
            return probs, top_w, top_i

Notes on the Megatron mapping:

* Megatron's :class:`Router` base class allocates ``self.weight`` of shape
  ``[num_experts, hidden_size]``. We reuse that buffer as HF's
  ``proj.weight`` (same shape, same role) so the conversion script can copy
  it verbatim and we avoid duplicating a Parameter.
* ``self.scale`` and ``self.per_expert_scale`` are extra Parameters added on
  top of the base class.
* ``self.norm`` is the HF ``Gemma4RMSNorm(with_scale=False)`` — a parameter-
  free RMSNorm with the eps **inside** the sqrt (``1/sqrt(mean(x^2)+eps)``),
  which differs from Megatron's default ``MegatronRMSNorm``
  (``1/sqrt(mean(x^2))+eps``). We implement it inline as a private nn.Module
  so the numerics match HF byte-for-byte and there is no learnable weight to
  convert.
* The :meth:`forward` override returns ``(scores, routing_map)`` where:

  - ``scores`` is a dense ``[num_tokens, num_experts]`` tensor whose
    non-zero entries equal HF's ``top_w`` (so the dispatcher's
    ``out = scores * dispatched_input`` reproduces HF's gating exactly);
  - ``routing_map`` is a dense ``[num_tokens, num_experts]`` bool mask
    with ``top_k_experts`` True per row.

  The base class's ``gating()`` / ``routing()`` plumbing is bypassed entirely
  because (a) we need a custom pre-projection norm + scalar scale, and
  (b) HF's per-expert-scale post-multiplication has no equivalent in the
  ``moe_router_topk_scaling_factor`` scalar slot.

Conversion (P2): direct per-Parameter copy ``HF -> MG``::

    proj.weight       -> router.weight              [num_experts, hidden]
    scale             -> router.scale               [hidden]
    per_expert_scale  -> router.per_expert_scale    [num_experts]
"""

from __future__ import annotations

import torch
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_config import TransformerConfig


class _Gemma4ParameterFreeRMSNorm(torch.nn.Module):
    """HF ``Gemma4RMSNorm(with_scale=False)`` — parameter-free, eps-in-sqrt.

    Computes ``x * pow(mean(x^2) + eps, -0.5)`` in float32 and casts back to
    the input dtype. Matches HF byte-for-byte; intentionally different from
    Megatron's ``MegatronRMSNorm`` (which puts eps outside the sqrt).
    """

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        x = hidden_states.float()
        mean_sq = x.pow(2).mean(-1, keepdim=True) + self.eps
        out = x * torch.pow(mean_sq, -0.5)
        return out.to(in_dtype)


class Gemma4Router(Router):
    """Self-contained router matching HF ``Gemma4TextRouter`` semantics.

    Inherits Megatron's :class:`Router` so token dispatchers can consume the
    standard ``(scores, routing_map)`` tuple, but completely overrides the
    base ``forward`` (and does not use the base ``gating`` / ``routing``
    helpers).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        # Base class already created self.weight: [num_experts, hidden_size]
        # — we reuse it as HF's proj.weight. Sanity-check the shape.
        assert self.weight.shape == (config.num_moe_experts, config.hidden_size), (
            f"unexpected router weight shape {tuple(self.weight.shape)}; "
            f"expected ({config.num_moe_experts}, {config.hidden_size})"
        )

        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size ** -0.5
        self.topk = config.moe_router_topk

        # HF: Gemma4RMSNorm(hidden_size, eps=rms_norm_eps, with_scale=False)
        # `layernorm_epsilon` is Megatron's RMSNorm eps field name.
        self.norm = _Gemma4ParameterFreeRMSNorm(eps=config.layernorm_epsilon)

        # HF: scale = nn.Parameter(torch.ones(hidden_size))
        self.scale = torch.nn.Parameter(
            torch.ones(self.hidden_size, dtype=config.params_dtype)
        )
        setattr(self.scale, "sequence_parallel", config.sequence_parallel)

        # HF: per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.per_expert_scale = torch.nn.Parameter(
            torch.ones(self.num_experts, dtype=config.params_dtype)
        )
        setattr(self.per_expert_scale, "sequence_parallel", config.sequence_parallel)

    # The abstract ``routing`` is required by the base class but unused
    # (forward is fully overridden). Implement as a no-op assert so accidental
    # callers fail loudly instead of silently misbehaving.
    def routing(self, logits: torch.Tensor, ori_dtype=None):  # type: ignore[override]
        raise NotImplementedError(
            "Gemma4Router overrides forward() entirely; routing() is unused."
        )

    def forward(self, input: torch.Tensor):  # noqa: A002 - mirror base name
        """Compute HF-equivalent gating and emit Megatron-shaped outputs.

        Args:
            input: ``[seq_len, batch, hidden_size]`` activations from the
                post-attention residual (raw — *not* pre-FFN-normed; see plan
                §1.4 "self-contained router" bullet).

        Returns:
            (scores, routing_map):
                - ``scores: [num_tokens, num_experts]`` dense, non-zero only
                  at top-k positions, equal to HF's ``top_w`` after the
                  ``per_expert_scale`` multiplication.
                - ``routing_map: [num_tokens, num_experts]`` bool, True at
                  top-k positions.
        """
        # Move weights to GPU on first call (mirror base class behaviour).
        if self.weight.device.type == "cpu":
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())

        # HF: norm -> *scale -> *scalar_root_size -> linear(proj.weight)
        h = self.norm(input)
        h = h * self.scale * self.scalar_root_size
        # self.weight: [E, H], h: [..., H] -> logits: [..., E]
        logits = torch.nn.functional.linear(h, self.weight)

        # Flatten leading dims for the dispatcher contract.
        logits = logits.view(-1, self.num_experts)

        # HF: softmax then topk (post-softmax topk).
        probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_i = torch.topk(probs, k=self.topk, dim=-1)
        # Re-normalize so top-k weights sum to 1 per token.
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        # Apply per-expert scale (cast index gather to float32 to match probs).
        top_w = top_w * self.per_expert_scale.float()[top_i]
        top_w = top_w.to(input.dtype)

        # Build dense (scores, routing_map) of shape [num_tokens, num_experts].
        num_tokens = logits.shape[0]
        scores = torch.zeros(
            num_tokens, self.num_experts, dtype=top_w.dtype, device=logits.device
        )
        scores.scatter_(1, top_i, top_w)
        routing_map = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=logits.device
        )
        routing_map.scatter_(1, top_i, True)

        return scores, routing_map


__all__ = ["Gemma4Router"]
