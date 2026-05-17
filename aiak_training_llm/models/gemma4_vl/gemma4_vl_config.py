"""Configuration dataclasses and registrations for Gemma4-VL family.

Mirrors the structure of ``aiak_training_llm.models.llava_onevision2.llava_onevision2_config``
and adds the Gemma4-specific knobs:

- ``final_logit_softcapping`` (final logits tanh-cap, gated)
- ``use_layer_scalar`` (per-layer scalar buffer multiplied at end of layer forward)
- ``per_layer_kv_channels`` / ``layer_pattern`` (sliding vs global hybrid attention)
- ``attention_k_eq_v`` (Gemma4 K=V tying flag — combined with ``layer_pattern[i]=='global'``
  this drives the no-V-projection runtime aliasing on global layers)
- ``kv_tied_layers`` (precomputed list of global layer indices for fast runtime lookup)
- ``scale_emb_by_sqrt_hidden`` (Gemma family: x = embed(ids) * sqrt(hidden))
- ``sliding_window`` (sliding-attention window size)

Note: this checkpoint has dual RoPE — sliding layers use ``rope_type='default'``
with ``rope_theta=10000`` and full rotation; global layers use HF
``rope_type='proportional'`` with ``rope_theta=1_000_000``,
``partial_rotary_factor=0.25``, and the proportional zero-pad scheme over
``global_head_dim=512`` (rotate the first 64 angle pairs of 256, leave the
remaining 192 as identity). The sliding/global split lives in
``layer_pattern``; ``Gemma4Model`` constructs two ``RotaryEmbedding`` instances
and ships them as a dict keyed by layer-type to ``Gemma4SelfAttention``.

These fields land on ``TransformerConfig`` via the matching aiak_megatron patch
(P1.13A) so ``core_transformer_config_from_args`` can pump them straight from
the parsed args.
"""

from dataclasses import dataclass, field

import torch
from torch.nn.functional import gelu

from aiak_training_llm.models.factory import register_model_config
from aiak_training_llm.utils.constants import VisionLanguageModelFamilies


@dataclass
class AdapterConfig:
    """Configuration for the Gemma4-VL projector.

    The fields need to be consistent with the definitions in args.
    """

    normalization: str
    activation_func: torch.nn.Module = gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06


@dataclass
class Gemma4VLConfig:
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    group_query_attention: bool = True
    num_query_groups: int = 8
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = False
    vocab_size_in_config_file: int = 262144
    make_vocab_size_divisible_by: int = 128
    norm_epsilon: float = 1e-06
    rotary_base: int = 1000000
    kv_channels: int = 256
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    # ---- Gemma4-specific extensions (mirrored on TransformerConfig via P1.13A) ----
    final_logit_softcapping: float = None
    use_layer_scalar: bool = False
    sliding_window: int = None
    rotary_base_sliding: int = None
    partial_rotary_factor: float = 1.0
    # Per-layer hybrid attention overrides. ``layer_pattern`` is a list of
    # ``"sliding"`` / ``"global"`` strings of length ``num_layers``;
    # ``per_layer_kv_channels`` and ``per_layer_num_query_groups`` are dicts
    # keyed by layer-type ("sliding"/"global") -> int.
    layer_pattern: list = field(default_factory=list)
    per_layer_kv_channels: dict = field(default_factory=dict)
    per_layer_num_query_groups: dict = field(default_factory=dict)
    # K=V tying configuration. ``attention_k_eq_v=True`` enables HF's "no V projection,
    # alias V from K" mechanism on layers listed in ``kv_tied_layers`` (= global layers
    # under HF's ``use_alternative_attention = attention_k_eq_v AND not is_sliding``).
    attention_k_eq_v: bool = False
    kv_tied_layers: list = field(default_factory=list)
    scale_emb_by_sqrt_hidden: bool = False


@register_model_config(
    model_family=VisionLanguageModelFamilies.GEMMA4_VL,
    model_arch="gemma4-26b-a4b-vl",
)
def gemma4_26b_a4b_vl():
    """Gemma4-26B-A4B-it (instruction-tuned MoE VLM).

    Architecture (verified from /ov2/pretrain_models/google/gemma-4-26B-A4B-it/config.json):
      - 30 layers with repeating ``[s,s,s,s,s,g]`` pattern (5 sliding + 1 global)
      - hidden=2816, num_attention_heads=16
      - dense intermediate=2112, moe intermediate=704 (per routed expert)
      - 128 routed experts top-k=8; dense MLP runs as a parallel branch from the
        same post-attention residual (NOT a Megatron "shared expert" — see
        Gemma4ParallelDenseMoE / plan v5 §531 R5)
      - sliding layers: head_dim=256, num_kv=8,  RoPE theta=10000,  window=1024
      - global  layers: head_dim=512, num_kv=2,  RoPE theta=1000000, K=V tied (no V proj)
      - full RoPE (no partial), qk_layernorm, parameter-free V LayerNorm
      - final logit softcap 30.0, per-layer scalar buffer multiplied at layer end
      - vocab=262144, ctx_len=262144
      - tie_word_embeddings=True
    """
    layer_pattern = []
    for i in range(30):
        layer_pattern.append("global" if (i + 1) % 6 == 0 else "sliding")
    kv_tied_layers = [i for i, t in enumerate(layer_pattern) if t == "global"]
    return Gemma4VLConfig(
        num_layers=30,
        hidden_size=2816,
        ffn_hidden_size=2112,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=262144,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=256,
        add_qkv_bias=False,
        rotary_base=1000000,
        rotary_base_sliding=10000,
        partial_rotary_factor=0.25,
        sliding_window=1024,
        num_experts=128,
        moe_ffn_hidden_size=704,
        final_logit_softcapping=30.0,
        use_layer_scalar=True,
        scale_emb_by_sqrt_hidden=True,
        untie_embeddings_and_output_weights=False,
        layer_pattern=layer_pattern,
        per_layer_kv_channels={"sliding": 256, "global": 512},
        per_layer_num_query_groups={"sliding": 8, "global": 2},
        attention_k_eq_v=True,
        kv_tied_layers=kv_tied_layers,
    )


@dataclass
class VisionConfig:
    """Configuration for the Gemma4-VL vision tower.

    Verbatim mirror of HF ``Gemma4VisionConfig`` for ``gemma-4-26B-A4B-it``
    (see ``/ov2/pretrain_models/google/gemma-4-26B-A4B-it/config.json`` →
    ``vision_config``). This is NOT a SigLIP tower — Gemma4 uses its own ViT
    with: learned 2-D position embeddings via one-hot + matmul against a
    ``[2, position_embedding_size, hidden_size]`` table; sandwich RMSNorm per
    layer; v_proj followed by a parameterless RMSNorm (``with_scale=False``);
    multidimensional RoPE applied per-axis on the head_dim/2 split; pooling via
    a dynamic-kernel 2-D average pool whose kernel is derived from
    ``pooling_kernel_size`` and the input sequence length; and an optional
    output standardization stage gated by ``standardize``.

    Field semantics that DIFFER from OneVision-Encoder / SigLIP and MUST NOT
    be silently aliased to the OV2 defaults:

    - ``patch_size=16`` (NOT 14) and ``in_channels=3`` give patch_embed input
      ``in_features = 3 * 16 * 16 = 768``.
    - ``num_attention_heads = num_key_value_heads = 16`` → MHA (not GQA);
      ``head_dim=72`` → ``Q/K/V`` projection out_features ``= 16 * 72 = 1152``.
    - ``hidden_activation='gelu_pytorch_tanh'`` is the *only* activation that
      reproduces HF — exact ``F.gelu`` (erf form) silently diverges.
    - ``use_clipped_linears=False`` for this checkpoint; the buffer branch in
      :class:`Gemma4ClippableLinear` stays inactive.
    - ``standardize=True`` → :class:`Gemma4VisionTower` registers persistent
      ``std_bias``/``std_scale`` buffers and applies them after pooling.
    - ``rope_theta=100.0`` (not 10000) for the vision multidimensional RoPE.
    - ``position_embedding_size=10240`` is the per-axis position table length;
      ``pooling_kernel_size=3`` controls the *output* sequence length via
      ``pixel_values.shape[-2] // (pooling_kernel_size ** 2)``.
    """

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    patch_size: int
    pooling_kernel_size: int
    position_embedding_size: int
    rms_norm_eps: float
    rope_theta: float
    standardize: bool
    use_clipped_linears: bool
    hidden_activation: str
    in_channels: int = 3
    initializer_range: float = 0.02


def get_vision_config(model_family, model_name):
    del model_family, model_name
    return VisionConfig(
        num_hidden_layers=27,
        hidden_size=1152,
        intermediate_size=4304,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=72,
        patch_size=16,
        pooling_kernel_size=3,
        position_embedding_size=10240,
        rms_norm_eps=1e-06,
        rope_theta=100.0,
        standardize=True,
        use_clipped_linears=False,
        hidden_activation="gelu_pytorch_tanh",
        in_channels=3,
        initializer_range=0.02,
    )


def get_adapeter_config(model_family, model_name=None):
    """Adapter config for Gemma4-VL: parameterless RMSNorm + bias-free Linear.

    The Gemma4 adapter is intentionally minimal — a single trainable tensor
    (``embedding_projection.weight``) plus a parameterless input RMSNorm
    (``with_scale=False``, no ``weight``). Mirrors HF ``Gemma4MultimodalEmbedder``
    (lines 2023-2047 of ``modeling_gemma4``); the LLM-side hidden_size used for
    the projection out_features is derived at construction time from the LLM
    config rather than carried here.
    """
    del model_family, model_name
    return AdapterConfig(
        normalization="RMSNorm",
        add_bias_linear=False,
        layernorm_epsilon=1e-06,
    )
