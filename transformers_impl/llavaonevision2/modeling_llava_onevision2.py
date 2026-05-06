from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    replace_return_docstrings,
)
from transformers.utils.generic import is_flash_attention_requested

from .configuration_llava_onevision2 import LlavaOnevision2Config, LlavaOnevision2VisionConfig


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava-Onevision-1.5 outputs, with hidden states and attentions.
    """
)
class LlavaOnevision2ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava-Onevision-1.5 causal language model (or autoregressive) outputs.
    """
)
class LlavaOnevision2CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


# ---------------------------------------------------------------------------
# Vision Rotary Embedding
# ---------------------------------------------------------------------------


class VisionRotaryEmbedding(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    Supports both grid_thw-based and explicit position-based RoPE computation.
    """

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half
        self.base = base

        # 4:6:6 split for T:H:W
        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from grid_thw (Qwen2VL style).

        Args:
            grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample

        Returns:
            freqs: [total_seq_len, half] tensor of position frequencies
        """
        device = grid_thw.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        all_freqs = []
        for sample_thw in grid_thw:
            t, h, w = sample_thw[0].item(), sample_thw[1].item(), sample_thw[2].item()

            # Compute frequency tables
            ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
            fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
            fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

            # Build position indices for this sample
            t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

            # Concatenate frequencies: [seq_len, half]
            sample_freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
            all_freqs.append(sample_freqs)

        return torch.cat(all_freqs, dim=0)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [seq_len, 3] tensor with [t, h, w] positions for each patch

        Returns:
            freqs: [seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)

    def forward_with_thw(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit t, h, w dimensions.

        Args:
            t: Number of temporal frames
            h: Number of height patches
            w: Number of width patches
            device: Target device

        Returns:
            freqs: [t*h*w, half] tensor of position frequencies
        """
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class OneVisionEncoderEmbeddings(nn.Module):
    """
    Patch embedding layer that converts pre-processed patches to embeddings.

    This module is designed to receive patches that have already been extracted
    and arranged by the Qwen2VL image processor in 3x3 block spatial order.

    Input format: [total_patches, num_channels, patch_size, patch_size]
    Output format: [total_patches, embed_dim]
    """

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.patch_embedding(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class LlavaOnevision2VisionPatchMerger(nn.Module):
    """
    Patch merger that merges spatial_merge_size x spatial_merge_size patches into one.

    This module is designed to work with Qwen2VL-style patch processing where patches
    are already arranged in 3x3 block order by the image processor.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 3,
        layer_norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        self.spatial_merge_size = spatial_merge_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)
        return x


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)
    # Keep the same dtype as q, k to avoid memory doubling from float32 promotion
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention; query/key/value are expected as ``(B, H, L, D)``."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L, H, D)
    return attn_output, attn_weights


class OneVisionEncoderAttention(nn.Module):
    """
    Multi-headed attention with RoPE support, dispatched through
    :data:`ALL_ATTENTION_FUNCTIONS` (``eager`` / ``sdpa`` / ``flash_attention_2``)
    based on ``config._attn_implementation``.
    """

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.num_key_value_groups = 1  # required by repeat_kv-aware eager paths
        self.scale = self.head_dim**-0.5
        self.scaling = self.scale  # alias expected by some attention interfaces
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()
        # (B, L, 3*H*D) -> (B, L, 3, H, D) -> 3 x (B, L, H, D) -> 3 x (B, H, L, D)
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(batch_size, q_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )
        query_states = q.transpose(1, 2)
        key_states = k.transpose(1, 2)
        value_states = v.transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        dropout = 0.0 if not self.training else self.attention_dropout

        if cu_seqlens is not None and is_flash_attention_requested(self.config):
            # Flash Attention varlen path: pass cu_seq_lens / max_length kwargs.
            if max_seqlen is None:
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scale,
                dropout=dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        elif cu_seqlens is not None:
            # Non-FA implementations do not understand cu_seqlens directly; mirror
            # Qwen3-VL by splitting the packed sequence into per-sample chunks
            # along the L dim of (B, H, L, D) and running attention per chunk.
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [torch.split(t, lengths, dim=2) for t in (query_states, key_states, value_states)]
            attn_outputs = [
                attention_interface(
                    self,
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q_chunk, k_chunk, v_chunk in zip(*splits)
            ]
            # interface output is (B, l_i, H, D); concat along the L axis
            attn_output = torch.cat(attn_outputs, dim=1)
        else:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.shape[0] == 1 and batch_size > 1:
                    attn_mask = attn_mask.expand(batch_size, -1, -1)
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, L, L)
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attn_mask,
                scaling=self.scale,
                dropout=dropout,
                is_causal=False,
                **kwargs,
            )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.proj(attn_output)

        return attn_output, None


class OneVisionEncoderEncoderLayer(nn.Module):
    """Vision encoder layer with pre-norm and Flash Attention 2."""

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OneVisionEncoderAttention(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


class OneVisionEncoderEncoder(nn.Module):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([OneVisionEncoderEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Gradient checkpointing support
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    output_attentions,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LlavaOnevision2PreTrainedModel(PreTrainedModel):
    config_class = LlavaOnevision2Config
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["OneVisionEncoderEncoderLayer", "Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        # Re-initialize VisionRotaryEmbedding inv_freq buffers.
        # These are registered with persistent=False, so they are not in the checkpoint
        # state_dict. When ``from_pretrained`` materializes the model from meta tensors,
        # the values in these buffers end up uninitialized. Mirror Qwen3-VL by explicitly
        # filling them here so RoPE produces the correct frequencies post-load.
        if isinstance(module, VisionRotaryEmbedding):
            base = module.base
            with torch.no_grad():
                inv_t = 1.0 / (base ** (torch.arange(module.t_size, dtype=torch.float32) / module.t_size))
                inv_h = 1.0 / (base ** (torch.arange(module.h_size, dtype=torch.float32) / module.h_size))
                inv_w = 1.0 / (base ** (torch.arange(module.w_size, dtype=torch.float32) / module.w_size))
                module.inv_freq_t.copy_(inv_t.to(module.inv_freq_t.device))
                module.inv_freq_h.copy_(inv_h.to(module.inv_freq_h.device))
                module.inv_freq_w.copy_(inv_w.to(module.inv_freq_w.device))


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        attn_output, _ = self.attention(probe, hidden_states, hidden_states)

        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)

        return attn_output[:, 0]


# ---------------------------------------------------------------------------
# Vision Model
# ---------------------------------------------------------------------------


class LlavaOnevision2VisionPretrainedModel(LlavaOnevision2PreTrainedModel):
    """
    LLaVA-OneVision 2.0 Vision Model.

    This vision model is designed to work with Qwen2VL-style image processing:
        - Receives pre-processed patches in 3x3 block spatial order
        - Applies RoPE with matching 3x3 block layout conversion
        - Accepts explicit patch_positions for RoPE computation

    Input format:
        hidden_state: [total_patches, num_channels, patch_size, patch_size]
        grid_thw: [num_samples, 3] with [t, h, w] for each sample
    """

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        # Vision components
        self.embeddings = OneVisionEncoderEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = OneVisionEncoderEncoder(config)
        self.video_rope = VisionRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.merger = LlavaOnevision2VisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.post_init()

    def _build_cu_seqlens(
        self,
        grid_thw: torch.Tensor,
        total_patches: int,
        fixed_t: Optional[int] = 4,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, int]:
        if grid_thw is None or grid_thw.numel() == 0:
            # Fallback for no grid_thw: treat as single sequence
            return torch.tensor([0, total_patches], dtype=torch.int32, device=device), total_patches

        if device is None:
            device = grid_thw.device

        cu_seqlens = [0]
        max_seqlen = 0
        total_entries = grid_thw.shape[0]
        current_len = 0

        # Calculate cumulative lengths: split sequences based on fixed_t if provided
        for idx in range(total_entries):
            t_val = grid_thw[idx, 0].item()
            h_val = grid_thw[idx, 1].item()
            w_val = grid_thw[idx, 2].item()

            if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
                # Split large t into chunks of fixed_t
                num_full_windows = t_val // fixed_t
                remainder = t_val % fixed_t

                # Add full windows
                for _ in range(num_full_windows):
                    chunk_patches = fixed_t * int(h_val) * int(w_val)
                    current_len += chunk_patches
                    max_seqlen = max(max_seqlen, chunk_patches)
                    cu_seqlens.append(current_len)

                # Add remainder if any
                if remainder > 0:
                    chunk_patches = remainder * int(h_val) * int(w_val)
                    current_len += chunk_patches
                    max_seqlen = max(max_seqlen, chunk_patches)
                    cu_seqlens.append(current_len)
            else:
                # Standard case: add as one chunk
                chunk_patches = t_val * int(h_val) * int(w_val)
                current_len += chunk_patches
                max_seqlen = max(max_seqlen, chunk_patches)
                cu_seqlens.append(current_len)

        last_len = cu_seqlens[-1]
        if last_len != total_patches:
            raise ValueError(
                "cu_seqlens calculation mismatch:\n"
                f"- total_patches: {total_patches}\n"
                f"- calculated total: {last_len}\n"
                f"- grid_thw: {grid_thw}"
            )

        return torch.tensor(cu_seqlens, dtype=torch.int32, device=device), max_seqlen

    def _build_block_attention_mask(
        self,
        grid_thw: torch.Tensor,
        total_patches: int,
        fixed_t: Optional[int] = 4,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        if grid_thw is None or grid_thw.numel() == 0:
            return None

        if device is None:
            device = grid_thw.device

        lengths = []
        total_entries = grid_thw.shape[0]

        for idx in range(total_entries):
            t_val = grid_thw[idx, 0].item()
            h_val = grid_thw[idx, 1].item()
            w_val = grid_thw[idx, 2].item()

            if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
                # Split large t into chunks of fixed_t
                num_full_windows = t_val // fixed_t
                remainder = t_val % fixed_t

                # Add full windows
                for _ in range(num_full_windows):
                    lengths.append(fixed_t * int(h_val) * int(w_val))

                # Add remainder if any
                if remainder > 0:
                    lengths.append(remainder * int(h_val) * int(w_val))
            else:
                lengths.append(t_val * int(h_val) * int(w_val))

        total_len = sum(lengths)
        if total_len != total_patches:
            raise ValueError(
                "Block attention mask length mismatch:\n"
                f"- total_patches: {total_patches}\n"
                f"- total_len: {total_len}\n"
                f"- grid_thw: {grid_thw}"
            )

        attn_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        start = 0
        for size in lengths:
            end = start + size
            attn_mask[start:end, start:end] = False
            start = end

        return attn_mask

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=LlavaOnevision2VisionConfig)
    def forward(
        self,
        hidden_state: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_merger: Optional[bool] = False,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        Forward pass for vision model.

        This method accepts pre-processed patches from Qwen2VL image processor and applies
        RoPE (Rotary Position Embedding) in 3x3 block layout to match the spatial arrangement
        of patches.

        Args:
            hidden_state: Pre-processed patches from Qwen2VL processor.
                Shape: [total_patches, num_channels, patch_size, patch_size]
            grid_thw: Grid sizes tensor of shape [num_samples, 3] with [t, h, w] for each sample.
                Required for computing RoPE and handling visible indices.
            patch_positions: Optional explicit patch positions for RoPE computation.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a ModelOutput instead of tuple.
            skip_merger: If True, skip patch merger (useful for consistency checking).

        Returns:
            BaseModelOutputWithPooling with last_hidden_state containing merged features.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        # 1. Embeddings
        # Note: embeddings returns [total_patches, embed_dim], we need to add batch dimension
        hidden_states = self.embeddings(hidden_state)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [1, total_patches, embed_dim]
        batch_size, total_patches, _ = hidden_states.shape

        # 2. RoPE Construction
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        freqs_visible = self.video_rope.forward_from_positions(patch_positions)

        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)
        if freqs_visible.dim() == 2:
            freqs_visible = freqs_visible.unsqueeze(0)

        # 3. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)

        cu_seqlens, max_seqlen = self._build_cu_seqlens(
            grid_thw=grid_thw,
            total_patches=total_patches,
            fixed_t=getattr(self.config, "frame_windows_size", 4),
            device=hidden_states.device,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states to use -2 layer
            return_dict=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Use second-to-last layer output for better feature representation
        if encoder_outputs.hidden_states is not None and len(encoder_outputs.hidden_states) >= 2 and not skip_merger:
            sequence_output = encoder_outputs.hidden_states[-1]
        else:
            sequence_output = encoder_outputs[0]

        # Post-Norm
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # Skip merger for consistency check with original ViT
        if skip_merger:
            pooled_output = None
            if self.head is not None:
                pooled_output = self.head(sequence_output)

            if not return_dict:
                return (sequence_output, pooled_output) + (
                    encoder_outputs.hidden_states if output_hidden_states else None,
                )
            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions if output_attentions else None,
            )

        # Patch merger: input patches are already in 3x3 block order from Qwen2VL processor
        merged_output = self.merger(sequence_output)

        if not return_dict:
            return (merged_output,) + (encoder_outputs.hidden_states if output_hidden_states else None,)

        return BaseModelOutputWithPooling(
            last_hidden_state=merged_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )

    @torch.no_grad()
    def forward_debug(
        self,
        hidden_state: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
    ) -> dict:
        """Stage-by-stage forward used by HF<->Megatron consistency tests.

        Mirrors the production ``forward`` exactly EXCEPT that ``patch_positions``
        passed in are assumed to be in row-major (t, h, w) order — matching the
        Megatron-side ``onevision_encoder_model.forward_debug`` test contract,
        which also receives row-major positions and applies a block-layout
        reorder. Production ``forward`` skips that reorder because production
        ``patch_positions`` come pre-arranged in block layout from the
        Qwen2VL processor (`build_patch_positions`).

        Layer-wise outputs match the schema produced by Megatron-side
        ``onevision_encoder_model.forward_debug``.
        """
        output: dict[str, torch.Tensor] = {}

        hidden_states = self.embeddings(hidden_state)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        output["after_patch_embed"] = hidden_states.clone()
        batch_size, total_patches, _ = hidden_states.shape

        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        freqs_visible = self.video_rope.forward_from_positions(patch_positions)

        if grid_thw is not None:
            sms = getattr(self.config, "spatial_merge_size", 2)
            if sms > 1:
                offset = 0
                converted_chunks: list[torch.Tensor] = []
                for sample_idx in range(grid_thw.shape[0]):
                    t_val = int(grid_thw[sample_idx, 0].item())
                    h_val = int(grid_thw[sample_idx, 1].item())
                    w_val = int(grid_thw[sample_idx, 2].item())
                    n_patches = t_val * h_val * w_val
                    chunk = freqs_visible[offset : offset + n_patches]
                    half = chunk.shape[-1]
                    chunk = chunk.view(t_val, h_val, w_val, half)
                    h_merged = h_val // sms
                    w_merged = w_val // sms
                    chunk = chunk.view(t_val, h_merged, sms, w_merged, sms, half)
                    chunk = chunk.permute(0, 1, 3, 2, 4, 5).contiguous()
                    chunk = chunk.view(n_patches, half)
                    converted_chunks.append(chunk)
                    offset += n_patches
                freqs_visible = torch.cat(converted_chunks, dim=0)

        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)
        if freqs_visible.dim() == 2:
            freqs_visible = freqs_visible.unsqueeze(0)
        output["rotary_pos_emb"] = freqs_visible.clone()

        hidden_states = self.layernorm_pre(hidden_states)
        output["after_pre_layernorm"] = hidden_states.clone()

        cu_seqlens, max_seqlen = self._build_cu_seqlens(
            grid_thw=grid_thw,
            total_patches=total_patches,
            fixed_t=getattr(self.config, "frame_windows_size", 4),
            device=hidden_states.device,
        )

        layer_outputs: dict[str, torch.Tensor] = {}
        layer_outputs["input_hidden_states"] = hidden_states.clone()
        for i, layer in enumerate(self.encoder.layers):
            layer_outputs[f"layer_{i}_input"] = hidden_states.clone()
            layer_out = layer(
                hidden_states,
                attention_mask=None,
                rotary_pos_emb=freqs_visible,
                output_attentions=False,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hidden_states = layer_out[0]
            layer_outputs[f"layer_{i}_output"] = hidden_states.clone()
        layer_outputs["final_output"] = hidden_states.clone()
        output["layer_outputs"] = layer_outputs

        if self.layernorm_post is not None:
            hidden_states = self.layernorm_post(hidden_states)
        output["before_adapter"] = hidden_states.clone()

        merger_input = hidden_states.squeeze(0) if hidden_states.dim() == 3 else hidden_states
        merged_output = self.merger(merger_input)
        output["after_merger"] = merged_output.clone()

        return output


@auto_docstring
class LlavaOnevision2Model(LlavaOnevision2PreTrainedModel):
    base_model_prefix = ""
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: LlavaOnevision2Config
    _no_split_modules = ["OneVisionEncoderEncoderLayer", "Qwen3DecoderLayer"]

    def __init__(self, config: LlavaOnevision2Config):
        super().__init__(config)
        self.visual = LlavaOnevision2VisionPretrainedModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions=None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos: Pre-processed patches from Qwen2VL processor.
                `torch.FloatTensor` of shape `(total_patches, num_channels, patch_size, patch_size)`
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Convert to correct dtype
        pixel_values_videos = pixel_values_videos.type(self.visual.embeddings.patch_embedding.weight.dtype)

        # Forward through vision model with grid_thw
        vision_output = self.visual(pixel_values_videos, grid_thw=video_grid_thw, patch_positions=patch_positions)

        # Extract the actual tensor from BaseModelOutputWithPooling
        if hasattr(vision_output, "last_hidden_state"):
            video_embeds = vision_output.last_hidden_state
        else:
            video_embeds = vision_output[0]  # Fallback for tuple output

        # Compute split sizes from video_grid_thw or from input shape
        if video_grid_thw is not None:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        else:
            # Compute from input shape
            batch_size = pixel_values_videos.shape[0]
            split_sizes = [video_embeds.shape[1]] * batch_size

        # Split embeddings per video
        if len(split_sizes) > 1:
            video_embeds = torch.split(video_embeds.view(-1, video_embeds.shape[-1]), split_sizes)
        else:
            video_embeds = [video_embeds.view(-1, video_embeds.shape[-1])]

        return video_embeds

    def get_image_features(
        self, pixel_values, image_grid_thw: Optional[torch.LongTensor] = None, patch_positions=None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values: Pre-processed patches from Qwen2VL processor.
                - `torch.FloatTensor` of shape `(total_patches, num_channels, patch_size, patch_size)`
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # Standard format from Qwen2VL processor
        if pixel_values.dim() == 2:
            # Convert to correct dtype
            pixel_values = pixel_values.type(self.visual.embeddings.patch_embedding.weight.dtype)

            # Forward through vision model with grid_thw
            vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, patch_positions=patch_positions)

            # Extract the actual tensor from BaseModelOutputWithPooling
            if hasattr(vision_output, "last_hidden_state"):
                image_embeds = vision_output.last_hidden_state
            else:
                image_embeds = vision_output[0]

            # Compute split sizes from grid_thw
            if image_grid_thw is not None:
                split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            else:
                # Fallback: assume single image
                split_sizes = [image_embeds.shape[0] if image_embeds.dim() == 2 else image_embeds.shape[1]]

            # Split embeddings per image
            image_embeds_flat = image_embeds.view(-1, image_embeds.shape[-1])
            if len(split_sizes) > 1:
                image_embeds = list(torch.split(image_embeds_flat, split_sizes))
            else:
                image_embeds = [image_embeds_flat]

            return image_embeds
        else:
            raise ValueError(
                f"Unsupported pixel_values shape: expected 4D tensor [total_patches, C, H, W], "
                f"got {pixel_values.shape if hasattr(pixel_values, 'shape') else type(pixel_values)}"
            )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevision2ModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        patch_positions (`torch.LongTensor` of shape `(total_patches, 3)` or `(1, total_patches, 3)`, *optional*):
            Explicit per-patch `(t, h, w)` position indices used by the vision tower to compute 3D rotary
            position embeddings (and the optional absolute position embedding inside the patch merger).
            `total_patches` is the sum of `t * h * w` across all images and videos in the batch, matching
            the layout produced by the Qwen2VL-style image processor.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
              Indices depicting the position of the input sequence tokens in the sequence. Contrarily to
              `position_ids`, this tensor is not affected by padding.

        Note: see the top-level ``LlavaOnevision2ForConditionalGeneration.forward``
        docstring; in P0 video flows in via the ``image_grid_thw`` / ``pixel_values``
        alias, so ``pixel_values_videos`` / ``video_grid_thw`` /
        ``second_per_grid_ts`` are unused at this layer.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, patch_positions=patch_positions)

        if image_embeds is not None:
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw, patch_positions=patch_positions
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Use simple 1D position_ids
        if position_ids is None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = (
                    torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
                )

            # Handle cache_position for generation
            if cache_position is not None and cache_position[0] != 0:
                position_ids = position_ids + cache_position[0]

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = LlavaOnevision2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()


@auto_docstring
class LlavaOnevision2ForConditionalGeneration(LlavaOnevision2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaOnevision2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions=None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, patch_positions=patch_positions)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevision2CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        patch_positions (`torch.LongTensor` of shape `(total_patches, 3)` or `(1, total_patches, 3)`, *optional*):
            Explicit per-patch `(t, h, w)` position indices used by the vision tower to compute 3D rotary
            position embeddings (and the optional absolute position embedding inside the patch merger).
            `total_patches` is the sum of `t * h * w` across all images and videos in the batch, matching
            the layout produced by the Qwen2VL-style image processor.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
              Indices depicting the position of the input sequence tokens in the sequence. Contrarily to
              `position_ids`, this tensor is not affected by padding.

        Note (P0 native-video alias):
            The companion ``Llava_Onevision2Processor.__call__(videos=...)`` does NOT
            pass ``pixel_values_videos`` / ``video_grid_thw`` / ``second_per_grid_ts``
            to this forward. Instead it aliases the video patch tensor as
            ``pixel_values=`` and ``image_grid_thw=``, so video inputs share the
            same code path as multi-image inputs (the OneVision encoder is purely
            spatial; temporal information is carried by per-frame ``<X.X seconds>``
            text tags emitted by the processor). The ``*_videos`` and
            ``second_per_grid_ts`` kwargs are kept declared here only for API
            completeness and future use (e.g. 3D mRoPE / ``get_rope_index``); they
            are NOT consumed by the current OneVision encoder. See
            NATIVE_VIDEO_PLAN.md (stage 3.1) for the full rationale.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaOnevision2ForConditionalGeneration

        >>> model = LlavaOnevision2ForConditionalGeneration.from_pretrained("Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf", trust_remote_code=True)
        >>> processor = AutoProcessor.from_pretrained("Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf", trust_remote_code=True)

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LlavaOnevision2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        patch_positions=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            patch_positions=patch_positions,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # After the prefill iteration, drop image inputs so the vision tower
        # isn't re-run on decode steps. Gating on `is_first_iteration` (the
        # Qwen3-VL convention) is the only reliable signal in transformers
        # 5.x: `past_key_values` is non-None even on the first call (an empty
        # DynamicCache is created up-front by `generate`), and `cache_position`
        # may be `None` for remote-code models.
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "patch_positions",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
                elif key == "patch_positions":
                    if image_grid_thw is not None and image_grid_thw.numel() > 0 and image_nums.sum() > 0:
                        samples = torch.split(image_grid_thw, list(image_nums))
                        lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    elif video_grid_thw is not None and video_grid_thw.numel() > 0 and video_nums.sum() > 0:
                        samples = torch.split(video_grid_thw, list(video_nums))
                        lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    else:
                        continue
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "LlavaOnevision2ForConditionalGeneration",
    "LlavaOnevision2Model",
    "LlavaOnevision2PreTrainedModel",
]
