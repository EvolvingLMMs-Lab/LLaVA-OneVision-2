from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import auto_docstring, can_return_tuple, logging

from .configuration_onevision_encoder import OneVisionEncoderConfig

logger = logging.get_logger(__name__)


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def rotate_half(x):
    # Interleaved rotation: (x1, x2, x3, x4) -> (-x2, x1, -x4, x3) to match source model.
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D); freqs: (B, L, D)
    # CRITICAL FIX (lang variant): Cast cos/sin to q.dtype (bf16/fp16) immediately.
    # freqs are float32, so cos()/sin() return float32. Without this cast,
    # (q * cos) upcasts q to float32, breaking FlashAttention dtype contract.
    cos = freqs.cos().unsqueeze(1).to(q.dtype)
    sin = freqs.sin().unsqueeze(1).to(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
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
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class VideoRotaryEmbeddingSplit466(nn.Module):
    """3D (T,H,W) Rotary frequency constructor with 4:6:6 split."""

    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.rope_base = base
        self.register_buffer("inv_freq_t", self._compute_inv_freq(self.t_size), persistent=False)
        self.register_buffer("inv_freq_h", self._compute_inv_freq(self.h_size), persistent=False)
        self.register_buffer("inv_freq_w", self._compute_inv_freq(self.w_size), persistent=False)

    def _compute_inv_freq(self, size: int) -> torch.Tensor:
        return 1.0 / (self.rope_base ** (torch.arange(size, dtype=torch.float32) / size))

    def reset_inv_freqs(self):
        for name, size in (("inv_freq_t", self.t_size), ("inv_freq_h", self.h_size), ("inv_freq_w", self.w_size)):
            buf = getattr(self, name)
            buf.copy_(self._compute_inv_freq(size).to(device=buf.device, dtype=buf.dtype))

    def forward(self, t: int, h: int, w: int, device=None):
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

        return torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        r"""
        Compute rotary frequencies from explicit per-patch (t, h, w) positions instead of a regular (T, H, W) grid.

        Args:
            patch_positions: `(batch_size, seq_len, 3)` int/float tensor with `[t, h, w]` per patch.

        Returns:
            freqs: `(batch_size, seq_len, dim_t + dim_h + dim_w)` position frequencies for RoPE.
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[..., 0].float()
        h_pos = patch_positions[..., 1].float()
        w_pos = patch_positions[..., 2].float()

        ft = torch.einsum("bs,d->bsd", t_pos, inv_t)
        fh = torch.einsum("bs,d->bsd", h_pos, inv_h)
        fw = torch.einsum("bs,d->bsd", w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multi-Head Attention Pooling with a learned probe (PMA-style)."""

    def __init__(self, config: OneVisionEncoderConfig):
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


class OneVisionEncoderEmbeddings(nn.Module):
    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)

        batch_size, channels, t_frames, height, width = pixel_values.shape

        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)

        embeddings = self.patch_embedding(x_2d)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)

        return embeddings


class OneVisionEncoderAttention(nn.Module):
    """Multi-headed attention with RoPE support, dispatching to v5 attention interface."""

    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class OneVisionEncoderEncoderLayer(nn.Module):
    def __init__(self, config: OneVisionEncoderConfig):
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
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states, None


class OneVisionEncoderEncoder(nn.Module):
    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([OneVisionEncoderEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_self_attentions = all_self_attentions + (attn_weights,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class OneVisionEncoderPreTrainedModel(PreTrainedModel):
    config_class = OneVisionEncoderConfig
    base_model_prefix = "onevision_encoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OneVisionEncoderEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, VideoRotaryEmbeddingSplit466):
            module.reset_inv_freqs()


@auto_docstring
class OneVisionEncoderModel(OneVisionEncoderPreTrainedModel):
    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = OneVisionEncoderEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = OneVisionEncoderEncoder(config)
        self.video_rope = VideoRotaryEmbeddingSplit466(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.post_init()
        self.video_rope.reset_inv_freqs()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        visible_indices (`torch.Tensor` of shape `(batch_size, num_visible)`, *optional*):
            Indices of patches to keep after token dropping. When provided, only the selected
            patches participate in attention; positional rotary frequencies are gathered at
            these indices so RoPE remains spatially correct after dropping. When `None`, all
            patches are used (no dropping).
        patch_positions (`torch.Tensor` of shape `(batch_size, seq_len, 3)`, *optional*):
            Explicit `[t, h, w]` position per patch. When provided, RoPE frequencies are computed
            directly from these positions via `forward_from_positions`, bypassing the regular
            (T, H, W) grid. Used by language-aligned multimodal callers that pass arbitrary
            patch layouts. Mutually exclusive with the default grid path.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values.dim() == 5:
            t_frames = (
                self.config.rope_temporal_size
                if self.config.rope_temporal_size is not None
                else pixel_values.shape[2]
            )
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]
        else:
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]

        hidden_states = self.embeddings(pixel_values)
        batch_size, total_patches, _ = hidden_states.shape

        if visible_indices is None:
            visible_indices = (
                torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)
            )

        if patch_positions is not None:
            freqs_visible = self.video_rope.forward_from_positions(patch_positions)
        else:
            freqs_full = self.video_rope(
                t=t_frames,
                h=height // self.config.patch_size,
                w=width // self.config.patch_size,
                device=pixel_values.device,
            )
            freqs_visible = freqs_full[visible_indices]
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        hidden_states = self.layernorm_pre(hidden_states)

        num_visible = visible_indices.shape[1]
        if num_visible != total_patches:
            hidden_states = hidden_states.gather(
                1, visible_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            )

        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        sequence_output = encoder_outputs.last_hidden_state

        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        pooled_output = None
        if self.head is not None:
            pooled_output = self.head(sequence_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
