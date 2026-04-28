from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_onevision_encoder import OneVisionEncoderConfig


try:
    from flash_attn import flash_attn_func

    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Model Docstrings
# ---------------------------------------------------------------------------

ONEVISION_ENCODER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OneVisionEncoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ONEVISION_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch_size, num_channels, num_frames, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`].
        visible_indices (`torch.Tensor`, *optional*):
            Indices of visible patches for masking. Used in MAE-style pretraining or inference.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# ---------------------------------------------------------------------------
# Helper Functions & Layers
# ---------------------------------------------------------------------------


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)
    # Use float32 for RoPE computation to maintain precision across layers,
    # then cast back to original dtype for FlashAttention compatibility.
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VideoRotaryEmbeddingSplit466(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    """

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

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs

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


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

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


# ---------------------------------------------------------------------------
# Modeling Components
# ---------------------------------------------------------------------------


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
        # Handle 4D (B, C, H, W) or 5D (B, C, T, H, W) inputs
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)  # (B, C, 1, H, W)

        batch_size, channels, t_frames, height, width = pixel_values.shape

        # Merge time into batch for Conv2d
        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)

        # Patch Embed
        embeddings = self.patch_embedding(x_2d)  # (B*T, C, Hp, Wp)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B*T, L_frame, C)

        # Flatten all patches
        total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)

        return embeddings


class OneVisionEncoderAttention(nn.Module):
    """Multi-headed attention with RoPE support"""

    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (B, L, H, D) -> Transpose to (B, H, L, D)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, q_len):
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        # FIX: Remove dtype=torch.float32 to stay in original dtype (bf16/fp16)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


class OneVisionEncoderFlashAttention2(nn.Module):
    """
    Multi-headed attention with RoPE support using Flash Attention 2.
    This module implements the same attention mechanism as OneVisionEncoderAttention but uses
    Flash Attention for improved performance and memory efficiency.
    """

    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using Flash Attention 2.
        """
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention requires (B, L, H, D) format
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            # Transpose for RoPE application: (B, L, H, D) -> (B, H, L, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            # NOTE: apply_rotary_pos_emb uses float32 internally, casts back to input dtype
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)
            # Transpose back: (B, H, L, D) -> (B, L, H, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        # Flash Attention forward pass
        if not _flash_attn_available:
            raise ImportError("flash_attn is not installed. Please install it to use OneVisionEncoderFlashAttention2.")

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Reshape to (B, L, embed_dim)
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # No extra casting here.
        attn_output = self.out_proj(attn_output)

        return attn_output, None


ONEVISION_ENCODER_ATTENTION_CLASSES = {
    "eager": OneVisionEncoderAttention,
    "flash_attention_2": OneVisionEncoderFlashAttention2,
}


class OneVisionEncoderEncoderLayer(nn.Module):
    def __init__(self, config: OneVisionEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Get attention implementation from config, default to "flash_attention_2"
        attn_implementation = getattr(config, "_attn_implementation", "flash_attention_2")
        if attn_implementation not in ONEVISION_ENCODER_ATTENTION_CLASSES:
            # Fallback to eager if flash_attention_2 is not available
            if not _flash_attn_available and attn_implementation == "flash_attention_2":
                attn_implementation = "eager"
            else:
                raise ValueError(
                    f"Unknown attention implementation: {attn_implementation}. "
                    f"Available implementations: {list(ONEVISION_ENCODER_ATTENTION_CLASSES.keys())}"
                )
        self.self_attn = ONEVISION_ENCODER_ATTENTION_CLASSES[attn_implementation](config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


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
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=output_attentions,
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


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


@add_start_docstrings(
    "The bare OneVision Encoder Model outputting raw hidden-states without any specific head on top.",
    ONEVISION_ENCODER_START_DOCSTRING,
)
class OneVisionEncoderPreTrainedModel(PreTrainedModel):
    config_class = OneVisionEncoderConfig
    base_model_prefix = "onevision_encoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OneVisionEncoderEncoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights"""
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
            # Fix: RMSNorm doesn't have bias, must check hasattr first
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings(
    "OneVision Encoder Model with a vision transformer encoder.",
    ONEVISION_ENCODER_START_DOCSTRING,
)
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

    @add_start_docstrings_to_model_forward(ONEVISION_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OneVisionEncoderConfig)
    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_postions: Optional[torch.Tensor] = None,
        visible_indices: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoModel, AutoImageProcessor
        >>> from PIL import Image

        >>> model = AutoModel.from_pretrained("lmms-lab-encoder/onevision-encoder-large", trust_remote_code=True)
        >>> preprocessor = AutoImageProcessor.from_pretrained("lmms-lab-encoder/onevision-encoder-large", trust_remote_code=True)
        >>> image = Image.open("path/to/your/image.jpg")  # Replace with your image path
        >>> pixel_values = preprocessor(images=image, return_tensors="pt")["pixel_values"]
        >>> outputs = model(pixel_values)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Determine video dimensions for RoPE
        # Note: pixel_values passed to embeddings can be 4D or 5D
        if pixel_values.dim() == 5:
            # Use config.rope_temporal_size if set, otherwise use actual frame count
            t_frames = (
                self.config.rope_temporal_size if self.config.rope_temporal_size is not None else pixel_values.shape[2]
            )
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]
        else:
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]

        # 1. Embeddings
        hidden_states = self.embeddings(pixel_values)
        batch_size, total_patches, _ = hidden_states.shape

        # 2. Visible Indices Handling
        if visible_indices is None:
            visible_indices = (
                torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)
            )

        # 3. RoPE Construction
        if patch_postions is not None:
            freqs_visible = self.video_rope.forward_from_positions(patch_postions)
        else:
            freqs_full = self.video_rope(
                t=t_frames,
                h=height // self.config.patch_size,
                w=width // self.config.patch_size,
                device=pixel_values.device,
            )
            freqs_visible = freqs_full[visible_indices]

        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        # 4. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)

        # fix: gather hidden_states to match freqs_visible when using sparse visible_indices
        num_visible = visible_indices.shape[1]
        if num_visible != total_patches:
            # sparse mode: select only visible patches
            hidden_states = hidden_states.gather(
                1, visible_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        # Apply post-norm if configured
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # 5. Pooling Head
        pooled_output = None
        if self.head is not None:
            pooled_output = self.head(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )