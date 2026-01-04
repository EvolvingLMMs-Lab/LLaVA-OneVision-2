""" VisionTransformer module """

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from aiak_training_llm.models.qwen_vl.vision_transformer_block import TransformerBlock
from aiak_training_llm.models.qwen_vl.vision_model import _rotate_half, apply_rotary_pos_emb_vision
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_config import VisionConfig
from megatron.training import print_rank_0
from aiak_training_llm.models.custom.common.local_norm import FusedLayerNorm

def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb_vision(t, freqs, config, cu_seqlens=None, rotary_interleaved=False):
#     """" Apply rotation to positional embedding """
#     orig_dtype = t.dtype
#     t = t.float()
#     if cu_seqlens is not None:
#         freqs = freqs.squeeze(1)
#         cos_ = freqs.cos().float().repeat(1, 1, 2)
#         sin_ = freqs.sin().float().repeat(1, 1, 2)
#     else:
#         cos_ = freqs.cos().float().repeat(1, 1, 1, 2)
#         sin_ = freqs.sin().float().repeat(1, 1, 1, 2)
#     t = (t * cos_) + (_rotate_half(t) * sin_)
#     return t.to(orig_dtype)

def apply_rotary_pos_emb_vision(t, freqs, config, cu_seqlens=None, rotary_interleaved=False):
    """
    Apply rotary positional embeddings to t in a robust way:
    - build cos_/sin_ from freqs as before
    - slice cos_/sin_ to match t sequence length
    - expand/unsqueeze so broadcasting works for common layouts
    """
    # DEBUG: Check input
    if t is None:
        print(f"[apply_rotary_pos_emb_vision] ERROR: input t is None!", flush=True)
        return None
    
    orig_dtype = t.dtype
    t = t.float()

    # Build cos_/sin_ as in your original code
    if cu_seqlens is not None:
        # freqs shape expected something like [B, 1, S, C] or similar; you squeezed axis 1
        freqs_work = freqs.squeeze(1)
        cos_ = freqs_work.cos().float().repeat(1, 1, 2)   # -> [?, ?, 2*?] depending on freqs_work layout
        sin_ = freqs_work.sin().float().repeat(1, 1, 2)
    else:
        cos_ = freqs.cos().float().repeat(1, 1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 1, 2)

    # --- Robust alignment code: make cos_/sin_ match t's sequence axis & length ----
    # Determine candidate sequence axis in t. Prefer axis 0, then 1, then others.
    # We'll detect the likely seq axis by comparing sizes; fallback to axis 0.
    def _find_seq_axis_and_len(t, cos_):
        t_shape = tuple(t.shape)
        # Try to find an axis i where cos_.shape[0] matches t.shape[i]
        for axis in range(len(t_shape)):
            if cos_.shape[0] == t_shape[axis]:
                return axis, t_shape[axis]
        # If no exact match, prefer axis 0 if it's smaller or cos_ is larger and can be sliced
        return 0, t_shape[0]

    seq_axis, seq_len = _find_seq_axis_and_len(t, cos_)

    # Slice cos_/sin_ on their first dimension to match seq_len (safe even if cos_ is smaller)
    if cos_.shape[0] != seq_len:
        # if cos_ is shorter than seq_len, we'll let broadcasting fail later (better to raise)
        if cos_.shape[0] < seq_len:
            # Recompute or raise — here we raise a helpful error so you can detect upstream bug
            raise RuntimeError(
                f"freqs/cos_ length ({cos_.shape[0]}) is smaller than the tensor sequence length ({seq_len}). "
                "Recompute freqs for the required length or provide larger cached freqs."
            )
        # Slice axis 0
        sl = [slice(None)] * cos_.ndim
        sl[0] = slice(0, seq_len)
        cos_ = cos_[tuple(sl)].contiguous()
        sin_ = sin_[tuple(sl)].contiguous()

    # Now make cos_/sin_ broadcast-compatible with t.
    # Typical cases:
    #  - t: [S, B, D] and cos_: [S, D] -> unsqueeze(1) -> [S,1,D]
    #  - t: [B, S, D] and cos_: [S, D] -> unsqueeze(0) -> [1,S,D]
    #  - t: [B, H, S, D] and cos_: [S, D] -> unsqueeze to [1,1,S,1,D...] not common
    # We'll handle the common 2/3-dim cases and otherwise attempt to add singleton dims to the left.
    if cos_.ndim == 2 and t.ndim == 3:
        # cos_: [S, D]
        if seq_axis == 0:
            # t: [S, B, D] -> need [S, 1, D]
            cos_ = cos_.unsqueeze(1)
            sin_ = sin_.unsqueeze(1)
        else:
            # seq_axis == 1: t: [B, S, D] -> need [1, S, D]
            cos_ = cos_.unsqueeze(0)
            sin_ = sin_.unsqueeze(0)
    elif cos_.ndim == 2 and t.ndim == 2:
        # both [S, D] vs [S, D] -> ok
        pass
    elif cos_.ndim == 3 and t.ndim == 3:
        # cos_ might already be [S, 1, D] or [B, S, D] style; try to match by permuting if necessary
        # If cos_.shape[0] == seq_len then assume first dim is seq and cos_ probably is [S,1,D]
        # else attempt no-op; broadcasting should handle remaining dims
        pass
    else:
        # Generic approach: try to prefix cos_ with singleton dims so that cos_.ndim == t.ndim or cos_.ndim == t.ndim-1
        # We want sequence dim at position seq_axis in t; ensure cos_ has seq at dim 0 currently (we sliced it)
        # So add singleton dims to the left until cos_.ndim matches t.ndim with seq at that position.
        if cos_.ndim < t.ndim:
            # number of dims to add on left
            to_add = t.ndim - cos_.ndim
            for _ in range(to_add):
                cos_ = cos_.unsqueeze(0)
                sin_ = sin_.unsqueeze(0)
        # If still not aligned, rely on broadcasting but print a debug warning:
        # (you can remove/disable the print in production)
        # debug print
        # print(f"[rotary] after align: t.shape={tuple(t.shape)}, cos_.shape={tuple(cos_.shape)}")

    # Final safety check: ensure cos_ can broadcast over t along seq_axis
    # Map cos_ seq dim (we assumed it's axis 0) to the seq_axis in t via unsqueezing above.
    # If the seq axis in t is not axis 0, the unsqueeze logic handled typical 3-D case above.

    # Debug print (optional; remove when satisfied)
    # print("rotary debug: t.shape", tuple(t.shape), "cos.shape", tuple(cos_.shape), "sin.shape", tuple(sin_.shape))

    # Apply rotary
    t = (t * cos_) + (_rotate_half(t) * sin_)

    result = t.to(orig_dtype)
    
    # DEBUG: Check output
    if result is None:
        print(f"[apply_rotary_pos_emb_vision] ERROR: result is None after conversion!", flush=True)
    
    return result


class PatchEmbed(torch.nn.Module):
    """" Patch Embedding """
    def __init__(
        self,
        patch_size: int = 14,
        # temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        # self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # kernel_size = [temporal_patch_size, patch_size, patch_size]
        # self.proj = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """" Forward pass """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            # -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(torch.nn.Module):
    """" Rotary Position Embedding """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = inv_freq.to(torch.cuda.current_device())

    def forward(self, seqlen: int) -> torch.Tensor:
        """ Forward Pass """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionModel(VisionModule):
    """VisionTransformer model. """
    def __init__(self, 
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2
    ) -> None:
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.spatial_merge_size = spatial_merge_size

        self.rotary_pos_emb = VisionRotaryEmbedding(config.kv_channels // 2)

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        """ rotation position embedding """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """ forward function """
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(2).float()

        x = self.patch_embed(x)
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x = self.decoder(x, rotary_pos_emb=rotary_pos_emb, attention_mask=None, attn_mask_type=AttnMaskType.no_mask)
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        return x, None


class RiceViTModel(VisionModel):
    """"""
    def __init__(self,
        config: VisionConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
        # window_size: int = 112,
    ) -> None:
        super().__init__(config, transformer_layer_spec, spatial_merge_size)
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = list(range(config.num_layers))
        # self.window_size = window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.register_buffer('class_embedding', torch.randn(config.hidden_size))
        self.register_buffer('class_pos_emb', torch.randn(1, config.kv_channels // 2))
        # self.class_embedding = torch.nn.Parameter(torch.randn(config.hidden_size))
        # self.class_pos_emb = torch.nn.Parameter(torch.randn(1, config.kv_channels // 2))

        self.pre_layernorm = torch.nn.LayerNorm(
            config.hidden_size,
            eps=1e-4)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):
            new_x.append(class_tokens[i:i+1])
            new_x.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)

        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i:i+1])
            new_rotary_pos_emb.append(rotary_pos_emb[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)

        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # 起始为0
        for length in tokens_per_sample:

            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)


        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)

        x = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [原始seq_len, hidden_size]
        return patch_output, None


    def forward_debug(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        output = {}

        x = self.patch_embed(x)
        output["after_patch_embed"] = x.clone()

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):

            new_x.append(class_tokens[i:i+1])

            new_x.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)
        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i:i+1])
            new_rotary_pos_emb.append(rotary_pos_emb[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)
        output["rotary_pos_emb"] = rotary_pos_emb.clone()
        output["class_embedding"] = self.class_embedding.clone()
        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # 起始为0
        for length in tokens_per_sample:

            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)


        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)
        output["after_pre_layernorm"] = x.clone()
        x = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [原始seq_len, hidden_size]
        output["before_adapter"] = patch_output.clone()
        return output
