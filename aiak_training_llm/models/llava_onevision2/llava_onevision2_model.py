import logging
import os
from collections import namedtuple
from functools import partial

import torch
from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import make_viewless_tensor

from aiak_training_llm.models.llava_onevision2.onevision_encoder_model import OneVisionEncoderModel
from aiak_training_llm.models.qwen import QwenModel
from aiak_training_llm.models.llava_onevision2.adapter import Adapter
from aiak_training_llm.models.llava_onevision2 import vision_balance
from aiak_training_llm.models.qwen_vl.utils import get_inputs_on_this_cp_rank


def vision_balance_group():
    """DP group over which --vision-dp-balance spreads vision work (same TP/PP rank), or None if DP == 1."""
    if parallel_state.get_data_parallel_world_size() <= 1:
        return None
    return parallel_state.get_data_parallel_group()

def get_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    spatial_merge_size=2,
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    tokens_per_second=2,
    second_per_grid_ts=None,
    attention_mask=None,
):
    """Compute Qwen 3D multimodal M-RoPE position ids.

    Ported from HuggingFace ``Qwen2_5_VLForConditionalGeneration.get_rope_index``
    (HF-standard mask convention: 1 = valid, 0 = padding). For single-frame
    images this is bit-identical to the Qwen3.5 ``get_rope_index`` (verified on
    toy inputs); the two only diverge on multi-frame video temporal handling
    (Qwen3.5 uses per-frame timestamps, which this OV2 data pipeline does not
    emit), where this falls back to the Qwen2.5-VL ``second_per_grid`` scheme.

    Args:
        input_ids: LongTensor [batch, seq_len].
        image_grid_thw: LongTensor [num_images, 3] or None.
        video_grid_thw: LongTensor [num_videos, 3] or None.
        spatial_merge_size: vision spatial merge factor (H/W divided by this in LLM).
        image_token_id / video_token_id / vision_start_token_id: special token ids.
        tokens_per_second: temporal-index scaling for video frames.
        second_per_grid_ts: optional [num_videos] seconds-per-temporal-grid.
        attention_mask: [batch, seq_len] with 1 = valid, 0 = pad (None -> all valid).

    Returns:
        position_ids: LongTensor [3, batch, seq_len] (T, H, W rows).
        mrope_position_deltas: LongTensor [batch, 1].
    """
    mrope_position_deltas = []
    if image_grid_thw is None and video_grid_thw is None:
        # Pure text: T == H == W == arange (caller normally handles this fallback).
        if attention_mask is not None:
            position_ids = (
                attention_mask.long().cumsum(-1) - 1
            )
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            s = input_ids.shape[1]
            position_ids = (
                torch.arange(s, device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
            )
        return position_ids, mrope_position_deltas

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index, video_index = 0, 0
    attention_mask = attention_mask.to(total_input_ids.device)
    for i, cur_ids in enumerate(total_input_ids):
        cur_ids = cur_ids[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(cur_ids == vision_start_token_id).squeeze(1)
        vision_tokens = cur_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = cur_ids.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            range_tensor = torch.arange(llm_grid_t).view(-1, 1)
            expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
            time_tensor = expanded_range * second_per_grid_t * tokens_per_second
            time_tensor_long = time_tensor.long()
            t_index = time_tensor_long.flatten()

            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


_MROPE_A_LOGGED = False
_MROPE_FB_LOGGED = False


def _rope_from_patch_positions(
    ids, cu, patch_positions, merge, image_token_id, video_token_id, device, total_len
):
    """Build packed 3D M-RoPE positions from real per-patch positions.

    patch_positions arrives BLOCK-LAYOUT + PATCH-LEVEL (the task encoder already
    ran convert_positions_to_block_layout): every ``merge*merge`` consecutive
    patches form one merged vision token, in the same order as the <image_pad>/
    <video_pad> tokens appear in the packed sequence. We reduce it to one
    (t, h//merge, w//merge) per merged token, then walk the sequence:

      * text token          -> running scalar position (cur, cur, cur)
      * vision token         -> (base + frame_order, base + h_m, base + w_m)
                                where base is captured at the visual start and
                                frame_order is the dense rank of the real t
                                within that visual (0,30,60 -> 0,1,2).

    A new visual is detected when a block start (h_m==0 and w_m==0) has t that
    did NOT advance past the visual's first t (image after image, or a new video
    whose t restarts). Returns [3, 1, S] LongTensor, or None to signal the
    caller to fall back to the grid path.
    """
    pp = patch_positions
    if isinstance(pp, (list, tuple)):
        pp = [p for p in pp if p is not None and (p.numel() if torch.is_tensor(p) else len(p)) > 0]
        if not pp:
            return None
        pp = torch.cat([p if torch.is_tensor(p) else torch.as_tensor(p) for p in pp], dim=0)
    if pp is None or (torch.is_tensor(pp) and pp.numel() == 0):
        return None
    pp = pp.detach().to("cpu").long()
    if pp.dim() != 2 or pp.shape[1] != 3:
        return None
    m2 = merge * merge
    if pp.shape[0] % m2 != 0:
        return None
    # representative patch of each merge*merge block; assert intra-block consistency
    rep = pp[::m2]
    if merge > 1:
        blk = pp[: (pp.shape[0] // m2) * m2].view(-1, m2, 3)
        bt = blk[:, :, 0]
        bh = blk[:, :, 1] // merge
        bw = blk[:, :, 2] // merge
        same = (
            (bt == bt[:, :1]).all()
            and (bh == bh[:, :1]).all()
            and (bw == bw[:, :1]).all()
        )
        if not same:
            # layout is not the expected 3x3 block order -> do not trust it
            return None
    merged_t = rep[:, 0].tolist()
    merged_h = (rep[:, 1] // merge).tolist()
    merged_w = (rep[:, 2] // merge).tolist()
    n_tok = len(merged_t)

    vp = 0  # global cursor into merged vision tokens (sequence order)
    pos_chunks = []
    n_vision_tokens_seen = 0
    for k in range(len(cu) - 1):
        s0, s1 = cu[k], cu[k + 1]
        if s1 <= s0:
            continue
        seg = ids[s0:s1].tolist()
        n = len(seg)
        t_row = [0] * n
        h_row = [0] * n
        w_row = [0] * n
        cur = 0
        vis_base = None      # baseline captured at the current visual's start
        vis_first_t = None   # t of the current visual's first frame
        vis_rank = {}        # real_t -> dense frame order within current visual
        vis_max = 0          # max offset used inside the current visual
        for i in range(n):
            tok = seg[i]
            if tok == image_token_id or tok == video_token_id:
                if vp < n_tok:
                    tt, hh, ww = merged_t[vp], merged_h[vp], merged_w[vp]
                    vp += 1
                else:
                    tt, hh, ww = (vis_first_t or 0), 0, 0
                n_vision_tokens_seen += 1
                block_start = (hh == 0 and ww == 0)
                new_visual = (
                    vis_base is None
                    or (block_start and vis_first_t is not None and tt <= vis_first_t)
                )
                if new_visual:
                    if vis_base is not None:
                        cur = max(cur, vis_base + vis_max + 1)
                    vis_base = cur
                    vis_first_t = tt
                    vis_rank = {}
                    vis_max = 0
                if tt not in vis_rank:
                    vis_rank[tt] = len(vis_rank)
                f = vis_rank[tt]
                t_row[i] = vis_base + f
                h_row[i] = vis_base + hh
                w_row[i] = vis_base + ww
                if f > vis_max:
                    vis_max = f
                if hh > vis_max:
                    vis_max = hh
                if ww > vis_max:
                    vis_max = ww
            else:
                # text token (includes <X.X seconds>, vision_start/end, newlines)
                t_row[i] = cur
                h_row[i] = cur
                w_row[i] = cur
                cur += 1
        pos_chunks.append(torch.tensor([t_row, h_row, w_row], device=device))

    # if the sequence had no vision tokens at all, this path adds nothing
    if n_vision_tokens_seen == 0:
        return None
    if not pos_chunks:
        return None
    positions = torch.cat(pos_chunks, dim=1)
    if positions.shape[1] < total_len:
        pad = total_len - positions.shape[1]
        extra = torch.arange(pad, device=device).view(1, -1).expand(3, -1)
        positions = torch.cat([positions, extra], dim=1)
    global _MROPE_A_LOGGED
    if not _MROPE_A_LOGGED:
        import sys as _sys
        _sys.stderr.write("[MROPE-A] real patch_positions path ACTIVE (merged_tok=%d, vision_tok_seen=%d)\n" % (n_tok, n_vision_tokens_seen))
        _MROPE_A_LOGGED = True
    return positions.unsqueeze(1).contiguous()


def get_rope_index_packed(
    input_ids,
    cu_seqlens,
    image_grid_thw=None,
    video_grid_thw=None,
    spatial_merge_size=2,
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    tokens_per_second=2,
    second_per_grid_ts=None,
    patch_positions=None,
):
    """Packed-mode Qwen 3D M-RoPE position ids.

    In packed (thd) mode the microbatch holds a single [1, S] row that is the
    concatenation of many independent samples; ``cu_seqlens`` gives the segment
    boundaries. Each segment is an independent sample: its (T, H, W) positions
    must RESTART at 0 and its image/video spans must get the 2D H/W grid layout.
    Image / video grids are laid out in sequence order across ALL segments, so
    we walk segments left-to-right and consume ``image_grid_thw`` /
    ``video_grid_thw`` rows with running cursors.

    Args:
        input_ids: LongTensor [1, S] or [S] (the full packed sequence).
        cu_seqlens: 1D LongTensor [nseg + 1] of segment boundaries (cu_seqlens_q).
        image_grid_thw / video_grid_thw: [num_*, 3] grids across all segments.
        (other args identical to get_rope_index)

    Returns:
        position_ids: LongTensor [3, 1, S] (T, H, W rows), positions restart per
        segment; image/video spans carry the 2D grid layout.
    """
    # Packing requires micro-batch-size == 1 (single packed row), matching the
    # data-side guard in get_batch (`cu_lengths.shape[0] == 1`) and LoongForge.
    if input_ids.dim() == 2:
        assert input_ids.shape[0] == 1, (
            "packed 3D M-RoPE requires micro-batch-size == 1 (got batch "
            + str(input_ids.shape[0]) + ")"
        )
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    device = ids.device
    total_len = ids.shape[0]
    cu = cu_seqlens.detach().to("cpu").long().tolist()

    # ----------------------------------------------------------------------
    # Preferred path: real per-patch (t, h, w) from patch_positions.npy.
    # Qwen3.5 M-RoPE temporal is the FRAME ORDER (arange), spatial is the
    # merged (h//merge, w//merge) grid; absolute timing is carried separately
    # by the <X.X seconds> timestamp TEXT tokens (not by M-RoPE). We therefore
    # use patch_positions only to recover, per merged vision token in sequence
    # order, its (frame_order, h//merge, w//merge). Real absolute frame numbers
    # (e.g. 0,30,60) are dense-ranked to 0,1,2 per visual.
    _pp_pos = _rope_from_patch_positions(
        ids, cu, patch_positions, spatial_merge_size,
        image_token_id, video_token_id, device, total_len,
    )
    if _pp_pos is not None:
        return _pp_pos
    global _MROPE_FB_LOGGED
    if not _MROPE_FB_LOGGED:
        import sys as _sys
        _sys.stderr.write("[MROPE-A] FELL BACK to grid path (patch_positions absent/unusable)\n")
        _MROPE_FB_LOGGED = True
    # Fallback (no usable patch_positions): reconstruct from grid_thw.

    # OV2 gives EACH frame/tile of a visual its own <vision_start> token, while
    # image_grid_thw / video_grid_thw store ONE row [t, h, w] per visual (t =
    # number of frames/tiles). Qwen get_rope_index assumes 1 <vision_start> <->
    # 1 grid row <-> 1 contiguous token run, so expand every [t, h, w] row into t
    # rows of [1, h, w]: then the per-segment <vision_start> count equals the grid
    # row count and each image/video run is contiguous, restoring that invariant.
    def _expand_grid(grid, spgts):
        if grid is None or grid.numel() == 0:
            return grid, spgts
        rows, sp = [], []
        for r in range(grid.shape[0]):
            t = int(grid[r, 0].item())
            one = grid[r:r + 1].clone()
            one[0, 0] = 1
            for _ in range(max(t, 1)):
                rows.append(one)
                if spgts is not None:
                    sp.append(spgts[r:r + 1])
        g = torch.cat(rows, dim=0) if rows else grid
        s = torch.cat(sp, dim=0) if (spgts is not None and sp) else spgts
        return g, s
    image_grid_thw, _ = _expand_grid(image_grid_thw, None)
    video_grid_thw, second_per_grid_ts = _expand_grid(video_grid_thw, second_per_grid_ts)

    pos_chunks = []
    img_cursor, vid_cursor = 0, 0
    for k in range(len(cu) - 1):
        s0, s1 = cu[k], cu[k + 1]
        if s1 <= s0:
            continue
        seg = ids[s0:s1]
        # After the expansion above, each <vision_start> maps to exactly one grid row.
        vstart = torch.argwhere(seg == vision_start_token_id).squeeze(1)
        n_img, n_vid = 0, 0
        if vstart.numel() > 0:
            nxt = (vstart + 1).clamp(max=seg.shape[0] - 1)
            vtok = seg[nxt]
            n_img = int((vtok == image_token_id).sum().item())
            n_vid = int((vtok == video_token_id).sum().item())
        seg_img = None
        seg_vid = None
        seg_spgts = None
        if n_img > 0 and image_grid_thw is not None:
            seg_img = image_grid_thw[img_cursor:img_cursor + n_img]
        if n_vid > 0 and video_grid_thw is not None:
            seg_vid = video_grid_thw[vid_cursor:vid_cursor + n_vid]
            if second_per_grid_ts is not None:
                seg_spgts = second_per_grid_ts[vid_cursor:vid_cursor + n_vid]
        try:
            seg_pos, _ = get_rope_index(
                input_ids=seg.unsqueeze(0),
                image_grid_thw=seg_img,
                video_grid_thw=seg_vid,
                spatial_merge_size=spatial_merge_size,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
                tokens_per_second=tokens_per_second,
                second_per_grid_ts=seg_spgts,
                attention_mask=None,
            )
            seg_chunk = seg_pos[:, 0, :].to(device)
        except Exception as _e:
            # Safety net: a malformed segment must not kill a long run. Fall back
            # to a plain arange for this segment; cursors still advance by the
            # per-segment vision counts so later segments stay grid-aligned.
            import sys
            sys.stderr.write(
                "[MROPE] segment k=%d fell back to arange (n_img=%d n_vid=%d): %s\n"
                % (k, n_img, n_vid, repr(_e))
            )
            seg_len = int(seg.shape[0])
            seg_chunk = (
                torch.arange(seg_len, device=device).view(1, -1).expand(3, -1).contiguous()
            )
        pos_chunks.append(seg_chunk)  # [3, seg_len]
        img_cursor += n_img
        vid_cursor += n_vid

    if len(pos_chunks) == 0:
        return (
            torch.arange(total_len, device=device).view(1, 1, -1).expand(3, 1, -1).contiguous()
        )
    positions = torch.cat(pos_chunks, dim=1)  # [3, covered_len]
    # Safety: if cu_seqlens does not cover the full physical width (shouldn't
    # happen after the get_batch tail fix), pad the remainder with a local arange.
    if positions.shape[1] < total_len:
        pad_len = total_len - positions.shape[1]
        extra = torch.arange(pad_len, device=device).view(1, -1).expand(3, -1)
        positions = torch.cat([positions, extra], dim=1)
    return positions.unsqueeze(1).contiguous()  # [3, 1, S]


class LlavaOnevision2(MegatronModule):
    """
    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
            language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for positional
            embedding.
        vision_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        adapter_config (TransformerConfig): Config for the projection from vision model outputs to language
            model inputs.
        adapter_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        adapter_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        allow_missing_adapter_checkpoint (bool): Allow vision projection weights to be missing when
            loading a checkpoint. Default False.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks.
            This is typically True for training and False for inference.
        language_position_embedding_type (str): Position embedding type to use in the language model.
            Default learned absolute.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings in the
            language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder
            (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder will live on only a subset of the pipeline stages
            (specifically, every stage after the first one).
        img_h (int): The height of each image that the ViT will see.
        img_w (int): The width of each image that the ViT will see.
        patch_dim (int): The size of each patch side.
    """

    def __init__(
        self,
        language_config,
        vision_config,
        adapter_config,
        language_layer_spec: ModuleSpec,
        vision_layer_spec: ModuleSpec,
        adapter_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        allow_missing_adapter_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = "rope",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 1000000,
        fp16_lm_cross_entropy: bool = False,
        share_embeddings_and_output_weights: bool = True,
        seq_len_interpolation_factor: float = None,
    ) -> None:
        super().__init__(config=language_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.adapter = None
        self.language_model = None

        #  define the vision model and the projection from vision model outputs to language model inputs.
        if self.add_encoder:
            spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
            self.vision_model = OneVisionEncoderModel(
                vision_config,
                vision_layer_spec,
                spatial_merge_size=spatial_merge_size,
            )

            self.adapter = Adapter(
                adapter_config,
                adapter_layer_spec,
                input_size=vision_config.hidden_size,  # input size to the adapter.
                output_size=language_config.hidden_size,  # output size of the adapter.
                spatial_merge_size=spatial_merge_size,
            )
            # This allows ignoring missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains pretrained
            # vision and language models but not the projection from vision model outputs to language model inputs.
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [f"adapter.{name}" for name in self.adapter.state_dict().keys()]
                self.adapter.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

        # # This attribute is needed to check if an all-reduce is required
        # # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        if self.add_decoder:
            self.language_model = QwenModel(
                config=language_config,
                transformer_layer_spec=language_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                scatter_embedding_sequence_parallel=False,
            )
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """set input tensor"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_adapter: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_adapter (bool): Freeze the vision adapter module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_adapter and self.adapter is not None:
            modules.append(self.adapter)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _vision_chunk_patches(self, images, image_grid_thw, patch_positions):
        """Patch budget per vision chunk (VISION_CHUNK_PATCHES), or 0 to run unchunked.

        Chunking is only used when the flat patch tensor is exactly the concatenation
        of the images described by image_grid_thw (and patch_positions matches it), so
        chunks can be cut at image boundaries.
        """
        max_patches = int(os.environ.get("VISION_CHUNK_PATCHES", "0"))
        if max_patches <= 0 or image_grid_thw is None or image_grid_thw.shape[0] <= 1:
            return 0
        n_patches = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum())
        if n_patches != images.shape[0]:
            return 0
        if patch_positions is not None and patch_positions.shape[0] != images.shape[0]:
            return 0
        return max_patches

    def _encode_images_chunked(self, images, image_grid_thw, patch_positions, max_patches):
        """Run vision tower + adapter over image-aligned chunks of <= max_patches patches.

        Packed bins with hundreds of images/frames (100k+ patches) crash the varlen
        vision kernels in a single call. Vision attention never crosses an image, and
        the adapter is per token, so cutting at image boundaries gives every image the
        same inputs as the unchunked call; the chunk outputs are concatenated in order.
        An image larger than max_patches gets a chunk of its own.
        """
        sizes = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        merge = self.vision_model.spatial_merge_size ** 2
        outs = []
        img_start, patch_start = 0, 0
        while img_start < len(sizes):
            img_end, n_patches = img_start, 0
            while img_end < len(sizes) and (n_patches == 0 or n_patches + sizes[img_end] <= max_patches):
                n_patches += int(sizes[img_end])
                img_end += 1
            if n_patches <= 0:  # zero-patch grid row: nothing to encode
                img_start = img_end
                continue
            emb = self.vision_model(
                images[patch_start:patch_start + n_patches],
                grid_thw=image_grid_thw[img_start:img_end],
                patch_positions=(
                    patch_positions[patch_start:patch_start + n_patches]
                    if patch_positions is not None else None
                ),
            )
            emb = self.adapter(emb)
            outs.append(emb[: n_patches // merge])  # drop per-chunk SP padding, if any
            img_start, patch_start = img_end, patch_start + n_patches
        return torch.cat(outs, dim=0)

    def _encode_local(self, images, image_grid_thw, patch_positions):
        """Vision tower + adapter on these images -> exactly n_patches // merge embeddings."""
        vision_chunk = self._vision_chunk_patches(images, image_grid_thw, patch_positions)
        if vision_chunk:
            return self._encode_images_chunked(images, image_grid_thw, patch_positions, vision_chunk)
        emb = self.adapter(self.vision_model(images, grid_thw=image_grid_thw, patch_positions=patch_positions))
        return emb[: images.shape[0] // self.vision_model.spatial_merge_size ** 2]

    def forward(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType | None = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        inference_params: InferenceParams = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        patch_positions: list[torch.tensor] | None = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen-VL model.

        Args:
            images (torch.Tensor): input image of shape [image_size / 3d_patch_size, in_channels * 3d_patch_size].
            image_grid_thw (torch.Tensor): image grid tensor of shape [num_images, 3]
            pixel_values_videos (torch.Tensor): The tensors corresponding to the input videos,
                            shape:[seq_length, num_channels * temporal_size * patch_size * patch_size]
            video_grid_thw (torch.Tensor): video grid tensor of shape [num_videos, 3]
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """

        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip image token computation if they were computed already
        # earlier for this sample.
        balance_anchor = None  # --vision-dp-balance: keeps text-only ranks in the backward all_to_all
        if use_inference_kv_cache:
            image_embeddings = None
        elif (self.add_encoder and self.training and getattr(self.config, "vision_dp_balance", False)
              and vision_balance_group() is not None):
            # --vision-dp-balance: vision tower + adapter work spread evenly over the DP group. Every DP
            # rank takes this branch for every microbatch, with or without images.
            _vm = self.vision_model
            _imgs = images if images is not None else torch.zeros(
                (0, _vm.patch_embed.proj.weight.shape[1]), dtype=next(_vm.parameters()).dtype,
                device=next(_vm.parameters()).device)
            _grid = image_grid_thw if images is not None else torch.zeros((0, 3), dtype=torch.int64, device=_imgs.device)
            _pp = patch_positions if images is not None else None
            image_embeddings = vision_balance.balance(
                _imgs, _grid, _pp, self._encode_local,
                merge=_vm.spatial_merge_size ** 2, out_dim=self.config.hidden_size,
                group=vision_balance_group(),
            )
            if images is None:
                balance_anchor, image_embeddings = image_embeddings, None
            else:
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                if image_embeddings.shape[0] != n_image_tokens:
                    raise ValueError(f"Image features {image_embeddings.shape[0]} != image tokens {n_image_tokens}")
        elif self.add_encoder:
            if images is not None:
                _prof = __import__('os').environ.get('PROF_TAGS') == '1'
                if _prof:
                    torch.cuda.synchronize(); _vt0 = __import__('time').perf_counter()
                if __import__('os').environ.get('VISION_RECOMPUTE') == '1' and self.training:
                    from megatron.core import tensor_parallel as _mpu_tp
                    def _vis_ckpt_fwd(_imgs):
                        return self.vision_model(_imgs, grid_thw=image_grid_thw, patch_positions=patch_positions)
                    image_embeddings = _mpu_tp.checkpoint(_vis_ckpt_fwd, False, images)  # VISION_RECOMPUTE: main_grad-aware megatron ckpt
                    image_embeddings = self.adapter(image_embeddings)
                elif (vision_chunk := self._vision_chunk_patches(images, image_grid_thw, patch_positions)):
                    # VISION_CHUNK_PATCHES: vision tower + adapter in image-aligned chunks.
                    image_embeddings = self._encode_images_chunked(
                        images, image_grid_thw, patch_positions, vision_chunk
                    )
                else:
                    image_embeddings = self.vision_model(
                        images, grid_thw=image_grid_thw, patch_positions=patch_positions
                    )  # [img_len, h_vision]
                    image_embeddings = self.adapter(image_embeddings)
                if _prof:
                    torch.cuda.synchronize()
                    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                        print('[PROF_TAGS] vision+adapter fwd: %.1f ms' % ((__import__('time').perf_counter() - _vt0) * 1000), flush=True)

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeddings.shape[0]
                if n_image_features > n_image_tokens:
                    # Trim any extra embeddings produced by ViT SP-padding tokens that
                    # passed through the adapter.  The language sequence has a fixed
                    # number of image-token slots (n_image_tokens); any embeddings
                    # beyond that count must not enter the fusion step.
                    logging.getLogger(__name__).warning(
                        "Trimming %d extra image embedding(s) (n_image_features=%d, "
                        "n_image_tokens=%d); likely caused by ViT SP-padding tokens "
                        "passing through the adapter.",
                        n_image_features - n_image_tokens,
                        n_image_features,
                        n_image_tokens,
                    )
                    image_embeddings = image_embeddings[:n_image_tokens]
                elif n_image_features < n_image_tokens:
                    raise ValueError(f"Image features {n_image_features} < image tokens {n_image_tokens}")

                # If running inference, the language model KV cache will be updated for image token positions.
                # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
                if inference_params is not None:
                    inference_params.key_value_memory_dict["image_tokens_count"] = image_embeddings.shape[0]
            if pixel_values_videos is not None:
                raise NotImplementedError(
                    "In LLaVA-OneVision2, video and images are treated as the same input type. "
                    "Please use the 'images' argument for both."
                )

        else:
            vision_embeddings = self.encoder_hidden_state.squeeze(0) if self.encoder_hidden_state is not None else None

        if not self.add_decoder:
            # p2p_communicate_shapes requires dim=3
            vision_embeddings = make_viewless_tensor(
                inp=vision_embeddings.unsqueeze(0), requires_grad=True, keep_graph=True
            )
            return vision_embeddings

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None
            )  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already
            # earlier for this sample.
            if use_inference_kv_cache or (images is None and pixel_values_videos is None):
                combined_embeddings = language_embeddings
                if balance_anchor is not None:  # numerically a no-op; keeps this rank in the backward all_to_all
                    combined_embeddings = combined_embeddings + 0.0 * balance_anchor.sum().to(combined_embeddings.dtype)
                if (
                    self.training
                    and self.add_encoder
                    and images is None
                    and not use_inference_kv_cache
                    and __import__('os').environ.get('DUMMY_VISION') == '1'
                ):
                    # DUMMY_VISION: text-only microbatch -- run a minimal dummy image
                    # through vision_model + adapter so this DP rank still produces
                    # grads for every vision/adapter param (keeps DDP overlap grad
                    # allreduce consistent across DP ranks). Numerically a no-op.
                    _vm = self.vision_model
                    _p = _vm.patch_size
                    _sms = _vm.spatial_merge_size
                    _in_ch = getattr(_vm.patch_embed, 'in_channels', 3)
                    _ref = next(_vm.parameters())
                    _dummy_imgs = torch.zeros(
                        _sms * _sms, _in_ch * _p * _p, dtype=_ref.dtype, device=_ref.device
                    )
                    _dummy_thw = torch.tensor(
                        [[1, _sms, _sms]], dtype=torch.int64, device=_ref.device
                    )
                    _dummy_out = self.adapter(
                        _vm(_dummy_imgs, grid_thw=_dummy_thw, patch_positions=None)
                    )
                    combined_embeddings = combined_embeddings + 0.0 * _dummy_out.sum().to(
                        combined_embeddings.dtype
                    )
                    if not getattr(self, '_dummy_vision_logged', False):
                        self._dummy_vision_logged = True
                        print('[DUMMY_VISION] dummy vision forward active for text-only microbatch', flush=True)
            else:
                if images is not None and self.config.image_token_id in input_ids:
                    image_token_id = self.config.image_token_id
                    images_mask = (
                        (input_ids == image_token_id)
                        .transpose(0, 1)
                        .unsqueeze(-1)
                        .expand_as(language_embeddings)
                        .to(language_embeddings.device)
                    )
                    image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                    combined_embeddings = language_embeddings.masked_scatter(images_mask, image_embeddings)

            if self.config.context_parallel_size > 1:
                combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings)

            if self.config.sequence_parallel:
                # Ensure sequence length is divisible by TP world size before scattering.
                seq_len = combined_embeddings.size(0)
                tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
                remainder = seq_len % tp_world_size
                if remainder != 0:
                    pad = tp_world_size - remainder
                    pad_shape = (pad,) + combined_embeddings.shape[1:]
                    pad_tensor = combined_embeddings.new_zeros(pad_shape)
                    combined_embeddings = torch.cat((combined_embeddings, pad_tensor), dim=0)
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings
                )
        
        else:
            combined_embeddings = None
            input_tensor = self.language_model.decoder.input_tensor

        # Qwen3.5 interleaved 3D M-RoPE: compute per-sample (T, H, W) position ids from
        # the image/video grids + input_ids and stash them on the language model, so
        # QwenModel builds the correct rotary embedding (replacing the text-path
        # arange placeholder). Text-only samples fall back to arange (handled in
        # QwenModel). SP now runs get_rope_index too: input_ids here is full-length
        # and unsharded (SP scatter touches only combined_embeddings), so it yields
        # unpadded full-length positions and QwenModel pads them up to the SP-gathered
        # length. Packed uses get_rope_index_packed; CP still uses the placeholder +
        # qwen35_rope CP-slice.
        if self.add_decoder and getattr(self.language_model, "use_qwen35_mrope", False):
            _mrope_pos = None
            _has_vision = (image_grid_thw is not None) or (video_grid_thw is not None)
            _is_packed = (
                packed_seq_params is not None
                and getattr(packed_seq_params, "qkv_format", None) == "thd"
            )
            # Non-packed / non-CP "simple" full-length path (now includes SP).
            _simple = (
                not _is_packed
                and getattr(self.config, "context_parallel_size", 1) == 1
            )
            if _has_vision and _is_packed and input_ids is not None:
                # Packed multimodal: per-segment 3D positions (restart at 0 each
                # segment; image/video spans get the 2D H/W grid layout). Full
                # sequence length, matching the placeholder convention QwenModel
                # / Qwen35RotaryEmbedding expect (SP gathers, CP-slice skipped in
                # packed rope).
                _mrope_pos = get_rope_index_packed(
                    input_ids=input_ids,
                    cu_seqlens=packed_seq_params.cu_seqlens_q,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    spatial_merge_size=getattr(self.config, "spatial_merge_size", 2),
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=getattr(self.config, "vision_start_token_id", 151652),
                    tokens_per_second=getattr(self.config, "tokens_per_second", 2),
                    patch_positions=patch_positions,
                )
                _mrope_pos = _mrope_pos.to(input_ids.device)
            elif _has_vision and _simple and input_ids is not None:
                _mrope_pos, _ = get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    spatial_merge_size=getattr(self.config, "spatial_merge_size", 2),
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=getattr(self.config, "vision_start_token_id", 151652),
                    tokens_per_second=getattr(self.config, "tokens_per_second", 2),
                    attention_mask=None,
                )
                _mrope_pos = _mrope_pos.to(input_ids.device)
            if __import__('os').environ.get('PP2_DIAG') == '1':
                _ps = __import__('megatron.core', fromlist=['parallel_state']).parallel_state
                print(f"[PP2_DIAG vlm] pp_rank={_ps.get_pipeline_model_parallel_rank()} "
                      f"add_enc={self.add_encoder} add_dec={self.add_decoder} "
                      f"input_ids={input_ids is not None} img_grid={image_grid_thw is not None} "
                      f"vid_grid={video_grid_thw is not None} mrope_pos={_mrope_pos is not None}", flush=True)
            self.language_model._mrope_position_ids = _mrope_pos

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=combined_embeddings,
            labels=labels,
            # rotary_pos_emb=rotary_pos_emb,
            rotary_pos_emb=None,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs={},
        )

        return output

    def forward_debug(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        attn_mask_type: AttnMaskType | None = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        inference_params: InferenceParams = None,
    ) -> dict:
        """Forward pass with debug outputs for consistency checking.

        This method captures intermediate outputs at each stage of the forward pass
        to enable layer-by-layer comparison with HuggingFace implementation.

        Args:
            images (torch.Tensor): Input image patches of shape [total_patches, patch_dim].
            image_grid_thw (torch.Tensor): Grid dimensions [batch_size, 3] with (T, H, W).
            input_ids (torch.Tensor): Optional input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): Optional input text position ids.
            attention_mask (torch.Tensor): Optional attention mask.
            attn_mask_type: Optional attention mask type.
            labels (torch.Tensor): Optional target text labels.
            packed_seq_params: Optional packed sequence parameters.
            inference_params: Optional inference parameters.

        Returns:
            dict: Dictionary containing intermediate outputs:
                - "before_adapter": Vision encoder output before adapter/merger
                - "after_adapter": Vision encoder output after adapter/merger
                - "after_merger": Alias for "after_adapter"
                - "vision_encoder_debug": Debug output from vision encoder
                - "language_embeddings": Language model input embeddings (if input_ids provided)
                - "combined_embeddings": Combined vision+language embeddings (if input_ids provided)
                - "logits": Model output logits (if input_ids provided)
        """
        output = {}

        # Store inputs for consistency checking
        output["input_images"] = images.clone()
        output["input_grid_thw"] = image_grid_thw.clone()

        if self.add_encoder and images is not None:
            # Get vision encoder debug output
            vision_debug = self.vision_model.forward_debug(images, grid_thw=image_grid_thw)
            output["vision_encoder_debug"] = vision_debug

            # Vision encoder output (before adapter)
            vision_output = vision_debug.get("before_adapter")
            if vision_output is None:
                # Fallback: run normal forward
                vision_output = self.vision_model(images, grid_thw=image_grid_thw)
            output["before_adapter"] = vision_output.clone()

            # Apply adapter/merger
            image_embeddings = self.adapter(vision_output)
            output["after_adapter"] = image_embeddings.clone()
            output["after_merger"] = image_embeddings.clone()  # Alias for compatibility

        # If input_ids provided, continue with language model
        if input_ids is not None and self.add_decoder and self.pre_process:
            # Get language embeddings
            language_embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None)
            output["language_embeddings"] = language_embeddings.clone()

            # Combine vision and language embeddings
            if images is not None and self.config.image_token_id in input_ids:
                image_token_id = self.config.image_token_id
                images_mask = (
                    (input_ids == image_token_id)
                    .transpose(0, 1)
                    .unsqueeze(-1)
                    .expand_as(language_embeddings)
                    .to(language_embeddings.device)
                )
                image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                combined_embeddings = language_embeddings.masked_scatter(images_mask, image_embeddings)
            else:
                combined_embeddings = language_embeddings

            output["combined_embeddings"] = combined_embeddings.clone()

            # Run through language model
            logits = self.language_model(
                input_ids=None,
                position_ids=None,
                attention_mask=attention_mask,
                attn_mask_type=attn_mask_type,
                decoder_input=combined_embeddings,
                labels=labels,
                rotary_pos_emb=None,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs={},
            )
            output["logits"] = logits

        return output


def _load_state_dict_hook_ignore_param_names(
    param_names: list[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and
    language model weights but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the
            missing and unexpected keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
