"""LlavaOnevision2 multi-modal processor.

Combines:
  - ``Qwen2VLImageProcessor[Fast]``    (existing in checkpoint preprocessor_config)
  - ``LlavaOnevision2VideoProcessor``  (this checkpoint, video_processing_*)
  - ``AutoTokenizer``                  (existing tokenizer.json)
  - ``chat_template.jinja``            (existing, emits <|video_pad|>)

Public API:
    proc = LlavaOnevision2Processor(image_processor, tokenizer, video_processor)
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=[text], videos=[mp4_or_frames], return_tensors="pt")
    out = model.generate(**inputs)

Design choices:
  - Video path is "in-processor, transformed to multi-image + per-frame
    timestamps" — model.forward sees the image path only.
  - The chat_template's <|vision_start|><|video_pad|><|vision_end|> placeholder
    is rewritten in __call__ to per-frame blocks:
        <X.X seconds><|vision_start|><|image_pad|>*n<|vision_end|>\n
  - We DO NOT emit `second_per_grid_ts`; see plan §0.5.
  - Backward-compatible: `images=...` / pure-text usage matches the existing
    Qwen2_5_VLProcessor output.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

import torch

# Special-token strings used by the checkpoint's tokenizer / chat_template.
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"


def _format_seconds_tag(seconds: float) -> str:
    """Match training format: ``<X.X seconds>`` (one decimal place)."""
    return f"<{float(seconds):.1f} seconds>"


def _expand_video_block_for_frames(
    n_per_frame: int,
    frame_seconds: Sequence[float],
) -> str:
    """Build the per-frame expanded text that replaces a single
    ``<|vision_start|><|video_pad|><|vision_end|>`` block.

    Output (one block per frame, newline-separated):
        ``<X.X seconds><|vision_start|><|image_pad|>*n_per_frame<|vision_end|>\\n``
    """
    parts: List[str] = []
    for sec in frame_seconds:
        parts.append(_format_seconds_tag(sec))
        parts.append(VISION_START)
        parts.append(IMAGE_PAD * n_per_frame)
        parts.append(VISION_END)
    return "".join(parts)


class LlavaOnevision2Processor:
    """Native multi-modal processor for LlavaOnevision2.

    NOTE: We deliberately do NOT inherit ``transformers.ProcessorMixin``.
    This class is registered via ``auto_map`` so
    ``AutoProcessor.from_pretrained(..., trust_remote_code=True)`` returns it.
    """

    attributes = ["image_processor", "video_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template: Optional[str] = None,
        codec_config: Optional[dict] = None,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor

        # Inherit chat_template from the tokenizer if not given (matches Qwen2_5_VLProcessor).
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        self.chat_template = chat_template

        # Cache the merge size from image_processor for token-count math.
        self.spatial_merge_size = int(getattr(image_processor, "merge_size", 2) if image_processor is not None else 2)

        # Codec config defaults (overridden per-call via ``codec_config=``).
        self._codec_config_defaults: dict = dict(codec_config or {})

    # ------------------------------------------------------------------ utils

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """No-op stub so ``AutoProcessor.from_pretrained(..., trust_remote_code=True)``
        can call this on the dynamically-loaded class without erroring.
        Real ``ProcessorMixin`` uses this to remember the auto-class for
        ``push_to_hub``; we don't need that for inference-only use."""
        cls._auto_class = auto_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Convenience builder mirroring HF's ``from_pretrained`` pattern."""
        from transformers import AutoTokenizer, Qwen2VLImageProcessor

        # Drop kwargs that AutoProcessor injects but downstream constructors
        # don't accept (e.g. _from_auto / trust_remote_code propagation).
        kwargs.pop("_from_auto", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        codec_config_override = kwargs.pop("codec_config", None)

        # Use the SLOW Qwen2VLImageProcessor: the Fast variant has small
        # normalization rounding differences that change pixel_values bit-for-bit.
        image_processor = Qwen2VLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Use the bundled VideoProcessor. Try a relative import first (when
        # this module is loaded as part of a remote_code package), and fall
        # back to a top-level import (when loaded as a standalone file via
        # ``get_class_from_dynamic_module``, which places sibling files on
        # ``sys.path``).
        try:
            from .video_processing_llava_onevision2 import LlavaOnevision2VideoProcessor
        except ImportError:
            from video_processing_llava_onevision2 import LlavaOnevision2VideoProcessor

        video_processor = LlavaOnevision2VideoProcessor(
            image_processor=image_processor,
            min_pixels=getattr(image_processor, "min_pixels", 256 * 28 * 28),
            max_pixels=getattr(image_processor, "max_pixels", 1605632),
            patch_size=getattr(image_processor, "patch_size", 14),
            spatial_merge_size=getattr(image_processor, "merge_size", 2),
        )

        # Codec defaults are read from preprocessor_config.json's "codec" field.
        # We load the JSON directly because Qwen2VLImageProcessor.from_pretrained
        # may not preserve unknown top-level keys as attributes.
        if codec_config_override is not None:
            codec_defaults = codec_config_override
        else:
            codec_defaults = {}
            try:
                import json as _json
                import os as _os

                # Try local file first (downloaded snapshot), then HF Hub.
                cfg_path = _os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
                if _os.path.isfile(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as _f:
                        codec_defaults = _json.load(_f).get("codec", {}) or {}
                else:
                    from huggingface_hub import hf_hub_download

                    cfg_path = hf_hub_download(pretrained_model_name_or_path, "preprocessor_config.json")
                    with open(cfg_path, "r", encoding="utf-8") as _f:
                        codec_defaults = _json.load(_f).get("codec", {}) or {}
            except Exception:
                codec_defaults = {}
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            codec_config=codec_defaults,
        )

    # ------------------------------------------------------------- chat helpers

    def apply_chat_template(self, messages, **kwargs):
        """Delegate to the tokenizer (which already has ``chat_template``)."""
        if self.chat_template and "chat_template" not in kwargs:
            kwargs["chat_template"] = self.chat_template
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ----------------------------------------------------------- main __call__

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images=None,
        videos=None,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = False,
        num_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        # Codec video backend (in-processor codec preprocessing). When
        # ``video_backend="codec"`` and ``videos`` is set, the codec pipeline
        # (cv-preinfer) replaces the frame-sampling VideoProcessor. The codec
        # canvas pixel budget is taken from ``max_pixels`` so the user only
        # configures one pixel knob.
        video_backend: str = "frames",
        max_pixels: Optional[int] = None,
        codec_config: Optional[dict] = None,
        **kwargs,
    ):
        """Process an aligned (text, images, videos) batch.

        Behaviour:
          * ``videos is not None``: run the VideoProcessor, rewrite each
            ``<|video_pad|>`` block in ``text`` to per-frame ``<X.X seconds>``
            blocks, then alias the video patches as ``pixel_values`` /
            ``image_grid_thw`` so the model's image path consumes them.
          * ``images is not None``: passed through to the underlying
            ``image_processor``. (May coexist with ``videos``; expansion order
            in the prompt is determined by the chat_template / placeholders.)
          * Pure text: tokenize and return.

        Per-call frame-sampling overrides (apply only to ``videos`` path; do
        not mutate the underlying VideoProcessor's defaults):
          * ``num_frames``  : force exactly N frames per video
            (alias of ``fixed_num_frames``).
          * ``max_frames``  : cap on auto-selected frame count (long videos).
          * ``target_fps``  : sample at this FPS (capped by ``max_frames``).

        Returns a ``BatchFeature`` with at minimum ``input_ids`` and
        ``attention_mask``; plus ``pixel_values`` / ``image_grid_thw`` /
        ``patch_positions`` when visuals are present.
        """
        if text is None:
            raise ValueError("`text` is required.")
        if isinstance(text, str):
            text = [text]
        text = list(text)

        out: dict = {}

        # ---------------- CODEC VIDEO BACKEND ----------------
        # Codec path: replaces the frame-sampling VideoProcessor entirely.
        # Each video -> N canvases + src_patch_position; we rewrite the
        # <|vision_start|>...<|vision_end|> span in `text` based on the codec
        # patch_positions (one canvas worth of <|image_pad|>s per timestamp).
        if videos is not None and str(video_backend).lower() == "codec":
            try:
                from .codec_video_processing_llava_onevision2 import (
                    CodecConfig,
                    process_codec_video,
                    drop_padding_canvases,
                    codec_positions_for_processor,
                    rewrite_text_with_codec_positions,
                    codec_image_processor_outputs,
                )
            except ImportError:
                raise ImportError(
                    "Codec video backend requested, but codec_video_processing_llava_onevision2.py "
                    "is not packaged in this HF directory. Use video_backend='native' or add the codec file."
                )

            # Normalise to list[video_url].
            if isinstance(videos, str):
                videos_list = [videos]
            else:
                videos_list = list(videos)

            # Build effective codec config: defaults < class-level < per-call.
            cfg_kwargs = dict(self._codec_config_defaults)
            if codec_config:
                cfg_kwargs.update(codec_config)
            # Unify pixel budget with image_processor.
            effective_max_pixels = int(
                max_pixels
                if max_pixels is not None
                else cfg_kwargs.get("max_pixels", getattr(self.image_processor, "max_pixels", 150000))
            )
            cfg_kwargs["max_pixels"] = effective_max_pixels
            cfg = CodecConfig(**cfg_kwargs)

            all_pixel_values, all_grid_thw, all_patch_positions = [], [], []
            rewritten_texts = list(text)
            if len(rewritten_texts) != len(videos_list):
                if len(rewritten_texts) == 1 and len(videos_list) >= 1:
                    rewritten_texts = rewritten_texts * len(videos_list)
                else:
                    raise ValueError(
                        f"codec video backend: got {len(rewritten_texts)} texts but {len(videos_list)} videos"
                    )

            for idx, video_url in enumerate(videos_list):
                payload = process_codec_video(video_url, cfg)
                imgs, src_positions, _ = drop_padding_canvases(payload["images"], payload["src_positions"])
                if not imgs:
                    raise RuntimeError(f"codec produced no usable canvases for {video_url}")
                image_data = codec_image_processor_outputs(self.image_processor, imgs, max_pixels=effective_max_pixels)
                image_grid_thw = image_data["image_grid_thw"]
                patch_positions = codec_positions_for_processor(
                    src_positions,
                    image_grid_thw,
                    device=image_grid_thw.device,
                )
                rewritten_texts[idx] = rewrite_text_with_codec_positions(
                    rewritten_texts[idx],
                    patch_positions,
                    fps=float(payload["fps"]),
                    decimals=1,
                )
                all_pixel_values.append(image_data["pixel_values"])
                all_grid_thw.append(image_grid_thw)
                all_patch_positions.append(patch_positions)

            out["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            out["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)
            out["patch_positions"] = torch.cat(all_patch_positions, dim=0)
            text = rewritten_texts
            # Codec branch handled the video. Suppress the frame-sampling block below.
            videos = None

        # ---------------- VIDEO PATH ----------------
        # Process videos first so we can rewrite their placeholders into the
        # text before tokenization.
        video_outputs = None
        if videos is not None:
            if self.video_processor is None:
                raise ValueError("videos passed but no video_processor configured.")
            # Normalise to a list of videos.
            if isinstance(videos, (str,)):
                videos_list = [videos]
            elif isinstance(videos, list) and len(videos) > 0 and not isinstance(videos[0], (list, str)):
                # list[PIL]/[np.ndarray] = single video
                videos_list = [videos]
            else:
                videos_list = list(videos)

            # Per-call sampling overrides: temporarily swap the
            # VideoProcessor's attributes, then restore. Lets users do
            #     processor(videos=[mp4], num_frames=8)
            # without mutating processor.video_processor.
            vp = self.video_processor
            saved = (vp.fixed_num_frames, vp.max_frames, vp.target_fps)
            try:
                if num_frames is not None:
                    vp.fixed_num_frames = int(num_frames)
                if max_frames is not None:
                    vp.max_frames = int(max_frames)
                if target_fps is not None:
                    vp.target_fps = float(target_fps)
                video_outputs = vp(videos=videos_list, return_tensors="pt")
            finally:
                vp.fixed_num_frames, vp.max_frames, vp.target_fps = saved

            # Rewrite each <|video_pad|> in `text` into per-frame blocks.
            video_grid_thw = video_outputs["video_grid_thw"]  # [num_videos, 3]
            frame_timestamps = video_outputs["frame_timestamps"]
            sms = self.spatial_merge_size

            # We iterate placeholders globally across all texts (matching how
            # Qwen2_5_VLProcessor sources `image_grid_thw` rows).
            video_idx = 0

            def _rewrite_one_text(s: str) -> str:
                nonlocal video_idx
                pattern = re.compile(
                    re.escape(VISION_START) + r"\s*" + re.escape(VIDEO_PAD) + r"\s*" + re.escape(VISION_END)
                )

                def _sub(_match):
                    nonlocal video_idx
                    if video_idx >= video_grid_thw.shape[0]:
                        raise ValueError("More <|video_pad|> placeholders in text than videos provided.")
                    T_eff = int(video_grid_thw[video_idx, 0].item())
                    H_p = int(video_grid_thw[video_idx, 1].item())
                    W_p = int(video_grid_thw[video_idx, 2].item())
                    n_per_frame = (H_p * W_p) // (sms * sms)
                    frame_seconds = frame_timestamps[video_idx]
                    if len(frame_seconds) != T_eff:
                        # Defensive: pad/truncate so the count matches the grid.
                        if len(frame_seconds) < T_eff:
                            frame_seconds = list(frame_seconds) + [frame_seconds[-1] if frame_seconds else 0.0] * (
                                T_eff - len(frame_seconds)
                            )
                        else:
                            frame_seconds = list(frame_seconds[:T_eff])
                    expanded = _expand_video_block_for_frames(n_per_frame, frame_seconds)
                    video_idx += 1
                    # Strip trailing newline so we don't double-newline existing prompts.
                    return expanded.rstrip("\n")

                return pattern.sub(_sub, s)

            text = [_rewrite_one_text(s) for s in text]

            if video_idx != video_grid_thw.shape[0]:
                raise ValueError(
                    f"Provided {video_grid_thw.shape[0]} videos but only "
                    f"{video_idx} <|video_pad|> placeholders were found in text."
                )

            # Alias video tensors into the image path (NEW model only consumes the image path).
            # Option 1 (multi-image semantics, training-aligned): expand each
            # video_grid_thw row [T, H, W] into T rows of [1, H, W]. The
            # pixel_values rows are already laid out frame-by-frame (T*H*W per
            # video, with temporal_patch_size=1), so this row-expansion of
            # image_grid_thw is the only adjustment needed for the model's
            # forward to treat each frame as a separate image (matching the
            # multi-image inference path).
            out["pixel_values"] = video_outputs["pixel_values_videos"]
            vgthw = video_outputs["video_grid_thw"]
            expanded_rows = []
            for row in vgthw:
                T_v, H_v, W_v = int(row[0]), int(row[1]), int(row[2])
                expanded_rows.extend([[1, H_v, W_v]] * T_v)
            out["image_grid_thw"] = torch.tensor(expanded_rows, dtype=vgthw.dtype)
            out["patch_positions"] = video_outputs["patch_positions"]

        # ---------------- IMAGE PATH ----------------
        if images is not None:
            if self.image_processor is None:
                raise ValueError("images passed but no image_processor configured.")
            image_outputs = self.image_processor(images=images, return_tensors="pt")
            image_grid_thw = image_outputs["image_grid_thw"]

            # Expand each <|image_pad|> placeholder to the number of merged tokens.
            sms = self.spatial_merge_size
            merge_factor = sms * sms
            image_token_counts = (
                (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // merge_factor
            ).tolist()
            img_idx = 0

            def _expand_image_pads(s: str) -> str:
                nonlocal img_idx
                while IMAGE_PAD in s:
                    if img_idx >= len(image_token_counts):
                        break
                    n = int(image_token_counts[img_idx])
                    s = s.replace(IMAGE_PAD, "<|placeholder|>" * n, 1)
                    img_idx += 1
                return s.replace("<|placeholder|>", IMAGE_PAD)

            text = [_expand_image_pads(s) for s in text]

            # If videos and images coexist, prefer concatenation of patch tensors.
            if "pixel_values" in out:
                out["pixel_values"] = torch.cat([out["pixel_values"], image_outputs["pixel_values"]], dim=0)
                out["image_grid_thw"] = torch.cat([out["image_grid_thw"], image_outputs["image_grid_thw"]], dim=0)
                # Build image patch_positions and concat.
                from .video_processing_llava_onevision2 import build_patch_positions

                image_pp = build_patch_positions(image_outputs["image_grid_thw"], spatial_merge_size=sms)
                out["patch_positions"] = torch.cat([out["patch_positions"], image_pp], dim=0)
            else:
                out["pixel_values"] = image_outputs["pixel_values"]
                out["image_grid_thw"] = image_outputs["image_grid_thw"]
                from .video_processing_llava_onevision2 import build_patch_positions

                out["patch_positions"] = build_patch_positions(image_outputs["image_grid_thw"], spatial_merge_size=sms)

        # ---------------- VIDEO PATH FINAL EXPANSION ----------------
        # When `videos` was given (and possibly without `images`), the per-frame
        # rewrite above already produced runs of <|image_pad|> that need to be
        # treated like image placeholders (one per merged token). Because the
        # rewrite directly emits ``IMAGE_PAD * n_per_frame``, the texts are
        # already in their tokenize-ready form for the video portion. So nothing
        # more to do here — fall through to tokenize.

        # ---------------- TOKENIZE ----------------
        encoding = self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "max_length",
                    "truncation",
                    "add_special_tokens",
                    "return_attention_mask",
                    "return_token_type_ids",
                )
            },
        )
        out["input_ids"] = encoding["input_ids"]
        out["attention_mask"] = encoding.get(
            "attention_mask",
            torch.ones_like(encoding["input_ids"]),
        )

        try:
            from transformers.feature_extraction_utils import BatchFeature

            return BatchFeature(data=out)
        except Exception:
            return out

    # ---------------------------------------------------------------- decoding

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


__all__ = [
    "LlavaOnevision2Processor",
    "VISION_START",
    "VISION_END",
    "IMAGE_PAD",
    "VIDEO_PAD",
]
