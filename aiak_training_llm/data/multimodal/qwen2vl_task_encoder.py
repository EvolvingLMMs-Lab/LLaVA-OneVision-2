"""Qwen2VLTaskEncoder class."""

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeVar, Union

import numpy as np
import torch
from megatron.energon import CaptioningSample, SkipSample, VQASample
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeWebdataset
from megatron.energon.flavors.webdataset import VideoData
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.task_encoder.base import stateless
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BlendDataset,
    EpochizeDataset,
    LogSampleDataset,
    ShuffleBufferDataset,
)
from megatron.energon.wrappers.repeat_dataset import RepeatDataset
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing_extensions import override

from aiak_training_llm.data.multimodal import MultiMixQASample
from aiak_training_llm.data.multimodal.length_sort_dataset import LengthPoolSortDataset
from aiak_training_llm.data.multimodal.packed_sort_dataset import PackedSeparateSortDataset
from aiak_training_llm.utils import constants, get_chat_template
from transformers import AutoProcessor

from .canvas_timestamp_schema import (
    CanvasTimestampError,
    rewrite_existing_canvas_wrappers,
    summarize_canvases,
)
from .task_encoder import ImageTaskBatchPacked, ImageTaskSample, ImageTaskSamplePacked, TaskEncoder


T = TypeVar("T")
V = TypeVar("V")
T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_raw_batch = TypeVar("T_raw_batch")
T_batch = TypeVar("T_batch")


IGNORE_INDEX = -100  # ID for labels that should be ignored.
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]
VIDEO_TOKEN_WITH_TAGS = VISION_TAGS[0] + VIDEO_TOKEN + VISION_TAGS[1]
SKIP_LOG_LIMIT = 5
SKIP_LOG_INTERVAL = 1000
_SKIP_LOG_COUNTS: dict[str, int] = {}


def skip_malformed_multimodal_sample(sample_key: str, signature: str, detail: str) -> None:
    count = _SKIP_LOG_COUNTS.get(signature, 0) + 1
    _SKIP_LOG_COUNTS[signature] = count

    if count <= SKIP_LOG_LIMIT or count % SKIP_LOG_INTERVAL == 0:
        print(f"Skipping malformed multimodal sample {sample_key}: {detail}", file=sys.stderr)
        if count == SKIP_LOG_LIMIT:
            print(
                f"Further '{signature}' skip logs will be reported every "
                f"{SKIP_LOG_INTERVAL} samples for this worker.",
                file=sys.stderr,
            )

    raise SkipSample(f"{sample_key}: {detail}")


def get_stateless(fn: Callable[..., T_sample]) -> bool:
    """Get whether a function is stateless."""
    return getattr(fn, "__stateless__", False)


def convert_positions_to_block_layout(
    positions: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2
) -> torch.Tensor:
    """
    Convert patch positions from row-major order to 2x2 block layout.

    This function reorders patch positions to match the 2x2 block arrangement
    used by the image processor. Uses index-based reordering instead of reshape.

    Args:
        positions: Patch positions in row-major order, shape [t*h*w, 3]
        t: temporal dimension
        h: height (unmerged patch count)
        w: width (unmerged patch count)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: Patch positions in 2x2 block order, same shape [t*h*w, 3]
    """
    sms = spatial_merge_size
    if sms == 1:
        return positions

    device = positions.device
    total_patches = t * h * w

    # Generate row-major indices: [0, 1, 2, ..., t*h*w-1]
    # Reshape to [t, h, w]
    indices = torch.arange(total_patches, device=device).view(t, h, w)

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # Reshape to [t, h_merged, sms, w_merged, sms]
    indices = indices.view(t, h_merged, sms, w_merged, sms)

    # Permute to [t, h_merged, w_merged, sms_h, sms_w] - 2x2 block order
    indices = indices.permute(0, 1, 3, 2, 4).contiguous()

    # Flatten to get the reordering indices
    indices = indices.view(total_patches)

    # Apply the reordering to positions
    return positions[indices]


@dataclass
class Qwen2VLImageTaskSample(ImageTaskSample):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, image_grid_thw: str, video_grid_thw=None, **kwargs):
        super().__init__(**kwargs)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskSamplePacked(ImageTaskSamplePacked):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**vars(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskBatchPacked(ImageTaskBatchPacked):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**vars(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


class Qwen2VLTaskEncoder(TaskEncoder):
    """A simple task encoder for VLMs."""

    def __init__(self, args):
        super().__init__()
        if args.training_phase in ["sft"]:
            self.chat_template = get_chat_template()
        self.processor = AutoProcessor.from_pretrained(self.args.hf_tokenizer_path, trust_remote_code=True)

        # image
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels

    def _normalize_image_backed_video_placeholders(
        self,
        messages: list[dict[str, str]],
        image: list | None,
        video: list | None,
    ) -> list[dict[str, str]]:
        """Rewrite image-backed <video> placeholders into image placeholders."""
        has_images = image is not None and len(image) > 0
        has_video = video is not None and (
            (isinstance(video, list) and len(video) > 0) or (not isinstance(video, list))
        )

        if not has_images or has_video:
            return messages

        video_placeholder_count = sum(msg.get("content", "").count(constants.Placeholder.VIDEO) for msg in messages)
        if video_placeholder_count == 0:
            return messages

        image_block = "\n".join([constants.Placeholder.IMAGE] * len(image))
        for msg in messages:
            if constants.Placeholder.VIDEO in msg.get("content", ""):
                msg["content"] = msg["content"].replace(constants.Placeholder.VIDEO, image_block)

        return messages

    def _resize_image(self, image, size_factor=28):
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=size_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((resized_width, resized_height))

        return image

    def _process(self, image, text):
        """ " Process the data to get the model's input"""
        inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0].logical_not()
        image_grid_thw = None
        pixel = []
        if image is not None:
            image_grid_thw = inputs["image_grid_thw"]  # [t,h,w]
            pixel = [inputs["pixel_values"]]  # [hw, 2*3*14*14]

        target = input_ids.clone()
        vision_start_id, img_pad_id, vision_end_id = self.tokenizer.convert_tokens_to_ids(
            [VISION_TAGS[0], IMAGE_TOKEN, VISION_TAGS[1]]
        )
        target[target == vision_start_id] = IGNORE_INDEX
        target[target == img_pad_id] = IGNORE_INDEX
        target[target == vision_end_id] = IGNORE_INDEX

        return input_ids, target, pixel, image_grid_thw, attn_mask

    def process_sft_vqa(self, context, answer, image):
        """process the data for sft vqa"""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": context}, {"role": "assistant", "content": answer}], tokenize=False
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
        if text[-1] == "\n":
            text = text[:-1]
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answer = self.tokenizer.tokenize(answer)
        target[-len(answer) - 1 : -1] = torch.tensor(answer)

        return input_ids, target, attn_mask, imgs, image_grid_thw

    def compute_frame_timestamps(self, images, patch_positions, fps):
        """Compute timestamps for unique frames from patch positions.

        Args:
            images: List of images
            patch_positions: Tensor of shape (n, 3) with (t, h, w) coordinates,
                             where t is the frame index
            fps: Frames per second

        Returns:
            List of float timestamps for each unique frame
        """
        if patch_positions is None or len(patch_positions) == 0:
            return []

        t_values = patch_positions[:, 0]
        unique_t = torch.unique(t_values)
        timestamps = [float(t.item() / fps) for t in unique_t]

        return timestamps

    def _rewrap_vision_by_frame(
        self,
        messages: list[dict],
        patch_positions: list[torch.Tensor],
        timestamp_strings: list[str],
    ) -> list[dict]:
        """Rewrite vision blocks in message text from per-canvas to per-frame grouping.

        After mm_plugin.process_messages, the message content has one vision block
        per canvas image (e.g. 20 blocks of 144 pad tokens each):
            VS PAD*144 VE \n VS PAD*144 VE \n ...

        This rewrites to per-frame blocks with timestamps:
            <t1> VS PAD*K1 VE \n <t2> VS PAD*K2 VE \n ...

        Works at the string level BEFORE tokenization, avoiding error-prone
        token-level merge/split.

        Args:
            messages: Message dicts with 'content' strings (after mm_plugin).
            patch_positions: List of tensors in block layout, each [n_patches, 3].
            timestamp_strings: List of timestamp strings, one per unique frame,
                e.g. ["<0.0 seconds>", "<0.1 seconds>", ...].

        Returns:
            Modified messages list (same objects, content strings updated in-place).
        """
        # Compute per-frame merged token counts from block-layout patch positions.
        # After block layout, every (sms*sms) consecutive patches = 1 merged token,
        # and patches with the same t are contiguous.
        sms = getattr(self.processor.image_processor, "merge_size", 2)
        merge_unit = sms * sms
        flat_positions = torch.cat(patch_positions, dim=0)
        num_merged_tokens = len(flat_positions) // merge_unit
        token_t_values = flat_positions[::merge_unit, 0].tolist()

        frame_token_counts = []
        idx = 0
        while idx < num_merged_tokens:
            current_t = token_t_values[idx]
            j = idx + 1
            while j < num_merged_tokens and token_t_values[j] == current_t:
                j += 1
            frame_token_counts.append(j - idx)
            idx = j

        # Build new vision string: for each frame, [timestamp] VS PAD*N VE \n
        new_parts = []
        for frame_i, count in enumerate(frame_token_counts):
            if frame_i < len(timestamp_strings):
                new_parts.append(timestamp_strings[frame_i])
            new_parts.append(VISION_TAGS[0])
            new_parts.append(IMAGE_TOKEN * count)
            new_parts.append(VISION_TAGS[1])
            new_parts.append("\n")
        new_vision_str = "".join(new_parts)

        # Find and replace the entire vision region in the message content.
        # Original: VS...VE\nVS...VE\n...VS...VE\n\nDescribe...
        # We replace from the first VS through the last VE + one \n,
        # and our new_vision_str already ends with \n for each frame.
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or VISION_TAGS[0] not in content:
                continue

            first_vs = content.find(VISION_TAGS[0])
            last_ve_end = content.rfind(VISION_TAGS[1]) + len(VISION_TAGS[1])
            # Skip one \n after the last VE (our new string already provides it)
            tail_start = last_ve_end
            if tail_start < len(content) and content[tail_start] == "\n":
                tail_start += 1

            msg["content"] = content[:first_vs] + new_vision_str + content[tail_start:]
            break  # only process the first message with vision content
        return messages

    def process_sft_qa(
        self, messages: list, system: str, raw_video: list, raw_image: list, raw_patch_positions: list, **kwargs
    ):
        """process the data for sft qa"""
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        serializer_image_grid_thw = None
        pixel_values_images = []
        video = []
        image = []
        patch_positions = []

        has_image_inputs = raw_image is not None and len(raw_image) > 0
        if has_image_inputs:
            image = raw_image

        if raw_patch_positions is not None:
            try:
                for item in raw_patch_positions:
                    if item is not None:
                        patch_positions.append(torch.tensor(item, dtype=torch.int64))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CanvasTimestampError(
                    "raw patch positions cannot be converted to tensors"
                ) from exc

        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image,
            video if raw_video is not None else [],
            self.processor,
        )

        if has_image_inputs:
            image_grid_thw = mm_inputs.get("image_grid_thw")
            assert image_grid_thw is not None, (
                "Missing `image_grid_thw` in multimodal inputs:\n"
                f"- num_raw_image: {len(raw_image)}\n"
                f"- mm_input_keys: {list(mm_inputs.keys())}\n"
                f"- has_image_placeholder: {any('<image>' in msg.get('content', '') for msg in messages)}"
            )
            # Keep the processor's per-canvas rows. The legacy path below may
            # collapse them into one temporal grid for model position IDs.
            serializer_image_grid_thw = image_grid_thw.clone()
            # get image patch num:
            num_patches = 0
            for img_idx in range(len(image_grid_thw)):
                t_val, h_val, w_val = image_grid_thw[img_idx].tolist()
                num_patches += h_val * w_val
            pixel_values = mm_inputs.get("pixel_values")
            assert pixel_values is not None, (
                "Missing `pixel_values` in multimodal inputs:\n"
                f"- num_raw_image: {len(raw_image)}\n"
                f"- mm_input_keys: {list(mm_inputs.keys())}"
            )
            pixel_values_images = [pixel_values]
            if len(patch_positions) == 0:
                # Generate default patch_positions from image_grid_thw, with t=0 for all patches
                # image_grid_thw: [num_images, 3], each row is (t, h, w)
                for img_idx in range(len(image_grid_thw)):
                    t_val, h_val, w_val = image_grid_thw[img_idx].tolist()
                    cur_num_patches = t_val * h_val * w_val
                    # Generate (t, h, w) coordinates in row-major order, t is fixed to 0
                    h_coords = torch.arange(h_val, dtype=torch.int64).repeat_interleave(w_val).repeat(t_val)
                    w_coords = torch.arange(w_val, dtype=torch.int64).repeat(h_val).repeat(t_val)
                    t_coords = torch.zeros(cur_num_patches, dtype=torch.int64)
                    # Stack into [num_patches, 3] tensor
                    img_patch_positions = torch.stack([t_coords, h_coords, w_coords], dim=1)
                    # Apply block layout conversion to match pixel_values arrangement
                    img_patch_positions = convert_positions_to_block_layout(
                        img_patch_positions, t_val, h_val, w_val,
                        spatial_merge_size=getattr(self.processor.image_processor, "merge_size", 2),
                    )
                    patch_positions.append(img_patch_positions)
            else:
                image_grid_thw = torch.tensor([[len(image_grid_thw), image_grid_thw[0][1], image_grid_thw[0][2]]])
                # Apply block layout conversion for temporal contiguity after spatial merge
                flat_positions = torch.cat(patch_positions, dim=0)
                t_v, h_v, w_v = int(image_grid_thw[0][0]), int(image_grid_thw[0][1]), int(image_grid_thw[0][2])
                flat_positions = convert_positions_to_block_layout(
                    flat_positions, t_v, h_v, w_v,
                    spatial_merge_size=getattr(self.processor.image_processor, "merge_size", 2),
                )
                patch_positions = [flat_positions]
            patch_positions_sum = sum(len(p) for p in patch_positions)
            assert num_patches == patch_positions_sum, (
                "num_patches mismatch:\n"
                f"- num_patches: {num_patches}\n"
                f"- patch_positions_sum: {patch_positions_sum}\n"
                f"- len(image_grid_thw): {len(image_grid_thw)}\n"
                f"- patch_positions len: {len(patch_positions)}\n"
                f"- image_grid_thw[-1]: {image_grid_thw[-1]}\n"
                f"- image_sizes: {[img.size for img in image]}"
            )

        raw_timed_positions = (
            [item for item in raw_patch_positions if item is not None]
            if raw_patch_positions is not None
            else []
        )
        has_explicit_fps = kwargs.get("fps") is not None
        if serializer_image_grid_thw is not None and has_explicit_fps:
            if not raw_timed_positions:
                raise CanvasTimestampError(
                    "timed canvas sample has fps but no explicit raw patch positions"
                )
            fps = kwargs["fps"]
            if isinstance(fps, list):
                if len(fps) != 1:
                    raise CanvasTimestampError(
                        f"timed canvas sample requires exactly one fps, got {fps!r}"
                    )
                fps = fps[0]
            try:
                flat_raw = np.concatenate(
                    [np.asarray(item) for item in raw_timed_positions],
                    axis=0,
                )
            except ValueError as exc:
                raise CanvasTimestampError(
                    "raw patch-position arrays cannot be concatenated"
                ) from exc
            summaries = summarize_canvases(
                flat_raw,
                serializer_image_grid_thw,
                fps=fps,
                spatial_merge_size=getattr(self.processor.image_processor, "merge_size", 2),
            )
            vision_messages = [
                message
                for message in messages
                if isinstance(message.get("content", ""), str)
                and VISION_TAGS[0] in message["content"]
            ]
            if len(vision_messages) != 1:
                raise CanvasTimestampError(
                    f"timed canvas sample requires one unambiguous vision message, got {len(vision_messages)}"
                )
            vision_messages[0]["content"], _ = rewrite_existing_canvas_wrappers(
                vision_messages[0]["content"],
                summaries,
                decimals=kwargs.get("timestamp_decimal", 1),
            )
        else:
            # Frozen fallback for samples without explicit per-canvas timing.
            timestamp_strings = None
            if kwargs is not None and "fps" in kwargs and len(patch_positions) > 0:
                fps = kwargs["fps"][0] if isinstance(kwargs["fps"], list) else kwargs["fps"]
                if fps is not None and fps > 0:
                    td = kwargs.get("timestamp_decimal", 1) or 1
                    pt_patch_position = torch.cat(patch_positions)
                    timestamps = self.compute_frame_timestamps(patch_positions, pt_patch_position, fps)
                    timestamps = [round(t, td) for t in timestamps]
                    timestamp_strings = [f"<{t:.{td}f} seconds>" for t in timestamps]

            if timestamp_strings is not None and len(timestamp_strings) > 0 and len(patch_positions) > 0:
                messages = self._rewrap_vision_by_frame(messages, patch_positions, timestamp_strings)
        encode_pairs = self.chat_template.encode_multiturn(
            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )
        input_ids, target = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
            input_ids += source_ids + target_ids
            target += [IGNORE_INDEX] * len(source_ids) + target_ids
        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        attn_mask = torch.zeros_like(input_ids).bool()

        return (
            input_ids,
            target,
            attn_mask,
            pixel_values_images,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            patch_positions,
        )

    def encode_vqa4packing(self, sample: VQASample) -> ImageTaskSample:
        """Encode VQASample in Qwen2VL style."""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": sample.context}, {"role": "assistant", "content": sample.answers}],
            tokenize=False,
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)

        if text[-1] == "\n":
            text = text[:-1]
            pass

        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answers = self.tokenizer.tokenize(sample.answers)
        target[-len(answers) - 1 : -1] = torch.tensor(answers)
        target[-1] = input_ids[-1]
        # print(target[-1])

        num_tiles = [len(image_grid_thw)]
        if self.args.enable_discard_sample:
            if len(input_ids) > self.args.seq_length:
                skip_malformed_multimodal_sample(
                    sample.__key__,
                    "input_length_exceeds_seq_length",
                    f"input length {len(input_ids)} exceeds seq_length={self.args.seq_length}",
                )
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_multi_vid_qa(self, sample: VQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            input_ids, target, attn_mask, imgs, image_grid_thw, video, video_grid_thw = self.process_sft_qa(
                sample.messages, sample.system, sample.video, None
            )
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        if self.args.enable_discard_sample:
            if len(input_ids) > self.args.seq_length:
                skip_malformed_multimodal_sample(
                    sample.__key__,
                    "input_length_exceeds_seq_length",
                    f"input length {len(input_ids)} exceeds seq_length={self.args.seq_length}",
                )
        else:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, (
                f"{sample.__key__} grid_thw: {video_grid_thw}"
            )

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=video,
            video_grid_thw=video_grid_thw,
            num_tiles=[len(video_grid_thw)],
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""

        if self.args.training_phase == constants.TrainingPhase.SFT:
            num_tiles = []
            kwargs = {}
            if hasattr(sample, "fps") and sample.fps is not None:
                kwargs["fps"] = sample.fps
            if hasattr(sample, "timestamp_decimal") and sample.timestamp_decimal is not None:
                kwargs["timestamp_decimal"] = sample.timestamp_decimal

            def _remove_last_qa_round(messages: list[dict]) -> list[dict]:
                assistant_idx = -1
                for idx in range(len(messages) - 1, -1, -1):
                    if messages[idx].get("role") == constants.DataRoles.ASSISTANT:
                        assistant_idx = idx
                        break

                if assistant_idx == -1:
                    return []

                user_idx = -1
                for idx in range(assistant_idx - 1, -1, -1):
                    if messages[idx].get("role") == constants.DataRoles.USER:
                        user_idx = idx
                        break

                if user_idx == -1:
                    return []

                return messages[:user_idx] + messages[assistant_idx + 1 :]

            current_messages = self._normalize_image_backed_video_placeholders(
                [dict(message) for message in sample.messages],
                sample.image,
                sample.video,
            )
            current_image = sample.image
            current_video = sample.video
            current_patch_positions = sample.patch_positions

            while True:
                try:
                    (
                        input_ids,
                        target,
                        attn_mask,
                        imgs,
                        image_grid_thw,
                        pixel_values_videos,
                        video_grid_thw,
                        patch_positions,
                    ) = self.process_sft_qa(
                        current_messages,
                        sample.system,
                        current_video,
                        current_image,
                        current_patch_positions,
                        **kwargs,
                    )
                except CanvasTimestampError as exc:
                    skip_malformed_multimodal_sample(
                        sample.__key__,
                        "invalid_canvas_timing",
                        str(exc),
                    )

                if len(input_ids) <= self.args.seq_length:
                    break

                current_messages = _remove_last_qa_round(current_messages)
                if len(current_messages) == 0:
                    if self.args.enable_discard_sample:
                        skip_malformed_multimodal_sample(
                            sample.__key__,
                            "qa_truncation_exhausted",
                            (
                                "sample has no QA rounds left after truncation "
                                f"to fit seq_length={self.args.seq_length}"
                            ),
                        )
                    raise AssertionError(
                        "Sample has no QA rounds left after truncation to fit seq_length:\n"
                        f"- sample: {sample.__key__}\n"
                        f"- seq_length: {self.args.seq_length}"
                    )

                image_placeholder_count = sum(
                    message.get("content", "").count(constants.Placeholder.IMAGE) for message in current_messages
                )
                video_placeholder_count = sum(
                    message.get("content", "").count(constants.Placeholder.VIDEO) for message in current_messages
                )

                if current_image is not None:
                    current_image = current_image[:image_placeholder_count]

                if current_patch_positions is not None:
                    current_patch_positions = current_patch_positions[:image_placeholder_count]

                if isinstance(current_video, list):
                    current_video = current_video[:video_placeholder_count]
                elif current_video is not None and video_placeholder_count == 0:
                    current_video = None

            if video_grid_thw is not None:
                num_tiles = [len(video_grid_thw)]
            elif image_grid_thw is not None:
                num_tiles = [len(image_grid_thw)]
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        assert len(input_ids) > 0, f"input_ids is empty in {sample.__key__}"

        if self.args.enable_discard_sample:
            if len(input_ids) > self.args.seq_length:
                skip_malformed_multimodal_sample(
                    sample.__key__,
                    "input_length_exceeds_seq_length",
                    f"input length {len(input_ids)} exceeds seq_length={self.args.seq_length}",
                )
        elif video_grid_thw is not None:
            sms = getattr(self.processor.image_processor, "merge_size", 2)
            assert video_grid_thw.prod(dim=-1).sum() / (sms * sms) <= self.args.seq_length, (
                f"{sample.__key__} grid_thw: {video_grid_thw}"
            )
        elif image_grid_thw is not None:
            sms = getattr(self.processor.image_processor, "merge_size", 2)
            image_token_len = int(image_grid_thw.prod(dim=-1).sum().item() / (sms * sms))
            assert image_token_len <= self.args.seq_length, (
                "Image token length exceeds seq_length:\n"
                f"- sample: {sample.__key__}\n"
                f"- image_grid_thw: {image_grid_thw}\n"
                f"- image_token_len: {image_token_len}\n"
                f"- seq_length: {self.args.seq_length}"
            )

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
            patch_positions=patch_positions,
        )

    def process_samples_grid(self, samples):
        """concat grid_thw for image and video"""
        image_grid_thw = [x.image_grid_thw for x in samples if x.image_grid_thw is not None]
        video_grid_thw = [x.video_grid_thw for x in samples if x.video_grid_thw is not None]

        if len(image_grid_thw) > 0:
            image_grid_thw = torch.cat(image_grid_thw).to(dtype=torch.int32)
        else:
            image_grid_thw = None

        if len(video_grid_thw) > 0:
            video_grid_thw = torch.cat(video_grid_thw).to(dtype=torch.int32)
        else:
            video_grid_thw = None

        return image_grid_thw, video_grid_thw

    @override
    @stateless
    def pack_selected_samples(self, samples: list[Qwen2VLImageTaskSample]) -> list[Qwen2VLImageTaskSamplePacked]:
        """Pack selected samples into one big sample."""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskSamplePacked(
            super().pack_selected_samples(samples), image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )

    @override
    def batch(
        self, samples: list[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]
    ) -> Qwen2VLImageTaskBatchPacked:
        """Batch samples together"""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskBatchPacked(
            super().batch(samples), image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )

    @override
    def process_images(
        self, samples: list[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]
    ) -> torch.Tensor:
        """ " Process the data to get the model's input"""
        imgs = [img for s in samples if s.imgs is not None for img in s.imgs]
        if len(imgs) > 0:
            return torch.cat(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def process_videos(
        self, samples: list[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]
    ) -> torch.Tensor:
        """ " Process the data to get the model's input"""
        pixel_values_videos = [
            pixel_values_video
            for s in samples
            if s.pixel_values_videos is not None
            for pixel_values_video in s.pixel_values_videos
        ]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def build_train_datasets(
        self,
        *,
        datasets: list[tuple[BaseCoreDatasetFactory[T_sample], Union[float, int, None]]],
        worker_config: WorkerConfig,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
        blend_mode: DatasetBlendMode = DatasetBlendMode.NONE,
        repeat: bool = True,
    ) -> SavableDataset[T_batch]:
        """Combines train datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset, _ in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        global_workers = max(1, worker_config.num_workers) * worker_config.world_size
        rotation_lengths = [len(dataset) for dataset, _ in datasets]
        for i in range(1, len(rotation_lengths)):
            rotation_lengths[i] += rotation_lengths[i - 1]
        worker_rotation_offsets = [rotation_length % global_workers for rotation_length in [0] + rotation_lengths[:-1]]

        if repeat:
            inner_datasets = [
                (
                    RepeatDataset(
                        dataset.build(worker_rotation_offset=worker_rotation_offset),
                        worker_config=worker_config,
                    ),
                    1.0 if weight is None else float(weight),
                )
                for (dataset, weight), worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]
        else:
            assert blend_mode in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ) and all(isinstance(repetitions, int) for _dataset, repetitions in datasets), (
                "If repeat is False, the datasets must be repeated with integer weights."
            )
            inner_datasets = [
                (
                    (
                        dataset.build(worker_rotation_offset=worker_rotation_offset)
                        if repetition is None or repetition == 1
                        else RepeatDataset(
                            dataset.build(worker_rotation_offset=worker_rotation_offset),
                            repeats=int(repetition),
                            worker_config=worker_config,
                        )
                    ),
                    len(dataset) * (1 if repetition is None else int(repetition)),
                )
                for (dataset, repetition), worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]

        if len(inner_datasets) > 1:
            # The worker offset for each dataset is the cumsum of the dataset lengths, but modulo the
            # global number of workers.
            dataset = BlendDataset(
                *inner_datasets,
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = inner_datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        if shuffle_buffer_size is not None and shuffle_buffer_size > 1:
            dataset = ShuffleBufferDataset(
                dataset,
                size=shuffle_buffer_size,
                worker_config=worker_config,
            )
        dataset = self.build_cook_crude_sample(dataset, worker_config=worker_config)
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)

        # Insert pool sorting before entering BatchDataset
        if getattr(self.args, "length_sort_pool_size", 0) and self.args.length_sort_pool_size > 0:

            def sort_key_fn(s):
                return getattr(s, "total_len", len(getattr(s, "tokens")))

            sort_ascending = not getattr(self.args, "length_sort_desc", False)
            sort_cls = (
                PackedSeparateSortDataset
                if getattr(self.args, "length_sort_separate_packed", False)
                else LengthPoolSortDataset
            )
            dataset = sort_cls(
                dataset,
                pool_size=self.args.length_sort_pool_size,
                key_fn=sort_key_fn,
                ascending=sort_ascending,
                worker_config=worker_config,
                warmup_steps=getattr(self.args, "length_sort_warmup_steps", 0),
                initial_pool_size=getattr(self.args, "length_sort_initial_pool_size", 10),
            )
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        if worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, mode="train", worker_config=worker_config)
        return dataset
