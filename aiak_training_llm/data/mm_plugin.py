import math
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Type, TypedDict, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from typing_extensions import override

from aiak_training_llm.utils.constants import Placeholder
from transformers.image_utils import get_image_size, to_numpy_array


if TYPE_CHECKING:
    import torch

    from transformers.image_processing_utils import BaseImageProcessor
    from transformers.processing_utils import ProcessorMixin

    class EncodedImage(TypedDict):
        """Encoded image type."""

        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str


class MMPlugin:
    """MM Plugin"""

    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        # image_resolution: int = kwargs.get("image_resolution")
        # if max(image.width, image.height) > image_resolution:
        #     resize_factor = image_resolution / max(image.width, image.height)
        #     width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        #     image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                # image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        self._validate_input(images, videos)
        return {}


class Qwen2VLPlugin(MMPlugin):
    """Qwen2VL plugin"""

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])
        actual_num_images = len(image_grid_thw)

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)

        image_placeholder_count = sum(message["content"].count(Placeholder.IMAGE) for message in messages)
        video_placeholder_count = sum(message["content"].count(Placeholder.VIDEO) for message in messages)

        if actual_num_images > 0 and image_placeholder_count != actual_num_images:
            for message in messages:
                message["content"] = message["content"].replace(Placeholder.IMAGE, "")

            first_user_msg = None
            for message in messages:
                if message.get("role") == "user":
                    first_user_msg = message
                    break

            if first_user_msg is None:
                raise ValueError("Cannot rebuild image placeholders: no user message found.")

            image_placeholders = "\n".join([Placeholder.IMAGE] * actual_num_images)
            user_content = first_user_msg["content"].lstrip("\n")
            first_user_msg["content"] = "{}\n{}".format(image_placeholders, user_content)

        if len(videos) > 0 and video_placeholder_count == 0:
            raise ValueError("Found video inputs but no {} token in messages.".format(Placeholder.VIDEO))
        if video_placeholder_count > 0 and video_placeholder_count != len(videos):
            raise ValueError(
                "Found {} video(s) but {} {} token(s) in messages.".format(
                    len(videos), video_placeholder_count, Placeholder.VIDEO
                )
            )

        for message in messages:
            content = message["content"]
            while Placeholder.IMAGE in content:
                if num_image_tokens >= actual_num_images:
                    raise ValueError(
                        "The number of {} tokens is greater than available images.".format(Placeholder.IMAGE)
                    )

                content = content.replace(
                    Placeholder.IMAGE,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while Placeholder.VIDEO in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError("`len(videos)` is less than the number of {} tokens.".format(Placeholder.VIDEO))

                content = content.replace(
                    Placeholder.VIDEO,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if actual_num_images != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(Placeholder.IMAGE))

        if len(videos) != num_video_tokens:
            raise ValueError("The number of videos does not match the number of {} tokens".format(Placeholder.VIDEO))

        return messages, mm_inputs

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class Gemma4VLPlugin(MMPlugin):
    """Gemma4-VL passthrough plugin: process_messages returns (messages, mm_inputs)
    where mm_inputs follows the existing OV2 flattened-patch contract.

    Cross-module contract (downstream consumers depend on these invariants):
    - ``pixel_values`` shape ``[total_imgs_in_batch, P, D]`` — FLATTENED, NOT ``[B, ...]``.
      Indexing by batch index will silently return wrong tensor.
    - ``image_grid_thw`` is synthesized from HF ``image_position_ids`` as
      ``[num_images, 3]`` rows of ``[1, H_p, W_p]``.
    - Text-only batches OMIT multimodal keys entirely (no zero-shape sentinel).
      All downstream consumers MUST guard with ``in mm_inputs``.
    """

    @staticmethod
    def _flatten_gemma4_image_outputs(
        image_outputs: dict[str, "torch.Tensor"],
    ) -> dict[str, Union["torch.Tensor", list["torch.Tensor"]]]:
        pixel_values = image_outputs["pixel_values"]
        image_position_ids = image_outputs["image_position_ids"]
        num_soft_tokens_per_image = image_outputs["num_soft_tokens_per_image"]

        valid_mask = (image_position_ids != -1).all(dim=-1)
        flat_pixel_values = pixel_values[valid_mask]

        image_grid_rows: list[list[int]] = []
        patch_positions: list[torch.Tensor] = []
        for image_idx in range(image_position_ids.shape[0]):
            valid_positions = image_position_ids[image_idx][valid_mask[image_idx]].to(dtype=torch.int64)
            if valid_positions.numel() == 0:
                raise ValueError(f"Gemma4 image {image_idx} has no valid patch positions.")

            width = int(valid_positions[:, 0].max().item()) + 1
            height = int(valid_positions[:, 1].max().item()) + 1
            patch_count = int(valid_positions.shape[0])
            if patch_count != height * width:
                raise ValueError(
                    "Gemma4 image patch positions are not a dense single-frame grid: "
                    f"image_idx={image_idx}, patch_count={patch_count}, height={height}, width={width}."
                )

            image_grid_rows.append([1, height, width])
            patch_positions.append(
                torch.stack(
                    (
                        torch.zeros(patch_count, dtype=torch.int64, device=valid_positions.device),
                        valid_positions[:, 1],
                        valid_positions[:, 0],
                    ),
                    dim=-1,
                )
            )

        image_grid_thw = torch.tensor(
            image_grid_rows,
            dtype=torch.int32,
            device=image_position_ids.device,
        )

        return {
            "pixel_values": flat_pixel_values,
            "image_grid_thw": image_grid_thw,
            "patch_positions": patch_positions,
            "image_position_ids": image_position_ids,
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }

    def _build_gemma4_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        processor: Optional["ProcessorMixin"],
    ) -> tuple[Optional[list["ImageObject"]], dict[str, Union["torch.Tensor", list["torch.Tensor"]]]]:
        regularized_images = self._regularize_images(images) if len(images) != 0 else None
        if regularized_images is None:
            return None, {}

        image_outputs = processor.image_processor(regularized_images, return_tensors="pt")
        return regularized_images, self._flatten_gemma4_image_outputs(dict(image_outputs))

    def _expand_image_placeholders(
        self,
        messages: Sequence[dict[str, str]],
        num_soft_tokens_per_image: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> list[dict[str, str]]:
        messages = deepcopy(messages)
        actual_num_images = len(num_soft_tokens_per_image)

        image_placeholder_count = sum(message["content"].count(Placeholder.IMAGE) for message in messages)
        if actual_num_images > 0 and image_placeholder_count != actual_num_images:
            for message in messages:
                message["content"] = message["content"].replace(Placeholder.IMAGE, "")

            first_user_msg = None
            for message in messages:
                if message.get("role") == "user":
                    first_user_msg = message
                    break

            if first_user_msg is None:
                raise ValueError("Cannot rebuild Gemma4 image placeholders: no user message found.")

            image_placeholders = "\n".join([Placeholder.IMAGE] * actual_num_images)
            user_content = first_user_msg["content"].lstrip("\n")
            first_user_msg["content"] = f"{image_placeholders}\n{user_content}"

        image_idx = 0
        for message in messages:
            content = message["content"]
            while Placeholder.IMAGE in content:
                if image_idx >= actual_num_images:
                    raise ValueError(
                        f"The number of {Placeholder.IMAGE} tokens is greater than available images."
                    )

                n_soft_tokens = int(num_soft_tokens_per_image[image_idx])
                replacement = (
                    f"{processor.boi_token}{self.image_token * n_soft_tokens}{processor.eoi_token}"
                )
                content = content.replace(Placeholder.IMAGE, replacement, 1)
                image_idx += 1

            if Placeholder.VIDEO in content:
                raise ValueError("Gemma4-VL video placeholders are not supported in this OV2 path yet.")

            message["content"] = content

        if image_idx != actual_num_images:
            raise ValueError(
                f"The number of images ({actual_num_images}) does not match expanded placeholders ({image_idx})."
            )

        return messages

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        return super()._preprocess_image(image, **kwargs)

    @override
    def process_messages(
        self,
        messages: Sequence[dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> tuple[list[dict[str, str]], dict[str, "torch.Tensor"]]:
        self._validate_input(images, videos)
        _regularized_images, mm_inputs = self._build_gemma4_mm_inputs(images, processor)

        if "num_soft_tokens_per_image" in mm_inputs:
            messages = self._expand_image_placeholders(
                messages,
                mm_inputs["num_soft_tokens_per_image"],
                processor,
            )
        else:
            messages = list(messages)

        return messages, dict(mm_inputs)

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        del imglens, vidlens, seqlens
        _regularized_images, mm_inputs = self._build_gemma4_mm_inputs(images, processor)
        return dict(mm_inputs)
