""" Qwen2VLTaskEncoder class."""
# The Qwen2VLTaskEncoder is responsible for converting raw data samples (images + text conversations) into model-ready tensors. It's called by Megatron Energon during data loading.

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, TypeVar

import numpy as np
import torch
from megatron.energon import CaptioningSample, VQASample
from megatron.energon.flavors.webdataset import VideoData
from megatron.energon.task_encoder.base import stateless
from PIL import Image
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor
from typing_extensions import override
#Fast ViT imports
#   Custom Image processor for FastViT
from aiak_training_llm.models.fastvit.fastvit_preprocessor import FastViTImageProcessor
# Utilities for FastViT multimodal image preprocessing
from aiak_training_llm.models.fastvit.mm_utils import (
    expand2square, # pads image to square with background color
    process_anyres_image, # handles variable resolution with patches
    process_images, #Main dispatcher based on aspect ratio mode
)

from aiak_training_llm.data.multimodal import MultiMixQASample
from aiak_training_llm.data.multimodal.length_sort_dataset import LengthPoolSortDataset
from aiak_training_llm.utils import constants, get_chat_template, get_tokenizer
from aiak_training_llm.tokenizer.special_tokens import ensure_multimodal_special_tokens

from .task_encoder import (ImageTaskBatchPacked, ImageTaskSample,
                           ImageTaskSamplePacked, TaskEncoder)

# Print helper: only emit from rank-0 process so multi-GPU runs don't duplicate output.
import os as _os
def _rank0_print(*args, **kwargs):
    if _os.environ.get("RANK", "0") == "0":
        print(*args, **kwargs)

from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    PinMemoryMixin,
    Sample,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeSample, CrudeWebdataset
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.rng import SystemRng
from megatron.energon.task_encoder.cooking import Cooker
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BlendDataset,
    ConcatDataset,
    BatchDataset,
    EpochizeDataset,
    GroupBatchDataset,
    LimitDataset,
    LogSampleDataset,
    MapDataset,
    PackingDataset,
    ShuffleBufferDataset,
)
# from .batch_dataset import BatchDataset
from megatron.energon.wrappers.repeat_dataset import RepeatDataset

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



def get_stateless(fn: Callable[..., T_sample]) -> bool:
    """Get whether a function is stateless."""
    return getattr(fn, "__stateless__", False)


@dataclass
class Qwen2VLImageTaskSample(ImageTaskSample):
    """ An image task sample with a grid of tokens and their corresponding pixel values."""
    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, image_grid_thw: str, video_grid_thw=None, **kwargs):
        super().__init__(**kwargs)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskSamplePacked(ImageTaskSamplePacked):
    """ An image task sample with a grid of tokens and their corresponding pixel values."""
    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**vars(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskBatchPacked(ImageTaskBatchPacked):
    """ An image task sample with a grid of tokens and their corresponding pixel values."""
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
        self.args = args
        if args.training_phase in ['sft']:
            self.chat_template = get_chat_template() # Load conversation template
            #template for formatting conversations (user/assistant turns)
        
        # OLD APPROACH (commented out - didn't work for MobileLLM):
        # Load HuggingFace processor (tokenizer + image/ video processor from Qwen2-VL)
        # self.processor = AutoProcessor.from_pretrained(self.args.hf_tokenizer_path, trust_remote_code=True, local_files_only=False)
        # print(f"Loaded processor from {self.args.hf_tokenizer_path}")
        # print(f"Processor config: {self.processor}")
        
        # NEW APPROACH: Handle FastVLM (MobileLLM + FastViT) vs Qwen2-VL separately
        # FastViT image processor (following FastVLM repo)
        # Use this for vision encoding instead of Qwen2-VL's processor
        self.use_fastvit = getattr(args, 'use_fastvit', False)
        
        if self.use_fastvit:
            # For FastVLM: Load only the language model tokenizer (e.g., MobileLLM)
            # No need for a full processor since FastViT handles vision separately
            from transformers import AutoTokenizer
            shared_tokenizer = get_tokenizer()
            if shared_tokenizer is not None and hasattr(shared_tokenizer, "hf_tokenizer"):
                self.tokenizer_hf = shared_tokenizer.hf_tokenizer()
                _rank0_print("Using shared training tokenizer in FastVLM mode")
            else:
                self.tokenizer_hf = AutoTokenizer.from_pretrained(
                    self.args.hf_tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=False,
                )

            added_mm_tokens = ensure_multimodal_special_tokens(self.tokenizer_hf)
            _rank0_print(f"Ensured multimodal special tokens for FastVLM tokenizer; added={added_mm_tokens}")

            # Set padding token if not present (required for batch processing)
            if self.tokenizer_hf.pad_token is None:
                self.tokenizer_hf.pad_token = self.tokenizer_hf.eos_token
                _rank0_print(f"Set pad_token to eos_token: {self.tokenizer_hf.eos_token}")

            # Create a wrapper that properly delegates all methods to the tokenizer
            class TokenizerWrapper:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer

                def __getattr__(self, name):
                    # Delegate all other attributes/methods to the underlying tokenizer
                    return getattr(self.tokenizer, name)

            self.processor = TokenizerWrapper(self.tokenizer_hf)
            _rank0_print(f"Loaded tokenizer (FastVLM mode) from {self.args.hf_tokenizer_path}")

            # Initialize FastViT image processor
            fastvit_image_size = getattr(args, 'fastvit_image_size', 1024)
            self.fastvit_processor = FastViTImageProcessor(image_size=fastvit_image_size)
            _rank0_print(f"Initialized FastViT processor with image_size={fastvit_image_size}")
        else:
            # For Qwen2-VL: Load full processor (tokenizer + image/video processor)
            self.processor = AutoProcessor.from_pretrained(
                self.args.hf_tokenizer_path,
                trust_remote_code=True,
                local_files_only=False
            )
            _rank0_print(f"Loaded Qwen2-VL processor from {self.args.hf_tokenizer_path}")

        _rank0_print(f"Processor config: {self.processor}")

        #Resolution parameters for resizing images/videos
        if args.image_resolution:
            setattr(self.processor, 'image_resolution', args.image_resolution)
            # resolution parameters for resizing images/videos
        _rank0_print("image_resolution:", getattr(self.processor, 'image_resolution', None))
        # Video processing parameters
        self.frame_min_pixels = args.frame_min_pixels
        self.frame_max_pixels = args.frame_max_pixels
        self.video_max_pixels = args.video_max_pixels
        self.fps = args.fps
        self.fps_min_frames = args.fps_min_frames
        self.fps_max_frames = args.fps_max_frames
        # Image processing parameters
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels
        self._logged_multimodal_token_debug_once = False

    def _reisize_video(self, vision: VideoData, image_factor=28, frame_factor=2):
        """ Resize video: frame number, height, width """
        total_frames = len(vision.frames)
        video_fps = vision.info['video_fps']
        vision.info['fps'] = self.fps
        vision.info['min_frames'] = self.fps_min_frames
        vision.info['max_frames'] = self.fps_max_frames

        # resize frame
        nframes = smart_nframes(vision.info, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        video = vision.frames[idx]
        # resize height, width
        nframes, _, height, width = video.shape
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=int(self.frame_min_pixels * 1.05),
            max_pixels=min(self.frame_max_pixels, self.video_max_pixels / nframes * frame_factor),
        )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        return video

    def _resize_image(self, image, size_factor=28):
        """
        Resize image based on vision encoder type.
        - For FastViT: Use FastVLM's preprocessing (pad/anyres)
        - For Rice/SigLIP: Dynamic resize with smart_resize
        """
        if self.use_fastvit:
            # FastViT: Use FastVLM's preprocessing approach
            # For single image, we'll handle aspect ratio in _process
            # Just return the PIL image as-is for now
            return image
        
        # Original Rice/SigLIP preprocessing
        #calculate optimal size using smart_resize 
        # constraints:
        # 1- width and height must be multiple of size_factor (28)
        # 2- Total pixels (height × width) are ≥ min_pixels
        # 3- Total pixels (height × width) are ≤ max_pixels
        # 4- Aspect ratio is preserved as much as possible

        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=size_factor, 
            # why factor of 28?
            # Input Image
            #     ↓
            # Split into 14×14 patches  (Patch Embedding)
            #     ↓
            # 2×2 patch merging  (Reduce spatial dimensions by 2)
            #     ↓
            # Effective patch size = 14 × 2 = 28×28 pixels
            # Example:

            # Image size: 1120 × 784 pixels
            # Number of patches: (1120/28) × (784/28) = 40 × 28 = 1,120 patches
            # Each patch → 1 vision token
            min_pixels=self.min_pixels, # e.g., 256*28*28
            max_pixels=self.max_pixels,  # e.g., 1280*28*28
        )
        image = image.resize((resized_width, resized_height))

        return image # return resized PIL image

    def _process(self, image, text):
        """" Process the data to get the model's input """
        if self.use_fastvit and image is not None:
            # FastViT preprocessing using FastVLM's approach
            # Tokenize text only
            text_inputs = self.processor.tokenizer(
                text=text,
                padding=True,
                return_tensors="pt",
            )
            input_ids = text_inputs['input_ids'][0]
            attn_mask = text_inputs['attention_mask'][0].logical_not()
            fastvit_tokenizer = self.processor.tokenizer
            vision_start_id, img_pad_id, vision_end_id = fastvit_tokenizer.convert_tokens_to_ids([
                VISION_TAGS[0],
                IMAGE_TOKEN,
                VISION_TAGS[1]
            ])
            if vision_start_id is None or img_pad_id is None or vision_end_id is None:
                vocab = fastvit_tokenizer.get_vocab()
                if img_pad_id is None:
                    img_pad_id = vocab.get(IMAGE_TOKEN)
                if vision_start_id is None:
                    vision_start_id = vocab.get(VISION_TAGS[0])
                if vision_end_id is None:
                    vision_end_id = vocab.get(VISION_TAGS[1])
            if not self._logged_multimodal_token_debug_once:
                img_count = int((input_ids == img_pad_id).sum().item()) if img_pad_id is not None else 0
                vstart_count = int((input_ids == vision_start_id).sum().item()) if vision_start_id is not None else 0
                _rank0_print(
                    f"[DEBUG PREPROCESS TOKENS] image_token='{IMAGE_TOKEN}' id={img_pad_id}, "
                    f"vision_start='{VISION_TAGS[0]}' id={vision_start_id}, vision_end='{VISION_TAGS[1]}' id={vision_end_id}, "
                    f"counts_in_input_ids: image={img_count}, vision_start={vstart_count}"
                )
                self._logged_multimodal_token_debug_once = True

            # Process image with FastVLM's preprocessing utilities
            # Default to 'pad' aspect ratio (expand to square with padding)
            image_aspect_ratio = getattr(self.args, 'image_aspect_ratio', 'pad')
            if image_aspect_ratio == 'pad':
                # Expand to square with mean color padding
                mean_color = tuple(int(x * 255) for x in self.fastvit_processor.image_mean)
                image = expand2square(image, mean_color)
                pixel_values = self.fastvit_processor(image)
                
                
            elif image_aspect_ratio == 'anyres':
                # Process with variable resolution (patches)
                grid_pinpoints = getattr(self.args, 'image_grid_pinpoints', '[(384, 384), (768, 384), (384, 768), (768, 768)]')
                pixel_values = process_anyres_image(
                    image,
                    self.fastvit_processor.processor,  # CLIPImageProcessor
                    grid_pinpoints
                )
            else:
                # Direct resize to target size
                pixel_values = self.fastvit_processor(image)
            
            pixel = [pixel_values]
            
            # FastViT: Set grid_thw to represent 1 tile (no dynamic grid)
            # Format: tensor([[num_tiles, height, width]])
            image_grid_thw = torch.tensor([[1, 1, 1]])
            
            # Create target tensor (same as Qwen2-VL path)
            target = input_ids.clone()
            target[target == vision_start_id] = IGNORE_INDEX
            target[target == img_pad_id] = IGNORE_INDEX
            target[target == vision_end_id] = IGNORE_INDEX
            
            return input_ids, target, pixel, image_grid_thw, attn_mask
        
        # Original Qwen2-VL processing
        inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs['input_ids'][0]
        attn_mask = inputs['attention_mask'][0].logical_not()
        image_grid_thw = None
        pixel = []
        if image is not None:
            image_grid_thw = inputs['image_grid_thw'] # [t,h,w]
            pixel = [inputs['pixel_values']] # [hw, 2*3*14*14]

        target = input_ids.clone()
        vision_start_id, img_pad_id, vision_end_id = self.tokenizer.convert_tokens_to_ids([
            VISION_TAGS[0],
            IMAGE_TOKEN,
            VISION_TAGS[1]
        ])
        if not self._logged_multimodal_token_debug_once:
            img_count = int((input_ids == img_pad_id).sum().item()) if img_pad_id is not None else 0
            vstart_count = int((input_ids == vision_start_id).sum().item()) if vision_start_id is not None else 0
            _rank0_print(
                f"[DEBUG PREPROCESS TOKENS] image_token='{IMAGE_TOKEN}' id={img_pad_id}, "
                f"vision_start='{VISION_TAGS[0]}' id={vision_start_id}, vision_end='{VISION_TAGS[1]}' id={vision_end_id}, "
                f"counts_in_input_ids: image={img_count}, vision_start={vstart_count}"
            )
            self._logged_multimodal_token_debug_once = True
        target[target == vision_start_id] = IGNORE_INDEX
        target[target == img_pad_id] = IGNORE_INDEX
        target[target == vision_end_id] = IGNORE_INDEX

        return input_ids, target, pixel, image_grid_thw, attn_mask


    def process_sft_vqa(self, context, answer, image):
        """ process the data for sft vqa """
        text = self.processor.apply_chat_template(
            [{
                'role': 'user',
                'content': context
            }, {
                'role': 'assistant',
                'content': answer
            }],
            tokenize=False
        ).replace(
            "<image>", IMAGE_TOKEN_WITH_TAGS
        )
        if text[-1] == '\n':
            text = text[:-1]
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answer = self.tokenizer.tokenize(answer)
        target[-len(answer) - 1: -1] = torch.tensor(answer)

        return input_ids, target, attn_mask, imgs, image_grid_thw


    def process_sft_qa(self, messages: list, system: str, raw_video: list, raw_image: list):
        """ process the data for sft qa """
        # messages: List of conversation turns (user/assistant)
        # system: System prompt
        # raw_video: List of PIL/VideoData objects (or None)
        # raw_image: List of PIL.Image objects (or None)

        #Initialize output containers
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        pixel_values_images = []
        video = []
        image = []

        # resize image 
        if raw_image is not None:
            # loop through each image and resize
            for i in raw_image:
                image.append(self._resize_image(i))
        # resize video
        if raw_video is not None:
            for v in raw_video:
                video.append(self._reisize_video(v))


        # input messages:
        # [
        #     {"from": "human", "value": "<image>\nWhat's in this image?"},
        #     {"from": "gpt", "value": "A cat sitting on a mat."}
        # ]
        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image if image is not None else [],
            video if raw_video is not None else [],
            self.processor
        )
        # output messages:
        # [
        #     {"from": "human", "value": "<|vision_start|><|image_pad|><|image_pad|>...<|vision_end|>\nWhat's in this image?"},
        #     {"from": "gpt", "value": "A cat sitting on a mat."}
        # ]
        # Output mm_inputs:
        # {
        #     "pixel_values": Tensor([1, 1176, 1176]),  # Vision encoder input
        #     "image_grid_thw": Tensor([[1, 28, 42]]),  # 1 temporal, 28 height patches, 42 width patches
        # }


        # assert raw_image is not None, f'No image found in {messages}' 确实有纯文本对话
        
        #extracting the multi-modal inputs
        if raw_video is not None:
            video_grid_thw = mm_inputs["video_grid_thw"]
            pixel_values_videos = [mm_inputs["pixel_values_videos"]]
        if raw_image is not None:
            image_grid_thw = mm_inputs["image_grid_thw"]
            pixel_values_images = [mm_inputs["pixel_values"]]

        encode_pairs = self.chat_template.encode_multiturn(
            # 1. Applies chat template to format conversation:
                # example:
                    # <|im_start|>system
                    # You are a helpful assistant.<|im_end|>
                    # <|im_start|>user
                    # <|vision_start|><|image_pad|><|image_pad|>...<|vision_end|>
                    # What's in this image?<|im_end|>
                    # <|im_start|>assistant
                    # A cat sitting on a mat.<|im_end|>

            # 2. Tokenizes the entire formatted conversation

            # 3. Splits into (source, target) pairs for each turn:
                # example:
                    # encode_pairs = [
                    #     (
                    #         [151644, 8948, 198, ...],  # source_ids: system + user prompt (don't train)
                    #         [8122, 4758, 6134, ...]    # target_ids: assistant response (train on this)
                    #     )
                    # ]


            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )

        input_ids, target = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
            input_ids += source_ids + target_ids # Append both source and target to input_ids
            # input_ids = [151644, 8948, ..., 151645, 8122, 4758, ...]
            #              ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
            #              system + user           assistant response

            # Mask source (user prompt), keep target (assistant response)
            target += [IGNORE_INDEX] * len(source_ids) + target_ids
            # target = [-100, -100, ..., -100, 8122, 4758, ...]
            #          ^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
            #          Masked (don't train on)    Train on this!
            # Why mask source_ids? 
            # We don't want the model to learn to predict the user's input 
            # only the assistant's response!

        # Convert to Tensors and Create Attention Mask
        input_ids = torch.tensor(input_ids) # Shape: [seq_len]
        # input_ids: Tensor([151644, 8948, 198, ..., 8122, 4758, ...]) 
        target = torch.tensor(target)
        # target: Tensor([-100, -100, ..., 8122, 4758, ...])

        # Create attention mask (all False = attend to all tokens)
        attn_mask = torch.zeros_like(input_ids).bool()
        # attn_mask: Tensor([False, False, False, ..., False, False])  # Shape: [seq_len]
        

        return input_ids, target, attn_mask, pixel_values_images, image_grid_thw, \
                    pixel_values_videos, video_grid_thw

        # input_ids: Tokenized conversation - Shape: [seq_len]
        # target: Labels with masking - Shape: [seq_len]
        # attn_mask: Attention mask - Shape: [seq_len]
        # pixel_values_images: Image tensors - List of Tensors
        # image_grid_thw: Image grid info - Tensor [[1, 28, 42]]
        # pixel_values_videos: Video tensors - Empty list
        # video_grid_thw: Video grid info - None


    def encode_captioning(self, sample: CaptioningSample) -> ImageTaskSample:
        """Encode CaptioningSample."""
        """Preprocessing function for datasets like COCO, containing image-caption pairs.
        See Energon codebase for more details on CaptioningSample.
        https://github.com/NVIDIA/Megatron-Energon/blob/develop/src/megatron/energon/flavors/captioning.py
        """

        # assert self.args.training_phase == constants.TrainingPhase.PRETRAIN, "Only support PRETRAIN phase"

        # text = IMAGE_TOKEN_WITH_TAGS + sample.caption + self.tokenizer.tokenizer.eos_token
        text = IMAGE_TOKEN_WITH_TAGS + sample.caption + self.tokenizer.eos_token

        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else [1]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        elif image_grid_thw is not None:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} thw {image_grid_thw}"

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


    def encode_vqa4packing(self, sample: VQASample) -> ImageTaskSample:
        """Encode VQASample in Qwen2VL style."""
        
        # 构建 chat_template
        text = self.processor.apply_chat_template(
            [{
                'role': 'user',
                'content': sample.context
            }, {
                'role': 'assistant',
                'content': sample.answers
            }],
            tokenize=False
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)        

        if text[-1] == '\n':
            text = text[:-1]
            pass  
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answers = self.tokenizer.tokenize(sample.answers)
        target[-len(answers) - 1: -1] = torch.tensor(answers)
        target[-1] = input_ids[-1]     
        # print(target[-1])
        
        num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else [1]
        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        elif image_grid_thw is not None:
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
            # call main processing function process_sft_qa
            input_ids, target, attn_mask, imgs, image_grid_thw, video, video_grid_thw = \
                        self.process_sft_qa(sample.messages, sample.system, sample.video, None)
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        elif video_grid_thw is not None:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, \
                    f"{sample.__key__} grid_thw: {video_grid_thw}"

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

    # Energon automatically calls the appropriate encode method based on sample type.
    # For SFT with multi-modal data, it calls:
    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            num_tiles = [] #store number of tiles for each image/ video after processing
            # call main processing function process_sft_qa
            input_ids, target, attn_mask, imgs, image_grid_thw, pixel_values_videos, video_grid_thw = \
                        self.process_sft_qa(sample.messages, sample.system, sample.video, sample.image)
            if sample.video is not None:
                num_tiles = [len(video_grid_thw)] if video_grid_thw is not None else [1]
            elif sample.image is not None:
                num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else [1]
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")


        if len(input_ids) == 0:
            raise ValueError(f"input_ids is empty in {sample.__key__}")

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        elif sample.video is not None and video_grid_thw is not None:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, \
                        f"{sample.__key__} grid_thw: {video_grid_thw}"
        elif sample.image is not None and image_grid_thw is not None:
            assert image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, \
                        f"{sample.__key__} grid_thw: {image_grid_thw}"

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
        )


    def encode_vaq(self, sample: VQASample) -> ImageTaskSample:
        """Encode pretrain sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                text = (sample.context + sample.answers).replace(
                    "<image>",
                    IMAGE_TOKEN_WITH_TAGS
                )
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            # text = text + self.tokenizer.tokenizer.eos_token
            text = text + self.tokenizer.eos_token
            input_ids, target, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        elif self.args.training_phase == constants.TrainingPhase.SFT:


            if len(sample.answers) < 1:
                raise ValueError("sample.answers < 1!")

            # Add image resize check for PIL.Image
            if sample.image is not None:

                img_arr = np.array(sample.image)
                if np.sum(img_arr) == 0:
                    raise ValueError("Image pixels are all zero!")

            # Truncate answer to the last full sentence if it exceeds the max length.
            max_answer_length = self.args.training_rice_vl_max_answer_length
            if len(sample.answers) > max_answer_length:
                original_length = len(sample.answers)

                # Perform a preliminary cut at the maximum allowed length.
                preliminary_cut = sample.answers[:max_answer_length]

                # Clean up trailing punctuation and whitespace from the preliminary cut
                cleaned_cut = preliminary_cut.rstrip('.。 \t\n')

                # Find the last occurrence of a sentence-ending punctuation mark followed by a space or the end of the string.
                # This pattern looks for sentence enders (. or 。)
                sentence_enders_pattern = r'[.。]'

                # Find all matches and get the end position of the last match
                matches = list(re.finditer(sentence_enders_pattern, cleaned_cut))

                if matches:
                    # Get the end position of the last match
                    last_end_index = matches[-1].end()
                    # Truncate at the end of the last full sentence.
                    sample.answers = cleaned_cut[:last_end_index]
                else:
                    # Fallback to a hard cut of the original preliminary string if no sentence ender is found.
                    sample.answers = preliminary_cut

                _rank0_print(
                    f"Answer truncated to a full sentence. "
                    f"Original length: {original_length}, New length: {len(sample.answers)}"
                )

            text = self.processor.apply_chat_template(
                [{
                    'role': 'user',
                    'content': sample.context
                }, {
                    'role': 'assistant',
                    'content': sample.answers
                }],
                tokenize=False
            ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            if text[-1] == '\n':
                text = text[:-1]
            input_ids, _, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
            target = torch.ones_like(input_ids) * IGNORE_INDEX
            answers = self.tokenizer.tokenize(sample.answers)
            target[-len(answers) - 1: -1] = torch.tensor(answers)
            target[-1] = input_ids[-1]
            # print(target[-1])
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else [1]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
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

    def process_samples_grid(self, samples):
        """ concat grid_thw for image and video """
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
    def pack_selected_samples(self, samples: List[Qwen2VLImageTaskSample]) -> List[Qwen2VLImageTaskSamplePacked]:
        """ Pack selected samples into one big sample."""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskSamplePacked(
            super().pack_selected_samples(samples),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )

    @override
    # After encoding , multiple  encoded samples are batched together
    def batch(self, samples: List[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]) \
                                                                                    -> Qwen2VLImageTaskBatchPacked:
        """ Batch samples together """
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        # Create batch
        return Qwen2VLImageTaskBatchPacked(
            super().batch(samples), # # Stacks tokens, labels, etc. with padding
            # Pads tokens to same length → [batch_size, max_seq_len]
            # Pads labels to same length → [batch_size, max_seq_len]
            # Concatenates imgs → [total_images_in_batch, C, H, W]
            # Creates batch attention masks
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )

    @override
    def process_images(self, samples: List[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]) \
                                                                                    -> torch.Tensor:
        """" Process the data to get the model's input """
        imgs = [img for s in samples if s.imgs is not None for img in s.imgs]
        if len(imgs) > 0:
            return torch.cat(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def process_videos(self, samples: List[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]) \
                                                                                    -> torch.Tensor:
        """" Process the data to get the model's input """
        pixel_values_videos = [pixel_values_video for s in samples if s.pixel_values_videos is not None \
                for pixel_values_video in s.pixel_values_videos]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)


    @override
    def build_train_datasets(
        self,
        *,
        datasets: List[Tuple[BaseCoreDatasetFactory[T_sample], Union[float, int, None]]],
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
        worker_rotation_offsets = [
            rotation_length % global_workers for rotation_length in [0] + rotation_lengths[:-1]
        ]

        if repeat:
            inner_datasets = [
                (
                    RepeatDataset(
                        dataset.build(worker_rotation_offset=worker_rotation_offset),
                        worker_config=worker_config,
                    ),
                    1.0 if weight is None else float(weight),
                )
                for (dataset, weight), worker_rotation_offset in zip(
                    datasets, worker_rotation_offsets
                )
            ]
        else:
            assert blend_mode in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ) and all(
                isinstance(repetitions, int) for _dataset, repetitions in datasets
            ), "If repeat is False, the datasets must be repeated with integer weights."
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
                for (dataset, repetition), worker_rotation_offset in zip(
                    datasets, worker_rotation_offsets
                )
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

         # 在进入 BatchDataset 之前插入池化排序
        if getattr(self.args, "length_sort_pool_size", 0) and self.args.length_sort_pool_size > 0:
            dataset = LengthPoolSortDataset(
                dataset,
                pool_size=self.args.length_sort_pool_size,
                key_fn=lambda s: getattr(s, "total_len", len(getattr(s, "tokens"))),
                ascending=not getattr(self.args, "length_sort_desc", False),
                worker_config=worker_config,
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