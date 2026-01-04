""" Tasks related to vision models."""
from abc import ABC, abstractmethod
import bisect
import dataclasses
import json
import re
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np
import torch

from aiak_training_llm.data.multimodal import MultiVidQASample, MultiMixQASample
from aiak_training_llm.data.multimodal import PackedCaptioningSample
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    Sample,
    SimilarityInterleavedSample,
    VQASample,
    MultiChoiceVQASample
)
from megatron.energon.task_encoder.base import stateless
from aiak_training_llm.utils import get_args, get_tokenizer


IGNORE_INDEX = -100  # ID for labels that should be ignored.

@dataclass
class ImageTaskSample(Sample):
    """ Dataclass to store a single unbatched sample. """
    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Dict
    __subflavors__: Dict
    num_tiles: List[int]
    tokens: torch.Tensor
    total_len: int  # Total token count in the sample, including text and image tokens
    labels: torch.Tensor = None
    attn_mask: torch.Tensor = None
    # (c, h, w)
    imgs: List[torch.Tensor] = None
    pixel_values_videos: List[torch.Tensor] = None

@dataclass
class ImageTaskSamplePacked(Sample):
    """Dataclass to store a single packed sample (not a batch).

        P = Number of sub-samples in the packed sample
        seq_len = Total sequence length
        num_imgs = Number of images across all samples in the packed sample
    """

    __key__: str    # Sample name
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Dict     # Sample metadata. Deprecated.
    __subflavors__: Dict    # Sample metadata.
    tokens: torch.Tensor  # Input tokens packed into a single tensor (seq_len,)
    labels: torch.Tensor # Target tokens packed into a single tensor (seq_len,)
    num_tiles: List[int]  # Number of tiles for each image of each sample (num_imgs)
    max_length: int    # Maximum length across sub-samples.
    cu_lengths: List[int]  # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    attn_mask: torch.Tensor = None
    imgs: List[torch.Tensor] = None   # Input images
    pixel_values_videos: List[torch.Tensor] = None

# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatchPacked(Batch):
    """Dataclass to store a batch of packed samples.

        N = Batch size
        P = Number of samples in the packed sample
        seq_len = Maximum sequence length
        num_imgs = Number of images across all samples in the packed sample
    """

    __key__: List[str]  # Sample names
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Dict     # Sample metadata. Deprecated.
    __subflavors__: List[Dict]  # Sample metadatas.
    tokens: torch.Tensor  # Input tokens packed and padded (N, seq_len)
    labels: torch.Tensor # Target tokens packed and padded (N, seq_len)
    num_tiles: List[List[int]]  # Number of tiles per image (N, num_imgs)
    max_lengths: List[int]  # Maximum length across sub-samples (N,)
    cu_lengths: List[List[int]]  # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    attn_mask: torch.Tensor = None
    imgs: torch.Tensor = None # All image tiles stacked into a single tensor (num_tiles, C, H, W)
    pixel_values_videos: torch.Tensor = None

# Based on https://github.com/hiyouga/LLaMA-Factory/
#          blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L19
def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


# Based on https://github.com/hiyouga/LLaMA-Factory/
#          blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L27
def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List:
    """Greedy algorithm with binary search for the knapsack problem.

    Pack as many samples as possible given a maximum capacity and capacities of individual samples.
    Used if sequence packing is enabled.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have the same length."

    knapsacks = []

    if len(item_sizes) == 0:
        return knapsacks

    # Sort sample lengths and samples together.
    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)

    # Check if all samples fit in the knapsack capacity.
    if sorted_item_sizes[-1] > max_capacity:
        raise ValueError(f"knapsack: {sorted_samples[-1].__key__} is larger {sorted_item_sizes[-1]} \
            than the max_sequence_length {max_capacity}.")

    while sorted_item_sizes:
        current_knapsack = []
        remaining_capacity = max_capacity

        while True:
            idx = search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break   # Can't fit more samples.

            remaining_capacity -= sorted_item_sizes[idx]

            sorted_item_sizes.pop(idx)
            sample = sorted_samples.pop(idx)
            current_knapsack.append(sample)

        knapsacks.append(current_knapsack)

    return knapsacks


class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatchPacked, dict]):
    """A simple task encoder for VLMs."""

    def __init__(self):
        super().__init__()

        self.args = get_args()

        self.tokenizer = get_tokenizer()
        self.is_packing_enabled = self.args.packing_pretrain_data or self.args.packing_sft_data

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: Union[CaptioningSample, OCRSample, VQASample, SimilarityInterleavedSample]):
        """ Generates an encoded sample from a raw sample. """
        if isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            yield self.encode_vaq(sample)
        elif isinstance(sample, MultiVidQASample):
            yield self.encode_multi_vid_qa(sample)
        elif isinstance(sample, MultiMixQASample):
            yield self.encode_multi_mix_qa(sample)
        elif isinstance(sample, PackedCaptioningSample):
            # print(f"-------------PackedCaptioningSample---------------")
            # print(sample)
            # print(f"-------------PackedCaptioningSample---------------")
            n_orig_sample = len(sample.images)
            l_Qwen2VLImageTaskSample = []
            for idx in range(n_orig_sample):
                if int(os.environ.get("OFFLINE_PACKING_VQA",0))==1:
                    cur_capsample = VQASample(
                        __key__=f"{sample.__key__}.img{idx:03d}_jpg",
                        __restore_key__=sample.__restore_key__,
                        __subflavor__=None,
                        __subflavors__=sample.__subflavors__,
                        image=sample.images[idx],
                        answers=sample.captions[idx],
                        context=sample.prompts[idx]
                    )                    
                    l_Qwen2VLImageTaskSample.append(self.encode_vqa4packing(cur_capsample))
                else:
                    cur_capsample = CaptioningSample(
                        __key__=f"{sample.__key__}.img{idx:03d}_jpg",
                        __restore_key__=sample.__restore_key__,
                        __subflavor__=None,
                        __subflavors__=sample.__subflavors__,
                        image=sample.images[idx],
                        caption=sample.captions[idx]
                    )
                    l_Qwen2VLImageTaskSample.append(self.encode_captioning(cur_capsample))
            l_sample_packed = self.pack_selected_samples(l_Qwen2VLImageTaskSample)
            yield l_sample_packed
            pass
        else:
            raise NotImplementedError("Sample format not supported", sample)

    @abstractmethod
    def encode_captioning(self, sample: CaptioningSample) -> ImageTaskSample:
        """ Generates an encoded captioning sample from a raw sample. """
        pass

    @abstractmethod
    def encode_vaq(self, sample: VQASample) -> ImageTaskSample:
        """ Generates an encoded vqa sample from a raw sample. """
        pass

    @abstractmethod
    def encode_multi_vid_qa(self, sample: MultiVidQASample) -> ImageTaskSample:
        """ Generates an encoded vid_qa sample from a raw sample. """
        pass

    @abstractmethod
    def encode_multi_vid_qa(self, sample: MultiMixQASample) -> ImageTaskSample:
        """ Generates an encoded multimodal mix sample from a raw sample. """
        pass

    def process_images(self, samples: List[Union[ImageTaskSample, ImageTaskSamplePacked]]) -> torch.Tensor:
        """ Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image. """
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            return torch.stack(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    def process_videos(self, samples: List[Union[ImageTaskSample, ImageTaskSamplePacked]]) \
                                                                                    -> torch.Tensor:
        """" Process the data to get the model's input """
        pixel_values_videos = [pixel_values_video for s in samples if s.pixel_values_videos is not None \
                for pixel_values_video in s.pixel_values_videos]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    def batch(self, samples: List[Union[ImageTaskSample, ImageTaskSamplePacked]]) -> ImageTaskBatchPacked:
        """ Generates a batched version of the provided samples. """
        imgs = self.process_images(samples)
        pixel_values_videos = self.process_videos(samples)
        max_seq_len = max(len(s.tokens) for s in samples)
        max_seq_len = min(max_seq_len, self.args.seq_length)

        tokens = np.full((len(samples), max_seq_len), self.tokenizer.pad, dtype=np.int64)
        labels = np.full((len(samples), max_seq_len), IGNORE_INDEX, dtype=np.int64)
        attn_masks = np.full((len(samples), max_seq_len), True, dtype=bool)

        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.tokens))
            target_len = min(max_seq_len, len(s.labels))

            tokens[i, :text_len] = s.tokens[:text_len]
            labels[i, :target_len] = s.labels[:target_len]
            attn_masks[i, :text_len] = s.attn_mask[:text_len]

        num_tiles = [n for s in samples for n in s.num_tiles]
        if len(num_tiles) > 0:
            num_tiles = torch.tensor(num_tiles, dtype=torch.int32)
        else:
            num_tiles = torch.tensor([[0]], dtype=torch.int32)

        # Cumulative sample lengths are needed for packing, otherwise use dummy values.
        cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        max_lengths = torch.tensor([[0]], dtype=torch.int32)

        if self.is_packing_enabled  or int(os.environ.get("OFFLINE_PACKED_DATA",0))==1:
            cu_lengths = torch.stack([s.cu_lengths for s in samples])
            max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)

        return ImageTaskBatchPacked(
            __key__=[s.__key__ for s in samples],
            __restore_key__=[s.__restore_key__ for s in samples],
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            tokens=tokens,
            labels=labels,
            attn_mask=attn_masks,
            imgs=imgs,
            pixel_values_videos=pixel_values_videos,
            num_tiles=num_tiles,
            cu_lengths=cu_lengths,
            max_lengths=max_lengths,
        )

    def encode_batch(self, batch: ImageTaskBatchPacked) -> dict:
        """ Generates a dictionary containing the data required by the model. """
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw

    def select_samples_to_pack(self, samples: List[ImageTaskSample]) -> List[List[ImageTaskSample]]:
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        lengths = [sample.total_len for sample in samples]

        packed_samples = greedy_knapsack(lengths, samples, self.args.seq_length)

        return packed_samples

    # @stateless
    # def pack_selected_samples(self, samples: List[ImageTaskSample]) -> List[ImageTaskSamplePacked]:
    #     """
    #     Function to pack a list of ImageTaskSample into a single ImageTaskSamplePacked.

    #     NOTE: Energon dataloader calls this method internally if packing is used.
    #     Please see https://nvidia.github.io/Megatron-Energon/packing.html

    #     Args:
    #         samples: List of ImageTaskSample instances to pack into one sample.

    #     Returns:
    #         ImageTaskSamplePacked instance.
    #     """

    #     packing_seq_len = self.args.seq_length

    #     packed_tokens = []
    #     packed_labels = []
    #     packed_masks = []
    #     packed_imgs = []
    #     packed_videos = []

    #     current_length = 0
    #     max_length = 0
    #     cu_lengths = [0]

    #     # Process each sample and build lists that we will concatenate to create the packed sample.
    #     for _, sample in enumerate(samples):
    #         sample_len = sample.total_len

    #         if sample_len > max_length:
    #             max_length = sample_len


    #             # If total_len property is used, update it to reflect truncation
    #             sample_len = min(sample_len, packing_seq_len)
    #             try:
    #                 sample.total_len = sample_len
    #             except Exception:
    #                 # If dataclass is frozen or doesn't allow assign, ignore;
    #                 # we will use sample_len variable for logic below.
    #                 pass

    #         # update max_length to be the maximum sample length after truncation
    #         if sample_len > max_length:
    #             max_length = sample_len



    #         # If adding this sample exceeds the max length, stop.
    #         # This should not happen. 
    #         # The select_samples_to_pack method should have already ensured that the samples fit.
    #         if current_length + sample_len > packing_seq_len:
    #             print(f"packing_seq_len:{packing_seq_len}----<<<<<----{current_length + sample_len}")
    #             raise ValueError(f"Packed sample exceeds the maximum sequence length of {packing_seq_len}: {samples}")

    #         # Add the sample's tokens and labels
    #         packed_tokens.append(sample.tokens)
    #         packed_labels.append(sample.labels)
    #         packed_masks.append(sample.attn_mask)

    #         # Add the images
    #         if sample.imgs is not None:
    #             packed_imgs += sample.imgs
    #         if sample.pixel_values_videos is not None:
    #             packed_videos += sample.pixel_values_videos
    #         current_length += sample_len
    #         cu_lengths.append(current_length)

    #     # Concatenate packed tokens and labels.
    #     packed_tokens = torch.cat(packed_tokens, dim=0)
    #     packed_labels = torch.cat(packed_labels, dim=0)
    #     packed_masks = torch.cat(packed_masks, dim=0)

    #     return ImageTaskSamplePacked(
    #         __key__=",".join([s.__key__ for s in samples]),
    #         __restore_key__=(),  # Will be set by energon based on `samples`
    #         __subflavor__=None,
    #         __subflavors__=samples[0].__subflavors__,
    #         tokens=packed_tokens,
    #         labels=packed_labels,
    #         attn_mask=packed_masks,
    #         imgs=packed_imgs,
    #         pixel_values_videos=packed_videos,
    #         cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
    #         max_length=max_length,
    #         num_tiles=[n for s in samples for n in s.num_tiles],
    #     )

    # @stateless
    # def pack_selected_samples(self, samples: List[ImageTaskSample]) -> ImageTaskSamplePacked:
    #     """
    #     Pack a list of ImageTaskSample into a single ImageTaskSamplePacked.
    #     Truncates samples as needed so the packed sample never exceeds packing_seq_len.
    #     """
    #     packing_seq_len = self.args.seq_length

    #     # Lists of per-token pieces (will be concatenated at the end).
    #     packed_tokens_parts = []
    #     packed_labels_parts = []
    #     packed_masks_parts = []
    #     packed_imgs = []
    #     packed_videos = []

    #     current_length = 0
    #     max_length = 0
    #     cu_lengths = [0]

    #     for sample in samples:
    #         # Determine sample token length (use available fields)
    #         sample_len = int(getattr(sample, "total_len", None) or
    #                          (len(sample.tokens) if hasattr(sample, "tokens") and sample.tokens is not None else 0))

    #         # If the sample itself is longer than the global packing_seq_len, allow truncation
    #         if sample_len > packing_seq_len:
    #             logging.getLogger(__name__).warning(
    #                 f"Sample {getattr(sample, '__key__', '<unknown>')} length {sample_len} > packing_seq_len {packing_seq_len}. "
    #                 "Truncating token fields to packing_seq_len."
    #             )
    #             sample_len = packing_seq_len

    #         remaining_capacity = packing_seq_len - current_length
    #         if remaining_capacity <= 0:
    #             # no capacity left (shouldn't normally happen if select_samples_to_pack worked),
    #             # stop adding further samples.
    #             logging.getLogger(__name__).warning(
    #                 "No remaining packing capacity; stopping packing of further samples."
    #             )
    #             break

    #         # If this sample would overflow the remaining capacity, we will truncate it to fit.
    #         if sample_len > remaining_capacity:
    #             logging.getLogger(__name__).warning(
    #                 f"Truncating sample {getattr(sample, '__key__', '<unknown>')} from {sample_len} to fit remaining capacity {remaining_capacity}."
    #             )
    #             sample_len = remaining_capacity

    #         # Slice the per-token fields safely (they might be torch tensors or numpy arrays)
    #         # If a field is missing, create an appropriate dummy slice of length sample_len.
    #         def slice_field(field, length):
    #             if field is None:
    #                 # return a zero-length tensor/array (we'll handle concatenation later)
    #                 return None
    #             try:
    #                 return field[:length]
    #             except Exception:
    #                 # fallback: convert to torch tensor then slice
    #                 try:
    #                     t = torch.as_tensor(field)
    #                     return t[:length]
    #                 except Exception:
    #                     return None

    #         tokens_slice = slice_field(getattr(sample, "tokens", None), sample_len)
    #         labels_slice = slice_field(getattr(sample, "labels", None), sample_len)
    #         mask_slice = slice_field(getattr(sample, "attn_mask", None), sample_len)

    #         # Append slices (if None, we will handle later by creating empty tensors)
    #         packed_tokens_parts.append(tokens_slice)
    #         packed_labels_parts.append(labels_slice)
    #         packed_masks_parts.append(mask_slice)

    #         # For images/videos: keep them as-is (we assume images are not token-aligned)
    #         if getattr(sample, "imgs", None) is not None:
    #             packed_imgs += list(sample.imgs)
    #         if getattr(sample, "pixel_values_videos", None) is not None:
    #             packed_videos += list(sample.pixel_values_videos)

    #         current_length += sample_len
    #         cu_lengths.append(current_length)
    #         if sample_len > max_length:
    #             max_length = sample_len

    #     # If nothing was added, return a minimal packed sample
    #     if not packed_tokens_parts:
    #         return ImageTaskSamplePacked(
    #             __key__="",
    #             __restore_key__=(),
    #             __subflavor__=None,
    #             __subflavors__=samples[0].__subflavors__ if samples else {},
    #             tokens=torch.zeros((0,), dtype=torch.int64),
    #             labels=torch.zeros((0,), dtype=torch.int64),
    #             attn_mask=torch.zeros((0,), dtype=torch.bool),
    #             imgs=packed_imgs,
    #             pixel_values_videos=packed_videos,
    #             cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
    #             max_length=0,
    #             num_tiles=[n for s in samples for n in s.num_tiles] if samples else [],
    #         )

    #     # For concatenation we need to ensure same dtypes. Replace None slices with zero-length tensors.
    #     def to_tensor_or_empty(x, dtype=None):
    #         if x is None:
    #             return torch.zeros((0,), dtype=dtype if dtype is not None else torch.int64)
    #         if isinstance(x, torch.Tensor):
    #             return x
    #         # try to convert numpy/other to tensor
    #         try:
    #             return torch.as_tensor(x)
    #         except Exception:
    #             return torch.zeros((0,), dtype=dtype if dtype is not None else torch.int64)

    #     # infer dtypes from first available slices
    #     first_tokens = next((p for p in packed_tokens_parts if p is not None), None)
    #     first_labels = next((p for p in packed_labels_parts if p is not None), None)
    #     first_masks = next((p for p in packed_masks_parts if p is not None), None)

    #     tokens_dtype = first_tokens.dtype if isinstance(first_tokens, torch.Tensor) else torch.int64
    #     labels_dtype = first_labels.dtype if isinstance(first_labels, torch.Tensor) else torch.int64
    #     masks_dtype = first_masks.dtype if isinstance(first_masks, torch.Tensor) else torch.bool

    #     packed_tokens = torch.cat([to_tensor_or_empty(p, dtype=tokens_dtype) for p in packed_tokens_parts], dim=0)
    #     packed_labels = torch.cat([to_tensor_or_empty(p, dtype=labels_dtype) for p in packed_labels_parts], dim=0)
    #     packed_masks = torch.cat([to_tensor_or_empty(p, dtype=masks_dtype) for p in packed_masks_parts], dim=0)

    #     return ImageTaskSamplePacked(
    #         __key__=",".join([s.__key__ for s in samples]),
    #         __restore_key__=(),  # Will be set by energon if needed
    #         __subflavor__=None,
    #         __subflavors__=samples[0].__subflavors__ if samples else {},
    #         tokens=packed_tokens,
    #         labels=packed_labels,
    #         attn_mask=packed_masks,
    #         imgs=packed_imgs,
    #         pixel_values_videos=packed_videos,
    #         cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
    #         max_length=max_length,
    #         num_tiles=[n for s in samples for n in s.num_tiles],
    #     )
    @stateless
    def pack_selected_samples(self, samples: List[ImageTaskSample]) -> List[ImageTaskSamplePacked]:
        packing_seq_len = int(self.args.seq_length)

        packed_tokens = []
        packed_labels = []
        packed_masks = []
        packed_imgs = []
        packed_videos = []

        current_length = 0
        max_length = 0
        cu_lengths = [0]

        def seq_len_from_tensor(tensor):
            if tensor is None:
                return 0
            if tensor.ndim == 1:
                return tensor.shape[0]
            if tensor.ndim >= 2:
                if tensor.shape[0] == 1 and tensor.shape[1] > 1:
                    return tensor.shape[1]
                return tensor.shape[0]
            return tensor.shape[0]

        def slice_by_seq(tensor, keep_len):
            if tensor is None:
                return None
            if keep_len <= 0:
                if tensor.ndim == 1:
                    return tensor.new_empty((0,))
                seq_axis = 1 if (tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] > 1) else 0
                shape = list(tensor.shape)
                shape[seq_axis] = 0
                return tensor.new_empty(tuple(shape))
            if tensor.ndim == 1:
                return tensor[:keep_len]
            seq_axis = 1 if (tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] > 1) else 0
            if seq_axis == 0:
                return tensor[:keep_len, ...]
            else:
                sl = [slice(None)] * tensor.ndim
                sl[1] = slice(0, keep_len)
                return tensor[tuple(sl)]

        for idx, sample in enumerate(samples):
            tokens = getattr(sample, "tokens", None)
            labels = getattr(sample, "labels", None)
            attn_mask = getattr(sample, "attn_mask", None)

            candidate_lens = []
            if tokens is not None:
                candidate_lens.append(seq_len_from_tensor(tokens))
            if labels is not None:
                candidate_lens.append(seq_len_from_tensor(labels))
            if attn_mask is not None:
                candidate_lens.append(seq_len_from_tensor(attn_mask))
            if hasattr(sample, "total_len"):
                try:
                    candidate_lens.append(int(sample.total_len))
                except Exception:
                    pass

            if len(candidate_lens) == 0:
                raise ValueError(f"Sample {idx} has no sequence info (tokens/labels/attn_mask/total_len).")

            orig_len = max(candidate_lens)
            remaining = packing_seq_len - current_length
            if remaining <= 0:
                break

            keep_len = min(orig_len, remaining, packing_seq_len)

            if keep_len < orig_len:
                print(f"[pack] truncating sample idx={idx} from {orig_len} -> {keep_len} (remaining={remaining}) key={getattr(sample, '__key__', 'N/A')})")

            t_tokens = slice_by_seq(tokens, keep_len) if tokens is not None else None
            t_labels = slice_by_seq(labels, keep_len) if labels is not None else None
            t_mask = slice_by_seq(attn_mask, keep_len) if attn_mask is not None else None

            # --- FIXED: choose first non-None tensor explicitly, don't use `or` with tensors ---
            first_available = None
            for cand in (t_tokens, t_labels, t_mask):
                if cand is not None:
                    first_available = cand
                    break

            tlen = 0 if first_available is None else seq_len_from_tensor(first_available)

            # Ensure tlen <= keep_len (safety)
            if tlen > keep_len:
                # Force-slice to keep_len
                if t_tokens is not None:
                    t_tokens = slice_by_seq(t_tokens, keep_len)
                if t_labels is not None:
                    t_labels = slice_by_seq(t_labels, keep_len)
                if t_mask is not None:
                    t_mask = slice_by_seq(t_mask, keep_len)
                tlen = keep_len

            if tlen > max_length:
                max_length = tlen

            if t_tokens is None:
                raise ValueError(f"Sample {idx} missing tokens after slicing (key={getattr(sample,'__key__','N/A')}).")
            if t_labels is None:
                raise ValueError(f"Sample {idx} missing labels after slicing (key={getattr(sample,'__key__','N/A')}).")
            if t_mask is None:
                raise ValueError(f"Sample {idx} missing attn_mask after slicing (key={getattr(sample,'__key__','N/A')}).")

            packed_tokens.append(t_tokens)
            packed_labels.append(t_labels)
            packed_masks.append(t_mask)

            if getattr(sample, "imgs", None):
                packed_imgs += list(sample.imgs)
            if getattr(sample, "pixel_values_videos", None):
                packed_videos += list(sample.pixel_values_videos)

            current_length += tlen
            cu_lengths.append(current_length)

            if current_length >= packing_seq_len:
                break

        if len(packed_tokens) == 0:
            packed_tokens = torch.empty((0,), dtype=torch.long)
            packed_labels = torch.empty((0,), dtype=torch.long)
            packed_masks = torch.empty((0,), dtype=torch.bool)
        else:
            packed_tokens = torch.cat(packed_tokens, dim=0)
            packed_labels = torch.cat(packed_labels, dim=0)
            packed_masks = torch.cat(packed_masks, dim=0)

        final_len = packed_tokens.shape[0] if packed_tokens is not None else 0
        if final_len > packing_seq_len:
            packed_tokens = packed_tokens[:packing_seq_len]
            packed_labels = packed_labels[:packing_seq_len]
            packed_masks = packed_masks[:packing_seq_len]
            final_len = packing_seq_len
            # best-effort adjust cu_lengths (keep it simple)
            if cu_lengths:
                cu_lengths[-1] = final_len

        return ImageTaskSamplePacked(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__ if samples else {},
            tokens=packed_tokens,
            labels=packed_labels,
            attn_mask=packed_masks,
            imgs=packed_imgs,
            pixel_values_videos=packed_videos,
            cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
            max_length=max_length,
            num_tiles=[n for s in samples for n in s.num_tiles],
        )

def print_error_handler(exc: Exception, key: Optional[str]):
    """ Print error handler function called when an exception occurs during loading. """
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
