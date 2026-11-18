"""MultiMixQASample"""

from dataclasses import dataclass
from typing import List, Optional, Union
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import VideoData
import torch
import numpy as np


@dataclass
class MultiMixQASample(Sample):
    """Sample type for mix question answering."""

    #: The context/question for the video, image or pure text QA.
    messages: List[dict]

    #: The video data containing the image and audio info.
    video: List[VideoData] = None

    #: The input image tensor in the shape (C, H, W)
    image: List[torch.Tensor] = None

    # system
    system: Optional[str] = None

    # patch positions for each image: List of np.ndarray with shape (num_patches, 3) containing [T, H, W]
    patch_positions: Optional[List[np.ndarray]] = None

    #: The frames per second of the video
    fps: Optional[Union[float, int]] = None

    #: Number of decimal places for frame timestamps (1 or 2, default 1)
    timestamp_decimal: Optional[int] = None

    #: Sample-level modality marker, e.g. "streaming". None means ordinary offline data.
    data_type: Optional[str] = None
