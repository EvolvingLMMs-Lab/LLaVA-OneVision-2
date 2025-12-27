""" PackedCaptioningSample """

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
import torch

@dataclass
class PackedCaptioningSample(Sample):
    """Sample type for packed captioning."""
    # sample_id: str
    images: Union[
        str,                    # A single image path, e.g., 'img001.jpg'
        torch.Tensor,           # A single image tensor
        List[str],              # A list of image paths, e.g., ['imgs001.jpg', 'imgs002.png']
        List[List[str]],        # A nested list of image paths, e.g., [['imgs001.tif'], [], ['imgs005.png']]
        List[torch.Tensor],     # A list of image tensors
        List[List[torch.Tensor]] 
    ]
    prompts: Union[
        Optional[List[str]],
        List[List[str]]
    ]
    captions: Union[
        List[str],
        List[List[str]]
    ]
    