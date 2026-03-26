#
# FastViT Image Preprocessing
# Following Apple FastVLM preprocessing approach
#

from PIL import Image
import torch
from transformers import CLIPImageProcessor


class FastViTImageProcessor:
    """
    Image processor for FastViT following FastVLM's approach.
    Uses CLIPImageProcessor with specific settings for FastViT.
    """
    
    def __init__(self, image_size=1024):
        """
        Initialize FastViT image processor.
        
        Args:
            image_size: Input image size (default 384 as in FastVLM)
        """
        self.image_size = image_size
        
        # Create CLIPImageProcessor with FastViT settings
        # Following mobileclip_encoder.py from FastVLM
        # Source: ml-fastvlm/llava/model/multimodal_encoder/mobileclip_encoder.py (lines 45-49)
        self.image_mean = [0.0, 0.0, 0.0]  # No normalization (black padding for 'pad' mode)
        self.image_std = [1.0, 1.0, 1.0]
        
        self.processor = CLIPImageProcessor(
            crop_size={"height": image_size, "width": image_size},
            image_mean=self.image_mean,
            image_std=self.image_std,
            size={"shortest_edge": image_size}
        )
    
    def preprocess(self, image):
        """
        Preprocess a single image for FastViT.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, Image.Image):
            # Process with CLIPImageProcessor
            processed = self.processor(images=image, return_tensors="pt")
            return processed['pixel_values'][0]  # Remove batch dimension
        else:
            raise ValueError(f"Expected PIL Image, got {type(image)}")
    
    def __call__(self, images):
        """
        Process single image or batch of images.
        
        Args:
            images: PIL Image or list of PIL Images
            
        Returns:
            Tensor of preprocessed images
        """
        if isinstance(images, list):
            return torch.stack([self.preprocess(img) for img in images])
        else:
            return self.preprocess(images).unsqueeze(0)
