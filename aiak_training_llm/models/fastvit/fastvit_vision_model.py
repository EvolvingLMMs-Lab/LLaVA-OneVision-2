#
# FastViT Vision Model wrapper for Megatron integration
# Adapted from Apple's FastVLM
#

import torch
import torch.nn as nn
from megatron.core.transformer import MegatronModule

from aiak_training_llm.models.fastvit.mobileclip_encoder import MobileCLIPVisionTower


class FastViTModel(MegatronModule):
    """
    FastViT Vision Model wrapper for Megatron.
    
    This wraps the MobileCLIPVisionTower (which contains FastViTHD)
    to be compatible with Megatron's training framework.
    
    Args:
        config: Megatron TransformerConfig
        layer_spec: Module specification (not used for FastViT)
    """
    
    def __init__(self, config, layer_spec):
        # Initialize MegatronModule (will set self.config)
        super().__init__(config=config)
        
        # Create a simple args object for MobileCLIPVisionTower
        class Args:
            def __init__(self):
                self.unfreeze_mm_vision_tower = False
        
        args = Args()
        
        # FastViTHD model name with resolution
        # Format: mobileclip_l_{resolution}
        # The first two parts must match the config file name (mobileclip_l.json)
        # The resolution is extracted from the last part
        vision_tower_name = "mobileclip_l_1024"  
        
        # Override with config if provided
        if hasattr(config, 'vision_tower_name'):
            vision_tower_name = config.vision_tower_name
        elif hasattr(config, 'img_h') and config.img_w == config.img_h:
            # Use square image size from config
            vision_tower_name = f"mobileclip_l_{config.img_h}"
        
        # Initialize the FastViT vision tower
        self.vision_tower = MobileCLIPVisionTower(
            vision_tower=vision_tower_name,
            args=args,
            delay_load=False
        )
        
        self._hidden_size = self.vision_tower.hidden_size
        
    def forward(self, images, grid_thw=None):
        """
        Forward pass through FastViT.
        
        Args:
            images: Image tensor [batch, channels, height, width]
            grid_thw: Grid dimensions (not used by FastViT, kept for API compatibility)
            
        Returns:
            Vision features [batch, num_tokens, hidden_size]
            window_index: None (FastViT doesn't use windowing)
        """
        print(f"[FastViTModel] Input images shape: {images.shape}, grid_thw: {grid_thw}")
        # MobileCLIPVisionTower expects single images or list of images
        # For batch processing, we need to handle it appropriately
        if images.dim() == 4:
            # Batch of images [B, C, H, W]
            image_features = self.vision_tower.forward_images(images)
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")
        
        print(f"[FastViTModel] Output features shape: {image_features.shape}")
        # Return features and None for window_index (compatibility with Qwen2-VL API)
        return image_features, None
    
    def set_input_tensor(self, input_tensor):
        """
        Set input tensor. 
        
        FastViT doesn't use pipeline parallelism, so this is a no-op.
        Required for compatibility with Megatron's pipeline parallel interface.
        """
        pass
    
    @property
    def hidden_size(self):
        """Return the hidden size of the vision model output."""
        return self._hidden_size
    
    @property
    def vision_config(self):
        """Return vision tower config."""
        return self.vision_tower.config
