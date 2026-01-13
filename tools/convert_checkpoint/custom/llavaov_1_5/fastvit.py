"""
Convert FastViT vision encoder weights from HuggingFace to Megatron format.
"""
import argparse
import os
import sys
import torch
from safetensors import safe_open

def convert_fastvit_weights(load_path, save_path, config):
    """
    Convert FastViT weights from HF format to Megatron format.
    
    HF format: model.vision_tower.vision_tower.model.*
    Megatron format: vision_model.vision_tower.model.*
    """
    print(f"Loading FastViT weights from {load_path}")
    
    # Load weights from safetensors
    hf_weights = {}
    safetensors_path = os.path.join(load_path, "model.safetensors")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model.vision_tower.vision_tower"):
                hf_weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(hf_weights)} FastViT weight tensors")
    
    # Convert keys from HF to Megatron format
    # HF: model.vision_tower.vision_tower.model.*
    # Megatron: vision_model.vision_tower.model.*
    mcore_weights = {}
    
    vision_model_from = config.get("vision_model_from", "model.vision_tower.vision_tower")
    vision_model_to = config.get("vision_model_to", "vision_model.vision_tower")
    
    for hf_key, weight in hf_weights.items():
        if hf_key.startswith(vision_model_from):
            # Strip the HF prefix and add Megatron prefix
            suffix = hf_key[len(vision_model_from):]
            mcore_key = vision_model_to + suffix
            mcore_weights[mcore_key] = weight
            
            if len(mcore_weights) <= 5:  # Show first 5 conversions
                print(f"  {hf_key} -> {mcore_key}")
    
    print(f"Converted {len(mcore_weights)} weight tensors")
    
    # Save in Megatron format
    os.makedirs(os.path.join(save_path, "release"), exist_ok=True)
    save_file = os.path.join(save_path, "release", "mp_rank_00", "model_optim_rng.pt")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    checkpoint = {
        'model': mcore_weights,
        'checkpoint_version': 3.0,
        'args': {
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1,
        }
    }
    
    torch.save(checkpoint, save_file)
    print(f"Saved Megatron checkpoint to {save_file}")
    
    return mcore_weights


def main():
    parser = argparse.ArgumentParser(description="Convert FastViT weights to Megatron format")
    parser.add_argument("--load_platform", type=str, default="huggingface")
    parser.add_argument("--save_platform", type=str, default="mcore")
    parser.add_argument("--common_config_path", type=str, required=True)
    parser.add_argument("--load_ckpt_path", type=str, required=True)
    parser.add_argument("--save_ckpt_path", type=str, required=True)
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--safetensors", action="store_true")
    parser.add_argument("--no_save_optim", action="store_true")
    parser.add_argument("--no_load_optim", action="store_true")
    
    args = parser.parse_args()
    
    # Load config
    import json
    with open(args.common_config_path, 'r') as f:
        config = json.load(f)
    
    # Convert weights
    convert_fastvit_weights(args.load_ckpt_path, args.save_ckpt_path, config)
    
    print("FastViT conversion completed successfully!")


if __name__ == "__main__":
    main()
