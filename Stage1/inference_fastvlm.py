"""
FastVLM Inference Script
Example run:
python inference_fastvlm.py --checkpoint_path ./stage_1_alignment_llava_ov_4b/iter_0000020 \
                            --image_path ./test_image.jpg \
                            --prompt "What is in this image?"
"""

import os
import sys
import torch
from PIL import Image
from argparse import ArgumentParser

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'aiak_megatron'))

from transformers import AutoProcessor
from aiak_training_llm.models.fastvit.fastvit_preprocessor import FastViTImageProcessor
from aiak_training_llm.models.fastvit.mm_utils import expand2square

# Argument parser
parser = ArgumentParser(description="FastVLM Inference")
parser.add_argument('--checkpoint_path', type=str, 
                    default='/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5/stage_1_alignment_llava_ov_4b/iter_0000020',
                    help='Path to trained checkpoint directory')
parser.add_argument('--tokenizer_path', type=str,
                    default='/share/data/drive_3/mobile_vlm/LLaVA-OneVision-1.5/checkpoints/LLaVA-OneVision-1.5-4B-stage0',
                    help='Path to tokenizer')
parser.add_argument('--image_path', type=str, default='test_image.jpg',
                    help='Path to input image')
parser.add_argument('--prompt', type=str, default='What is in this image?',
                    help='Text prompt for the model')
parser.add_argument('--image_size', type=int, default=1024,
                    help='FastViT image size (384 or 1024)')
parser.add_argument('--use_gpu', action='store_true', default=True,
                    help='Use GPU for inference')
args = parser.parse_args()

print("=" * 80)
print("FastVLM Inference")
print("=" * 80)
print(f"Checkpoint: {args.checkpoint_path}")
print(f"Tokenizer: {args.tokenizer_path}")
print(f"Image: {args.image_path}")
print(f"Prompt: {args.prompt}")
print(f"Image Size: {args.image_size}")
print("=" * 80)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
print(f"Using device: {device}")

# Load tokenizer/processor
print("\nLoading tokenizer and processor...")
processor = AutoProcessor.from_pretrained(args.tokenizer_path, trust_remote_code=True)
tokenizer = processor.tokenizer

# Initialize FastViT image processor
fastvit_processor = FastViTImageProcessor(image_size=args.image_size)
print(f"FastViT processor initialized with image_size={args.image_size}")

# Load and preprocess image
print(f"\nLoading image from: {args.image_path}")
if not os.path.exists(args.image_path):
    print(f"ERROR: Image file not found: {args.image_path}")
    print("Please provide a valid image path using --image_path")
    sys.exit(1)

image = Image.open(args.image_path).convert('RGB')
print(f"Original image size: {image.size}")

# Preprocess with FastViT (pad to square)
mean_color = tuple(int(x * 255) for x in fastvit_processor.image_mean)
image_padded = expand2square(image, mean_color)
print(f"Padded to square: {image_padded.size}")

pixel_values = fastvit_processor(image_padded)
print(f"Preprocessed image shape: {pixel_values.shape}")

# Create prompt with vision tokens
IMAGE_TOKEN = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"

# Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>\nPROMPT<|im_end|>\n<|im_start|>assistant\n
conversation = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{VISION_START}{IMAGE_TOKEN}{VISION_END}\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

print("\nTokenizing prompt...")
input_ids = tokenizer(conversation, return_tensors="pt")["input_ids"]
print(f"Input IDs shape: {input_ids.shape}")
print(f"Prompt tokens: {input_ids.shape[1]}")

# TODO: Load your trained model checkpoint here
# This requires implementing model loading from Megatron checkpoint
print("\n" + "=" * 80)
print("NOTE: Model loading from Megatron checkpoint not yet implemented.")
print("This script currently only demonstrates preprocessing.")
print("\nTo complete inference, you need to:")
print("1. Load the model from checkpoint using Megatron utilities")
print("2. Convert distributed checkpoint to single GPU format")
print("3. Call model.forward() with preprocessed inputs")
print("=" * 80)

# Placeholder for model inference
print("\nPreprocessed inputs ready:")
print(f"  - pixel_values: {pixel_values.shape} ({pixel_values.dtype})")
print(f"  - input_ids: {input_ids.shape}")
print(f"  - Device: {device}")

print("\nInference complete (preprocessing only).")
 