"""
Test FastVLM inference with MobileLLM-140M + FastViT
Uses the same model setup as training but loads from checkpoint for inference.
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser

# Add paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'aiak_megatron'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'aiak_training_llm'))

from transformers import AutoTokenizer
from aiak_training_llm.models.fastvit.fastvit_preprocessor import FastViTImageProcessor
from aiak_training_llm.models.fastvit.mm_utils import expand2square
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_config import llava_ov_mobilellm_140m
from aiak_training_llm.models.llavaov_1_5.llavaov_1_5_model import LlavaOnevision1_5
from aiak_training_llm.models.mobilellm.mobilellm_layer_spec import get_mobilellm_layer_with_te_spec
from aiak_training_llm.models.qwen.layer_spec import get_adapeter_layer_with_spec

def main():
    parser = ArgumentParser(description="FastVLM + MobileLLM Inference Test")
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/mobilellm-fastvit-merged-tp1-pp1',
                        help='Path to merged checkpoint')
    parser.add_argument('--tokenizer', type=str,
                        default='checkpoints/MobileLLM-R1-140M',
                        help='Path to tokenizer')
    parser.add_argument('--image', type=str, default='test_image.jpg',
                        help='Path to test image')
    parser.add_argument('--prompt', type=str, default='What is in this image?',
                        help='Text prompt')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='FastViT image size')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    args = parser.parse_args()

    print("=" * 80)
    print("FastVLM + MobileLLM-140M Inference Test")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Image: {args.image}")
    print(f"Image Size: {args.image_size}")
    print("=" * 80)

    # Check checkpoint exists
    checkpoint_path = os.path.join(REPO_ROOT, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Please run training first or provide a valid checkpoint path.")
        return

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer_path = os.path.join(REPO_ROOT, args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"✓ Loaded tokenizer from {tokenizer_path}")
    print(f"  Vocab size: {len(tokenizer)}")

    # Load and preprocess image
    print("\n[2/4] Loading and preprocessing image...")
    image_path = os.path.join(os.path.dirname(__file__), args.image) if not os.path.isabs(args.image) else args.image
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert('RGB')
    print(f"  Original size: {image.size}")
    
    # Initialize FastViT processor
    fastvit_processor = FastViTImageProcessor(image_size=args.image_size)
    mean_color = tuple(int(x * 255) for x in fastvit_processor.image_mean)
    image_padded = expand2square(image, mean_color)
    pixel_values = fastvit_processor(image_padded).unsqueeze(0)  # Add batch dim
    print(f"  Padded to square: {image_padded.size}")
    print(f"  Preprocessed shape: {pixel_values.shape}")

    # Create prompt
    print("\n[3/4] Tokenizing prompt...")
    IMAGE_TOKEN = "<|image_pad|>"
    VISION_START = "<|vision_start|>"
    VISION_END = "<|vision_end|>"
    
    conversation = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{VISION_START}{IMAGE_TOKEN}{VISION_END}\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    input_ids = tokenizer(conversation, return_tensors="pt")["input_ids"]
    print(f"  Prompt: {args.prompt}")
    print(f"  Input IDs shape: {input_ids.shape}")

    # Initialize model
    print("\n[4/4] Loading model and running inference...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # Get model configs
    print("  Building model...")
    language_config, vision_config, adapter_config = llava_ov_mobilellm_140m()
    
    # Get layer specs (FastViT doesn't use layer spec, uses nn.Module directly)
    language_layer_spec = get_mobilellm_layer_with_te_spec()
    vision_layer_spec = None  # FastViT uses direct nn.Module
    adapter_layer_spec = get_adapeter_layer_with_spec()
    
    # Build model
    model = LlavaOnevision1_5(
        language_config=language_config,
        vision_config=vision_config,
        adapter_config=adapter_config,
        language_layer_spec=language_layer_spec,
        vision_layer_spec=vision_layer_spec,
        adapter_layer_spec=adapter_layer_spec,
        language_vocab_size=128256,
        language_max_sequence_length=512,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=False,
        share_embeddings_and_output_weights=True
    )
    
    # Load checkpoint
    print(f"  Loading weights from {checkpoint_path}...")
    checkpoint_file = os.path.join(checkpoint_path, 'release', 'mp_rank_00', 'model_optim_rng.pt')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        # Remove 'module.' prefix if present
        state_dict = {}
        for k, v in checkpoint['model'].items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            state_dict[new_key] = v
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Checkpoint loaded")
        print(f"    Missing keys: {len(missing_keys)} (adapter weights expected)")
        print(f"    Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"  WARNING: Checkpoint file not found: {checkpoint_file}")
        print("  Using randomly initialized weights")
    
    model = model.to(device)
    model.eval()
    print("  ✓ Model ready")
    
    # Move inputs to device
    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)
    
    # Generate
    print("\n" + "=" * 80)
    print("Generating response...")
    print("=" * 80)
    
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for step in range(args.max_new_tokens):
            # Forward pass
            outputs = model(
                input_ids=generated_ids,
                image=pixel_values,
                labels=None
            )
            
            # Get logits for next token
            logits = outputs[0]  # [batch, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :] / args.temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() in [tokenizer.eos_token_id, 128009]:  # llama3 eos
                break
            
            # Print progress
            if (step + 1) % 10 == 0:
                print(f"  Generated {step + 1} tokens...")
    
    # Decode output
    output_ids = generated_ids[0, input_ids.shape[1]:]  # Remove prompt
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"\nGenerated ({output_ids.shape[0]} tokens):")
    print(output_text)
    print("=" * 80)
    
    print("\n✓ Inference complete!")

if __name__ == "__main__":
    main()
