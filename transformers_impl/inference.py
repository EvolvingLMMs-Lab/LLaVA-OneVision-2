import sys
import os
import argparse
import torch
from transformers import AutoProcessor
from llavaonevision1_5.modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration
from qwen_vl_utils import process_vision_info



def generate_for_messages(model, processor, messages):
    """
    A helper function to run the full generation pipeline for a given set of messages.
    """
    # --- Preparation for inference ---
    # Apply the chat template to format the prompt
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process visual information (images/videos) from the messages
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Combine text, image, and video inputs into a single model input
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move inputs to the same device as the model
    inputs = inputs.to(model.device)

    # --- Inference: Generation of the output ---
    generated_ids = model.generate(**inputs, max_new_tokens=512, eos_token_id=151645)
    
    # Trim the generated IDs to remove the prompt portion
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode the output text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Print the prompt and the generated text
    prompt = messages[0]['content'][1]['text']
    print(f"\n> Prompt: {prompt}")
    print(f"> Generated Text: {output_text[0].strip()}")


def main(args):
    """
    Main function to load the model and generate captions for an image in English and Chinese.
    """
    print(f"Loading model from path: {args.model_path}")
    
    # Load the model and processor from the specified path
    # device_map="auto" will handle placing the model on available GPUs
    model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=False # Recommended for custom models
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=False
    )
    print("✓ Model and processor loaded successfully.")
    print(f"Using image: {args.image_path}")

    # --- Test with English Prompt ---
    print("\n--- Testing with English Prompt ---")
    english_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    generate_for_messages(model, processor, english_messages)

    # --- Test with Chinese Prompt ---
    print("\n--- Testing with Chinese Prompt ---")
    chinese_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": "请用中文详细描述这张图片。"},
            ],
        }
    ]
    generate_for_messages(model, processor, chinese_messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from an image using a Qwen2-VL model.")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to the directory containing the pretrained model and processor."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Optional: Path or URL to the image. Defaults to a demo image."
    )
    
    args = parser.parse_args()
    main(args)
    
