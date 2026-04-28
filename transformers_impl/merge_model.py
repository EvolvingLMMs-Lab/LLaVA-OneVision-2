from llavaonevision1_5.configuration_llavaonevision1_5 import Llavaonevision1_5Config
from llavaonevision1_5.modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration
from transformers import Qwen2Tokenizer, AutoProcessor, AutoConfig
from transformers import MLCDVisionModel
from transformers import CLIPImageProcessor
from transformers import Qwen2VLImageProcessor, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import numpy as np
from transformers import Qwen2Tokenizer, logging
from safetensors.torch import load_file
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, snapshot_download
import requests
from io import BytesIO

import argparse

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
CUDA_DEVICE=0

def cosine_similarity(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))

def create_test_image():
    img = Image.new('RGB', (560, 560), color='red')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 474, 474], fill='blue')
    draw.text((100, 100), "TEST", fill='white')
    return img

def load_empty_model(llm_path):
    print("Loading tokenizer and processor from Qwen2.5-VL and empty model...")
    tokenizer = Qwen2Tokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True, device_map={"": f"cuda:{CUDA_DEVICE}"}, use_fast=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    processor.image_processor.temporal_patch_size = 1
    processor.image_processor.max_pixels = 1600*1600
    llava_ov_config = Llavaonevision1_5Config()
    llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, use_fast=True)
    llava_ov_config.text_config.update(llm_config.to_dict())
    llava_ov_config.text_config.tie_word_embeddings = False
    llava_ov_config.vision_config.text_hidden_size = llava_ov_config.text_config.hidden_size
    model = LLaVAOneVision1_5_ForConditionalGeneration(llava_ov_config)
    return model, processor, tokenizer

def load_vit_weights(model, vit_path):
    """
    Load ViT weights and copy them to the vision part of LLaVAOneVision1_5_ForConditionalGeneration
    
    Args:
        model: LLaVAOneVision1_5_ForConditionalGeneration
        vit_path: ViT model path
    """
    print(f"Loading weight form: {vit_path}")

    if os.path.exists(vit_path):
        print(f"Loading weights from local file: {vit_path}")
        cache_path = os.path.join(vit_path, "model.safetensors")
    else:
        print(f"Loading weights from Hugging Face Hub: {vit_path}")
        cache_path = hf_hub_download(vit_path, "model.safetensors")

    vit_weights = load_file(cache_path)
    loaded_keys = 0
    VIT_KEYS_TO_MODIFY_MAPPING = {
        "vision_model.": "model.visual.",
        "model.visual.embeddings.": "model.visual.",
        "model.visual.patch_embedding.": "model.visual.patch_embed.proj.",
        "model.visual.encoder.layers.": "model.visual.blocks.",
        "model.visual.pre_layrnorm": "model.visual.pre_layernorm",
        ".layer_norm": ".norm",
        ".self_attn.out_proj.": ".attn.proj.",
    }
    def merge_qkv_weights(state_dict, block_prefix):
        # Merge q_proj, k_proj, v_proj weights and biases
        q_w = state_dict[f"{block_prefix}.self_attn.q_proj.weight"]
        k_w = state_dict[f"{block_prefix}.self_attn.k_proj.weight"]
        v_w = state_dict[f"{block_prefix}.self_attn.v_proj.weight"]
        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict[f"{block_prefix}.self_attn.q_proj.bias"]
        k_b = state_dict[f"{block_prefix}.self_attn.k_proj.bias"]
        v_b = state_dict[f"{block_prefix}.self_attn.v_proj.bias"]
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)
        return {f"{block_prefix}.attn.qkv.weight": qkv_weight, f"{block_prefix}.attn.qkv.bias": qkv_bias}

    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in VIT_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value

        new_state_dict2 = {}
        for key, value in new_state_dict.items():
            if key.startswith("model.visual.blocks.") and "self_attn" in key and ("q_proj" in key or "k_proj" in key or "v_proj" in key):
                block_index = key.split('.')[3]
                block_prefix = f"model.visual.blocks.{block_index}"
                if f"{block_prefix}.self_attn.q_proj.weight" in new_state_dict:
                    merge_res = merge_qkv_weights(new_state_dict, block_prefix)
                    new_state_dict2.update(merge_res)
            else:
                new_state_dict2[key] = value
        return new_state_dict2
    
    vit_weights = convert_state_dict(vit_weights)
    vit_weights.pop("model.visual.post_layernorm.weight")
    vit_weights.pop("model.visual.post_layernorm.bias")
    vit_keys = len(set(vit_weights.keys()))
    
    model_state_dict = model.state_dict()
    total_keys = len(model_state_dict.keys())
    for vit_key in vit_weights:
        if vit_key not in model_state_dict:
            logger.warning(f"ViT key {vit_key} not found in model, skipping...")
            continue
        model_state_dict[vit_key] = vit_weights[vit_key].clone()
        loaded_keys += 1
    assert loaded_keys == vit_keys, f"ViT weight loading incomplete: {loaded_keys}/{vit_keys} parameters loaded"
    model.load_state_dict(model_state_dict)
    print(f"ViT weights loaded successfully: {loaded_keys}/{total_keys} parameters loaded")

    return vit_weights, loaded_keys

def load_adapter_weights(model, adapter_path, cur_len):
    """
    Load Adapter weights and copy them to the corresponding part of LLaVAOneVision1_5_ForConditionalGeneration
    
    Args:
        model: LLaVAOneVision1_5_ForConditionalGeneration model
        adapter_path: Adapter model path
    """
    print(f"Loading Adapter weights from: {adapter_path}")

    # Load Adapter weights
    if adapter_path.endswith('.safetensors'):
        adapter_weights = load_file(adapter_path)
    else:
        adapter_weights = torch.load(adapter_path, map_location="cpu")
        if "state_dict" in adapter_weights:
            adapter_weights = adapter_weights["state_dict"]

    # Count successfully loaded parameters
    loaded_keys = 0
    total_keys = 0
    ADAPTER_KEYS_TO_MODIFY_MAPPING = {
        "model.mm_projector": "model.visual.merger"
    }
    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in ADAPTER_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value
        return new_state_dict
    
    adapter_weights = convert_state_dict(adapter_weights)
    adapter_keys = len(set(adapter_weights.keys()))

    # Load weights into model
    model_state_dict = model.state_dict()
    total_keys = len(model_state_dict.keys())
    for adapter_key in adapter_weights:
        if adapter_key not in model_state_dict:
            logger.warning(f"Adapter key {adapter_key} not found in model, skipping...")
            continue
        model_state_dict[adapter_key] = adapter_weights[adapter_key].clone()
        loaded_keys += 1
    assert loaded_keys == adapter_keys, f"Adapter weight loading incomplete: {loaded_keys}/{adapter_keys} parameters loaded"
    model.load_state_dict(model_state_dict)
    print(f"Adapter weights loaded successfully: {loaded_keys+cur_len}/{total_keys} parameters loaded")

    return adapter_weights, cur_len + loaded_keys

def load_llm_weights(model, llm_path, cur_len):
    """
    Load LLM model weights and copy them to the language model part of Qwen2VL

    Args:
        model: LLaVAOneVision1_5_ForConditionalGeneration model
        llm_path: LLM model path
    """
    print(f"Loading weight form: {llm_path}")
    if os.path.exists(llm_path):
        cache_path = llm_path
    else:
        cache_path = snapshot_download(llm_path, allow_patterns="*.safetensors")

    llm_weights = {}
    if os.path.isdir(cache_path):
        for filename in os.listdir(cache_path):
            if filename.endswith('.safetensors'):
                filepath = os.path.join(cache_path, filename)
                weights = load_file(filepath)
                llm_weights.update(weights)
    elif cache_path.endswith('.safetensors'):
        llm_weights = load_file(cache_path)
    else:
        llm_weights = torch.load(cache_path, map_location="cpu")
        if "state_dict" in llm_weights:
            llm_weights = llm_weights["state_dict"]
    
    loaded_keys = 0

    ADAPTER_KEYS_TO_MODIFY_MAPPING = {
        "model.": "model.language_model.",
    }
    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in ADAPTER_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value
        return new_state_dict
    
    llm_weights = convert_state_dict(llm_weights)
    if 'lm_head.weight' not in llm_weights:
        llm_weights['lm_head.weight'] = llm_weights['model.language_model.embed_tokens.weight']
    llm_keys = len(set(llm_weights.keys()))
    
    model_state_dict = model.state_dict()
    for llm_key in llm_weights:
        if llm_key not in model_state_dict:
            logger.warning(f"LLM key {llm_key} not found in model, skipping...")
            continue
        model_state_dict[llm_key] = llm_weights[llm_key].clone()
        loaded_keys += 1
    assert loaded_keys == llm_keys, f"LLM weight loading incomplete: {loaded_keys}/{llm_keys} parameters loaded"
    
    return llm_weights

def validate_vit_consistency(model, vit_path, img_path):
    """
    Verify the consistency of the ViT component
    
    Args:
        model: LLaVAOneVision1_5_ForConditionalGeneration after merged
        vit_path: original ViT model path
        sample_image: sample image
    """
    print("Verifying consistency of ViT component...")
    response = requests.get(img_path)
    sample_image = Image.open(BytesIO(response.content)).convert("RGB")
    sample_image = sample_image.resize((560, 560))
    
    rice_model = MLCDVisionModel.from_pretrained(vit_path, device_map={"": f"cuda:{CUDA_DEVICE}"})
    processor = CLIPImageProcessor.from_pretrained(vit_path, device_map={"": f"cuda:{CUDA_DEVICE}"}, use_fast=True)
    rice_inputs = processor.preprocess(images=sample_image, return_tensors="pt").to(dtype=model.dtype, device=rice_model.device)
        
    rice_model = rice_model.eval()
    print(rice_inputs["pixel_values"].size())
    with torch.no_grad():
        output_list = rice_model(**rice_inputs, output_hidden_states=True).hidden_states
    reord_output_list = []
    def spatial_reorder(tensor):
        H, W, C = tensor.shape
        blocks = tensor.view(H//2, 2, W//2, 2, C)
        blocks = blocks.permute(0, 2, 1, 3, 4)
        blocks = blocks.reshape(H//2, W//2, 4, C)
        return blocks.view(-1, C)
    for output in output_list:
        output = output[0,1:].reshape(40, 40, -1).cpu()
        output = spatial_reorder(output)
        reord_output_list.append(output)
    rice_vit_features = reord_output_list[-1]

    image_grid_thw = torch.tensor([[1, 40, 40]], device=model.device, dtype=torch.long)
    image_processor = Qwen2VLImageProcessor()
    image_processor.temporal_patch_size=1
    processed_image = image_processor(sample_image, return_tensors="pt")
    with torch.no_grad():
        merged_output = model.visual(processed_image['pixel_values'].to(device=model.device,dtype=model.dtype), grid_thw=image_grid_thw, is_verifying=True)
        
    if isinstance(merged_output, torch.Tensor) and isinstance(rice_vit_features, torch.Tensor):
        diff = (merged_output - rice_vit_features).abs().mean().item()
        print(f"Mean difference of ViT outputs: {diff:.4f}")
        if diff < 5e-2:
            print("✅ ViT component consistency verification passed")
        else:
            print("❌ ViT component consistency verification failed")

def validate_llm_consistency(model, llm_path, sample_text):
    """
    Verify the consistency of the LLM component
    
    Args:
        model: Merged LLaVAOneVision1_5_ForConditionalGeneration model
        llm_path: Original LLM model path
        sample_text: Sample text
    """
    print("Verifying consistency of LLM component...")

    # Load original LLM model
    original_llm = AutoModelForCausalLM.from_pretrained(llm_path).to(dtype=model.dtype, device=model.device)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)

    # Prepare sample text
    inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        merged_output = model(**inputs).logits

        # Original LLM output
        original_output = original_llm(**inputs).logits

        cur_sim = cosine_similarity(merged_output.flatten(0,1).cpu(), original_output.flatten(0,1).cpu())

    # Compare results
    diff = (merged_output - original_output).abs().mean().item()
    print(f"LLM output mean difference: {diff:.8f}")
    if diff < 1e-3 or cur_sim < 0.99:
        print("✅ LLM component consistency verification passed")
    else:
        print("❌ LLM component consistency verification failed")

def save_merged_model(model, output_path, tokenizer, image_processor):
    """
    Save the merged model

    Args:
        model: Merged model
        output_path: Output path
    """
    print(f"Saving merged model to: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save model configuration
    tokenizer.save_pretrained(output_path)
    image_processor.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print("Model saving completed.")

def main(args):
    # model paths
    vit_path = args.vit_path
    adapter_path = args.adapter_path
    llm_path = args.llm_path
    output_path = args.output_path
    img_path = args.img_path
    sample_text = args.sample_text
    
    # 1. load empty model
    model, processor, tokenizer = load_empty_model(llm_path)
    model.to(dtype=torch.float32)
    
    pretrain_weights = {}
    # 2. load ViT weights
    vit_weights, cur_len = load_vit_weights(model, vit_path)
    pretrain_weights.update(vit_weights)

    # 3. load Adapter weights
    if adapter_path:
        adapter_weights, cur_len = load_adapter_weights(model, adapter_path, cur_len)
        pretrain_weights.update(adapter_weights)

    # 4. load LLM weights
    llm_weights = load_llm_weights(model, llm_path, cur_len)
    pretrain_weights.update(llm_weights)

    model.load_state_dict(pretrain_weights, strict=False)

    # 5. validate model consistency
    validate_vit_consistency(model, vit_path, img_path)
    validate_llm_consistency(model, llm_path, sample_text)

    # 6. save merged model
    save_merged_model(model.to(dtype=torch.bfloat16), output_path, tokenizer, processor)
    print("Model merging process completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ViT and LLM models")
    parser.add_argument("--vit_path", type=str, default="DeepGlint-AI/rice-vit-large-patch14-560", help="Path to the ViT model")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Path to the LLM model")
    parser.add_argument("--output_path", type=str, default="./checkpoints/merged/LLaVA-OneVision-1.5-4B-stage0", help="Path to save the merged model")
    parser.add_argument("--adapter_path", type=str, default="", help="Path to the Adapter model (optional)")
    parser.add_argument("--img_path", type=str, default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", help="Path to the image file")
    parser.add_argument("--sample_text", type=str, default="Hello, my dog is cute", help="Sample text for LLM consistency check")
    args = parser.parse_args()
    main(args)
