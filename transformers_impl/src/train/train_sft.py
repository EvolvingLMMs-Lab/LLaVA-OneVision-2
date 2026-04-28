import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from llavaonevision1_5.modeling_llavaonevision1_5.py import LLaVAOneVision1_5_ForConditionalGeneration
from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)
    
    if not training_args.freeze_vision_tower:
        rank0_print("Vision tower parameters are set to trainable")
    else:
        rank0_print("Vision tower parameters are frozen")
        
    if not training_args.freeze_merger:
        rank0_print("Merger parameters are set to trainable")
    else:
        rank0_print("Merger parameters are frozen")

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)
    
    if not training_args.freeze_llm:
        rank0_print("LLM parameters are set to trainable")
    else:
        rank0_print("LLM parameters are frozen")


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
    

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    rank0_print(f"Loading model from: {model_args.model_id}")
    rank0_print(f"Compute dtype: {compute_dtype}")
    rank0_print(f"BnB args: {bnb_model_from_pretrained_args}")
    
    try:
        if "Qwen2.5" in model_args.model_id:
            rank0_print("Loading Qwen2.5-VL model...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                torch_dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
                **bnb_model_from_pretrained_args
            )
        else:
            rank0_print("Loading LLaVAOneVision model...")
            model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
                model_args.model_id,
                torch_dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
                **bnb_model_from_pretrained_args
            )
    except Exception as e:
        rank0_print(f"ERROR loading model: {e}")
        raise

    # 立即检查模型是否正确加载
    rank0_print(f"Model loaded successfully: {type(model)}")
    
    # 快速参数计数来验证模型 - 使用更安全的方式
    try:
        quick_param_count = sum(1 for _ in model.named_parameters())
        rank0_print(f"Model has {quick_param_count} named parameters")
    except Exception as e:
        rank0_print(f"Warning: Could not count named parameters: {e}")
        quick_param_count = -1
    
    # 备用检查方式
    if quick_param_count <= 0:
        try:
            all_params = list(model.parameters())
            rank0_print(f"Model has {len(all_params)} parameters (backup count)")
            quick_param_count = len(all_params)
        except Exception as e:
            rank0_print(f"Warning: Could not count parameters with backup method: {e}")
    
    if quick_param_count == 0:
        rank0_print("WARNING: Model appears to have no parameters!")
        rank0_print("This might be normal in distributed training - continuing...")
    
    # 检查模型的主要组件
    if hasattr(model, 'visual'):
        rank0_print("✓ Model has visual component")
    if hasattr(model, 'model'):
        rank0_print("✓ Model has language model component") 
    if hasattr(model, 'lm_head'):
        rank0_print("✓ Model has lm_head component")

    model.config.use_cache = False
    model_to_configure = model
    
    rank0_print("Configuring model parameters...")
    rank0_print(f"freeze_llm: {training_args.freeze_llm}")
    rank0_print(f"freeze_vision_tower: {training_args.freeze_vision_tower}")
    rank0_print(f"freeze_merger: {training_args.freeze_merger}")
    
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    # 简单测试模型是否能正常前向传播
    rank0_print("Testing basic model functionality...")
    try:
        with torch.no_grad():
            # 创建一个简单的测试输入
            test_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            if torch.cuda.is_available():
                test_input_ids = test_input_ids.cuda()
            
            # 尝试前向传播
            test_output = model(input_ids=test_input_ids)
            rank0_print("✓ Model forward pass test successful")
            
    except Exception as e:
        rank0_print(f"Warning: Model forward pass test failed: {e}")
        rank0_print("This might be normal for distributed training")

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # 调试信息：检查可训练参数
    trainable_params = 0
    total_params = 0
    trainable_param_names = []
    
    rank0_print("Checking model parameters...")
    
    # 先检查模型是否正确加载
    if model is None:
        rank0_print("ERROR: Model is None!")
        raise ValueError("Model failed to load")
    
    # 使用更安全的方式统计参数
    param_count = 0
    param_iter_test = list(model.named_parameters())
    rank0_print(f"Parameters iterator length: {len(param_iter_test)}")
    
    if len(param_iter_test) == 0:
        rank0_print("ERROR: model.named_parameters() returns empty iterator!")
        # 尝试其他方式获取参数信息
        try:
            rank0_print("Trying alternative parameter access methods...")
            all_params = list(model.parameters())
            rank0_print(f"model.parameters() length: {len(all_params)}")
            
            # 检查模型状态字典
            state_dict = model.state_dict()
            rank0_print(f"State dict keys count: {len(state_dict.keys())}")
            rank0_print(f"Sample state dict keys: {list(state_dict.keys())[:5]}")
            
            # 计算state_dict中的参数总数
            total_params_from_state = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            rank0_print(f"Total parameters from state_dict: {total_params_from_state:,}")
            
        except Exception as e:
            rank0_print(f"Error accessing model parameters: {e}")
        
        rank0_print("Continuing with zero parameters assumption for now...")
        total_params = 0
        trainable_params = 0
    else:
        # 正常的参数统计
        for name, param in param_iter_test:
            param_count += 1
            if param is not None and hasattr(param, 'numel'):
                param_size = param.numel()
                total_params += param_size
                if param.requires_grad:
                    trainable_params += param_size
                    trainable_param_names.append(name)
                
                # 打印前几个参数的详细信息
                if param_count <= 5:
                    rank0_print(f"Param {param_count}: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}, numel: {param_size}")
    
    rank0_print(f"Total parameter count: {param_count}")
    rank0_print(f"Trainable parameters: {trainable_params:,}")
    rank0_print(f"Total parameters: {total_params:,}")
    
    # 只有在真正没有参数时才报错
    if total_params == 0 and len(param_iter_test) == 0:
        rank0_print("ERROR: Model appears to have no accessible parameters!")
        rank0_print("This might be a DeepSpeed/distributed training issue.")
        rank0_print("Model type:", type(model))
        rank0_print("Model config:", model.config if hasattr(model, 'config') else "No config")
        # 不直接报错，而是警告并继续
        rank0_print("WARNING: Continuing training despite parameter access issues...")
    
    if total_params > 0:
        trainable_percentage = 100 * trainable_params / total_params
        rank0_print(f"Trainable percentage: {trainable_percentage:.2f}%")
    else:
        rank0_print("Cannot calculate trainable percentage due to parameter access issues")
    
    # 打印一些可训练参数的名称
    if len(trainable_param_names) > 0:
        rank0_print(f"Sample trainable params: {trainable_param_names[:10]}")
    
    if trainable_params == 0 and len(param_iter_test) > 0:
        rank0_print("WARNING: No trainable parameters found!")
        rank0_print("Checking parameter requires_grad status...")
        sample_count = 0
        for name, param in model.named_parameters():
            if ("visual" in name or "lm_head" in name or "model.embed_tokens" in name) and sample_count < 10:
                rank0_print(f"{name}: requires_grad={param.requires_grad}")
                sample_count += 1

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    # 调试信息：检查数据集
    train_dataset = data_module['train_dataset']
    rank0_print(f"Training dataset size: {len(train_dataset)}")
    
    # # 检查第一个样本
    # if len(train_dataset) > 0:
    #     sample = train_dataset[0]
    #     rank0_print(f"Sample keys: {list(sample.keys())}")
    #     rank0_print(f"Input IDs shape: {sample['input_ids'].shape}")
    #     rank0_print(f"Labels shape: {sample['labels'].shape}")
    #     rank0_print(f"Labels min/max: {sample['labels'].min()}/{sample['labels'].max()}")
    #     # 检查是否有有效的标签
    #     valid_labels = (sample['labels'] != -100).sum()
    #     total_labels = sample['labels'].numel()
    #     rank0_print(f"Valid labels count: {valid_labels} / {total_labels}")
    #     if valid_labels == 0:
    #         rank0_print("ERROR: No valid labels found in sample! All labels are -100")
    #         rank0_print("This will cause loss=0. Check your data processing.")
    #     else:
    #         rank0_print(f"Data looks good: {valid_labels}/{total_labels} labels are valid")
    #     # 检查input_ids是否有效
    #     if sample['input_ids'].max() >= 151936:  # Qwen2-VL vocab size
    #         rank0_print(f"WARNING: Input IDs contain out-of-vocab tokens: max={sample['input_ids'].max()}")
    # else:
    #     rank0_print("ERROR: Empty dataset!")

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
