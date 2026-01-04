"""default pretrain for generative models like GPTS"""

import os
import torch

from functools import partial

from transformers.utils import PaddingStrategy
from transformers import AutoProcessor
from megatron.training import get_timers
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.transformer.enums import AttnMaskType

from aiak_training_llm.utils import (
    constants,
    get_args,
    get_chat_template,
    get_tokenizer,
    print_rank_0
)

from aiak_training_llm.models import get_model_provider, get_model_family

from aiak_training_llm.train.megatron_trainer import MegatronTrainer
from aiak_training_llm.train.trainer_builder import register_model_trainer
from aiak_training_llm.data import (
    SFTDataset,
    SFTDatasetConfig,
    BlendedHuggingFaceDatasetBuilder,
    MultiModalDataCollatorForSupervisedDataset
)
from .utils import build_sft_cyclic_iterators, get_dataset_blend_from_list, build_sft_data_collator


stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True):
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        MCoreModel: The returned model
    """
    args = get_args()
    model_family = get_model_family(args.model_name)
    model_provider = get_model_provider(model_family)
    assert model_provider is not None, f'model provider for {args.model_name} not found'
    return model_provider(pre_process, post_process)


def get_batch(data_iterator):
    """Generate a batch"""

    # get batches based on the TP rank you are on
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data([
        "input_ids", 
        # "position_ids", 
        "attention_mask", 
        "labels", 
        "loss_mask", 
        "image_grid_thw", 
        "loss_mask"
    ], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["images"], data, torch.float32)

    # slice batch along sequence dimension for context parallelism
    assert mpu.get_context_parallel_world_size() == 1, "not implemented"

    attention_mask = data_i['attention_mask'].logical_not()
    data_i['labels'] = torch.roll(data_i['labels'], shifts=-1, dims=1)
    data_i['loss_mask'][:, -1] = 0

    batch = (
        data_f['images'],
        data_i['image_grid_thw'],
        data_i['input_ids'],
        None,
        attention_mask,
        data_i['labels'],
        data_i['loss_mask'].float(),
        AttnMaskType.padding_causal if attention_mask.any() else AttnMaskType.causal
    )

    return batch


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """    
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
    
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    loss_reduced_dict = {'lm loss': (reporting_loss[0], reporting_loss[1])}

    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(num_input_tokens, dtype=torch.int, device=output_tensor.device)
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = input_tokens.item() * args.context_parallel_size

    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        loss_reduced_dict
    )


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()

    global stimer
    with stimer(bdata=True):
        images, image_grid_thw, input_ids, position_ids, attention_mask, labels, loss_mask, attn_mask_type \
            = get_batch(data_iterator)
        
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(images, image_grid_thw, input_ids, position_ids, attention_mask, attn_mask_type, labels)
 
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.
    
    For GPT-like models, if there are no special requirements, we should directly reuse the Megatron GPTDataset.
    """
    args = get_args()

    tokenizer = get_tokenizer()

    processor = AutoProcessor.from_pretrained(args.hf_tokenizer_path, trust_remote_code=True)
    if args.image_resolution:
        setattr(processor, "image_resolution", args.image_resolution)

    config = SFTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length, # max sequence length
        enable_discard_sample=args.enable_discard_sample,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
          get_blend_from_list(args.train_data_path),
          get_blend_from_list(args.valid_data_path),
          get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        dataset=get_dataset_blend_from_list(args.sft_dataset),
        dataset_per_split=[
            get_dataset_blend_from_list(args.sft_train_dataset),
            get_dataset_blend_from_list(args.sft_valid_dataset),
            get_dataset_blend_from_list(args.sft_test_dataset)
        ],
        dataset_config_file=args.sft_dataset_config,
        streaming=args.sft_data_streaming,
        streaming_buffer_size=args.streaming_buffer_size,
        mix_strategy=args.sft_data_mix_strategy,
        chat_template=get_chat_template(),
        processor=processor,
        num_preprocess_workers=args.sft_num_preprocess_workers,
        train_on_prompt=args.train_on_prompt,
        ignore_index=constants.IGNORE_INDEX,
        eod_mask_loss=args.eod_mask_loss,
        is_tokenized=args.is_tokenized_data,
        packing=args.packing_sft_data,
        sort_batch=args.sft_sort_batch,
        packing_batch_size=args.packing_batch_size,
        context_parallel_size=args.context_parallel_size,
    )

    train_ds, valid_ds, test_ds = BlendedHuggingFaceDatasetBuilder(
        cls=SFTDataset,
        sizes=train_val_test_num_samples, # NOTE: not use now!
        is_built_on_rank=lambda: mpu.get_tensor_model_parallel_rank() == 0,
        config=config,
    ).build()

    print_rank_0(f"> building sft train, validation, and test datasets for {args.model_name} ...")

    data_collator = build_sft_data_collator(
        MultiModalDataCollatorForSupervisedDataset,
        processor=config.processor,
        plugin=config.chat_template.mm_plugin,
    )

    train_iter, valid_iter, test_iter = build_sft_cyclic_iterators(train_ds, valid_ds, test_ds, data_collator)

    print_rank_0(f"> finished creating {args.model_name} sft datasets ...")

    return train_iter, valid_iter, test_iter

# This registers the function for model family "llava_ov_1_5" and phase "sft"
@register_model_trainer(model_family=[constants.VisionLanguageModelFamilies.LLAVA_OV_1_5],
                        training_phase=constants.TrainingPhase.SFT)
def default_pretrain_trainer(train_args):
    """build trainer"""
    from aiak_training_llm.train.pretrain import pretrain_llavaov_1_5
    #imports the module containing model/dataset/forward functions for LLaVA-OV-1.5
    if train_args.encoder_pipeline_model_parallel_size in [0, None]:
        #check if model use pipeline parallelism with separate encoder and decoder stages
        model_type = ModelType.encoder_or_decoder # unified model
        print_rank_0("Using unified model type for training.")
    else:
        model_type = ModelType.encoder_and_decoder
        print_rank_0("Using encoder-and-decoder model type for training.")
    #created megatron trainer instance   
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=pretrain_llavaov_1_5.train_valid_test_dataset_provider,
        model_provider=pretrain_llavaov_1_5.model_provider,
        model_type=model_type,
        forward_step_func=pretrain_llavaov_1_5.forward_step,
    )

    return trainer
