"""default pretrain for generative models like GPTS"""

import os
from functools import partial

import torch
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import StragglerDetector
from megatron.training import get_timers

from aiak_training_llm.data.multimodal.dataloader_provider import get_train_dataset, get_train_loader
from aiak_training_llm.data.multimodal.gemma4_vl_task_encoder import Gemma4VLTaskEncoder
from aiak_training_llm.models import get_model_family, get_model_provider
from aiak_training_llm.models.gemma4_vl.gemma4_vl_attention_mask import (
    build_gemma4_mm_attention_mask,
)
from aiak_training_llm.models.qwen_vl.utils import get_inputs_on_this_cp_rank
from aiak_training_llm.train.megatron_trainer import MegatronTrainer
from aiak_training_llm.train.sft.utils import build_sft_data_collator
from aiak_training_llm.train.trainer_builder import register_model_trainer
from aiak_training_llm.utils import constants, get_args
from transformers import DataCollatorForSeq2Seq


stimer = StragglerDetector()

# Gemma4-26B-A4B special token ids (from HF config.json).
image_token_id = 258880
video_token_id = 258884
vision_start_token_id = 255999


def gemma4_vl_embedding_ranks(pp_ranks):
    """gemma4_vl's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size or 0

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def gemma4_vl_position_embedding_ranks(pp_ranks):
    """gemma4_vl's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size or 0

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]


def model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
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
    assert model_provider is not None, f"model provider for {args.model_name} not found"
    return model_provider(pre_process, post_process, add_encoder, add_decoder)


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        if isinstance(data.get("tokens"), torch.Tensor):
            orig_dtype = data["tokens"].dtype
            if data["tokens"].dtype != torch.long:
                print(
                    f"[WARN] tokens dtype {orig_dtype} -> force cast to torch.long; shape={tuple(data['tokens'].shape)}"
                )
                data["tokens"] = data["tokens"].to(torch.long)
        if isinstance(data.get("labels"), torch.Tensor) and data["labels"].dtype != torch.long:
            data["labels"] = data["labels"].to(torch.long)

        assert isinstance(data["tokens"], torch.Tensor) and data["tokens"].dtype == torch.long, (
            f"Expected tokens torch.int64 but got {type(data['tokens'])} {getattr(data['tokens'], 'dtype', None)}"
        )
        assert isinstance(data["labels"], torch.Tensor) and data["labels"].dtype == torch.long, (
            f"Expected labels torch.int64 but got {type(data['labels'])} {getattr(data['labels'], 'dtype', None)}"
        )
    else:
        data = None

    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.bool)["attn_mask"]
    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"]

    has_video = video_token_id in tokens
    has_image = image_token_id in tokens
    thw = None
    video_grid_thw = None
    imgs = None
    pixel_values_videos = None
    patch_positions = None
    if has_image:
        imgs = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
        thw = tensor_parallel.broadcast_data(["image_grid_thw"], data, torch.int32)["image_grid_thw"]
        # Synchronize whether patch_positions is available across all TP ranks.
        # Only rank 0 of each TP group has `data`; other ranks have `data = None`.
        # Using a collective (all_reduce MAX) ensures every rank agrees before
        # deciding whether to call broadcast_data, preventing a collective mismatch
        # that would otherwise corrupt the communication state and cause NaN loss.
        has_pp = int(data is not None and "patch_positions" in data and data["patch_positions"] is not None)
        has_pp_tensor = torch.tensor([has_pp], dtype=torch.int32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(
            has_pp_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group(),
        )
        if has_pp_tensor.item():
            patch_positions = tensor_parallel.broadcast_data(["patch_positions"], data, torch.int64)["patch_positions"]
    if has_video:
        pixel_values_videos = tensor_parallel.broadcast_data(["pixel_values_videos"], data, torch.float32)[
            "pixel_values_videos"
        ]
        video_grid_thw = tensor_parallel.broadcast_data(["video_grid_thw"], data, torch.int32)["video_grid_thw"]

    packed_seq_params = None
    is_video = video_token_id in tokens

    attn_mask_type = AttnMaskType.padding_causal if attn_mask.any() else AttnMaskType.causal

    labels = torch.roll(labels, shifts=-1, dims=1)
    loss_mask = (labels != -100).long()

    if cu_lengths.shape == torch.Size([1, 1]):
        for i in range(attn_mask.shape[0]):
            loss_mask[i, (attn_mask[i] == False).sum() - 1] = 0
    else:
        assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
        # for i in range(cu_lengths.shape[0]):
        #     for j in range(1, cu_lengths[i].shape[0]):
        #         loss_mask[i, cu_lengths[i][j] - 1] = 0

        attn_mask = None
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            max_seqlen_q=max_lengths[0].item(),
            max_seqlen_kv=max_lengths[0].item(),
        )

    # Pad sequence for Sequence Parallelism (SP) divisibility.
    # SP requires sequence length to be divisible by TP size;
    # when combined with CP, it must be divisible by TP * CP * 2.
    # This padding is required for both non-packed and packed sequences because
    # scatter_to_sequence_parallel_region asserts divisibility by TP size.
    if args.sequence_parallel:
        tp_size = args.tensor_model_parallel_size
        cp_size = args.context_parallel_size
        if cp_size > 1:
            padding_factor = tp_size * cp_size * 2
        else:
            padding_factor = tp_size
        seq_len = tokens.shape[1]
        if seq_len % padding_factor != 0:
            padded_len = (seq_len + padding_factor - 1) // padding_factor * padding_factor
            pad_size = padded_len - seq_len
            pad_token_id = getattr(args, "pad_token_id", 0)
            tokens = F.pad(tokens, (0, pad_size), value=pad_token_id)
            labels = F.pad(labels, (0, pad_size), value=-100)
            loss_mask = F.pad(loss_mask, (0, pad_size))
            if packed_seq_params is not None:
                # Append a dummy sequence entry so the packed-sequence attention
                # kernel covers every element in the padded tensor.
                # cu_seqlens ends with the original total token count; adding a
                # new entry at (original_total + pad_size) represents the padding
                # tokens as one extra sequence.
                new_end_q = packed_seq_params.cu_seqlens_q[-1:] + pad_size
                new_end_kv = packed_seq_params.cu_seqlens_kv[-1:] + pad_size
                max_seqlen_q = max(packed_seq_params.max_seqlen_q, pad_size)
                max_seqlen_kv = max(packed_seq_params.max_seqlen_kv, pad_size)
                packed_seq_params = PackedSeqParams(
                    qkv_format=packed_seq_params.qkv_format,
                    cu_seqlens_q=torch.cat([packed_seq_params.cu_seqlens_q, new_end_q]),
                    cu_seqlens_kv=torch.cat([packed_seq_params.cu_seqlens_kv, new_end_kv]),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )

    if args.context_parallel_size > 1:
        labels = get_inputs_on_this_cp_rank(labels.transpose(0, 1)).transpose(0, 1)
        loss_mask = get_inputs_on_this_cp_rank(loss_mask.transpose(0, 1)).transpose(0, 1)

    if packed_seq_params is None and (has_image or has_video):
        mm_token_type_ids = torch.zeros_like(tokens, dtype=torch.int32)
        mm_token_type_ids[tokens == image_token_id] = 1
        mm_token_type_ids[tokens == video_token_id] = 2
        sliding_window = int(getattr(args, "sliding_window", 1024) or 1024)
        attn_mask = build_gemma4_mm_attention_mask(
            tokens, mm_token_type_ids, sliding_window=sliding_window
        )
        attn_mask_type = AttnMaskType.causal
    else:
        attn_mask_type = AttnMaskType.causal
        attn_mask = None
    position_ids = None
    return (
        imgs,
        thw,
        pixel_values_videos,
        video_grid_thw,
        tokens,
        position_ids,
        attn_mask,
        labels,
        loss_mask,
        attn_mask_type,
        packed_seq_params,
        patch_positions,
    )


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
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    loss_reduced_dict = {"lm loss": (reporting_loss[0], reporting_loss[1])}

    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(num_input_tokens, dtype=torch.int, device=output_tensor.device)
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = input_tokens.item() * args.context_parallel_size

    return (loss[0] * args.context_parallel_size, local_num_tokens, loss_reduced_dict)


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    global stimer
    with stimer(bdata=True):
        (
            images,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            attn_mask_type,
            packed_seq_params,
            patch_positions,
        ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            images,
            image_grid_thw,
            input_ids,
            position_ids,
            attention_mask,
            attn_mask_type,
            labels,
            packed_seq_params,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            patch_positions=patch_positions,
        )

    # HOTFIX (text-only-sample backward): when a packed sample contains no
    # images/video AND the LLM is frozen (stage-1: --trainable-modules adapter
    # vision_model), the vision_model+adapter are never invoked, so output_tensor
    # has no grad_fn and custom_backward (schedules.py:161) crashes — desyncing
    # NCCL collectives. Fix: tie a zero-magnitude contribution from the first
    # trainable param so backward succeeds with zero gradients and DP all-reduce
    # stays aligned. Looks like a no-op — DO NOT REMOVE.
    if torch.is_tensor(output_tensor) and not output_tensor.requires_grad:
        dummy_param = None
        for _p in model.parameters():
            if _p.requires_grad:
                dummy_param = _p
                break
        if dummy_param is not None:
            _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _host = os.uname()[1]
            print(
                f"[hotfix-zero-grad] rank={_rank} host={_host} "
                f"output_tensor.requires_grad=False -> tying dummy zero loss to "
                f"trainable param shape={tuple(dummy_param.shape)} dtype={dummy_param.dtype}",
                flush=True,
            )
            output_tensor = output_tensor + (dummy_param.sum() * 0.0).to(output_tensor.dtype)
        else:
            print(
                "[hotfix-zero-grad] WARNING: output_tensor has no grad_fn AND no "
                "trainable parameters found in model — backward will fail.",
                flush=True,
            )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Provides the datasets used by the trainer"""

    args = get_args()
    task_encoder = Gemma4VLTaskEncoder(args)
    train_dataset = get_train_dataset(task_encoder)
    collator = build_sft_data_collator(DataCollatorForSeq2Seq)
    train_dataloader = get_train_loader(train_dataset, collator)
    return train_dataloader, None, None


@register_model_trainer(
    model_family=[constants.VisionLanguageModelFamilies.GEMMA4_VL],
    training_phase=constants.TrainingPhase.PRETRAIN,
)
def default_pretrain_trainer(train_args):
    """build trainer"""
    if train_args.encoder_pipeline_model_parallel_size in [None, 0]:
        model_type = ModelType.encoder_or_decoder
    else:
        model_type = ModelType.encoder_and_decoder
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider,
        model_type=model_type,
        forward_step_func=forward_step,
        get_embedding_ranks=gemma4_vl_embedding_ranks,
        get_position_embedding_ranks=gemma4_vl_position_embedding_ranks,
    )

    return trainer
