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
from aiak_training_llm.data.multimodal.qwen2vl_task_encoder import Qwen2VLTaskEncoder
from aiak_training_llm.models import get_model_family, get_model_provider
from aiak_training_llm.models.qwen_vl.utils import get_inputs_on_this_cp_rank
from aiak_training_llm.train.megatron_trainer import MegatronTrainer
from aiak_training_llm.train.sft.utils import build_sft_data_collator
from aiak_training_llm.train.trainer_builder import register_model_trainer
from aiak_training_llm.utils import constants, get_args
from aiak_training_llm.utils import opt_switches
from transformers import DataCollatorForSeq2Seq


stimer = StragglerDetector()

# TODO: get token id from tokenizer
image_token_id = 151655
video_token_id = 151656
vision_start_token_id = 151652


def qwen2vl_embedding_ranks(pp_ranks):
    """qwen2vl's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
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


def qwen2vl_position_embedding_ranks(pp_ranks):
    """qwen2vl's embedding ranks consist of the singular rank of the model or the decoder's first rank.
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
    if opt_switches.flag('TRITON_CE'):
        # the language model's per-token cross-entropy in two Triton kernels (same fp32 math, no fp32 logits copy)
        from aiak_training_llm.models.custom.common import triton_cross_entropy
        triton_cross_entropy.install()
    if opt_switches.flag('FUSED_SWIGLU'):
        # the language model's SwiGLU (jit_fuser in Megatron) in two Triton kernels (bitwise as the jit version)
        from aiak_training_llm.models.custom.common import fused_swiglu
        fused_swiglu.install()
    if opt_switches.flag('FAST_BIAS_ADD'):
        # the vision MLP's broadcast fc1 bias add in a vectorised Triton kernel (bitwise as PyTorch's)
        from aiak_training_llm.models.custom.common import bias_add
        bias_add.install()
    if opt_switches.flag('FUSED_QK_NORM_ROPE'):
        # the language model's q/k RMSNorm and rotary embedding in one Triton kernel each way
        from aiak_training_llm.models.custom.common import fused_qk_norm_rope
        fused_qk_norm_rope.install()
    model_family = get_model_family(args.model_name)
    model_provider = get_model_provider(model_family)
    assert model_provider is not None, f"model provider for {args.model_name} not found"
    return model_provider(pre_process, post_process, add_encoder, add_decoder)


def _to_device_async(t):
    """Host tensor -> current GPU without a host synchronisation (pinned staging, non-blocking copy)."""
    t = t.contiguous()
    if not t.is_pinned():
        t = t.pin_memory()
    return t.to(torch.cuda.current_device(), non_blocking=True)


def _get_batch_host(data_iterator):
    """get_batch() for TP 1 / CP 1 without sequence parallelism, with every decision taken on the host copy of the
    micro-batch: the same tensors (values, dtypes, shapes) on the GPU, but no host<->device synchronisation (the
    default path moves each key with a synchronous copy and reads flags and lengths back with .item()). Host-side
    metadata the model would otherwise read back from the GPU is attached to the GPU tensors."""
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

    def move(key, dtype):
        # tensor_parallel.broadcast_data() asserts the dtype of every key
        assert data[key].dtype == dtype, (
            "{} has data type {} which is different than {}".format(key, data[key].dtype, dtype)
        )
        return _to_device_async(data[key])

    tokens_host = data["tokens"]
    tokens = move("tokens", torch.int64)
    labels = move("labels", torch.int64)
    attn_mask_host = data["attn_mask"]
    assert attn_mask_host.dtype == torch.bool
    cu_lengths_host = data["cu_lengths"]
    assert cu_lengths_host.dtype == torch.int32
    max_lengths_host = data["max_lengths"]
    assert max_lengths_host.dtype == torch.int32

    has_video = bool((tokens_host == video_token_id).any())
    n_image_tokens = int((tokens_host == image_token_id).sum())
    has_image = n_image_tokens > 0
    thw = None
    video_grid_thw = None
    imgs = None
    pixel_values_videos = None
    patch_positions = None
    if has_image:
        imgs = move("imgs", torch.float32)
        thw = move("image_grid_thw", torch.int32)
        thw._ov2_host = data["image_grid_thw"]
        if "patch_positions" in data and data["patch_positions"] is not None:
            patch_positions = move("patch_positions", torch.int64)
    if has_video:
        pixel_values_videos = move("pixel_values_videos", torch.float32)
        video_grid_thw = move("video_grid_thw", torch.int32)
    tokens._ov2_n_image_tokens = n_image_tokens
    if has_image and opt_switches.flag('INDEX_SCATTER') and tokens_host.shape[0] == 1:
        tokens._ov2_image_pos = _to_device_async(
            (tokens_host[0] == image_token_id).nonzero(as_tuple=True)[0]
        )

    packed_seq_params = None
    labels = torch.roll(labels, shifts=-1, dims=1)
    loss_mask = (labels != -100).long()

    if cu_lengths_host.shape == torch.Size([1, 1]):
        attn_mask = _to_device_async(attn_mask_host)
        for i in range(attn_mask_host.shape[0]):
            loss_mask[i, int((attn_mask_host[i] == False).sum()) - 1] = 0
    else:
        assert cu_lengths_host.shape[0] == 1, "micro-batch-size must be 1 for packing"
        cu_lengths = _to_device_async(cu_lengths_host)
        max_seqlen = int(max_lengths_host[0])
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

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


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    if (
        opt_switches.flag('HOST_SYNC_FREE')
        and data_iterator is not None
        and args.tensor_model_parallel_size == 1
        and args.context_parallel_size == 1
        and not args.sequence_parallel
    ):
        return _get_batch_host(data_iterator)
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

    # TODO
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
    deferred = opt_switches.flag('DEFER_LOSS_REDUCE') and torch.is_grad_enabled()
    if args.check_for_nan_in_loss_and_grad and deferred and opt_switches.flag('HOST_SYNC_FREE'):
        # the same check, read back once per iteration in flush_deferred_loss_reduce()
        _DEFERRED_NAN_CHECKS.append(loss[0].isnan())
    elif args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    if opt_switches.flag('DEFER_LOSS_REDUCE') and torch.is_grad_enabled():
        # Training: the same data-parallel all-reduces (same tensors, same order) are issued by
        # flush_deferred_loss_reduce() after the backward of the last micro-batch, so that the fast ranks are not
        # held at the end of every micro-batch's forward by the slowest one. The dict holds views of the tensors
        # that are reduced in place, exactly as before.
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        loss_reduced_dict = {"lm loss": (reporting_loss[0], reporting_loss[1])}
        input_tokens = None
        if args.variable_seq_lengths:
            num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
            if opt_switches.flag('ASYNC_TOKEN_COUNT'):
                # the same int tensor, filled on the device: torch.tensor(..., device=cuda) is a blocking
                # host-to-device copy, i.e. a stream synchronisation after every micro-batch's forward
                input_tokens = torch.full((), num_input_tokens, dtype=torch.int, device=output_tensor.device)
            else:
                input_tokens = torch.tensor(num_input_tokens, dtype=torch.int, device=output_tensor.device)
        _DEFERRED_LOSS_REDUCE.append((reporting_loss, input_tokens, loss_reduced_dict))
        return (loss[0] * args.context_parallel_size, local_num_tokens, loss_reduced_dict)

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


_DEFERRED_LOSS_REDUCE = []
_DEFERRED_NAN_CHECKS = []


def flush_deferred_loss_reduce():
    """Issue the data-parallel reductions loss_func deferred during training, in micro-batch order (per micro-batch:
    the reporting loss, then the input-token count), and fill in 'total_inputs'."""
    if _DEFERRED_NAN_CHECKS:
        flags = list(_DEFERRED_NAN_CHECKS)
        _DEFERRED_NAN_CHECKS.clear()
        if bool(torch.stack(flags).any()):
            global_rank = torch.distributed.get_rank()
            raise AssertionError(
                f"Rank {global_rank}: found NaN in local forward loss calculation. "
                f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
            )
    if not _DEFERRED_LOSS_REDUCE:
        return
    args = get_args()
    pending = list(_DEFERRED_LOSS_REDUCE)
    _DEFERRED_LOSS_REDUCE.clear()
    group = mpu.get_data_parallel_group()
    for reporting_loss, input_tokens, _ in pending:
        torch.distributed.all_reduce(reporting_loss, group=group)
        if input_tokens is not None:
            torch.distributed.all_reduce(input_tokens, group=group)
    for _, input_tokens, loss_reduced_dict in pending:
        if input_tokens is not None:
            loss_reduced_dict["total_inputs"] = input_tokens.item() * args.context_parallel_size


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
    task_encoder = Qwen2VLTaskEncoder(args)
    train_dataset = get_train_dataset(task_encoder)
    collator = build_sft_data_collator(DataCollatorForSeq2Seq)
    train_dataloader = get_train_loader(train_dataset, collator)
    return train_dataloader, None, None


@register_model_trainer(
    model_family=[constants.VisionLanguageModelFamilies.LLAVA_ONEVISION2],
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
        get_embedding_ranks=qwen2vl_embedding_ranks,
        get_position_embedding_ranks=qwen2vl_position_embedding_ranks,
    )

    return trainer
