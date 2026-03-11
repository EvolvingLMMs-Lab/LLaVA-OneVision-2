"""default pretrain for generative models like GPTS"""

import os
import torch
from typing import Tuple, Optional
from functools import partial

from megatron.training import get_timers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector

from megatron.core.transformer.enums import AttnMaskType

from transformers import DataCollatorForSeq2Seq

from aiak_training_llm.utils import constants, get_args, get_tokenizer, print_rank_0
from aiak_training_llm.models.qwen_vl.utils import get_inputs_on_this_cp_rank
from aiak_training_llm.models import get_model_provider, get_model_family
from aiak_training_llm.train.megatron_trainer import MegatronTrainer
from aiak_training_llm.train.trainer_builder import register_model_trainer
from aiak_training_llm.train.sft.utils import build_sft_data_collator
from aiak_training_llm.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader
)
from aiak_training_llm.data.multimodal.qwen2vl_task_encoder import Qwen2VLTaskEncoder

stimer = StragglerDetector()

# Resolve vision token ids lazily from the active tokenizer (works for Qwen and MobileLLM tokenizers)
image_token_id = None
video_token_id = None
vision_start_token_id = None
_logged_token_presence_once = False
_big_debug_step = 0


def _log_big_batch_debug(
    step_id: int,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    imgs: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    cu_lengths: torch.Tensor,
    max_lengths: torch.Tensor,
    has_image: bool,
    has_video: bool,
):
    sample_idx = 0
    sample_tokens = tokens[sample_idx]
    sample_labels = labels[sample_idx]
    sample_loss_mask = loss_mask[sample_idx]

    image_token_count = int((tokens == image_token_id).sum().item()) if image_token_id is not None else 0
    video_token_count = int((tokens == video_token_id).sum().item()) if video_token_id is not None else 0
    vision_start_count = int((tokens == vision_start_token_id).sum().item()) if vision_start_token_id is not None else 0
    sample_valid_label_count = int((sample_labels != -100).sum().item())

    sample_decoded = "<decode unavailable>"
    try:
        tokenizer = get_tokenizer()
        ids_for_decode = sample_tokens[:128].tolist()
        if hasattr(tokenizer, "decode"):
            sample_decoded = tokenizer.decode(ids_for_decode)
        elif hasattr(tokenizer, "detokenize"):
            sample_decoded = tokenizer.detokenize(ids_for_decode)
    except Exception as exc:
        sample_decoded = f"<decode failed: {exc}>"

    img_stats = "None"
    if imgs is not None:
        img_stats = (
            f"shape={tuple(imgs.shape)}, dtype={imgs.dtype}, "
            f"min={float(imgs.min().item()):.4f}, max={float(imgs.max().item()):.4f}, "
            f"mean={float(imgs.mean().item()):.4f}"
        )

    attn_stats = "None"
    if attn_mask is not None:
        attn_stats = (
            f"shape={tuple(attn_mask.shape)}, dtype={attn_mask.dtype}, "
            f"masked={(attn_mask == True).sum().item()}, unmasked={(attn_mask == False).sum().item()}"
        )

    print_rank_0("\n" + "=" * 120)
    print_rank_0(f"[BIG DEBUG][BATCH][STEP {step_id}]")
    print_rank_0(
        f"tokens.shape={tuple(tokens.shape)} dtype={tokens.dtype} | "
        f"labels.shape={tuple(labels.shape)} dtype={labels.dtype} | "
        f"loss_mask.shape={tuple(loss_mask.shape)} dtype={loss_mask.dtype}"
    )
    print_rank_0(
        f"token_ids: image={image_token_id}, video={video_token_id}, vision_start={vision_start_token_id} | "
        f"counts: image={image_token_count}, video={video_token_count}, vision_start={vision_start_count}"
    )
    print_rank_0(
        f"has_image={has_image}, has_video={has_video} | imgs={img_stats} | "
        f"image_grid_thw={tuple(image_grid_thw.shape) if image_grid_thw is not None else None}"
    )
    print_rank_0(
        f"attn_mask={attn_stats} | cu_lengths.shape={tuple(cu_lengths.shape)} values={cu_lengths[0].tolist()} | "
        f"max_lengths.shape={tuple(max_lengths.shape)} values={max_lengths[0].tolist()}"
    )
    print_rank_0(
        f"sample[{sample_idx}] first_64_tokens={sample_tokens[:64].tolist()} | "
        f"first_64_labels={sample_labels[:64].tolist()}"
    )
    print_rank_0(
        f"sample[{sample_idx}] valid_label_count={sample_valid_label_count} | "
        f"loss_mask_active={(sample_loss_mask == 1).sum().item()} | decoded_prefix={sample_decoded}"
    )
    print_rank_0("=" * 120)


def _ensure_vision_token_ids():
    """Resolve multimodal special token ids once from the active tokenizer."""
    global image_token_id, video_token_id, vision_start_token_id
    if image_token_id is not None and video_token_id is not None and vision_start_token_id is not None:
        return

    tokenizer = get_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

    if image_token_id is None or video_token_id is None or vision_start_token_id is None:
        vocab = getattr(tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            if image_token_id is None:
                image_token_id = vocab.get("<|image_pad|>")
            if video_token_id is None:
                video_token_id = vocab.get("<|video_pad|>")
            if vision_start_token_id is None:
                vision_start_token_id = vocab.get("<|vision_start|>")

    print_rank_0(
        f"[DEBUG TOKEN IDS] image_token_id={image_token_id}, "
        f"video_token_id={video_token_id}, vision_start_token_id={vision_start_token_id}"
    )


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
    args = get_args() #get training arguments

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
    # Get model family: "llava-ov-1.5-4b" → "llava_ov_1_5"
    model_family = get_model_family(args.model_name)
    # Lookup the registered provider
    model_provider = get_model_provider(model_family)
        #   = MODEL_FAMILY_TO_PROVIDER["llava_ov_1_5"]
    #   = rice_vl_model_provider
    assert model_provider is not None, f'model provider for {args.model_name} not found'
    # call the provider to build the model and return the model
    return model_provider(pre_process, post_process, add_encoder, add_decoder)


def get_batch(data_iterator):
    """Generate a batch"""
    global _logged_token_presence_once, _big_debug_step
    args = get_args()
    _ensure_vision_token_ids()

    print("=" * 80)
    print("[DEBUG GET_BATCH - pretrain_llavaov_1_5.py]")
    
    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        
        print(f"  Data from iterator is dict: {isinstance(data, dict)}")
        if isinstance(data, dict):
            print(f"  Data keys: {list(data.keys())}")
            print(f"  'imgs' in data: {'imgs' in data}")
            print(f"  'image_grid_thw' in data: {'image_grid_thw' in data}")
            if 'imgs' in data:
                print(f"  data['imgs'] is None: {data['imgs'] is None}")
                print(f"  data['imgs'] shape: {data['imgs'].shape if data['imgs'] is not None else 'None'}")
        
        if isinstance(data.get('tokens'), torch.Tensor):
            orig_dtype = data['tokens'].dtype
            if data['tokens'].dtype != torch.long:
                print(f"[WARN] tokens dtype {orig_dtype} -> force cast to torch.long; shape={tuple(data['tokens'].shape)}")
                data['tokens'] = data['tokens'].to(torch.long)
        if isinstance(data.get('labels'), torch.Tensor) and data['labels'].dtype != torch.long:
            data['labels'] = data['labels'].to(torch.long)

        assert isinstance(data['tokens'], torch.Tensor) and data['tokens'].dtype == torch.long, \
            f"Expected tokens torch.int64 but got {type(data['tokens'])} {getattr(data['tokens'],'dtype',None)}"
        assert isinstance(data['labels'], torch.Tensor) and data['labels'].dtype == torch.long, \
            f"Expected labels torch.int64 but got {type(data['labels'])} {getattr(data['labels'],'dtype',None)}"
    else:
        data = None

    print(f"  Broadcasting data...")
    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.bool)["attn_mask"]
    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"]
    
    has_video = False  # Video path intentionally disabled for this training setup.
    has_image = image_token_id is not None and image_token_id in tokens
    if not _logged_token_presence_once:
        image_token_count = int((tokens == image_token_id).sum().item()) if image_token_id is not None else 0
        vision_start_count = int((tokens == vision_start_token_id).sum().item()) if vision_start_token_id is not None else 0
        print_rank_0(
            f"[DEBUG TRAIN TOKENS] image_token_id={image_token_id} count={image_token_count}, "
            f"vision_start_token_id={vision_start_token_id} count={vision_start_count}, "
            f"first_32_ids={tokens[0, :32].tolist()}"
        )
        _logged_token_presence_once = True
    print(f"  has_image token in batch: {has_image}")
    print(f"  has_video token in batch: {has_video}")
    thw = None
    video_grid_thw = None
    imgs = None
    pixel_values_videos = None
    if has_image:
        imgs = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
        thw = tensor_parallel.broadcast_data(["image_grid_thw"], data, torch.int32)["image_grid_thw"]
        print(f"  Broadcasted imgs: {imgs.shape if imgs is not None else 'None'}")
        print(f"  Broadcasted image_grid_thw: {thw.shape if thw is not None else 'None'}")
    else:
        print("  No image tokens in this packed batch (imgs not forwarded to model).")
    if has_video:
        pixel_values_videos = tensor_parallel.broadcast_data(
            ["pixel_values_videos"],
            data,
            torch.float32)["pixel_values_videos"]
        video_grid_thw = tensor_parallel.broadcast_data(
            ["video_grid_thw"],
            data,
            torch.int32)["video_grid_thw"]

    print("=" * 80)

    packed_seq_params = None
    is_video = False

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

    if args.context_parallel_size > 1:
        labels = get_inputs_on_this_cp_rank(labels.transpose(0, 1)).transpose(0, 1)
        loss_mask = get_inputs_on_this_cp_rank(loss_mask.transpose(0, 1)).transpose(0, 1)

    _big_debug_step += 1
    _log_big_batch_debug(
        step_id=_big_debug_step,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attn_mask=attn_mask,
        imgs=imgs,
        image_grid_thw=thw,
        cu_lengths=cu_lengths,
        max_lengths=max_lengths,
        has_image=bool(has_image),
        has_video=bool(has_video),
    )

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
        packed_seq_params
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

#called by megatron training loop during each training step (called once per micro batch during training)
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
        #get batch from data iterator
        images, image_grid_thw, pixel_values_videos, video_grid_thw, \
        input_ids, position_ids, attention_mask, \
        labels, loss_mask, attn_mask_type, packed_seq_params \
            = get_batch(data_iterator)
        #returns:
        # images: Tensor([num_images, C, H, W])          # Processed image pixels
        # image_grid_thw: Tensor([num_images, 3])        # Grid dimensions [t, h, w]
        # pixel_values_videos: Tensor or None            # Video frames (if present)
        # video_grid_thw: Tensor or None                 # Video grid dimensions
        # input_ids: Tensor([batch, seq_len])            # Token IDs (with <|image_pad|>)
        # position_ids: Tensor([batch, seq_len]) or None # Position indices
        # attention_mask: Tensor([batch, seq_len])       # Attention mask (False=attend, True=mask)
        # labels: Tensor([batch, seq_len])               # Target labels for loss
        # loss_mask: Tensor([batch, seq_len])            # Which tokens to compute loss on
        # attn_mask_type: AttnMaskType                   # Causal or padding_causal
        # packed_seq_params: PackedSeqParams or None     # For packed sequences
        
    timers('batch-generator').stop()

    with stimer:
        # MODEL FORWARD PASS
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
            video_grid_thw=video_grid_thw
        )
 
    return output_tensor, partial(loss_func, loss_mask)

# Factory function that creates and returns train/valid/test data loaders for the Megatron trainer.
def train_valid_test_dataset_provider(train_val_test_num_samples):
    """ Provides the datasets used by the trainer """

    args = get_args()
    # create task encoder
    task_encoder = Qwen2VLTaskEncoder(args)
    # Processes images/videos using processor
    # Applies chat template to conversations
    # Tokenizes text
    # Creates attention masks
    # Handles vision token placement (<|image_pad|>, <|video_pad|>)
    # Packs sequences efficiently

    #get train dataset
    train_dataset = get_train_dataset(task_encoder)

    #Purpose: Combines multiple samples into a batch
    # Input: List of individual samples (each with different sequence lengths)
    # Output: Batched tensors with padding
    collator = build_sft_data_collator(DataCollatorForSeq2Seq)
    # Example:
    # Sample 1: [151652, 8932, 1234, ...]      # Length 50
    # Sample 2: [151652, 9821, 5567, ...]      # Length 120
                    #    ↓  (collate)
    # Batch: [[151652, 8932, 1234, ..., <pad>, <pad>],   # Padded to 120
    #         [151652, 9821, 5567, ..., ...., ....]]      # Already 120

    #create dataloader
    train_dataloader = get_train_loader(train_dataset, collator)

    return train_dataloader, None, None 
    # For SFT, typically only training loop is needed
    # valid_iterator and test_iterator are set to None
    

@register_model_trainer(
    model_family=[
        constants.VisionLanguageModelFamilies.LLAVA_OV_1_5],
        training_phase=constants.TrainingPhase.PRETRAIN)
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