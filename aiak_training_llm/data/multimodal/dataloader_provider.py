"""Dataset and DataLoader related utilities"""

import os
import torch
import torch.nn.functional as F

from megatron import energon
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name
from .task_encoder import print_error_handler


def get_train_dataset(task_encoder):
    """Get the training dataset"""
    args = get_args()
    worker_config = energon.WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
        worker_debug_path=None,
        worker_log_level=0,
    )
    train_ds = energon.get_train_dataset(
        args.data_path[0],
        batch_size=args.micro_batch_size,
        task_encoder=task_encoder,
        worker_config=worker_config,
        # None = each tar shard is ONE endless sequential read; its reader (and C-level
        # read buffers) stays resident per touched dataset -> unbounded worker RSS growth
        # on many-dataset mixes (s4 dynD leak, 08-13). A finite value splits shards into
        # re-openable sequences so readers recycle. Env-tunable for A/B.
        max_samples_per_sequence=int(__import__("os").environ.get("ENERGON_MAX_SAMPLES_PER_SEQ", "0")) or None,
        shuffle_buffer_size=None,
        packing_buffer_size=args.packing_batch_size,
        handler=print_error_handler,
        image_decode="pil",
    )
    return train_ds


def get_train_loader(train_ds, collator=None):
    """Get the training loader"""
    args = get_args()
    train_dataloader = energon.get_savable_loader(train_ds)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                tensor_rank=0,  # Dataloader checkpoint is saved by TP rank 0 only.
                pipeline_rank=0,  # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu", weights_only=False)
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print("loading dataset state failed. Skipping. " + str(e))
            else:
                print(f"dataset state {data_save_name} does not exist")
    return EnergonDataloader(train_dataloader, collator)


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""

    def __init__(self, dataloader, collator=None):
        self._dataloader = dataloader
        self._collator = collator
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        features = self._iter.__next__()
        if self._collator is not None:
            padded = self._collator.tokenizer.pad(
                {"input_ids": features["tokens"]},
                padding=self._collator.padding,
                max_length=self._collator.max_length,
                pad_to_multiple_of=self._collator.pad_to_multiple_of,
            )
            paded_length = padded["input_ids"].shape[1] - features["tokens"].shape[1]
            features["tokens"] = padded["input_ids"]
            features["labels"] = F.pad(
                features["labels"], (0, paded_length), "constant", self._collator.label_pad_token_id
            )
            features["attn_mask"] = F.pad(features["attn_mask"], (0, paded_length), "constant", True)
        return features

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        """Save the current state of this dataloader"""
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    """Infinite iteration over an iterator"""
    while True:
        for x in iter:
            yield x
