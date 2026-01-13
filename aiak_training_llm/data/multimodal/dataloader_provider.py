""" Dataset and DataLoader related utilities """
import os
import torch
import torch.nn.functional as F

from megatron import energon
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name
from .task_encoder import print_error_handler


def get_train_dataset(task_encoder):
    """ Get the training dataset """
    args = get_args()
    worker_config = energon.WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
        worker_debug_path=None,
        worker_log_level=0
    )
    #use megatron energon to get the training dataset
    # a high performance data loading library for large-scale distributed training
    #each gpu gets its shard of the data based on data parallel rank
    # Supports packing: Multiple short sequences can be packed into one to maximize GPU utilization
    # streaming: can load data on-the-fly without pre-downloading entire dataset into memory
    train_ds = energon.get_train_dataset(
        args.data_path[0], # Path to dataset (LLaVA-558K-Webdataset)
        batch_size=args.micro_batch_size, #Batch size per GPU
        task_encoder=task_encoder, # The Qwen2VLTaskEncoder we just created
        worker_config=worker_config, #Distributed Config
        max_samples_per_sequence=None,
        shuffle_buffer_size=None,
        packing_buffer_size=args.packing_batch_size, #Buffer size for packing sequences
        handler=print_error_handler, #Error handler function
        image_decode="pil", #Decode images using PIL
    )
    return train_ds


def get_train_loader(train_ds, collator=None):
    """ Get the training loader """
    args = get_args()
    train_dataloader = energon.get_savable_loader(train_ds)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,    # Only the first pipeline parallel rank stores the dataloader checkpoint.
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
        self._iter = iter(cyclic_iter(dataloader)) # Infinite iterator!

    def __next__(self):
        features = self._iter.__next__() #get next batch
        if self._collator is not None:
            # Apply padding using the collator
            padded = self._collator.tokenizer.pad(
                {"input_ids": features['tokens']},
                padding=self._collator.padding,
                max_length=self._collator.max_length,
                pad_to_multiple_of=self._collator.pad_to_multiple_of,
            )
            paded_length = padded['input_ids'].shape[1] - features['tokens'].shape[1]
            
            # Update features with padded versions
            features['tokens'] = padded["input_ids"]
            #pad labels and attention mask too
            features['labels'] = F.pad(
                features['labels'],
                (0, paded_length),
                "constant",
                self._collator.label_pad_token_id
            )
            features['attn_mask'] = F.pad(features['attn_mask'], (0, paded_length), "constant", True)
        # return a batch 
        print("Dataloader batch - tokens shape:", features['tokens'].shape)
        print("Dataloader batch - labels shape:", features['labels'].shape)
        print("Dataloader batch - attn_mask shape:", features['attn_mask'].shape)
        print("Dataloader batch - image_grid_thw shape:", features['image_grid_thw'].shape)
        return features

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        """ Save the current state of this dataloader """
        return self._dataloader.save_state_rank()

# This means training never runs out of data - it just loops back to the beginning.
def cyclic_iter(iter):
    """ Infinite iteration over an iterator """
    while True:
        for x in iter:
            yield x
