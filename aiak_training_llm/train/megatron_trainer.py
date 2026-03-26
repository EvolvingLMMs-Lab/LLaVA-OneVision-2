"""model trainer"""

from .training_utils import pretrain


class MegatronTrainer(object):
    """megatron trainer"""
    def __init__(
            self,
            train_args,
            train_valid_test_dataset_provider,
            model_provider,
            model_type,
            forward_step_func,
            process_non_loss_data_func=None,
            get_embedding_ranks=None,
            get_position_embedding_ranks=None,
    ):
        """
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        """
        self.train_args = train_args
        self.train_valid_test_datasets_provider = train_valid_test_dataset_provider
        self.model_provider = model_provider
        self.model_type = model_type
        self.forward_step_func = forward_step_func
        self.process_non_loss_data_func = process_non_loss_data_func
        self.get_embedding_ranks = get_embedding_ranks
        self.get_position_embedding_ranks = get_position_embedding_ranks

    def train(self):
        """start training"""
        # Mark dataset provider as distributed (multi-GPU)
        self.train_valid_test_datasets_provider.is_distributed = True

        pretrain(
            train_args=self.train_args, #All training arguments (like batch size, lr, etc)
            train_valid_test_dataset_provider=self.train_valid_test_datasets_provider,
            # function that provides train/valid/test datasets (preprocessing handled inside)
            model_provider=self.model_provider, #function that provides the model architecture
            model_type=self.model_type, #type of model being trained (e.g., language model, vision model, etc)(encoder or decoder)
            forward_step_func=self.forward_step_func, #function that computes loss for one batch
            process_non_loss_data_func=self.process_non_loss_data_func, #Optional function for logging/visualization
            get_embedding_ranks=self.get_embedding_ranks, #for distributed embedding tables
            get_position_embedding_ranks=self.get_position_embedding_ranks, # for distributed position embeddings
        )
