"""Training Entry"""

from aiak_training_llm.train import parse_train_args
from aiak_training_llm.train import build_model_trainer


def main():
    """train cmd"""

    # parse args
    args = parse_train_args()
    print("Building model trainer...")
    # get model trainer
    trainer = build_model_trainer(args)
    print(trainer) # <aiak_training_llm.train.megatron_trainer.MegatronTrainer object>
    print("Starting training...")
    # start training
    trainer.train()


if __name__ == '__main__':
    main()
