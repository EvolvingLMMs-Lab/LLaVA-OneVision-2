"""Training Entry"""

from aiak_training_llm.train import parse_train_args
from aiak_training_llm.train import build_model_trainer
from aiak_training_llm.utils.triton_autotune import apply_triton_autotune_cap


def main():
    """train cmd"""

    # Before any fla/Triton kernel runs: mid-run autotune sweeps OOM on new packed shapes.
    apply_triton_autotune_cap()

    # parse args
    args = parse_train_args()

    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == '__main__':
    main()
