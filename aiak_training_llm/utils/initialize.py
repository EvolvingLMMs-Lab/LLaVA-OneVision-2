# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, AIAK team. All rights reserved.
# This code was adopted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/initialize.py

"""AIAK-Training-LLM initialization."""

import os
import logging

import torch

from megatron.core import mpu, tensor_parallel

from megatron.training.arguments import (
    parse_args,
    validate_args as validate_megatron_args
)

from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.global_vars import set_global_variables as set_megatron_global_variables
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)

from megatron.training.initialize import (
    _initialize_distributed,
    _set_random_seed,
    _init_autoresume,
    _compile_dependencies,
    _initialize_tp_communicators
)

from .global_vars import set_aiak_extra_global_vars

logger = logging.getLogger(__name__)


def parse_arguments(
    extra_args_provider=None,
    validate_extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
):
    """Parse arguments."""
    print("parse arguments function called")
    args = parse_args(extra_args_provider, ignore_unknown_args)
    
    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        assert args.non_persistent_ckpt_type != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    # Validate arguments.
    if validate_extra_args_provider is not None:
        validate_extra_args_provider(args)

    for key in args_defaults:
        # just overwrite the args with defaults
        setattr(args, key, args_defaults[key])

    assert args.yaml_cfg is None, "yaml_cfg is not supported in AIAK-Training-LLM yet"
    validate_megatron_args(args)
    
    return args


def initialize_aiak_megatron(
    args,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker()

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_megatron_global_variables(args, build_tokenizer=False)

    # set aiak extra global args
    set_aiak_extra_global_vars(args, build_tokenizer=True)

    # set logging level
    setup_logging(args)

    # init rerun state
    def state_save_func():
        return {'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict['rng_tracker_states']:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(args.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=args.error_injection_rate,
            error_injection_type=RerunDiagnostic(args.error_injection_type),
        ),
        result_rejected_tracker_filename=args.result_rejected_tracker_filename,
    )
    
    # torch.distributed initialization
    def finish_mpu_init():
        """torch.distributed initialization"""

        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
        )

    if skip_mpu_initialization:
        return None
    
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
           _initialize_tp_communicators()

        # No continuation function
        return None


def setup_logging(args) -> None:
    """ Sets the default logging level based on cmdline args and env vars.

    Precedence:
    1. Command line argument `--logging-level`
    2. Env var `MEGATRON_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    logging_level = None
    env_logging_level = os.getenv('MEGATRON_LOGGING_LEVEL', None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if args.logging_level is not None:
        logging_level = args.logging_level

    if logging_level is not None:
        logger.info(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
