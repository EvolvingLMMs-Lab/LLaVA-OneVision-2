"""qwen model"""
from collections import OrderedDict
from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from aiak_training_llm.models.qwen35.qwen35_rope import Qwen35RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    keys_to_remove = [
        key for key in incompatible_keys.missing_keys
        if "input_layernorm._extra_state" in key 
        or "pre_mlp_layernorm._extra_state" in key
        or "enorm._extra_state" in key
        or "hnorm._extra_state" in key
        or "eh_proj._extra_state" in key
        or "output_layernorm._extra_state" in key
        or "self_attention.q_layernorm._extra_state" in key
        or "self_attention.k_layernorm._extra_state" in key
        or "linear_fc1._extra_state" in key
        or "linear_fc2._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class QwenModel(LanguageModule):
    """Qwen Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to True.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights
            are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional): Position embedding type, Defaults to 'rope'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type
            is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer
            sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'rope',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )

        # Qwen3.5 hybrid-attention backbone uses interleaved 3D M-RoPE (partial rotary)
        # instead of the standard 1D RotaryEmbedding. Selected via config flag set by
        # the provider (apply_qwen35_language_config) / arch config.
        self.use_qwen35_mrope = getattr(config, 'language_model_type', None) in ('qwen3_5', 'qwen35')
        if self.use_qwen35_mrope:
            # Proven parity recipe flags (idempotent if provider already set them).
            config.rotary_interleaved = False
            config.apply_rope_fusion = False
            config.multi_latent_attention = False

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if self.use_qwen35_mrope:
                self.rotary_pos_emb = Qwen35RotaryEmbedding(self.config)
            else:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=self.config.kv_channels,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=rotary_base,
                    rope_scaling=rope_scaling,
                    rope_scaling_factor=rope_scaling_factor,
                    use_cpu_initialization=self.config.use_cpu_initialization,
                )

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )

        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is None:
            if self.pre_process:
                decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            else:
                # intermediate stage of pipeline
                # decoder will get hidden_states from encoder.input_tensor
                decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        if rotary_pos_emb is None and self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if self.use_qwen35_mrope:
                # Qwen3.5 interleaved 3D M-RoPE. Derive (seq_len, batch) from whichever
                # input is present (decoder_input is [S, B, H]; input_ids is [B, S]).
                if decoder_input is not None:
                    S, B = decoder_input.shape[0], decoder_input.shape[1]
                    dev = decoder_input.device
                elif input_ids is not None:
                    B, S = input_ids.shape[0], input_ids.shape[1]
                    dev = input_ids.device
                else:
                    S, B, dev = position_ids.shape[-1], position_ids.shape[0], position_ids.device
                # Under sequence-parallel, decoder_input is the SP-scattered [S/tp, B, H];
                # rope must be built at the FULL gathered length because attention's
                # SP->TP all-gather restores the full sequence before rope is applied.
                # Mirror the standard get_rotary_seq_len up-scale (rotary_pos_embedding.py).
                # Guard on decoder_input so full-length input_ids/position_ids fallbacks
                # are not double-scaled.
                if decoder_input is not None and self.config.sequence_parallel:
                    S = S * self.config.tensor_model_parallel_size
                mrope_position_ids = getattr(self, '_mrope_position_ids', None)
                if mrope_position_ids is None:
                    # TODO(multimodal): for image/video inputs the 3 (T,H,W) position-id
                    # rows must be computed per-sample from image_grid_thw/video_grid_thw
                    # (cf. qwen2_vl get_rope_index). Until the LlavaOnevision2 model wires
                    # that in and stashes it on self.language_model._mrope_position_ids,
                    # fall back to the text-path positions (T==H==W==arange), which is
                    # exact for pure-text and a valid placeholder for the smoke forward.
                    mrope_position_ids = (
                        torch.arange(S, device=dev).view(1, 1, S).expand(3, B, S).contiguous()
                    )
                elif mrope_position_ids.shape[-1] != S:
                    # get_rope_index returns unpadded full-length positions; pad (by
                    # replicating the last position, whose value is irrelevant since pad
                    # tokens are masked/discarded and unattended) or truncate to the
                    # SP-padded gathered length S so cos_/sin_ broadcast against the query.
                    L = mrope_position_ids.shape[-1]
                    if L < S:
                        pad = mrope_position_ids[..., -1:].expand(3, B, S - L)
                        mrope_position_ids = torch.cat((mrope_position_ids, pad), dim=-1).contiguous()
                    else:
                        mrope_position_ids = mrope_position_ids[..., :S].contiguous()
                rotary_pos_emb = self.rotary_pos_emb(
                    mrope_position_ids,
                    packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
                )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        _prof = __import__('os').environ.get('PROF_TAGS') == '1'
        if _prof:
            __import__('torch').cuda.synchronize(); _ct0 = __import__('time').perf_counter()
        loss = self.compute_language_model_loss(labels, logits)
        if _prof:
            __import__('torch').cuda.synchronize()
            if (not __import__('torch').distributed.is_initialized()) or __import__('torch').distributed.get_rank() == 0:
                print('[PROF_TAGS] cross_entropy fwd: %.1f ms' % ((__import__('time').perf_counter() - _ct0) * 1000), flush=True)

        return loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility
        (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict
