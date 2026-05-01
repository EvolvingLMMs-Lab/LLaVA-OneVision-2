"""MobileLLM Model - Adapted from Qwen model for MobileLLM-R1-140M architecture"""
from collections import OrderedDict
from typing import Dict, Literal, Optional

from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Hook to ignore Transformer Engine _extra_state used for FP8."""
    keys_to_remove = [
        key for key in incompatible_keys.missing_keys
        if "input_layernorm._extra_state" in key 
        or "pre_mlp_layernorm._extra_state" in key
        or "output_layernorm._extra_state" in key
        or "self_attention.q_layernorm._extra_state" in key
        or "self_attention.k_layernorm._extra_state" in key
        or "linear_fc1._extra_state" in key
        or "linear_fc2._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class MobileLLMModel(LanguageModule):
    """MobileLLM Transformer language model.
    
    Based on facebook/MobileLLM-R1-140M architecture (LLaMA-style):
    - 15 layers
    - 576 hidden size
    - 9 attention heads with 3 KV heads (GQA)
    - 2048 FFN hidden size
    - RoPE position embeddings
    - RMSNorm, SwiGLU activation
    - Shared input/output embeddings

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size (128256 for MobileLLM)
        max_sequence_length (int): Maximum sequence length (32768 for MobileLLM)
        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include output layer (used with pipeline parallelism)
        fp16_lm_cross_entropy (bool): Use fp16 for cross entropy
        parallel_output (bool): Keep outputs split across tensor parallel ranks
        share_embeddings_and_output_weights (bool): Share input and output embeddings
        position_embedding_type (str): Position embedding type ('rope' for MobileLLM)
        rotary_percent (float): Percent of rotary dimension (1.0 for MobileLLM)
        rotary_base (int): Base period for RoPE (8000000 for MobileLLM)
        rope_scaling (bool): Enable RoPE scaling
        rope_scaling_factor (float): RoPE scaling factor
        scatter_embedding_sequence_parallel (bool): Scatter embeddings in sequence parallel
        seq_len_interpolation_factor (float): Sequence length interpolation factor
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
        rotary_base: int = 8000000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 1.0,
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

        # Model type for Megatron pipelining
        self.model_type = ModelType.encoder_or_decoder

        # Attributes for TensorRT-LLM export
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling

        # Embedding layer
        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )

        # RoPE embeddings
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
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

        # Cache for RoPE tensors
        self.rotary_pos_emb_cache = {}

        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output layer
        if self.post_process:
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
            )

        # Register hook for TE compatibility
        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Set input tensor for pipeline parallelism."""
        self.decoder.set_input_tensor(input_tensor)

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
        """Forward pass of the model.
        
        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        
        # Decoder embedding
        if decoder_input is None:
            if self.pre_process:
                decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            else:
                # intermediate stage of pipeline
                # decoder will get hidden_states from encoder.input_tensor
                decoder_input = None

        # Rotary positional embeddings.
        if (
            rotary_pos_emb is None
            and self.position_embedding_type == 'rope'
            and not self.config.multi_latent_attention
        ):
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, packed_seq_params
            )
            cache_key = (
                rotary_seq_len,
                packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
            )
            if cache_key not in self.rotary_pos_emb_cache:
                self.rotary_pos_emb_cache[cache_key] = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
                )
            rotary_pos_emb = self.rotary_pos_emb_cache[cache_key]

        # Forward through transformer
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        # Output layer
        if not self.post_process:
            return hidden_states

        # Get logits
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        
        # If labels are provided, compute and return loss
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()
        
        loss = self.compute_language_model_loss(labels, logits)
        return loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """Provide sharded state dict for distributed checkpointing."""
        sharded_state_dict = {}

        if self.pre_process:
            embedding_prefix = f'{prefix}embedding.'
            embedding_sharded_state_dict = self.embedding.sharded_state_dict(
                prefix=embedding_prefix, sharded_offsets=sharded_offsets, metadata=metadata
            )
            sharded_state_dict.update(embedding_sharded_state_dict)

        decoder_prefix = f'{prefix}decoder.'
        decoder_sharded_state_dict = self.decoder.sharded_state_dict(
            prefix=decoder_prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        sharded_state_dict.update(decoder_sharded_state_dict)

        if self.post_process:
            output_layer_prefix = f'{prefix}output_layer.'
            output_layer_key = f'{output_layer_prefix}weight'
            if self.share_embeddings_and_output_weights:
                if not self.pre_process:
                    sharded_output_layer_tensor = metadata['embedding'][0]
                else:
                    sharded_embedding_tensor = embedding_sharded_state_dict[
                        f'{embedding_prefix}word_embeddings.weight'
                    ]
                    sharded_output_layer_tensor = sharded_embedding_tensor

                sharded_state_dict[output_layer_key] = sharded_output_layer_tensor
            else:
                output_layer_state_dict = self.output_layer.sharded_state_dict(
                    prefix=output_layer_prefix,
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(output_layer_state_dict)

        return sharded_state_dict

    def shared_embedding_or_output_weight(self):
        """Get shared embedding/output weight for gradient all-reduce."""
        if self.share_embeddings_and_output_weights:
            if self.pre_process:
                return self.embedding.word_embeddings.weight
            elif self.post_process:
                return self.output_layer.weight
        return None
