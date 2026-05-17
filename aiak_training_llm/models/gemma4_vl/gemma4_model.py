"""Gemma4 LLM backbone wrapper (Megatron LanguageModule subclass).

Mirrors :class:`aiak_training_llm.models.qwen.qwen_model.QwenModel` and adds three
Gemma4-specific behaviors to ``forward``:

  1. Embedding scaling: decoder_input *= sqrt(hidden_size). HF reference:
     ``Gemma4TextScaledWordEmbedding.forward`` multiplies token embeddings by
     ``embed_scale = config.hidden_size ** 0.5``.

  2. Final logit soft-capping: ``logits = softcap * tanh(logits / softcap)``,
     gated by ``config.final_logit_softcapping`` (=30.0 for Gemma4-26B-A4B).
     Disabled when the value is None or <= 0.

  3. Dual RoPE: when ``config.layer_pattern`` is non-empty (Gemma4 hybrid
     attention), construct two independent rotary embeddings — sliding layers
     get the stock :class:`RotaryEmbedding` (``head_dim=256``, ``theta=1e4``,
     full rotation), global layers get :class:`Gemma4ProportionalRotaryEmbedding`
     (``head_dim=512``, ``theta=1e6``, ``partial_rotary_factor=0.25`` with HF
     proportional zero-padding). Both are evaluated each step and shipped as
     a ``dict[str, Tensor]`` keyed by layer-type. The unmodified
     :class:`TransformerBlock` passes the dict through verbatim;
     :class:`Gemma4SelfAttention.forward` selects the matching tensor based on
     its own ``self.layer_number`` and ``config.layer_pattern``.
"""

import math
from collections import OrderedDict
from typing import Literal, Optional

import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from aiak_training_llm.models.gemma4_vl.gemma4_proportional_rotary import (
    Gemma4ProportionalRotaryEmbedding,
)
from aiak_training_llm.models.qwen.qwen_model import _load_state_dict_hook_ignore_extra_state


class Gemma4Model(LanguageModule):
    """Gemma4 transformer language model backbone."""

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
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rotary_base_sliding: Optional[int] = None,
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

        self.model_type = ModelType.encoder_or_decoder

        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling

        # Gemma4 embedding scale factor: applied after embedding lookup.
        # See HF Gemma4TextScaledWordEmbedding.embed_scale.
        self._embed_scale = math.sqrt(self.config.hidden_size)

        # Final logit soft-capping; off when <= 0 / None.
        self._final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )

        if self.position_embedding_type == "rope" and not self.config.multi_latent_attention:
            sliding_kv_channels = (
                self.config.per_layer_kv_channels.get("sliding")
                if self.config.per_layer_kv_channels
                else None
            ) or self.config.kv_channels
            global_kv_channels = (
                self.config.per_layer_kv_channels.get("global")
                if self.config.per_layer_kv_channels
                else None
            ) or self.config.kv_channels

            sliding_base = rotary_base_sliding if rotary_base_sliding is not None else rotary_base
            global_base = rotary_base

            partial_global = getattr(self.config, "partial_rotary_factor", 1.0) or 1.0

            self.rotary_pos_emb_sliding = RotaryEmbedding(
                kv_channels=sliding_kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=sliding_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

            if self.config.layer_pattern and "global" in self.config.layer_pattern:
                self.rotary_pos_emb_global = Gemma4ProportionalRotaryEmbedding(
                    head_dim=global_kv_channels,
                    partial_rotary_factor=partial_global,
                    rotary_base=global_base,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    use_cpu_initialization=self.config.use_cpu_initialization,
                )
            else:
                self.rotary_pos_emb_global = None

        self.rotary_pos_emb_cache = {}

        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if post_process:
            if self.config.defer_embedding_wgrad_compute:
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
                self.config, self.state_dict(), prefix=f"{type(self).__name__}_init_ckpt"
            )

        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """See :meth:`LanguageModule.set_input_tensor`."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"
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
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:
        if decoder_input is None:
            if self.pre_process:
                decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
                # Gemma4 embedding scaling: in-dtype multiply preserves the
                # downstream casting policy (do NOT promote to fp32 here).
                decoder_input = decoder_input * self._embed_scale
            else:
                decoder_input = None

        if (
            rotary_pos_emb is None
            and self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            rotary_seq_len = self.rotary_pos_emb_sliding.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, packed_seq_params
            )
            packed = (
                packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            )
            sliding_emb = self.rotary_pos_emb_sliding(rotary_seq_len, packed_seq=packed)
            if self.rotary_pos_emb_global is not None:
                global_emb = self.rotary_pos_emb_global(rotary_seq_len, packed_seq=packed)
                rotary_pos_emb = {"sliding": sliding_emb, "global": global_emb}
            else:
                rotary_pos_emb = sliding_emb

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

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        # Gemma4 final-logit soft-capping: logits = c * tanh(logits / c).
        # Bounds extreme logits before cross-entropy. Off when c is None / <= 0.
        if self._final_logit_softcapping and self._final_logit_softcapping > 0:
            cap = float(self._final_logit_softcapping)
            logits = cap * torch.tanh(logits / cap)

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "decoder_input": decoder_input,
                    "logits": logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix="input_and_logits")

        if labels is None:
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        return loss

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f"{prefix}output_layer._extra_state"
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f"Expected output layer extra state to be empty, got: {output_extra_state}"
        return sharded_state_dict
