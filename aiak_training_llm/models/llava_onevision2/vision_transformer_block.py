import logging
import os
from contextlib import nullcontext

import torch
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import TransformerBlock as MegatronTransformerBlock
from megatron.core.utils import make_viewless_tensor
from torch import Tensor


try:
    from megatron.core.transformer.custom_layers.transformer_engine import TEDelayedScaling
except ImportError:
    pass


logger = logging.getLogger(__name__)

_TORCH_COMPILE_LAYERS = os.environ.get("TORCH_COMPILE_LAYERS", "0") == "1"
_TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE", "default")
_TORCH_COMPILE_DYNAMIC = os.environ.get("TORCH_COMPILE_DYNAMIC", "1") == "1"


class TransformerBlock(MegatronTransformerBlock):
    """Transformer class."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Compiled-layer cache. MUST stay a plain dict (not an nn.Module /
        # Parameter) so nn.Module.__setattr__ routes it into instance __dict__
        # rather than self._modules. As a result state_dict() / load_state_dict
        # enumerate `self.layers` only (not the wrapped OptimizedModule's
        # `_orig_mod` subtree), keeping checkpoint keys identical to the
        # uncompiled path. The OptimizedModule anyway shares the original
        # layer's Parameter objects, so grad/optimizer state flow unchanged.
        self._compiled_layer_cache: dict[int, torch.nn.Module] = {}

    def _get_forward_layer(self, i: int):
        """Return the i-th layer, lazily wrapped in torch.compile when enabled."""
        layer = self.layers[i]
        if not _TORCH_COMPILE_LAYERS:
            return layer

        compiled = self._compiled_layer_cache.get(i)
        if compiled is None:
            if self.config.recompute_granularity is not None and _TORCH_COMPILE_MODE == "reduce-overhead":
                logger.warning(
                    "TORCH_COMPILE_MODE=reduce-overhead (CUDA graphs) is incompatible with "
                    "activation recompute (recompute_granularity=%r); falling back to mode='default'. "
                    "Set TORCH_COMPILE_MODE=default or disable recompute to use reduce-overhead.",
                    self.config.recompute_granularity,
                )
                compile_mode = "default"
            else:
                compile_mode = _TORCH_COMPILE_MODE

            compiled = torch.compile(
                layer,
                dynamic=_TORCH_COMPILE_DYNAMIC,
                mode=compile_mode,
            )
            self._compiled_layer_cache[i] = compiled
            logger.info(
                "TransformerBlock: torch.compiled layer %d (mode=%s, dynamic=%s).",
                i,
                compile_mode,
                _TORCH_COMPILE_DYNAMIC,
            )
        return compiled

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: list[PackedSeqParams] | None = None,
        **kwargs,
    ):
        """forward with list of packed params"""
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context and fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_type=attn_mask_type,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    **kwargs,
                )
            else:
                for l_no in range(len(self.layers)):
                    layer = self._get_forward_layer(l_no)
                    with self.offload_context:
                        if (len(self.cuda_graphs) == 0) or (not self.training):
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                attn_mask_type=attn_mask_type,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                inference_params=inference_params,
                                packed_seq_params=packed_seq_params[l_no] if packed_seq_params is not None else None,
                                **kwargs,
                            )
                            # CUDA graph doesn't output context and is expected to be None
                            assert (context is None) or (not self.config.enable_cuda_graph) or (not self.training)
                        else:
                            # CUDA graph replay for layer `l_no` and microbatch `self.current_microbatch`
                            # CUDA graph requires positional arguments with the exception of is_first_microbatch.
                            # Also CUDA graph accepts only Tensor inputs and outputs. Hence, the arg list and
                            # returned list is limited to `hidden_states`.
                            assert (len(self.cuda_graphs) > l_no) and (
                                self.current_microbatch < len(self.cuda_graphs[l_no])
                            )
                            hidden_states = self.cuda_graphs[l_no][self.current_microbatch](
                                hidden_states,
                                is_first_microbatch=(self.current_microbatch == 0),
                            )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states,
                requires_grad=True,
                keep_graph=True,
            )

        return hidden_states

    def forward_debug(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: list[PackedSeqParams] | None = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass with layer-by-layer debug outputs for consistency checking.

        Returns:
            dict: Contains:
                - 'input_hidden_states': Input to the transformer block
                - 'input_rotary_pos_emb': Rotary position embeddings input
                - 'layer_outputs': Dict mapping layer index to output after that layer
                - 'final_output': Final output (after final_layernorm if present)
                - 'before_final_layernorm': Output before final_layernorm (if present)
        """
        output = {}

        if not self.pre_process:
            hidden_states = self.input_tensor

        # Save input
        output["input_hidden_states"] = hidden_states.clone()
        if rotary_pos_emb is not None:
            output["input_rotary_pos_emb"] = rotary_pos_emb.clone()

        # Make viewless tensor
        hidden_states = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        # Layer-by-layer outputs
        layer_outputs = {}

        with rng_context and fp8_context:
            for l_no in range(len(self.layers)):
                layer = self._get_forward_layer(l_no)
                # Save input to this layer
                layer_outputs[f"layer_{l_no}_input"] = hidden_states.clone()

                with self.offload_context:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        attn_mask_type=attn_mask_type,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params[l_no] if packed_seq_params is not None else None,
                        **kwargs,
                    )

                # Save output of this layer
                layer_outputs[f"layer_{l_no}_output"] = hidden_states.clone()

                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        output["layer_outputs"] = layer_outputs
        output["before_final_layernorm"] = hidden_states.clone()

        # Final layer norm
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states,
                requires_grad=True,
                keep_graph=True,
            )

        output["final_output"] = hidden_states.clone()

        return output

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        **kwargs,
    ):
        """Forward method with activation checkpointing."""
        if attn_mask_type is not None:
            attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                attn_mask_type,
                context,
                context_mask,
                rotary_pos_emb,
            ):
                if attn_mask_type is not None:
                    attn_mask_type = AttnMaskType(attn_mask_type.item())

                for index in range(start, end):
                    layer = self._get_forward_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        attn_mask_type=attn_mask_type,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        inference_params=None,
                        packed_seq_params=packed_seq_params[index] if packed_seq_params is not None else None,
                        **kwargs,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    attn_mask_type,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    **kwargs,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    attn_mask_type,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    **kwargs,
                )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        attn_mask_type,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        **kwargs,
                    )

        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states
