"""Top-level Gemma4-VL VLM (Gemma4 vision tower + Gemma4 adapter + Gemma4 LLM).

Subclasses :class:`LlavaOnevision2` so the provider/freeze/set_input_tensor
plumbing is reused, but overrides ``__init__`` and ``forward`` to swap in
the HF-verbatim Gemma4 vision tower / adapter and the Gemma4 MoE LLM.

``vision_layer_spec`` and ``adapter_layer_spec`` arguments are accepted for
provider-call uniformity but are IGNORED — the Gemma4 vision tower / adapter
are plain ``nn.Module`` stacks (see ``gemma4_vision_tower.py`` /
``gemma4_adapter.py``), not Megatron ``ModuleSpec``-driven blocks.

The vision data path (``images is not None``) currently raises
``NotImplementedError``: the production Gemma4 image processor / dataloader
that produces ``pixel_values`` + ``pixel_position_ids`` in the HF layout the
tower expects has not yet been wired into the OV2 dataloader (P3.x). LM-only
forward (``images is None``, P1 smoke + pure-text training) is fully
functional.
"""

import logging
from functools import partial
from typing import Optional

import torch
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec

from aiak_training_llm.models.gemma4_vl.gemma4_adapter import Gemma4Adapter
from aiak_training_llm.models.gemma4_vl.gemma4_model import Gemma4Model
from aiak_training_llm.models.gemma4_vl.gemma4_vision_tower import Gemma4VisionTower
from aiak_training_llm.models.llava_onevision2.llava_onevision2_model import (
    LlavaOnevision2,
    _load_state_dict_hook_ignore_param_names,
)


class Gemma4VL(LlavaOnevision2):
    def __init__(
        self,
        language_config,
        vision_config,
        adapter_config,
        language_layer_spec: ModuleSpec,
        vision_layer_spec: ModuleSpec,
        adapter_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        allow_missing_adapter_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = "rope",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        language_rotary_base_sliding: Optional[int] = None,
        fp16_lm_cross_entropy: bool = False,
        share_embeddings_and_output_weights: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        del vision_layer_spec, adapter_layer_spec

        from megatron.core.transformer import MegatronModule

        MegatronModule.__init__(self, config=language_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.adapter = None
        self.language_model = None

        if self.add_encoder:
            self.vision_model = Gemma4VisionTower(
                transformer_config=vision_config,
                hidden_size=vision_config.hidden_size,
                num_hidden_layers=vision_config.num_hidden_layers,
                num_attention_heads=vision_config.num_attention_heads,
                num_key_value_heads=vision_config.num_key_value_heads,
                head_dim=vision_config.head_dim,
                intermediate_size=vision_config.intermediate_size,
                patch_size=vision_config.patch_size,
                in_channels=vision_config.in_channels,
                position_embedding_size=vision_config.position_embedding_size,
                pooling_kernel_size=vision_config.pooling_kernel_size,
                rope_theta=vision_config.rope_theta,
                rms_norm_eps=vision_config.rms_norm_eps,
                hidden_activation=vision_config.hidden_activation,
                use_clipped_linears=vision_config.use_clipped_linears,
                standardize=vision_config.standardize,
            )
            self.adapter = Gemma4Adapter(
                vision_hidden_size=vision_config.hidden_size,
                text_hidden_size=language_config.hidden_size,
                rms_norm_eps=adapter_config.layernorm_epsilon,
            )
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"adapter.{name}" for name in self.adapter.state_dict().keys()
                ]
                self.adapter.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

        if self.add_decoder:
            self.language_model = Gemma4Model(
                config=language_config,
                transformer_layer_spec=language_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                rotary_base_sliding=language_rotary_base_sliding,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                scatter_embedding_sequence_parallel=False,
                fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
            )
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, (
            "input_tensor should only be length 1 for Gemma4-VL"
        )

        if self.add_encoder and self.add_decoder:
            raise NotImplementedError(
                "Gemma4-VL encoder-stage set_input_tensor requires Gemma4VisionTower "
                "to expose a Megatron-style set_input_tensor; deferred to P3.x along "
                "with the vision data path."
            )
        if self.add_encoder:
            raise NotImplementedError(
                "Gemma4-VL encoder-only pipeline stage requires Gemma4VisionTower "
                "to expose a Megatron-style set_input_tensor; deferred to P3.x."
            )
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def forward_debug(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma4-VL forward_debug is deferred to P3.x; the inherited "
            "LlavaOnevision2.forward_debug calls Gemma4VisionTower.forward_debug "
            "which does not exist (the Gemma4 vision tower is a plain nn.Module, "
            "not a Megatron block). The HF↔Megatron consistency check in Phase C "
            "must call Gemma4VisionTower.forward / Gemma4Adapter.forward directly."
        )

    @staticmethod
    def _prepare_vision_inputs(
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        patch_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_counts = image_grid_thw.prod(dim=-1).to(torch.long)
        if int(patch_counts.sum().item()) != images.shape[0]:
            raise ValueError(
                f"Gemma4-VL image patch count mismatch: images has {images.shape[0]} "
                f"patches but image_grid_thw sums to {int(patch_counts.sum().item())}."
            )

        max_patches = int(patch_counts.max().item())
        batch_size = int(patch_counts.numel())
        pixel_values = images.new_zeros((batch_size, max_patches, images.shape[-1]))
        pixel_position_ids = torch.full(
            (batch_size, max_patches, 2),
            -1,
            dtype=torch.long,
            device=images.device,
        )

        offset = 0
        for batch_idx, patch_count_tensor in enumerate(patch_counts):
            patch_count = int(patch_count_tensor.item())
            next_offset = offset + patch_count
            pixel_values[batch_idx, :patch_count] = images[offset:next_offset]

            if patch_positions is None:
                _t, h, w = image_grid_thw[batch_idx].tolist()
                h_coords = torch.arange(h, dtype=torch.long, device=images.device).repeat_interleave(w)
                w_coords = torch.arange(w, dtype=torch.long, device=images.device).repeat(h)
                coords = torch.stack((w_coords, h_coords), dim=-1)
                if coords.shape[0] != patch_count:
                    raise ValueError(
                        f"Default Gemma4-VL pixel positions only support single-frame images; "
                        f"got image_grid_thw={image_grid_thw[batch_idx].tolist()}."
                    )
            else:
                coords = patch_positions[offset:next_offset, -2:].to(device=images.device, dtype=torch.long)
                coords = torch.stack((coords[:, 1], coords[:, 0]), dim=-1)

            pixel_position_ids[batch_idx, :patch_count] = coords
            offset = next_offset

        return pixel_values, pixel_position_ids

    def forward(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType | None = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        inference_params: InferenceParams = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        patch_positions: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del position_ids

        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "Gemma4-VL video path not implemented; pixel_values_videos / "
                "video_grid_thw must be None."
            )

        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )

        if images is not None and self.add_encoder and not use_inference_kv_cache:
            if image_grid_thw is None:
                raise ValueError("Gemma4-VL image_grid_thw is required when images are provided.")
            pixel_values, pixel_position_ids = self._prepare_vision_inputs(
                images, image_grid_thw, patch_positions
            )
            image_embeddings = self.vision_model(pixel_values, pixel_position_ids)
            image_embeddings = self.adapter(image_embeddings)

            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeddings.shape[0]
            if n_image_features > n_image_tokens:
                logging.getLogger(__name__).warning(
                    "Trimming %d extra Gemma4 image embedding(s) "
                    "(n_image_features=%d, n_image_tokens=%d).",
                    n_image_features - n_image_tokens,
                    n_image_features,
                    n_image_tokens,
                )
                image_embeddings = image_embeddings[:n_image_tokens]
            elif n_image_features < n_image_tokens:
                raise ValueError(
                    f"Gemma4 image features {n_image_features} < image tokens {n_image_tokens}"
                )

            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = image_embeddings.shape[0]
        else:
            image_embeddings = None

        if not self.add_decoder:
            raise NotImplementedError(
                "Gemma4-VL encoder-only pipeline stage requires the vision data "
                "path which is not yet implemented; see images-not-None branch."
            )

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None
            )
            if use_inference_kv_cache or images is None:
                combined_embeddings = language_embeddings
            elif (input_ids == self.config.image_token_id).any().item():
                images_mask = (
                    (input_ids == self.config.image_token_id)
                    .transpose(0, 1)
                    .unsqueeze(-1)
                    .expand_as(language_embeddings)
                    .to(language_embeddings.device)
                )
                image_embeddings = image_embeddings.to(
                    language_embeddings.device, language_embeddings.dtype
                )
                combined_embeddings = language_embeddings.masked_scatter(
                    images_mask, image_embeddings
                )
            else:
                combined_embeddings = language_embeddings

            if self.config.sequence_parallel:
                seq_len = combined_embeddings.size(0)
                tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
                remainder = seq_len % tp_world_size
                if remainder != 0:
                    pad = tp_world_size - remainder
                    pad_shape = (pad,) + combined_embeddings.shape[1:]
                    pad_tensor = combined_embeddings.new_zeros(pad_shape)
                    combined_embeddings = torch.cat(
                        (combined_embeddings, pad_tensor), dim=0
                    )
                combined_embeddings = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        combined_embeddings
                    )
                )
        else:
            combined_embeddings = None

        return self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=combined_embeddings,
            labels=labels,
            rotary_pos_emb=None,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs={},
        )
