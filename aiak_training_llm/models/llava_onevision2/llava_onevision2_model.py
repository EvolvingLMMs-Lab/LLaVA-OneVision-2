import logging
from collections import namedtuple
from functools import partial

import torch
from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import make_viewless_tensor

from aiak_training_llm.models.llava_onevision2.onevision_encoder_model import OneVisionEncoderModel
from aiak_training_llm.models.qwen import QwenModel
from aiak_training_llm.models.llava_onevision2.adapter import Adapter
from aiak_training_llm.models.qwen_vl.utils import get_inputs_on_this_cp_rank


class LlavaOnevision2(MegatronModule):
    """
    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
            language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for positional
            embedding.
        vision_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        adapter_config (TransformerConfig): Config for the projection from vision model outputs to language
            model inputs.
        adapter_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        adapter_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        allow_missing_adapter_checkpoint (bool): Allow vision projection weights to be missing when
            loading a checkpoint. Default False.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks.
            This is typically True for training and False for inference.
        language_position_embedding_type (str): Position embedding type to use in the language model.
            Default learned absolute.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings in the
            language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder
            (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder will live on only a subset of the pipeline stages
            (specifically, every stage after the first one).
        img_h (int): The height of each image that the ViT will see.
        img_w (int): The width of each image that the ViT will see.
        patch_dim (int): The size of each patch side.
    """

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
        language_rotary_base: int = 1000000,
        fp16_lm_cross_entropy: bool = False,
        share_embeddings_and_output_weights: bool = True,
        seq_len_interpolation_factor: float = None,
    ) -> None:
        super().__init__(config=language_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.adapter = None
        self.language_model = None

        #  define the vision model and the projection from vision model outputs to language model inputs.
        if self.add_encoder:
            spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
            self.vision_model = OneVisionEncoderModel(
                vision_config,
                vision_layer_spec,
                spatial_merge_size=spatial_merge_size,
            )

            self.adapter = Adapter(
                adapter_config,
                adapter_layer_spec,
                input_size=vision_config.hidden_size,  # input size to the adapter.
                output_size=language_config.hidden_size,  # output size of the adapter.
                spatial_merge_size=spatial_merge_size,
            )
            # This allows ignoring missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains pretrained
            # vision and language models but not the projection from vision model outputs to language model inputs.
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [f"adapter.{name}" for name in self.adapter.state_dict().keys()]
                self.adapter.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

        # # This attribute is needed to check if an all-reduce is required
        # # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        if self.add_decoder:
            self.language_model = QwenModel(
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
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                scatter_embedding_sequence_parallel=False,
            )
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """set input tensor"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_adapter: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_adapter (bool): Freeze the vision adapter module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_adapter and self.adapter is not None:
            modules.append(self.adapter)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

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
        patch_positions: list[torch.tensor] | None = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen-VL model.

        Args:
            images (torch.Tensor): input image of shape [image_size / 3d_patch_size, in_channels * 3d_patch_size].
            image_grid_thw (torch.Tensor): image grid tensor of shape [num_images, 3]
            pixel_values_videos (torch.Tensor): The tensors corresponding to the input videos,
                            shape:[seq_length, num_channels * temporal_size * patch_size * patch_size]
            video_grid_thw (torch.Tensor): video grid tensor of shape [num_videos, 3]
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """

        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip image token computation if they were computed already
        # earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder:
            if images is not None:
                image_embeddings = self.vision_model(
                    images, grid_thw=image_grid_thw, patch_positions=patch_positions
                )  # [img_len, h_vision]
                image_embeddings = self.adapter(image_embeddings)

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeddings.shape[0]
                if n_image_features > n_image_tokens:
                    # Trim any extra embeddings produced by ViT SP-padding tokens that
                    # passed through the adapter.  The language sequence has a fixed
                    # number of image-token slots (n_image_tokens); any embeddings
                    # beyond that count must not enter the fusion step.
                    logging.getLogger(__name__).warning(
                        "Trimming %d extra image embedding(s) (n_image_features=%d, "
                        "n_image_tokens=%d); likely caused by ViT SP-padding tokens "
                        "passing through the adapter.",
                        n_image_features - n_image_tokens,
                        n_image_features,
                        n_image_tokens,
                    )
                    image_embeddings = image_embeddings[:n_image_tokens]
                elif n_image_features < n_image_tokens:
                    raise ValueError(f"Image features {n_image_features} < image tokens {n_image_tokens}")

                # If running inference, the language model KV cache will be updated for image token positions.
                # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
                if inference_params is not None:
                    inference_params.key_value_memory_dict["image_tokens_count"] = image_embeddings.shape[0]
            if pixel_values_videos is not None:
                raise NotImplementedError(
                    "In LLaVA-OneVision2, video and images are treated as the same input type. "
                    "Please use the 'images' argument for both."
                )

        else:
            vision_embeddings = self.encoder_hidden_state.squeeze(0) if self.encoder_hidden_state is not None else None

        if not self.add_decoder:
            # p2p_communicate_shapes requires dim=3
            vision_embeddings = make_viewless_tensor(
                inp=vision_embeddings.unsqueeze(0), requires_grad=True, keep_graph=True
            )
            return vision_embeddings

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None
            )  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already
            # earlier for this sample.
            if use_inference_kv_cache or (images is None and pixel_values_videos is None):
                combined_embeddings = language_embeddings
            else:
                if images is not None and self.config.image_token_id in input_ids:
                    image_token_id = self.config.image_token_id
                    images_mask = (
                        (input_ids == image_token_id)
                        .transpose(0, 1)
                        .unsqueeze(-1)
                        .expand_as(language_embeddings)
                        .to(language_embeddings.device)
                    )
                    image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                    combined_embeddings = language_embeddings.masked_scatter(images_mask, image_embeddings)

            if self.config.context_parallel_size > 1:
                combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings)

            if self.config.sequence_parallel:
                # Ensure sequence length is divisible by TP world size before scattering.
                seq_len = combined_embeddings.size(0)
                tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
                remainder = seq_len % tp_world_size
                if remainder != 0:
                    pad = tp_world_size - remainder
                    pad_shape = (pad,) + combined_embeddings.shape[1:]
                    pad_tensor = combined_embeddings.new_zeros(pad_shape)
                    combined_embeddings = torch.cat((combined_embeddings, pad_tensor), dim=0)
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings
                )
        
        else:
            combined_embeddings = None
            input_tensor = self.language_model.decoder.input_tensor

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=combined_embeddings,
            labels=labels,
            # rotary_pos_emb=rotary_pos_emb,
            rotary_pos_emb=None,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs={},
        )

        return output

    def forward_debug(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        attn_mask_type: AttnMaskType | None = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        inference_params: InferenceParams = None,
    ) -> dict:
        """Forward pass with debug outputs for consistency checking.

        This method captures intermediate outputs at each stage of the forward pass
        to enable layer-by-layer comparison with HuggingFace implementation.

        Args:
            images (torch.Tensor): Input image patches of shape [total_patches, patch_dim].
            image_grid_thw (torch.Tensor): Grid dimensions [batch_size, 3] with (T, H, W).
            input_ids (torch.Tensor): Optional input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): Optional input text position ids.
            attention_mask (torch.Tensor): Optional attention mask.
            attn_mask_type: Optional attention mask type.
            labels (torch.Tensor): Optional target text labels.
            packed_seq_params: Optional packed sequence parameters.
            inference_params: Optional inference parameters.

        Returns:
            dict: Dictionary containing intermediate outputs:
                - "before_adapter": Vision encoder output before adapter/merger
                - "after_adapter": Vision encoder output after adapter/merger
                - "after_merger": Alias for "after_adapter"
                - "vision_encoder_debug": Debug output from vision encoder
                - "language_embeddings": Language model input embeddings (if input_ids provided)
                - "combined_embeddings": Combined vision+language embeddings (if input_ids provided)
                - "logits": Model output logits (if input_ids provided)
        """
        output = {}

        # Store inputs for consistency checking
        output["input_images"] = images.clone()
        output["input_grid_thw"] = image_grid_thw.clone()

        if self.add_encoder and images is not None:
            # Get vision encoder debug output
            vision_debug = self.vision_model.forward_debug(images, grid_thw=image_grid_thw)
            output["vision_encoder_debug"] = vision_debug

            # Vision encoder output (before adapter)
            vision_output = vision_debug.get("before_adapter")
            if vision_output is None:
                # Fallback: run normal forward
                vision_output = self.vision_model(images, grid_thw=image_grid_thw)
            output["before_adapter"] = vision_output.clone()

            # Apply adapter/merger
            image_embeddings = self.adapter(vision_output)
            output["after_adapter"] = image_embeddings.clone()
            output["after_merger"] = image_embeddings.clone()  # Alias for compatibility

        # If input_ids provided, continue with language model
        if input_ids is not None and self.add_decoder and self.pre_process:
            # Get language embeddings
            language_embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None)
            output["language_embeddings"] = language_embeddings.clone()

            # Combine vision and language embeddings
            if images is not None and self.config.image_token_id in input_ids:
                image_token_id = self.config.image_token_id
                images_mask = (
                    (input_ids == image_token_id)
                    .transpose(0, 1)
                    .unsqueeze(-1)
                    .expand_as(language_embeddings)
                    .to(language_embeddings.device)
                )
                image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                combined_embeddings = language_embeddings.masked_scatter(images_mask, image_embeddings)
            else:
                combined_embeddings = language_embeddings

            output["combined_embeddings"] = combined_embeddings.clone()

            # Run through language model
            logits = self.language_model(
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
            output["logits"] = logits

        return output


def _load_state_dict_hook_ignore_param_names(
    param_names: list[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and
    language model weights but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the
            missing and unexpected keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
