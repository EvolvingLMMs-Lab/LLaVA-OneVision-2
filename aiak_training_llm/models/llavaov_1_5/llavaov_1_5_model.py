import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch
from megatron.core import InferenceParams, parallel_state
from megatron.core.models.common.embeddings.rope_utils import \
    get_pos_emb_on_this_cp_rank
from megatron.core.models.common.embeddings.rotary_pos_embedding import \
    RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from aiak_training_llm.models.llavaov_1_5.rice_vision_model import (
    RiceViTModel, VisionModel)
from aiak_training_llm.models.fastvit import FastViTModel
from aiak_training_llm.models.qwen import QwenModel
from aiak_training_llm.models.qwen_vl.adapter import Adapter
from aiak_training_llm.models.qwen_vl.utils import get_inputs_on_this_cp_rank


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _get_thd_freqs_on_this_cp_rank(cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor) -> Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    else:
        return freqs[: x.size(0)]


def _apply_mrope_bshd(t, freq, config, cu_seqlens=None, mrope_section=[16, 24, 24]):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    (https://qwenlm.github.io/blog/qwen2-vl/).
    Args:
        t (torch.Tensor): Input tensor of shape [S, B, heads, dim]
        freq (torch.Tensor): Frequency tensor of shape [S, B, 3, dim]
    """
    cos = freq.cos().to(dtype=t.dtype)
    sin = freq.sin().to(dtype=t.dtype)
    mrope_section = mrope_section * 2

    cos = torch.cat([m[..., i % 3, :] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(2)
    sin = torch.cat([m[..., i % 3, :] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(2)

    t = (t * cos) + (_rotate_half(t) * sin)
    return t


def apply_mrope(t, freq, config, cu_seqlens=None, mrope_section=[16, 24, 24]):
    """ mrope """
    if cu_seqlens is not None:
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        return torch.cat(
            [
                _apply_mrope_bshd(
                    x.unsqueeze(1),
                    _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freq),
                    config,
                    cu_seqlens,
                    mrope_section
                )
                for x in torch.split(t, seqlens)
            ]
        ).squeeze(1)
    else:
        return _apply_mrope_bshd(t, freq, config, cu_seqlens, mrope_section)



class LlavaOnevision1_5(MegatronModule):
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
        language_position_embedding_type: str = 'rope',
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
            # Use FastViT as the vision encoder (following FastVLM repo)
            self.vision_model = FastViTModel(
                vision_config,
                vision_layer_spec,
            )
            print(f'self,vision_model: {self.vision_model}')
            # Original Rice/SigLIP encoder (commented out):
            # if vision_config.normalization == "RMSNorm":
            #     self.vision_model = RiceViTModel(
            #         vision_config,
            #         vision_layer_spec,
            #     )
            # else:
            #     self.vision_model = VisionModel(
            #         vision_config,
            #         vision_layer_spec,
            #     )
            # Map (intermediate) vision model outputs to the language model input dimension.
            # from megatron.training import print_rank_0
            print(f"[DEBUG] Creating adapter: vision_config.hidden_size={vision_config.hidden_size}")
            print(f"[DEBUG] Creating adapter: language_config.hidden_size={language_config.hidden_size}")
            self.adapter = Adapter(
                adapter_config,
                adapter_layer_spec,
                input_size=vision_config.hidden_size,  # input size to the adapter.
                output_size=language_config.hidden_size,  # output size of the adapter.
            )
            # This allows ignoring missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains pretrained
            # vision and language models but not the projection from vision model outputs to language model inputs.
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"adapter.{name}"
                    for name in self.adapter.state_dict().keys()
                ]
                self.adapter.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

        # # This attribute is needed to check if an all-reduce is required
        # # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        if self.add_decoder:
            # self.rotary_emb = Qwen2VLRotaryEmbedding(
            #     dim=language_config.hidden_size // language_config.num_attention_heads,
            #     theta=language_rotary_base
            # )
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
            )
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )


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
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, 
        freeze_language_model: bool, 
        freeze_vision_model: bool, 
        freeze_adapter: bool
    ):
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
        attn_mask_type: Optional[AttnMaskType] = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        inference_params: InferenceParams = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None
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
        # from megatron.training import print_rank_0
        # print_rank_0(
        #     f"> forward step: input_ids shape {input_ids.shape}, "
        #     f"images shape {images.shape}, "
        #     f"image_grid_thw shape {image_grid_thw.shape}, "
        #     # f"labels shape {labels.shape}, "
        #     f"attn_mask_type {attn_mask_type}, "
        #     f"position_ids shape {position_ids.shape if position_ids is not None else None}"
        # )
        # print_rank_0(input_ids)
        # print_rank_0(position_ids)
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip image token computation if they were computed already
        # earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder:
            if images is not None:
                image_embeddings, window_index = self.vision_model(images, \
                                    grid_thw=image_grid_thw) # [img_len, h_vision]
                image_embeddings = self.adapter(image_embeddings, window_index)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeddings.shape[0]
                if n_image_tokens != n_image_features:
                    # raise ValueError(
                    #     f"Image features {n_image_features} != image tokens {n_image_tokens}"
                    # )
                    if n_image_features > n_image_tokens:
                        # Trim extra vision features if the model produced more than requested tokens.
                        image_embeddings = image_embeddings[:n_image_tokens]
                        if window_index is not None:
                            window_index = window_index[:n_image_tokens]
                    else:
                        # Pad missing vision features with zeros to align with token count.
                        pad_len = n_image_tokens - n_image_features
                        pad_emb = torch.zeros(
                            (pad_len, image_embeddings.size(1)),
                            device=image_embeddings.device,
                            dtype=image_embeddings.dtype,
                        )
                        image_embeddings = torch.cat([image_embeddings, pad_emb], dim=0)
                        if window_index is not None:
                            pad_idx = window_index[-1:].repeat(pad_len, *([1] * (window_index.dim() - 1)))
                            window_index = torch.cat([window_index, pad_idx], dim=0)

                # If running inference, the language model KV cache will be updated for image token positions.
                # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
                if inference_params is not None:
                    inference_params.key_value_memory_dict["image_tokens_count"] = (
                        image_embeddings.shape[0]
                    )
            if pixel_values_videos is not None:
                raise NotImplementedError(
                    "Video input is not supported in RiceVLModel. "
                    "Please use a different model that supports video input."
                )
                # pixel_values_videos: [video_seq_len, num_channels * temporal_size * patch_size * patch_size]
                # video_grid_thw: [num_videos, 3]
                # # Get the video embeddings from the vision model.
                # video_embeddings, window_index = self.vision_model(pixel_values_videos, grid_thw=video_grid_thw)
                # video_embeddings = self.adapter(video_embeddings, window_index)
                # n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                # n_video_features = video_embeddings.shape[0]
                # if n_video_tokens != n_video_features:
                #     raise ValueError(
                #         f"video features {n_video_features} != video tokens {n_video_tokens}"
                #     )

                # # If running inference, the language model KV cache will be updated for image token positions.
                # # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
                # if inference_params is not None:
                #     inference_params.key_value_memory_dict["video_tokens_count"] = (
                #         video_embeddings.shape[0]
                #     )
        else:
            vision_embeddings = self.encoder_hidden_state.squeeze(0) if self.encoder_hidden_state is not None else None

        if not self.add_decoder:
            # p2p_communicate_shapes requires dim=3
            vision_embeddings = make_viewless_tensor(
                inp=vision_embeddings.unsqueeze(0),
                requires_grad=True,
                keep_graph=True
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
                        (input_ids == image_token_id).transpose(0, 1)
                        .unsqueeze(-1)
                        .expand_as(language_embeddings)
                        .to(language_embeddings.device)
                    )
                    image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                    combined_embeddings = language_embeddings.masked_scatter(images_mask, image_embeddings)

                if pixel_values_videos is not None and self.config.video_token_id in input_ids:
                    video_token_id = self.config.video_token_id
                    videos_mask = (
                        (input_ids == video_token_id).transpose(0, 1)
                        .unsqueeze(-1)
                        .expand_as(language_embeddings)
                        .to(language_embeddings.device)
                    )
                    video_embeddings = video_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                    combined_embeddings = language_embeddings.masked_scatter(videos_mask, video_embeddings)

            if self.config.context_parallel_size > 1:
                combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings)

        else:
            combined_embeddings = None
            input_tensor = self.language_model.decoder.input_tensor

        # rotary_pos_emb = self.rotary_emb(position_ids).transpose(0, 2).contiguous()

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


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
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
