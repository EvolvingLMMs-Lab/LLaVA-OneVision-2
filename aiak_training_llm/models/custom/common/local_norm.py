"""megatron local norm"""

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from aiak_training_llm.utils import opt_switches
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as ApexFusedRMSNorm
    HAVE_FUSED_RMS_NORM = True
except:
    HAVE_FUSED_RMS_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as ApexFusedLayerNorm
    HAVE_FUSED_LAYER_NORM = True
except:
    HAVE_FUSED_LAYER_NORM = False


class FusedRMSNorm(ApexFusedRMSNorm):
    """Fused RMS Norm"""
    def __init__(self,
                 config: TransformerConfig,
                 hidden_size: int,
                 eps=1e-5,
                 elementwise_affine=True):

        if not HAVE_FUSED_RMS_NORM:
            # TODO: Add pytorch only rms norm
            raise ValueError(f'Apex must currently be installed to use FusedRMSNorm op.')

        super().__init__(hidden_size,
                         eps=eps,
                         elementwise_affine=elementwise_affine)

        self.config = config

        self.sequence_parallel = self.config.sequence_parallel

        # set sequence parallelism flag on weight parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def forward(self, input):
        rope = self.__dict__.get('_ov2_rope')
        if rope is not None:
            # OV2_OPT_FUSED_QK_NORM_ROPE: the attention layer handed over its rotary embedding; norm + RoPE fused
            from aiak_training_llm.models.custom.common import fused_qk_norm_rope
            freqs, cu_seqlens = rope
            if not (self.elementwise_affine and not self.memory_efficient
                    and fused_qk_norm_rope.supported(input, self.weight, freqs)):
                raise RuntimeError('fused q/k norm + RoPE: unsupported input (the separate RoPE call is disabled)')
            return fused_qk_norm_rope.fused_qk_norm_rope(input, self.weight, self.eps, freqs, cu_seqlens)
        if opt_switches.flag('BITEXACT_QK_RMSNORM'):
            from aiak_training_llm.models.custom.common import bitexact_rmsnorm
            if (self.elementwise_affine and not self.memory_efficient
                    and tuple(self.normalized_shape) == (bitexact_rmsnorm.N2,) and input.is_cuda
                    and not torch.jit.is_tracing() and not torch.jit.is_scripting()
                    and not torch.compiler.is_compiling() and not torch.is_autocast_enabled()
                    and bitexact_rmsnorm.supported(input, self.weight)):
                # same arithmetic as apex's fused_rms_norm_affine (bitwise), faster kernels, no contiguous copies
                return bitexact_rmsnorm.bitexact_rms_norm(input, self.weight, self.eps)
        return super().forward(input)


class LocalNorm:
    """
    A conditional wrapper to initialize an instance of Megatron Local `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5, elementwise_affine=True):
        if config.normalization == "LayerNorm":
            if elementwise_affine:
                instance = FusedLayerNorm(
                    config=config,
                    hidden_size=hidden_size,
                    eps=eps,
                )
            else:
                assert HAVE_FUSED_LAYER_NORM, "Apex must currently be installed to use FusedLayerNorm op."
                instance = ApexFusedLayerNorm(
                    hidden_size,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                )
        elif config.normalization == "RMSNorm":
            instance = FusedRMSNorm(
                config=config,
                hidden_size=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
