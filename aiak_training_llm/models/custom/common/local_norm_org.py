"""megatron local norm"""

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as ApexFusedRMSNorm
    HAVE_FUSED_RMS_NORM = True
except Exception as e:
    HAVE_FUSED_RMS_NORM = False
    print("Failed to import Apex FusedRMSNorm:", repr(e))
print("HAVE_FUSED_RMS_NORM =", HAVE_FUSED_RMS_NORM)
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as ApexFusedLayerNorm
    HAVE_FUSED_LAYER_NORM = True
except:
    HAVE_FUSED_LAYER_NORM = False
print("HAVE_FUSED_LAYER_NORM =", HAVE_FUSED_LAYER_NORM)

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


class LocalNorm:
    """
    A conditional wrapper to initialize an instance of Megatron Local `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5, elementwise_affine=True):
        if config.normalization == "LayerNorm":
            if elementwise_affine:
                print("LayerNorm")
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
