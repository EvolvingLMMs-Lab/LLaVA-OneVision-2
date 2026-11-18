# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/huggingface/transformers.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zero-centered RMSNorm variants for Qwen3.5 (ported verbatim from LoongForge)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer import TransformerConfig


class Qwen3NextRMSNormGated(nn.Module):
    """
    The RMSNorm layer with gating, used in the Qwen3-Next model.
    Args:
        hidden_size (int): The dimension size of the hidden layer
        eps (float, optional): Numerical stability parameter, default to 1e-6
        **kwargs: Other optional parameters
    """
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        """forward"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class Qwen3NextRMSNorm(torch.nn.Module):
    """
    Zero-Centered RMSNorm for Qwen3-Next.
    Uses (1 + weight) scaling to match HuggingFace implementation exactly.
    This eliminates the need for +1/-1 offset during weight conversion.

    Interface matches TENorm for compatibility with Megatron-Core build_module.
    """
    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.config = config
        self.eps = eps
        # Initialize weight to zeros (Zero-Centered), matching HuggingFace Qwen3NextRMSNorm
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states):
        """forward"""
        output = self._norm(hidden_states.float())
        # Zero-Centered: use (1 + weight) instead of weight
        # This matches HuggingFace's Qwen3NextRMSNorm exactly
        output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)
