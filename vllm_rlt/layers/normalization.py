# SPDX-License-Identifier: Apache-2.0
"""Shared normalization layers across models."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Retains FP32 precision during reduction to preserve numerical stability
    across all input activation dtypes (BF16/FP16/FP32).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(hidden.dtype)
