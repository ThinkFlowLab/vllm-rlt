# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared RMSNorm layer."""

import pytest
import torch

from vllm_rlt.layers import RMSNorm
from vllm_rlt.models.nanbeige import NanbeigeRMSNorm
from vllm_rlt.models.ouro import OuroRMSNorm


def test_backward_compatible_aliases():
    assert OuroRMSNorm is RMSNorm
    assert NanbeigeRMSNorm is RMSNorm


def test_rmsnorm_weight_parameter_and_state_dict():
    hidden_size = 128
    norm = RMSNorm(hidden_size, eps=1e-5)

    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.shape == (hidden_size,)
    assert torch.all(norm.weight == 1.0)
    assert set(norm.state_dict().keys()) == {"weight"}

    # Loading a standard state_dict
    state = {"weight": torch.randn(hidden_size)}
    norm.load_state_dict(state)
    assert torch.allclose(norm.weight, state["weight"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_forward_numerics_and_dtypes(dtype):
    torch.manual_seed(42)
    hidden_size = 64
    eps = 1e-5
    norm = RMSNorm(hidden_size, eps=eps).to(dtype)

    # Random weights and input
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size))

    x = torch.randn(2, 8, hidden_size, dtype=dtype)
    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == dtype

    # Verify exact reference computation matching forward()
    x_fp32 = x.float()
    variance = x_fp32.square().mean(-1, keepdim=True)
    normalized = (x_fp32 * torch.rsqrt(variance + eps)).to(dtype)
    ref = norm.weight * normalized

    assert torch.equal(out, ref)


def test_rmsnorm_explicit_epsilon():
    hidden_size = 16
    x = torch.ones(1, 4, hidden_size)

    norm_small_eps = RMSNorm(hidden_size, eps=1e-8)
    norm_large_eps = RMSNorm(hidden_size, eps=1.0)

    out_small = norm_small_eps(x)
    out_large = norm_large_eps(x)

    assert not torch.allclose(out_small, out_large)
    # Variance of ones is 1.0; 1.0 / sqrt(1.0 + 1.0) = 1.0 / sqrt(2) ≈ 0.7071
    expected_large = 1.0 / (2.0**0.5)
    assert torch.allclose(out_large, torch.full_like(x, expected_large), atol=1e-4)
