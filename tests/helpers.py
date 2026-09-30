# SPDX-License-Identifier: Apache-2.0
"""Test helper functions and small model configurations for CPU testing."""

from typing import Any

from vllm_rlt.models import NanbeigeConfig, OuroConfig


def tiny_ouro_config(**overrides: Any) -> OuroConfig:
    """Small randomly initialized Ouro architecture for tests."""
    values = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        total_ut_steps=4,
        bos_token_id=0,
        eos_token_id=0,
    )
    values.update(overrides)
    return OuroConfig(**values)


def tiny_nanbeige_config(**overrides: Any) -> NanbeigeConfig:
    """Small randomly initialized Nanbeige architecture for tests."""
    values = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        num_loops=2,
        total_ut_steps=2,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    values.update(overrides)
    return NanbeigeConfig(**values)
