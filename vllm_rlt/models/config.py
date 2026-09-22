# SPDX-License-Identifier: Apache-2.0
# Adapted and modified from ByteDance/Ouro-1.4B configuration_ouro.py at
# 574fa66cb8bf5abdc979642d01cf2b79b16bfab1; see NOTICE for upstream attribution.
"""Native configuration for the first supported checkpoint, ByteDance/Ouro-1.4B."""

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

OURO_MODEL_ID = "ByteDance/Ouro-1.4B"
OURO_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


@dataclass(frozen=True)
class OuroConfig:
    vocab_size: int = 49152
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    total_ut_steps: int = 4
    early_exit_threshold: float = 1.0
    bos_token_id: int | None = 0
    eos_token_id: int | None = 0
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    rope_scaling: dict[str, Any] | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None
    layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "total_ut_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for split-half RoPE")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        for name in ("rope_theta", "rms_norm_eps", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.early_exit_threshold) or not 0 <= self.early_exit_threshold <= 1:
            raise ValueError("early_exit_threshold must be in [0, 1]")
        if self.hidden_act != "silu":
            raise ValueError("Only Ouro's silu activation is supported")
        if self.rope_scaling is not None:
            raise ValueError("RoPE scaling is not supported")
        if self.use_sliding_window or self.sliding_window is not None:
            raise ValueError("Sliding-window attention is not supported")
        if self.layer_types is not None and (
            len(self.layer_types) != self.num_hidden_layers
            or any(layer != "full_attention" for layer in self.layer_types)
        ):
            raise ValueError("Every Ouro layer must use full_attention")
        if self.tie_word_embeddings:
            raise ValueError("Tied embeddings are not supported by the Ouro-1.4B loader")
        if self.attention_dropout != 0:
            raise ValueError("Inference requires attention_dropout=0")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ValueError(f"{name} must be a vocabulary index or None")

    @classmethod
    def tiny(cls, **overrides: Any) -> "OuroConfig":
        """Small randomly initialized architecture for CPU examples and correctness tests."""
        values = dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OuroConfig":
        if values.get("model_type", "ouro") != "ouro":
            raise ValueError("Only model_type='ouro' is supported")
        if values.get("architectures", ["OuroForCausalLM"]) != ["OuroForCausalLM"]:
            raise ValueError("Only the OuroForCausalLM architecture is supported")
        names = {field.name for field in fields(cls)}
        metadata = {
            "architectures",
            "auto_map",
            "model_type",
            "torch_dtype",
            "dtype",
            "transformers_version",
            "max_window_layers",
            "use_cache",
            "_name_or_path",
        }
        unknown = values.keys() - names - metadata
        if unknown:
            raise ValueError(f"Unsupported Ouro configuration fields: {sorted(unknown)}")
        config_values = {key: value for key, value in values.items() if key in names}
        if "head_dim" not in config_values and "hidden_size" in config_values:
            heads = config_values.get("num_attention_heads", cls.num_attention_heads)
            if config_values["hidden_size"] % heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            config_values["head_dim"] = config_values["hidden_size"] // heads
        if config_values.get("num_key_value_heads", 1) is None:
            config_values["num_key_value_heads"] = config_values.get(
                "num_attention_heads", cls.num_attention_heads
            )
        return cls(**config_values)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "ouro", "architectures": ["OuroForCausalLM"], **asdict(self)}
