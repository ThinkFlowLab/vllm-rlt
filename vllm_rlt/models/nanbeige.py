# SPDX-License-Identifier: Apache-2.0
"""Native Nanbeige execution split into embedding, recurrent core, and LM head.

Supports fixed-loop recurrent causal LM architectures such as Nanbeige/Nanbeige4.2-3B.
The scheduler owns loop depth and token progress. Each row can belong to a different
request or loop depth; the cache supplies the corresponding causal KV history.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch

NANBEIGE_MODEL_ID = "Nanbeige/Nanbeige4.2-3B"


@dataclass(frozen=True)
class NanbeigeConfig:
    vocab_size: int = 166144
    hidden_size: int = 3072
    intermediate_size: int = 10752
    num_hidden_layers: int = 22
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_theta: float = 70_000_000.0
    num_loops: int = 2
    total_ut_steps: int = 2
    early_exit_threshold: float = 1.0
    bos_token_id: int | None = 166100
    eos_token_id: int | None = 166101
    pad_token_id: int | None = 0
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    attention_bias: bool = False
    skip_loop_final_norm: bool = False
    rope_scaling: dict[str, Any] | None = None

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
            "num_loops",
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
            raise ValueError("Only silu hidden_act is supported")
        if self.rope_scaling is not None:
            raise ValueError("RoPE scaling is not supported")
        if self.tie_word_embeddings:
            raise ValueError("Tied embeddings are not supported")
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
    def tiny(cls, **overrides: Any) -> "NanbeigeConfig":
        """Small randomly initialized architecture for CPU tests."""
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
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NanbeigeConfig":
        if values.get("model_type", "nanbeige") != "nanbeige":
            raise ValueError("Only model_type='nanbeige' is supported")
        if values.get("architectures", ["NanbeigeForCausalLM"]) != ["NanbeigeForCausalLM"]:
            raise ValueError("Only the NanbeigeForCausalLM architecture is supported")
        names = {field.name for field in fields(cls)}
        config_values = {key: value for key, value in values.items() if key in names}
        if "num_loops" in values and "total_ut_steps" not in config_values:
            config_values["total_ut_steps"] = values["num_loops"]
        elif "total_ut_steps" in config_values and "num_loops" not in config_values:
            config_values["num_loops"] = config_values["total_ut_steps"]
        if "head_dim" not in config_values and "hidden_size" in config_values:
            heads = config_values.get("num_attention_heads", cls.num_attention_heads)
            config_values["head_dim"] = config_values.get(
                "kv_channels", config_values["hidden_size"] // heads
            )
        if config_values.get("num_key_value_heads") is None:
            config_values["num_key_value_heads"] = config_values.get(
                "num_attention_heads", cls.num_attention_heads
            )
        return cls(**config_values)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "nanbeige", "architectures": ["NanbeigeForCausalLM"], **asdict(self)}


class NanbeigeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(hidden.dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class NanbeigeRotaryEmbedding(nn.Module):
    def __init__(self, config: NanbeigeConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device: torch.device | str | None = None) -> torch.Tensor:
        return 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(0, self.config.head_dim, 2, dtype=torch.float32, device=device)
                / self.config.head_dim
            )
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self.frequencies(device=self.inv_freq.device)
        return self

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(hidden.dtype).unsqueeze(1), angles.sin().to(hidden.dtype).unsqueeze(
            1
        )


class NanbeigeAttention(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch: "_PreparedKVBatch",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        shape_q = (hidden.shape[0], self.config.num_attention_heads, self.config.head_dim)
        shape_kv = (hidden.shape[0], self.config.num_key_value_heads, self.config.head_dim)
        q = self.q_proj(hidden).view(shape_q)
        k = self.k_proj(hidden).view(shape_kv)
        v = self.v_proj(hidden).view(shape_kv)
        cos, sin = position_embeddings
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        cache._write_prepared(self.layer_idx, batch, k, v)
        output = cache._attend_prepared(self.layer_idx, batch, q)
        return self.o_proj(output.reshape(hidden.shape[0], -1))


class NanbeigeMLP(nn.Module):
    def __init__(self, config: NanbeigeConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class NanbeigeDecoderLayer(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = NanbeigeAttention(config, layer_idx)
        self.mlp = NanbeigeMLP(config)
        self.input_layernorm = NanbeigeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NanbeigeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch: "_PreparedKVBatch",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        attention = self.self_attn(hidden, position_embeddings, batch, cache)
        hidden = residual + attention

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        mlp = self.mlp(hidden)
        return residual + mlp


class NanbeigeModel(nn.Module):
    def __init__(self, config: NanbeigeConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            NanbeigeDecoderLayer(config, layer) for layer in range(config.num_hidden_layers)
        )
        self.norm = NanbeigeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = NanbeigeRotaryEmbedding(config)


class NanbeigeForCausalLM(nn.Module):
    """Nanbeige causal language model matching the vllm-rlt execution contract."""

    def __init__(self, config: NanbeigeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NanbeigeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def prelude(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(tokens)

    def recurrent(
        self,
        hidden: torch.Tensor,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int],
        cache: "KVCacheManager",
        *,
        compute_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden.ndim != 2 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("recurrent expects hidden with shape [N, hidden_size]")
        count = hidden.shape[0]
        if count == 0 or not (len(request_ids) == len(depths) == len(positions) == count):
            raise ValueError("recurrent requires matching, nonempty packed metadata")
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        """Run recurrent core with runner-owned metadata."""
        position_embeddings = self.model.rotary_emb(hidden, batch.position_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, position_embeddings, batch, cache)
        if not self.config.skip_loop_final_norm:
            hidden = self.model.norm(hidden)
        # Nanbeige is a fixed-loop recurrent model without an early-exit gate.
        # Returning a strongly negative logit ensures 0 exit probability until max_loops.
        gate = (
            torch.full((hidden.shape[0],), -1e4, device=hidden.device, dtype=hidden.dtype)
            if compute_gate
            else None
        )
        return hidden, gate

    def coda(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path = NANBEIGE_MODEL_ID,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "NanbeigeForCausalLM":
        """Stream strictly checked safetensors into a meta model; never execute Hub code."""
        from safetensors import safe_open

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from huggingface_hub import snapshot_download

            repo = str(path_or_repo)
            folder = Path(
                snapshot_download(
                    repo_id=repo,
                    revision=revision,
                    allow_patterns=["config.json", "*.safetensors", "model.safetensors.index.json"],
                )
            )
        config = NanbeigeConfig.from_dict(json.loads((folder / "config.json").read_text()))
        index_path = folder / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"] if index_path.is_file() else None
        if index is None:
            shards = [folder / "model.safetensors"]
        else:
            filenames = set(index.values())
            if any(
                Path(name).name != name or not name.endswith(".safetensors") for name in filenames
            ):
                raise ValueError("Checkpoint index must reference local safetensors basenames")
            shards = [folder / name for name in sorted(filenames)]
        if any(not shard.is_file() for shard in shards):
            raise FileNotFoundError(
                "A complete safetensors checkpoint is required (pickle .bin is unsupported)"
            )
        with torch.device("meta"):
            model = cls(config)
        expected = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        discovered: dict[str, Path] = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name in discovered:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    if name not in expected:
                        raise ValueError(f"Unexpected checkpoint tensor: {name}")
                    if tuple(checkpoint.get_slice(name).get_shape()) != expected[name]:
                        raise ValueError(f"Checkpoint shape mismatch for {name}")
                    if index is not None and index.get(name) != shard.name:
                        raise ValueError(f"Checkpoint index mismatch for {name}")
                    discovered[name] = shard
        missing = expected.keys() - discovered.keys()
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        if index is not None and set(index) != set(expected):
            raise ValueError("Checkpoint index does not match Nanbeige parameter names")
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    tensor = checkpoint.get_tensor(name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"Checkpoint tensor must be floating point: {name}")
                    module_path, parameter_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_path)
                    setattr(
                        module,
                        parameter_name,
                        nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False),
                    )
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
        return model.eval()
