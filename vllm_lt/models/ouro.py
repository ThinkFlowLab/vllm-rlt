# SPDX-License-Identifier: Apache-2.0
# Architecture adapted from ByteDance/Ouro-1.4B modeling_ouro.py, revision
# 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 (Apache-2.0).
# Original architecture includes code Copyright 2024 The Qwen team, Alibaba
# Group and the HuggingFace Inc. team. All rights reserved.
"""Native Ouro execution split into embedding, recurrent core, and LM head.

The scheduler owns loop depth and halting. Each row can belong to a different
request or loop depth; the cache supplies the corresponding causal KV history.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .config import OURO_MODEL_ID, OURO_REVISION, OuroConfig

if TYPE_CHECKING:
    from vllm_lt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch


class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(hidden.dtype)


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, config: OuroConfig) -> None:
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
        # Module.to(dtype) otherwise rounds these constants before forward's
        # float32 cast, changing long-context rotary angles irreversibly.
        self.inv_freq = self.frequencies(device=self.inv_freq.device)
        return self

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The reference forces fp32 here, including when the projections use bf16.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(hidden.dtype).unsqueeze(1), angles.sin().to(hidden.dtype).unsqueeze(
            1
        )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class OuroAttention(nn.Module):
    def __init__(self, config: OuroConfig, layer_idx: int) -> None:
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
        shape = (hidden.shape[0], -1, self.config.head_dim)
        q = self.q_proj(hidden).view(shape)
        k = self.k_proj(hidden).view(shape)
        v = self.v_proj(hidden).view(shape)
        cos, sin = position_embeddings
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        cache._write_prepared(self.layer_idx, batch, k, v)
        output = cache._attend_prepared(self.layer_idx, batch, q)
        return self.o_proj(output.reshape(hidden.shape[0], -1))


class OuroMLP(nn.Module):
    def __init__(self, config: OuroConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class OuroDecoderLayer(nn.Module):
    def __init__(self, config: OuroConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = OuroAttention(config, layer_idx)
        self.mlp = OuroMLP(config)
        self.input_layernorm = OuroRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = OuroRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm_2 = OuroRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch: "_PreparedKVBatch",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        attention = self.self_attn(self.input_layernorm(hidden), position_embeddings, batch, cache)
        hidden = hidden + self.input_layernorm_2(attention)
        return hidden + self.post_attention_layernorm_2(
            self.mlp(self.post_attention_layernorm(hidden))
        )


class OuroModel(nn.Module):
    def __init__(self, config: OuroConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            OuroDecoderLayer(config, layer) for layer in range(config.num_hidden_layers)
        )
        self.norm = OuroRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = OuroRotaryEmbedding(config)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1, bias=True)


class OuroForCausalLM(nn.Module):
    """Inference-only native Ouro-1.4B with checkpoint-compatible parameter names."""

    def __init__(self, config: OuroConfig) -> None:
        super().__init__()
        self.config = config
        self.model = OuroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.requires_grad_(False)
        self.eval()

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def prelude(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError("prelude expects packed token_ids with shape [N]")
        return self.model.embed_tokens(token_ids)

    def recurrent(
        self,
        hidden: torch.Tensor,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        cache: "KVCacheManager",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one full shared core; depths are zero-based cache namespaces."""
        if hidden.ndim != 2 or hidden.shape[1] != self.config.hidden_size:
            raise ValueError("recurrent expects hidden with shape [N, hidden_size]")
        count = hidden.shape[0]
        if count == 0 or not (len(request_ids) == len(depths) == len(positions) == count):
            raise ValueError("recurrent requires matching, nonempty packed metadata")
        # Internal model/cache traversal contract: descriptor ownership and
        # allocation identity are checked again by every prepared layer call.
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self._recurrent_prepared(hidden, batch, cache)

    def _recurrent_prepared(
        self, hidden: torch.Tensor, batch: "_PreparedKVBatch", cache: "KVCacheManager"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager borrowed traversal with explicit inactive physical rows.

        Host ownership and layer bookkeeping remain outside any graph contract.
        Padding changes the dense row shape, so active results need numerical
        qualification even though rows do not interact mathematically.
        """
        cache._require_live_batch(batch)
        if hidden.ndim != 2 or hidden.shape != (batch.row_count, self.config.hidden_size):
            raise ValueError(
                "prepared recurrent expects hidden with shape [row_count, hidden_size]"
            )
        if not batch.rows:
            return torch.zeros_like(hidden), hidden.new_zeros(batch.row_count)
        inactive = None
        if len(batch.rows) != batch.row_count:
            inactive = ~batch.active
            # Multiplication would preserve poison NaNs in inactive inputs.
            hidden = hidden.masked_fill(inactive[:, None], 0.0)
        position_embeddings = self.model.rotary_emb(hidden, batch.position_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, position_embeddings, batch, cache)
        # Norm is inside the recurrence in Ouro; this normalized state is the
        # next loop's input as well as the gate and LM head input.
        hidden = self.model.norm(hidden)
        gate_logits = self.model.early_exit_gate(hidden).squeeze(-1)
        if inactive is not None:
            hidden = hidden.masked_fill(inactive[:, None], 0.0)
            # The gate has a bias; zero hidden alone does not define padding.
            gate_logits = gate_logits.masked_fill(inactive, 0.0)
        return hidden, gate_logits

    def coda(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path = OURO_MODEL_ID,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "OuroForCausalLM":
        """Stream strictly checked safetensors into a meta model; never execute Hub code.

        The official repository defaults to a frozen revision. Local directories
        may contain a single model.safetensors or an indexed sharded checkpoint.
        """
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
                    revision=revision or (OURO_REVISION if repo == OURO_MODEL_ID else None),
                    allow_patterns=["config.json", "*.safetensors", "model.safetensors.index.json"],
                )
            )
        config = OuroConfig.from_dict(json.loads((folder / "config.json").read_text()))
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
        # Validate every key and shape before allocating model weight memory.
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
            raise ValueError("Checkpoint index does not match Ouro parameter names")
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
