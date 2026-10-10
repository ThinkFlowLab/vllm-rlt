# SPDX-License-Identifier: Apache-2.0
"""Model exports and unified AutoModelForCausalLM factory."""

import json
import sys
from pathlib import Path
from typing import Any

import torch

from .huginn import HuginnConfig, HuginnForCausalLM
from .nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from .ouro import OuroConfig, OuroForCausalLM

MODEL_MAPPING = {
    "ouro": OuroForCausalLM,
    "nanbeige": NanbeigeForCausalLM,
    "huginn_raven": HuginnForCausalLM,
}


def resolve_model_source(path_or_repo: str | Path) -> str:
    """Expand explicit local paths; preserve caller-provided repository IDs."""
    folder = Path(path_or_repo).expanduser()
    if folder.is_dir():
        return str(folder.resolve())
    return str(path_or_repo)


def resolve_local_model_path(path_or_repo: str | Path, revision=None) -> Path | None:
    """Find an explicit directory or a complete, revision-specific HF snapshot."""
    from huggingface_hub import try_to_load_from_cache

    source = resolve_model_source(path_or_repo)
    folder = Path(source)
    if folder.is_dir():
        return folder
    cached = try_to_load_from_cache(source, "config.json", revision=revision)
    if not isinstance(cached, str):
        return None
    folder = Path(cached).parent
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if names and all((folder / name).is_file() for name in names):
            return folder
    elif (folder / "model.safetensors").is_file():
        return folder
    return None


def _confirm_download(source):
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"Download requires approval for '{source}'. Run interactively and enter y/yes, "
            "or provide a local checkpoint."
        )
    try:
        answer = input(f"Download '{source}' from HuggingFace? (y/yes): ")
    except EOFError as error:
        raise RuntimeError("Download was not approved.") from error
    if answer.strip().lower() not in ("y", "yes"):
        raise RuntimeError("Download was not approved.")


def resolve_model_config(path_or_repo, *, revision=None):
    """Use a local checkpoint, or confirm interactively before downloading it."""
    from huggingface_hub import hf_hub_download, snapshot_download

    source = resolve_model_source(path_or_repo)
    local = resolve_local_model_path(source, revision=revision)
    if local is not None:
        config_path = local / "config.json"
    else:
        _confirm_download(source)
        config_path = Path(hf_hub_download(source, "config.json", revision=revision))
    config = json.loads(config_path.read_text())
    model_type = config.get("model_type", "").lower()
    if model_type not in MODEL_MAPPING:
        raise ValueError(f"Unsupported model_type {model_type!r} for {path_or_repo}")
    # Use the same resolved snapshot for weights and the default tokenizer.
    if config_path.parent.parent.name == "snapshots":
        revision = config_path.parent.name
    if local is None:
        local = Path(
            snapshot_download(
                repo_id=source,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "tokenizer.tiktoken",
                    "vocab.txt",
                    "merges.txt",
                ],
            )
        )
    return str(local), revision, config


def load_tokenizer(source, *, revision=None):
    """Load cached tokenizer assets, asking before any missing Hub download."""
    from transformers import AutoTokenizer

    source = resolve_model_source(source)
    try:
        return AutoTokenizer.from_pretrained(
            source, revision=revision, trust_remote_code=False, local_files_only=True
        )
    except OSError:
        if Path(source).is_dir():
            raise
    _confirm_download(source)
    return AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)


class AutoModelForCausalLM:
    """Factory for loading recurrent causal language models by configuration inspection."""

    @classmethod
    def _dispatch_from_config(
        cls,
        config_data: dict[str, Any],
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_type = config_data.get("model_type", "").lower()
        model_cls = MODEL_MAPPING.get(model_type)
        if model_cls is None:
            supported = sorted(MODEL_MAPPING.keys())
            raise ValueError(
                f"Unsupported model_type {model_type!r} for {path_or_repo}. "
                f"Supported types: {supported}"
            )
        return model_cls.from_pretrained(
            path_or_repo, revision=revision, device=device, dtype=dtype
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        source, resolved_revision, config = resolve_model_config(path_or_repo, revision=revision)
        return cls._dispatch_from_config(
            config, source, revision=resolved_revision, device=device, dtype=dtype
        )


__all__ = [
    "OuroConfig",
    "OuroForCausalLM",
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "HuginnConfig",
    "HuginnForCausalLM",
    "AutoModelForCausalLM",
    "resolve_local_model_path",
    "resolve_model_source",
    "resolve_model_config",
    "load_tokenizer",
]
