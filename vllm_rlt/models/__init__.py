import json
from pathlib import Path

import torch

from .config import OURO_MODEL_ID, OURO_REVISION, OuroConfig
from .nanbeige import NANBEIGE_MODEL_ID, NanbeigeConfig, NanbeigeForCausalLM
from .ouro import OuroForCausalLM


class AutoModelForCausalLM:
    """Factory for loading recurrent causal language models by configuration inspection."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        folder = Path(path_or_repo).expanduser()
        repo_str = str(path_or_repo).lower()

        if folder.is_dir() and (folder / "config.json").is_file():
            config_data = json.loads((folder / "config.json").read_text())
            model_type = config_data.get("model_type", "").lower()
            if model_type == "nanbeige":
                return NanbeigeForCausalLM.from_pretrained(
                    path_or_repo, revision=revision, device=device, dtype=dtype
                )
            if model_type == "ouro":
                return OuroForCausalLM.from_pretrained(
                    path_or_repo, revision=revision, device=device, dtype=dtype
                )
            raise ValueError(f"Unsupported model_type {model_type!r}")

        if "nanbeige" in repo_str:
            return NanbeigeForCausalLM.from_pretrained(
                path_or_repo, revision=revision, device=device, dtype=dtype
            )

        if "ouro" in repo_str:
            return OuroForCausalLM.from_pretrained(
                path_or_repo, revision=revision, device=device, dtype=dtype
            )

        # Try downloading config.json from Hub; fall back to OuroForCausalLM for tests or mocks
        try:
            from huggingface_hub import hf_hub_download

            config_file = Path(
                hf_hub_download(
                    repo_id=str(path_or_repo), filename="config.json", revision=revision
                )
            )
            config_data = json.loads(config_file.read_text())
            model_type = config_data.get("model_type", "").lower()
            if model_type == "nanbeige":
                return NanbeigeForCausalLM.from_pretrained(
                    path_or_repo, revision=revision, device=device, dtype=dtype
                )
        except Exception:
            pass

        return OuroForCausalLM.from_pretrained(
            path_or_repo, revision=revision, device=device, dtype=dtype
        )


__all__ = [
    "OURO_MODEL_ID",
    "OURO_REVISION",
    "OuroConfig",
    "OuroForCausalLM",
    "NANBEIGE_MODEL_ID",
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "AutoModelForCausalLM",
]
