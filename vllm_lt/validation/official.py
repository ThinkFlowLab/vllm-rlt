"""Reviewed official Ouro arithmetic for fixed-four-loop Q1 comparisons only.

No checkpoint or remote-code loading occurs here. Callers provide a frozen
configuration and immutable, already-loaded parameter tensors. Adaptive
LAST-EXITED histories require the separate serial oracle.
"""

import hashlib
import importlib.metadata
import importlib.util
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

OFFICIAL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
OFFICIAL_TRANSFORMERS_VERSION = "4.55.0"
OFFICIAL_SOURCE_SHA256 = {
    "modeling_ouro.py": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
    "configuration_ouro.py": "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb",
}
_SOURCE_DIR = Path(__file__).with_name("reference_code")


def official_provenance() -> dict[str, Any]:
    """Verify published source bytes and record dependencies without device queries."""
    for filename, expected in OFFICIAL_SOURCE_SHA256.items():
        observed = hashlib.sha256((_SOURCE_DIR / filename).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Official Ouro source hash mismatch: {filename}")
    dependencies = {}
    for name in (
        "transformers",
        "torch",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "kernels",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    kernels_spec = importlib.util.find_spec("kernels")
    return {
        "model_id": "ByteDance/Ouro-1.4B",
        "revision": OFFICIAL_REVISION,
        "source_sha256": dict(OFFICIAL_SOURCE_SHA256),
        "dependencies": dependencies,
        "required_transformers": OFFICIAL_TRANSFORMERS_VERSION,
        "optional_kernels_present": kernels_spec is not None,
        "attention_implementation": "eager",
        "total_ut_steps": 4,
        "exit_at_step": 3,
        "use_cache": False,
        "weight_storage": "shared immutable parameter mapping",
    }


def _official_classes():
    provenance = official_provenance()
    actual = provenance["dependencies"]["transformers"]
    if actual != OFFICIAL_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Official Ouro requires transformers=={OFFICIAL_TRANSFORMERS_VERSION}; "
            f"found {actual!r}. Use the prepared reference environment."
        )
    if provenance["optional_kernels_present"]:
        raise RuntimeError(
            "Official Ouro requires optional 'kernels' to be absent; "
            "Transformers 4.55.0's integration is incompatible with kernels 0.16.0"
        )
    # Import only after verifying source bytes and the decorator dependency.
    from .reference_code.configuration_ouro import OuroConfig
    from .reference_code.modeling_ouro import OuroForCausalLM, OuroRMSNorm, OuroRotaryEmbedding

    forward = OuroRMSNorm.forward
    if (
        Path(forward.__code__.co_filename).resolve() != (_SOURCE_DIR / "modeling_ouro.py").resolve()
        or forward.__module__ != OuroRMSNorm.__module__
    ):
        raise RuntimeError("Official RMSNorm forward has been replaced")
    return OuroConfig, OuroForCausalLM, OuroRotaryEmbedding, OuroRMSNorm, forward


def _verify_norms(model, norm_type, forward):
    norms = [model.model.norm]
    for layer in model.model.layers:
        norms.extend(
            getattr(layer, name)
            for name in (
                "input_layernorm",
                "input_layernorm_2",
                "post_attention_layernorm",
                "post_attention_layernorm_2",
            )
        )
    if any(
        type(norm) is not norm_type or getattr(norm.forward, "__func__", None) is not forward
        for norm in norms
    ):
        raise RuntimeError("Official RMSNorm forward has been replaced")


def _initialize_official(
    config: Mapping[str, Any],
    weights: Mapping[str, torch.Tensor],
    *,
    use_cache: bool = False,
    allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16),
):
    """Construct the pinned model with shared weights; callers select cache policy."""
    official_config, official_model, official_rotary, norm_type, norm_forward = _official_classes()
    values = deepcopy(dict(config))
    if values.get("total_ut_steps", 4) != 4:
        raise ValueError("The official reference supports exactly four loops")
    if values.get("tie_word_embeddings", False):
        raise ValueError("The official reference requires untied Ouro-1.4B weights")
    if not weights:
        raise ValueError("An already-loaded parameter mapping is required")
    first = next(iter(weights.values()))
    if not isinstance(first, torch.Tensor):
        raise ValueError("Weights must be tensors")
    device, dtype = first.device, first.dtype
    if device.type == "meta" or dtype not in allowed_dtypes:
        raise ValueError("Weights must be materialized tensors with a supported reference dtype")
    if any(
        not isinstance(value, torch.Tensor) or value.device != device or value.dtype != dtype
        for value in weights.values()
    ):
        raise ValueError("All official weights must share one device and dtype")
    values["use_cache"] = use_cache
    values["attn_implementation"] = "eager"
    resolved_config = official_config(**values)
    with torch.device("meta"):
        model = official_model(resolved_config)
    _verify_norms(model, norm_type, norm_forward)
    # detach prevents load_state_dict(assign=True) from modifying a caller's
    # Parameter requires_grad flag while retaining identical tensor storage.
    model.load_state_dict(
        {name: value.detach() for name, value in weights.items()}, strict=True, assign=True
    )
    # This nonpersistent buffer is absent from the parameter mapping. Rebuild
    # it with the unchanged official constructor, never Module.to(bfloat16).
    model.model.rotary_emb = official_rotary(resolved_config, device=device)
    model.requires_grad_(False)
    model.eval()
    return model, resolved_config, device, dtype, norm_type, norm_forward


class OfficialOuroReference:
    """Full-sequence, causal fixed-depth reference sharing caller-owned weights.

    ``predict(prompt, eight_inputs)`` returns nine genuine prediction logits at
    the final prompt position and the eight supplied continuation positions.
    Provided weights must remain alive and unchanged through the comparison.
    The adapter never changes their values, storage, or requires-grad flags.
    """

    def __init__(self, config: Mapping[str, Any], weights: Mapping[str, torch.Tensor]):
        (
            self.model,
            self.config,
            self.device,
            self.dtype,
            self._norm_type,
            self._norm_forward,
        ) = _initialize_official(config, weights)

    @torch.inference_mode()
    def predict(
        self, prompt_token_ids: Sequence[int], continuation_token_ids: Sequence[int]
    ) -> torch.Tensor:
        """Return unbatched ``[1 + continuation_count, vocab_size]`` logits."""
        if self.model is None:
            raise RuntimeError("Official reference is closed")
        _verify_norms(self.model, self._norm_type, self._norm_forward)
        prompt, continuation = list(prompt_token_ids), list(continuation_token_ids)
        if not prompt:
            raise ValueError("The prompt must contain at least one token")
        token_ids = prompt + continuation
        if any(
            type(token) is not int or not 0 <= token < self.config.vocab_size for token in token_ids
        ):
            raise ValueError("Token IDs must be integers inside the model vocabulary")
        if len(token_ids) > self.config.max_position_embeddings:
            raise ValueError("The teacher-forced input exceeds the configured context")
        inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(token_ids), dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                use_cache=False,
                exit_at_step=3,
                logits_to_keep=0,
                return_dict=True,
            )
        # Clone the selected rows so retaining nine predictions does not retain
        # the full prompt's vocabulary-sized logits allocation.
        return output.logits[0, len(prompt) - 1 :].clone()

    def close(self) -> None:
        """Release adapter references without altering caller-owned weights."""
        self.model = None
