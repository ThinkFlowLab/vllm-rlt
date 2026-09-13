"""Single-sequence cached adapter for unchanged, pinned official Ouro arithmetic.

Only the published cache's Transformers 4.55 interface is adapted. Sampling,
completion synchronization and detailed cache inspection belong to the host
driver. No checkpoint, remote code, device discovery or alternate kernels load.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

from .official import _initialize_official, _verify_norms


def _cache_class():
    # Called only after _initialize_official verifies dependencies and source.
    from .reference_code.modeling_ouro import UniversalTransformerCache

    class CompatibleUniversalTransformerCache(UniversalTransformerCache):
        """Writable published lists and the installed causal-mask interface."""

        @property
        def key_cache(self):
            return self._keys

        @key_cache.setter
        def key_cache(self, value):
            self._keys = value

        @property
        def value_cache(self):
            return self._values

        @value_cache.setter
        def value_cache(self, value):
            self._values = value

        def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
            # Full, unpadded attention; mask construction precedes every update.
            return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    return CompatibleUniversalTransformerCache


class OfficialOuroCachedReference:
    """Fixed-four FP32/BF16 generation with one prefill and single-token advances.

    Returned logits are unsampled ``[vocab]`` tensors. A successful last call
    awaits explicit host completion; it never forwards the final predicted ID.
    Failed calls poison the current request, including a partially updated cache.
    Reset/close require completion confirmation before discarding live state.
    """

    def __init__(self, config: Mapping[str, Any], weights: Mapping[str, torch.Tensor]):
        layers = config.get("num_hidden_layers", 24)
        if type(layers) is not int or layers <= 0:
            raise ValueError("num_hidden_layers must be a positive integer")
        layer_types = config.get("layer_types")
        if (
            config.get("use_sliding_window", False)
            or config.get("sliding_window") is not None
            or (
                layer_types is not None
                and (len(layer_types) != layers or any(t != "full_attention" for t in layer_types))
            )
        ):
            raise ValueError("Cached official reference requires full attention in every layer")
        (
            self.model,
            self.config,
            self.device,
            self.dtype,
            self._norm_type,
            self._norm_forward,
        ) = _initialize_official(config, weights, use_cache=True)
        self.slot_count = 4 * self.config.num_hidden_layers
        self._cache_type = _cache_class()
        self.cache = None
        self._clear_request()

    @property
    def cache_slots(self) -> int:
        """Current populated slot count; a constant-time cleanup observation."""
        return 0 if self.cache is None else len(self.cache.key_cache)

    def _clear_request(self) -> None:
        self.status = "ready"
        self.prompt_length = 0
        self.max_outputs = 0
        self.expected_position = 0
        self.output_count = 0
        self.forward_calls = 0
        self.calls: list[dict[str, int]] = []
        self.failure: dict[str, str] | None = None

    def _token(self, token: int) -> None:
        if type(token) is not int or not 0 <= token < self.config.vocab_size:
            raise ValueError("Token IDs must be integers inside the model vocabulary")

    def start(self, prompt: Sequence[int], max_outputs: int) -> torch.Tensor:
        if self.status != "ready":
            raise RuntimeError(f"Cannot start official request in state {self.status}")
        token_ids = list(prompt)
        if not token_ids:
            raise ValueError("The prompt must contain at least one token")
        for token in token_ids:
            self._token(token)
        if type(max_outputs) is not int or max_outputs <= 0:
            raise ValueError("max_outputs must be a positive integer")
        if len(token_ids) + max_outputs - 1 > self.config.max_position_embeddings:
            raise ValueError("The cached request exceeds the configured context")
        self.cache = self._cache_type(max_cache_size=self.slot_count)
        self.prompt_length, self.max_outputs = len(token_ids), max_outputs
        self.status = "active"
        return self._forward(token_ids)

    def advance(self, actual_previous_token: int) -> torch.Tensor:
        if self.status != "active":
            raise RuntimeError(f"Cannot advance official request in state {self.status}")
        self._token(actual_previous_token)
        return self._forward([actual_previous_token])

    @torch.inference_mode()
    def _forward(self, token_ids: list[int]) -> torch.Tensor:
        position, count = self.expected_position, len(token_ids)
        try:
            _verify_norms(self.model, self._norm_type, self._norm_forward)
            if type(self.cache) is not self._cache_type or self.cache.get_seq_length() != position:
                raise RuntimeError("Official cache identity or contiguous prefix changed")
            inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            positions = torch.arange(
                position, position + count, dtype=torch.long, device=self.device
            )
            mask = torch.ones((1, position + count), dtype=torch.long, device=self.device)
            self.forward_calls += 1
            with torch.autocast(device_type=self.device.type, enabled=False):
                output = self.model(
                    input_ids=inputs,
                    attention_mask=mask,
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    past_key_values=self.cache,
                    use_cache=True,
                    exit_at_step=3,
                    logits_to_keep=1,
                    use_weighted_exit=False,
                    return_dict=True,
                )
            if output.past_key_values is not self.cache:
                raise RuntimeError("Official forward replaced the supplied universal cache")
            if self.cache.get_seq_length() != position + count:
                raise RuntimeError("Official forward returned an unexpected cache length")
            if (
                output.logits.shape != (1, 1, self.config.vocab_size)
                or output.logits.dtype != self.dtype
                or output.logits.device != self.device
            ):
                raise RuntimeError("Official forward returned unexpected last-position logits")
            self.expected_position = position + count
            self.calls.append(
                {
                    "output_index": self.output_count,
                    "input_count": count,
                    "position": position,
                    "cache_length": self.expected_position,
                }
            )
            self.output_count += 1
            if self.output_count == self.max_outputs:
                self.status = "awaiting_completion"
            return output.logits[0, 0]
        except BaseException as error:
            self.status = "failed"
            self.failure = {"type": type(error).__name__, "message": str(error)[:512]}
            raise

    def complete(self, *, completion_confirmed: bool) -> None:
        """Confirm the driver's real final readback/synchronization, without one here."""
        if completion_confirmed is not True or self.status != "awaiting_completion":
            raise RuntimeError("Completion requires all outputs and confirmed device completion")
        self.status = "finished"

    def _may_discard(self, completion_confirmed: bool) -> None:
        if self.status not in ("ready", "finished", "closed") and completion_confirmed is not True:
            raise RuntimeError(
                "Discarding live/failed official state requires confirmed completion"
            )

    def reset(self, *, completion_confirmed: bool = False) -> None:
        if self.status == "closed":
            raise RuntimeError("Official reference is closed")
        self._may_discard(completion_confirmed)
        if self.cache is not None:
            self.cache.clear()
        self.cache = None
        self._clear_request()

    def close(self, *, completion_confirmed: bool = False) -> None:
        self._may_discard(completion_confirmed)
        if self.cache is not None:
            self.cache.clear()
        self.cache = None
        self.model = None
        self.status = "closed"

    def snapshot(
        self, *, inspect_cache: bool = False, check_finite: bool = False
    ) -> dict[str, Any]:
        """Detailed metadata/finite scans are opt-in and must run outside delivery."""
        if check_finite and not inspect_cache:
            raise ValueError("Finite checks require explicit cache inspection")
        result = {
            "status": self.status,
            "prompt_length": self.prompt_length,
            "max_outputs": self.max_outputs,
            "expected_position": self.expected_position,
            "output_count": self.output_count,
            "forward_calls": self.forward_calls,
            "cache_slots": self.cache_slots,
            "calls": deepcopy(self.calls),
            "failure": deepcopy(self.failure),
            "final_summary": None,
        }
        if inspect_cache:
            keys = [] if self.cache is None else self.cache.key_cache
            values = [] if self.cache is None else self.cache.value_cache
            if len(keys) != len(values) or any(t is None for t in keys + values):
                raise RuntimeError("Official cache has partially populated slots")
            tensors = keys + values
            if any(
                tensor.dtype != self.dtype or tensor.device != self.device for tensor in tensors
            ):
                raise RuntimeError("Official cache dtype or device changed")
            pointers = [tensor.untyped_storage().data_ptr() for tensor in tensors]
            key_lengths = [tensor.shape[2] for tensor in keys]
            value_lengths = [tensor.shape[2] for tensor in values]
            if key_lengths != value_lengths:
                raise RuntimeError("Official K/V prefix lengths disagree")
            result["final_summary"] = {
                "slot_count": len(keys),
                "max_cache_size": self.slot_count,
                "lengths": key_lengths,
                "key_shapes": [list(tensor.shape) for tensor in keys],
                "value_shapes": [list(tensor.shape) for tensor in values],
                "dtype": str(self.dtype),
                "device": str(self.device),
                "distinct_storage": len(set(pointers)) == len(pointers),
                "all_finite": (
                    all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
                    if check_finite
                    else None
                ),
            }
        return result
