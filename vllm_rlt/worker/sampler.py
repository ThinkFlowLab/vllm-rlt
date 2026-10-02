"""Greedy, top-k and top-p sampling for a single logits row.

The arithmetic is a verbatim move out of ``ModelRunner._sample_tensor`` so the
sampling contract can be pinned by unit tests without a runner. ``probabilities``
and ``draw`` were unified here from ``worker/sampling.py``, so the CODA and
speculative paths keep one copy of the distribution construction.

``Sampler`` itself holds no state: the generator is created on the distribution's
device and stored by the caller in an ``RngRegistry``, which the engine owns and
shares between its runners.
"""

import torch

from vllm_rlt.sampling_params import SamplingParams


class Sampler:
    """Greedy / top-k / top-p sampling for one logits row."""

    @staticmethod
    def probabilities(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Return the temperature/top-k/top-p distribution for one logits row.

        ``params`` is assumed to be validated by ``SamplingParams.__post_init__``.
        Greedy decoding has no temperature-scaled distribution, so it is a caller
        error rather than a silent argmax.
        """
        if params.temperature == 0:
            raise ValueError("greedy decoding has no temperature-scaled distribution")
        logits = logits.float() / params.temperature
        if params.top_k > 0:
            threshold = logits.topk(min(params.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if params.top_p < 1:
            sorted_logits, indices = logits.sort(descending=True)
            remove = sorted_logits.softmax(-1).cumsum(-1) > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
        return logits.softmax(-1)

    @staticmethod
    def draw(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Draw one token from a normalized distribution as a 0-dim long tensor."""
        return torch.multinomial(probs, 1, generator=generator).squeeze(0)

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Generator | None]:
        """Return a 0-dim long device tensor and the possibly created generator.

        No host synchronization happens here: the returned tensor stays on the
        logits device so the async CODA path can feed it to the next prelude.
        """
        if params.temperature == 0:
            return logits.argmax(), generator
        probs = self.probabilities(logits, params)
        if generator is None:
            # multinomial runs on the distribution's device, so the generator has
            # to be created there, not on whichever device the model was loaded on.
            generator = new_generator(logits.device, params)
        return (self.draw(probs, generator), generator)


def new_generator(device: torch.device, params: SamplingParams) -> torch.Generator:
    """Create a request's first sampling generator from its validated seed."""
    return torch.Generator(device=device).manual_seed(params.seed)


class RngRegistry:
    """Per-request sampling generators for one engine.

    The engine owns one registry and injects the same instance into every runner,
    so the CODA and speculative paths share a request's RNG slot. Greedy requests
    never get a slot, suspension keeps one, and termination drops it so a request
    ID that is reused later starts from a fresh stream.
    """

    def __init__(self):
        self.generators: dict[str, torch.Generator] = {}

    def peek(self, request_id: str) -> torch.Generator | None:
        """Return the request's generator, or None before its first random draw."""
        return self.generators.get(request_id)

    def store(self, request_id: str, generator: torch.Generator | None) -> None:
        """Record a generator the sampler created; None keeps the slot absent."""
        if generator is not None:
            self.generators[request_id] = generator

    def acquire(self, request_id: str, params: SamplingParams, device) -> torch.Generator:
        """Return the request's generator, creating it on ``device`` if needed."""
        generator = self.generators.get(request_id)
        if generator is None:
            generator = new_generator(device, params)
            self.generators[request_id] = generator
        return generator

    def drop(self, request_id: str) -> None:
        """Forget the request's generator; a suspended request must not reach here."""
        self.generators.pop(request_id, None)
