"""Greedy, top-k and top-p sampling for a single logits row.

The arithmetic is a verbatim move out of ``ModelRunner._sample_tensor`` so the
sampling contract can be pinned by unit tests without a runner. This class owns
no per-request state: the caller stores and passes in the RNG generator.
"""

import torch

from vllm_rlt.sampling_params import SamplingParams


class Sampler:
    """Greedy / top-k / top-p sampling for one logits row."""

    def __init__(self, device: torch.device):
        self.device = device

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Generator | None]:
        """Return a 0-dim long device tensor and the possibly created generator.

        ``params`` is assumed to be validated by ``SamplingParams.__post_init__``.
        No host synchronization happens here: the returned tensor stays on the
        logits device so the async CODA path can feed it to the next prelude.
        """
        if params.temperature == 0:
            return logits.argmax(), generator
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
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(0), generator
