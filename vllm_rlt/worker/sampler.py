"""Greedy, top-k and top-p sampling for a single logits row.

The arithmetic is a verbatim move out of ``ModelRunner._sample_tensor`` so the
sampling contract can be pinned by unit tests without a runner. This class owns
no per-request state: the caller stores and passes in the RNG generator.
"""

import torch

from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampling import check_seed, processed_logits


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

        ``params`` is assumed to be validated by ``SamplingParams.__post_init__``
        and to carry an engine-resolved seed. No host synchronization happens
        here: the returned tensor stays on the logits device so the async CODA
        path can feed it to the next prelude.
        """
        check_seed(params)
        if params.temperature == 0:
            return logits.argmax(), generator
        logits = processed_logits(logits, params)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(0), generator
