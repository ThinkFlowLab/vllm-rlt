"""Shared distribution construction and exact speculative rejection sampling."""

import torch

LOGPROBS_MODES = ("raw_logprobs", "processed_logprobs")


def check_logprobs_mode(mode):
    if mode not in LOGPROBS_MODES:
        raise ValueError(
            f"logprobs_mode must be one of {LOGPROBS_MODES}, got {mode!r}; "
            "raw_logits and processed_logits are not supported"
        )


def processed_logits(logits, params):
    """FP32 logits that sampling draws from: temperature, then top-k, then top-p.

    Excluded tokens are -inf. Requires temperature > 0. Sampler, speculative
    sampling and processed logprobs share this exact arithmetic and op order.
    """
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
    return logits


def probabilities(logits, params):
    if params.temperature == 0:
        raise ValueError("greedy decoding has no temperature-scaled distribution")
    return processed_logits(logits, params).softmax(-1)


def raw_logprobs(logits, token_ids):
    """FP32 [rows] log-probabilities of ``token_ids`` [rows] under ``logits`` [rows, vocab]."""
    logprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
    return logprobs.gather(-1, token_ids[:, None]).squeeze(-1)


def token_logprob(logits, params, token_id, mode):
    """0-dim FP32 log-probability of ``token_id`` (0-dim long tensor) for one logits row.

    raw_logprobs uses the model distribution. processed_logprobs uses the
    distribution actually sampled from; greedy rows have no temperature (vLLM).
    Consumes no RNG and keeps the result on the logits device.
    """
    if mode == "processed_logprobs" and params.temperature != 0:
        logits = processed_logits(logits, params)
    return raw_logprobs(logits[None], token_id.view(1))[0]


def check_seed(params):
    """Reject an unresolved seed; engines replace seed=None before creating requests."""
    if params.seed is None:
        raise ValueError("seed=None must be resolved by the engine before sampling")


def generator_for(request, device):
    check_seed(request.sampling_params)
    if request.generator is None:
        request.generator = torch.Generator(device=device).manual_seed(request.sampling_params.seed)
    return request.generator


def draw(probs, generator):
    return torch.multinomial(probs, 1, generator=generator).squeeze(0)


def sample_logits(logits, request):
    if request.sampling_params.temperature == 0:
        return logits.argmax()
    return draw(
        probabilities(logits, request.sampling_params), generator_for(request, logits.device)
    )


def rejection_sample(candidate, target, proposal, generator):
    """Return (token, accepted) for normalized, actually sampled p and q.

    Comparing u*q to p avoids division by tiny q. A zero residual after a real
    rejection is a numerical error, never a reason to silently sample from p.
    """
    mass = proposal[candidate]
    if not bool(mass > 0):
        raise ValueError("candidate has zero proposal probability")
    uniform = torch.rand((), device=target.device, generator=generator)
    if bool(uniform * mass < target[candidate]):
        return candidate, True
    residual = (target - proposal).clamp_min(0)
    total = residual.sum()
    if not bool(torch.isfinite(total) & (total > 0)):
        raise RuntimeError("invalid speculative residual probability mass")
    return int(draw(residual / total, generator).item()), False
