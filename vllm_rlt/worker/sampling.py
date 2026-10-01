"""Shared distribution construction and exact speculative rejection sampling."""

import torch


def probabilities(logits, params):
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


def generator_for(request, device):
    if request.generator is None:
        request.generator = torch.Generator(device=device)
        if request.sampling_params.seed is None:
            request.generator.seed()
        else:
            request.generator.manual_seed(request.sampling_params.seed)
    return request.generator


def selected_logprob(logits, token, params):
    if params.logprobs_mode == "processed" and params.temperature > 0:
        return probabilities(logits, params)[token].log()
    # Integer speculative tokens otherwise retain a view of the vocabulary row.
    return logits.float().log_softmax(-1)[token].clone()


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
