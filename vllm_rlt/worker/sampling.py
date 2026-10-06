"""Shared distribution construction, logprobs and exact speculative rejection sampling."""

from collections import defaultdict

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


def truncates(params):
    """Whether this row is drawn from top-k/top-p truncated logits (see ``processed_logits``).

    Greedy rows never are: ``argmax`` ignores top-k and top-p.
    """
    return params.temperature != 0 and (params.top_k > 0 or params.top_p < 1)


def scaled_logprobs(logits, token_ids, temperature):
    """``raw_logprobs`` of ``logits / temperature``, divided as ``processed_logits`` divides."""
    return raw_logprobs(logits if temperature == 1 else logits.float() / temperature, token_ids)


def processed_logprobs(logits, params, token_ids, truncated):
    """FP32 [rows] processed logprobs of ``token_ids``, equal to per-row ``token_logprob``.

    ``logits`` [rows, vocab] are the CODA rows of ``params`` (one per row), at least one of
    which requests logprobs. ``truncated`` maps every requesting row that ``truncates`` to
    the 0-dim logprob ``Sampler.sample_with_logprob`` computed from the processed logits it
    drew from. Other requesting rows are grouped by temperature: greedy and T == 1 rows are
    scored by a raw ``log_softmax`` (``x / 1.0 == x``), each other temperature by
    ``log_softmax(logits.float() / T)`` with the Python scalar T, as ``processed_logits``
    divides. A group is scored in one pass over its row span ``[first, last]`` when that span
    is at most twice its rows, else row by row, so the scored rows never exceed twice the
    requesting rows, however many temperatures the batch mixes. The only group, holding at
    least half the rows, is scored over the whole batch and returned as is; otherwise one
    stack of 0-dim views assembles the rows. No host sync, per-row device write or per-row
    device metadata (spans are host-int slices). Rows that did not request logprobs hold
    arbitrary values.
    """
    groups = defaultdict(list)  # Temperature (greedy: 1) -> requesting untruncated rows.
    for row, row_params in enumerate(params):
        if row_params.logprobs is not None and row not in truncated:
            groups[1 if row_params.temperature == 0 else row_params.temperature].append(row)
    values = dict(truncated)
    for temperature, rows in groups.items():
        if not truncated and len(groups) == 1 and len(params) <= 2 * len(rows):
            return scaled_logprobs(logits, token_ids, temperature)  # One pass, no assembly.
        first, end = rows[0], rows[-1] + 1
        if end - first > 2 * len(rows):  # Sparse group: a span pass would mostly score others.
            for row in rows:
                one = slice(row, row + 1)
                values[row] = scaled_logprobs(logits[one], token_ids[one], temperature)[0]
            continue
        scored = scaled_logprobs(logits[first:end], token_ids[first:end], temperature)
        values.update((row, scored[row - first]) for row in rows)
    filler = next(iter(values.values()))
    return torch.stack([values.get(row, filler) for row in range(len(params))])


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
