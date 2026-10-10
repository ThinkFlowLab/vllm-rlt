"""Contract tests for sampling: the extracted Sampler, per-token logprobs and seeds.

The Sampler tests (greedy, top-k, top-p, seeded generators) pin the arithmetic
exactly as it behaved inside ``ModelRunner._sample_tensor``; they are not a
specification of what sampling should do in the future. The rest cover
``token_logprob`` in raw and processed modes, the batched ``processed_logprobs``
and ``Sampler.sample_with_logprob`` against it, ``generator_for``, and the
rejection of unresolved ``seed=None`` params before any sampling.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker import sampling
from vllm_rlt.worker.sampler import Sampler
from vllm_rlt.worker.sampling import (
    generator_for,
    probabilities,
    processed_logprobs,
    raw_logprobs,
    token_logprob,
    truncates,
)


def draw(sampler, logits, params, generator=None, count=1):
    """Sample ``count`` tokens, threading the generator like the runner does."""
    tokens = []
    for _ in range(count):
        token, generator = sampler.sample(logits, params, generator)
        tokens.append(token)
    return tokens, generator


def test_greedy_returns_argmax_and_never_creates_a_generator():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.tensor([0.1, 2.0, 1.0, -1.0])
    # top_k/top_p are deliberately set: the temperature == 0 short circuit runs
    # first, so greedy ignores them and stays RNG-free.
    params = SamplingParams(temperature=0.0, top_k=1, top_p=0.5)
    tokens, generator = draw(sampler, logits, params)
    token = tokens[0]
    assert token.item() == int(logits.argmax())
    assert token.dim() == 0
    assert token.dtype == torch.long
    assert generator is None


GREEDY_CASES = [
    torch.tensor([0.1, 2.0, 1.0, -1.0]),
    torch.tensor([-3.0, -3.5, -2.0, -9.0]),
    torch.tensor([5.0, 4.999, 4.998]),
]


@pytest.mark.parametrize("logits", GREEDY_CASES)
def test_top_k_one_is_equivalent_to_greedy(logits):
    sampler = Sampler(torch.device("cpu"))
    greedy_tokens, _ = draw(sampler, logits, SamplingParams(temperature=0.0))
    # top_k=1 makes the threshold the maximum, so masked_fill leaves exactly one
    # candidate and the result cannot depend on the RNG stream. Checking several
    # logits rows and seeds pins that mechanism, not one coincidental argmax.
    for seed in (0, 7, 12345):
        params = SamplingParams(temperature=1.0, top_k=1, seed=seed)
        top_k_tokens, generator = draw(sampler, logits, params)
        assert generator is not None
        assert top_k_tokens[0].item() == greedy_tokens[0].item()


def test_top_k_keeps_every_token_tied_at_the_threshold():
    sampler = Sampler(torch.device("cpu"))
    # top_k=2 selects a threshold of 3.0, which three tokens share. The mask is
    # ``logits < threshold``, so the candidate set is larger than k.
    logits = torch.tensor([3.0, 3.0, 3.0, 0.0])
    params = SamplingParams(temperature=1.0, top_k=2, seed=11)
    tokens, _ = draw(sampler, logits, params, count=200)
    observed = {token.item() for token in tokens}
    assert observed <= {0, 1, 2}
    assert len(observed) > 1


def test_top_p_keeps_at_least_one_token():
    sampler = Sampler(torch.device("cpu"))
    # A top_p far below the head probability would empty the candidate set; the
    # mask shift keeps the single most likely token instead.
    logits = torch.linspace(1.0, 0.0, 100)
    params = SamplingParams(temperature=1.0, top_p=0.005, seed=3)
    tokens, _ = draw(sampler, logits, params, count=20)
    assert {token.item() for token in tokens} == {0}


def test_top_p_one_does_not_truncate_the_support():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7])
    truncated = SamplingParams(temperature=1.0, top_p=0.5, seed=13)
    untruncated = SamplingParams(temperature=1.0, top_p=1.0, seed=13)
    truncated_tokens, _ = draw(sampler, logits, truncated, count=200)
    untruncated_tokens, _ = draw(sampler, logits, untruncated, count=200)
    assert {token.item() for token in truncated_tokens} <= {0, 1}
    # Index 3 is the least likely token, so it is reachable only while the
    # support is intact. This asserts reachability over a fixed 200-draw stream
    # (p(miss) ~ 0.79**200), not the exact draw sequence: the exact sequence
    # would couple the test to torch's multinomial implementation.
    assert 3 in {token.item() for token in untruncated_tokens}


def sequence(sampler, logits, params, count=32):
    tokens, _ = draw(sampler, logits, params, count=count)
    return [token.item() for token in tokens]


def test_same_seed_reproduces_the_same_sequence():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=42)
    assert sequence(sampler, logits, params) == sequence(sampler, logits, params)


def test_different_seeds_produce_different_sequences():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.linspace(1.0, -1.0, 64)
    base = dict(temperature=0.7, top_k=8, top_p=0.9)
    assert sequence(sampler, logits, SamplingParams(seed=1, **base)) != sequence(
        sampler, logits, SamplingParams(seed=2, **base)
    )


def test_generator_is_reused_and_advanced_across_calls():
    sampler = Sampler(torch.device("cpu"))
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    _, generator = draw(sampler, logits, params)
    assert generator is not None
    state = generator.get_state().clone()
    # Later calls must advance the same generator instead of reseeding it, which
    # is what makes a multi-token request reproducible.
    _, same_generator = draw(sampler, logits, params, generator=generator)
    assert same_generator is generator
    assert not torch.equal(generator.get_state(), state)


def test_unresolved_seed_is_rejected_before_any_sampling():
    params = SamplingParams(seed=None, temperature=1.0)
    for sampled in (params, SamplingParams(seed=None)):
        with pytest.raises(ValueError, match="seed=None"):
            Sampler(torch.device("cpu")).sample(torch.zeros(4), sampled, None)
    with pytest.raises(ValueError, match="seed=None"):
        generator_for(SimpleNamespace(generator=None, sampling_params=params), "cpu")


LOGPROB_LOGITS = torch.tensor([1.5, 0.2, -0.7, 2.1, 0.0, -3.0])


@pytest.mark.parametrize("token", range(len(LOGPROB_LOGITS)), ids=lambda t: f"token={t}")
def test_raw_token_logprob_is_the_model_log_softmax_for_any_token(token):
    for dtype in (torch.float32, torch.bfloat16):
        logits = LOGPROB_LOGITS.to(dtype)
        params = SamplingParams(temperature=0.6, top_k=2, top_p=0.5, seed=1)
        value = token_logprob(logits, params, torch.tensor(token), "raw_logprobs")
        assert value.dim() == 0 and value.dtype == torch.float32
        expected = torch.log_softmax(logits.float(), -1)[token]
        assert torch.equal(value, expected)


PROCESSED_PARAMS = [
    SamplingParams(temperature=0.6, seed=1),
    SamplingParams(temperature=1.3, top_k=3, seed=1),
    SamplingParams(temperature=0.9, top_p=0.7, seed=1),
    SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=1),
]
PROCESSED_IDS = ["temperature", "temperature-top_k", "temperature-top_p", "temperature-top_k-top_p"]


@pytest.mark.parametrize("params", PROCESSED_PARAMS, ids=PROCESSED_IDS)
def test_processed_token_logprob_scores_the_sampling_distribution(params):
    probs = probabilities(LOGPROB_LOGITS, params)
    for token in range(len(LOGPROB_LOGITS)):
        value = token_logprob(LOGPROB_LOGITS, params, torch.tensor(token), "processed_logprobs")
        if probs[token] == 0:
            assert value.item() == -math.inf  # Filtered tokens cannot be sampled.
        else:
            assert value.item() == pytest.approx(probs[token].log().item(), abs=1e-6)


def test_processed_token_logprob_edge_cases():
    argmax = torch.tensor(int(LOGPROB_LOGITS.argmax()))
    greedy = SamplingParams(temperature=0.0, top_k=1)
    # Greedy requests report the unscaled model distribution, as vLLM does.
    assert torch.equal(
        token_logprob(LOGPROB_LOGITS, greedy, argmax, "processed_logprobs"),
        token_logprob(LOGPROB_LOGITS, greedy, argmax, "raw_logprobs"),
    )
    # One token survives top_k=1, or top_p=0.6 after top_k=2: top-p sees the renormalized
    # top-2 mass (argmax 0.65), not the full distribution (0.53, which would keep two).
    for single in (dict(temperature=0.7, top_k=1), dict(temperature=1.0, top_k=2, top_p=0.6)):
        params = SamplingParams(seed=1, **single)
        assert token_logprob(LOGPROB_LOGITS, params, argmax, "processed_logprobs").item() == 0.0


def test_token_logprob_consumes_no_rng():
    sampler = Sampler(torch.device("cpu"))
    params = SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=42)
    token, generator = sampler.sample(LOGPROB_LOGITS, params, None)
    state, global_state = generator.get_state().clone(), torch.get_rng_state()
    for logprobs_mode in ("raw_logprobs", "processed_logprobs"):
        token_logprob(LOGPROB_LOGITS, params, token, logprobs_mode)
    assert torch.equal(generator.get_state(), state)
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize(
    "params", [SamplingParams(top_k=1), *PROCESSED_PARAMS], ids=["greedy", *PROCESSED_IDS]
)
def test_sample_with_logprob_draws_like_sample_and_scores_the_drawn_logits(params):
    sampler, plain, scored = Sampler(torch.device("cpu")), None, None
    for _ in range(4):
        token, plain = sampler.sample(LOGPROB_LOGITS, params, plain)
        same, scored, logprob = sampler.sample_with_logprob(LOGPROB_LOGITS, params, scored)
        assert torch.equal(same, token)
        expected = token_logprob(LOGPROB_LOGITS, params, token, "processed_logprobs")
        assert torch.equal(logprob, expected)
    # Same generator creation and RNG consumption as sample().
    assert (plain is None) == (scored is None) == (params.temperature == 0)
    assert plain is None or torch.equal(plain.get_state(), scored.get_state())


# One row per processed-logprob path: batched raw (greedy, even with top-k, and T == 1), batched
# per temperature, the sampler's top-k/top-p logits, and (last) a row without logprobs.
BATCH_PARAMS = [
    SamplingParams(temperature=0.0, top_k=3, logprobs=0),
    SamplingParams(temperature=1.0, logprobs=0, seed=1),
    SamplingParams(temperature=0.7, logprobs=0, seed=2),
    SamplingParams(temperature=1.3, logprobs=0, seed=3),
    SamplingParams(temperature=0.9, top_k=8, logprobs=0, seed=4),
    SamplingParams(temperature=1.0, top_p=0.8, logprobs=0, seed=5),
    SamplingParams(temperature=0.7, top_k=16, top_p=0.9, logprobs=0, seed=6),
    SamplingParams(temperature=0.8, seed=7),
]


ROWS = 48  # Rows laid out by a BATCH_LAYOUTS entry; a row without logprobs follows them.


def at(temperature):
    return SamplingParams(temperature=temperature, logprobs=0, seed=1)


# Layout -> (row -> params, log_softmax passes of processed_logprobs or None: only bounded).
BATCH_LAYOUTS = {
    "mixed": (lambda row: BATCH_PARAMS[row % 8], None),
    "t1": (lambda row: BATCH_PARAMS[1], 1),  # The only group: one pass over the whole batch.
    "t0.7": (lambda row: BATCH_PARAMS[2], 1),
    "top_k-top_p": (lambda row: BATCH_PARAMS[4 + row % 3], 0),
    # Many temperatures: one per row; rows r and ROWS - 1 - r sharing one (nested spans, the
    # outer ones scored row by row); two alternating (one pass per overlapping span).
    "distinct_t": (lambda row: at(0.5 + 0.01 * row), ROWS),
    "nested_t": (lambda row: at(0.5 + 0.01 * min(row, ROWS - 1 - row)), None),
    "alternating_t": (lambda row: at((0.7, 1.3)[row % 2]), 2),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", BATCH_LAYOUTS.values(), ids=BATCH_LAYOUTS.keys())
def test_batched_processed_logprobs_equal_per_row_token_logprob(monkeypatch, layout, dtype):
    params = [layout[0](row) for row in range(ROWS)] + [BATCH_PARAMS[-1]]
    logits = 4 * torch.randn(len(params), 256, generator=torch.Generator().manual_seed(0))
    logits, sampler, tokens, truncated = logits.to(dtype), Sampler(torch.device("cpu")), [], {}
    for row, row_params in enumerate(params):  # As ModelRunner._sample_with_logprobs.
        if row_params.logprobs is not None and truncates(row_params):
            token, _, truncated[row] = sampler.sample_with_logprob(logits[row], row_params, None)
        else:
            token, _ = sampler.sample(logits[row], row_params, None)
        tokens.append(token)
    token_ids, passes = torch.stack(tokens), []  # passes: rows scored by each log_softmax.

    def counted(logits, token_ids):
        passes.append(len(logits))
        return raw_logprobs(logits, token_ids)

    monkeypatch.setattr(sampling, "raw_logprobs", counted)
    values = processed_logprobs(logits, params, token_ids, truncated)
    monkeypatch.undo()
    # However many temperatures mix, at most twice the batched rows are scored (57d2308 scored
    # each requesting row once, in a pass of its own).
    assert sum(passes) <= 2 * (sum(p.logprobs is not None for p in params) - len(truncated))
    assert layout[1] is None or len(passes) == layout[1]
    assert values.shape == (len(params),) and values.dtype == torch.float32
    for row, row_params in enumerate(params):
        if row_params.logprobs is not None:
            expected = token_logprob(logits[row], row_params, token_ids[row], "processed_logprobs")
            assert torch.equal(values[row], expected)
