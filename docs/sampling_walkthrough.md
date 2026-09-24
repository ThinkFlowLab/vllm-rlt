# Sampling Module Walkthrough

This note documents the M7 (Sampling) increment of RFC
[#32](https://github.com/hsliuustc0106/vllm-rlt/issues/32). It covers
[`worker/sampler.py`](../vllm_rlt/worker/sampler.py), the shared helpers in
[`worker/sampling.py`](../vllm_rlt/worker/sampling.py), the speculative call sites in
[`worker/speculative.py`](../vllm_rlt/worker/speculative.py) and the sampling call sites in
[`worker/model_runner.py`](../vllm_rlt/worker/model_runner.py),
[`request.py`](../vllm_rlt/request.py),
[`core/scheduler.py`](../vllm_rlt/core/scheduler.py),
[`engine/preemption.py`](../vllm_rlt/engine/preemption.py),
[`engine/llm_engine.py`](../vllm_rlt/engine/llm_engine.py) and
[`pd/worker.py`](../vllm_rlt/pd/worker.py).

M7 is split into three increments: `M7 1/3` extracts the algorithm and pins it with CPU
contract tests, `M7 2/3` collapses `worker/sampling.py` into `Sampler`, migrates RNG
ownership and splits the release path, and `M7 3/3` adds GPU validation. This document is
written during `M7 1/3`; sections that describe later increments say so explicitly.

## 1. Baseline and boundary

Pinned baseline: `upstream/main @ 3314c1b`, the merge of PR #44. PR #44 added
`worker/sampling.py` and routed `ModelRunner._sample_tensor` through
`sampling.sample_logits()`; this increment replaces that one-line delegate with `Sampler`
and leaves the speculative decoding path on `sampling.py`. Both modules therefore coexist
until `M7 2/3` (section 6).

### Call chain

```text
Scheduler.schedule() -> CODA batch
  -> ModelRunner.execute()                    # sync path
     -> _execute()
        -> model.coda(hidden)                 # lm_head logits rows
        -> _sample_tensor(row, request)       # model_runner.py:425
        -> result.cpu().tolist()              # the only host readback, in execute()
  -> ModelRunner.submit()                     # async path
     -> _execute() samples on the device
     -> routing.scatter(result, tokens=True)  # sampled IDs stay on the device
  -> SpeculativeRunner.execute()              # speculative path, sync eager only
     -> sampling.probabilities(logits, ...)   # worker/sampling.py:6
     -> sampling.draw() / rejection_sample()  # worker/sampling.py:28, :40
```

The async path requires the sampled value to remain a 0-dim device tensor: it is
`torch.stack`ed and scattered into the next prelude, so sampling must not call
`.item()`, `.cpu()`, `synchronize()` or add host work. The sync path's
`.cpu().tolist()` happens in `execute()`, outside the sampling function.

### In scope

| File | Role |
| --- | --- |
| `vllm_rlt/sampling_params.py` | Parameter contract and validation |
| `vllm_rlt/worker/sampler.py` | Greedy / top-k / top-p algorithm for one logits row; the M7 boundary |
| `vllm_rlt/worker/sampling.py` | Distribution helpers and rejection sampling added by PR #44; unchanged here |
| `vllm_rlt/worker/speculative.py` | Draft/verify loop that calls `sampling.py` directly; unchanged here |
| `vllm_rlt/worker/model_runner.py` | Sampling call site and RNG storage (`_sample_tensor`) |
| `vllm_rlt/request.py` | Current RNG location (`Request.generator`) |
| `vllm_rlt/core/scheduler.py` | RNG release on termination |
| `vllm_rlt/engine/preemption.py` | Preemption contract that must preserve the RNG |
| `tests/test_sampler.py` | Algorithm, seed and generator contracts |
| `tests/test_engine.py`, `tests/test_prefix_growth.py` | RNG lifecycle across termination and preemption |
| `tests/test_speculative.py` | Speculative-path distribution, replay and rejection-sampling contracts |

### Out of scope

Exit policy and stage transitions (M3), KV allocation and prepared metadata (M4),
attention backends (M8), the PD connector protocol (M9), new sampling features
(`min_p`, `repetition_penalty`, `logprobs`, `n > 1`), any change to the sampling
arithmetic or to user-visible defaults, the batched rewrite of the per-row CODA loop
(see "Known limitation" below), and collapsing `worker/sampling.py` into `Sampler`
(M7 2/3).

## 2. Classes and state

### `SamplingParams` (frozen dataclass)

Defaults are `max_tokens=16`, `temperature=0.0`, `top_p=1.0`, `top_k=-1`, `seed=0`,
plus the Ouro extensions `min_loops`, `max_loops`, `exit_threshold`, `ignore_eos` and
`priority`. `__post_init__` already validates everything sampling depends on:
`temperature` is finite and nonnegative, `top_p ∈ (0, 1]`, `top_k` is `-1` or a
positive integer, and `seed ∈ [0, 2**63)`. `Sampler` therefore assumes validated
input and does not re-check it.

### `Sampler`

`Sampler` owns only the device used to create a new generator. It holds no
per-request state: `sample()` receives the generator and returns the possibly created
one, so "where the RNG lives" stays visible at the call site. This is what lets
`M7 2/3` move RNG storage without touching the algorithm or its tests.

### `worker/sampling.py` (unchanged by this increment)

The same arithmetic also exists as module-level helpers, introduced by PR #44:
`probabilities()` builds the temperature/top-k/top-p distribution, `draw()` performs one
multinomial draw, and `generator_for()` performs the lazy generator creation for the
speculative path. `rejection_sample()` is speculative-only and has no counterpart in
`Sampler`.

### RNG lifecycle today

| Event | Current behavior | Location |
| --- | --- | --- |
| First random sample (CODA) | Lazily creates `torch.Generator(device=...)` seeded from `params.seed` | `sampler.py:43-44` |
| First random sample (speculative) | Lazily creates the same kind of generator on the `Request` | `sampling.py:22-25` |
| Greedy sample | `argmax`, no generator created or advanced | `sampler.py:31-32` |
| Request finishes | `scheduler.finish` sets `request.generator = None` | `scheduler.py:95` |
| Request preempted | Not reset; the `Request` object and its generator are retained | `preemption.py:3`, `preemption.py:102` |
| Request resumed | Not rebuilt; sampling continues on the same generator | `preemption.py:114-144` |

Both modules write the same `Request.generator` slot, so the two creation sites must stay
consistent until `M7 2/3` moves the RNG behind the runner (section 6). Speculative
decoding requires synchronous eager execution, refill scheduling and no preemption
(`llm_engine.py:37-50`), so only the CODA path exercises the preemption rows above.

## 3. Functions and parameters

```python
class Sampler:
    def __init__(self, device: torch.device): ...

    # logits: [vocab] 1-dim device tensor, any float dtype; caller guarantees ndim == 1
    # params: validated by SamplingParams.__post_init__
    # generator: None on the greedy path
    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Generator | None]:
        """Return a 0-dim long device tensor and the possibly created generator."""
```

`ModelRunner` keeps a thin delegate so the async path and existing tests are
unaffected:

```python
def _sample_tensor(self, logits: torch.Tensor, request: Request):
    token, generator = self.sampler.sample(logits, request.sampling_params, request.generator)
    request.generator = generator
    return token
```

## 4. Line-by-line coverage

`Sampler.sample` (`sampler.py:19-45`):

| Lines | Behavior |
| --- | --- |
| `31-32` | `temperature == 0` is a greedy short circuit. It runs **before** any RNG work, so greedy requests never create a generator. |
| `33` | Non-greedy logits are promoted with `float()` and divided by temperature. |
| `34-36` | `top_k`: the threshold is the k-th largest value and the mask is `logits < threshold`, so every token tied at the threshold survives and the candidate set can exceed `k`. |
| `37-42` | `top_p`: sort descending, cumulative softmax, then shift the mask by one position and force `remove[0] = False`, so at least one token always survives. |
| `43-44` | Lazy generator creation with `params.seed`, on `self.device`. |
| `45` | `torch.multinomial(logits.softmax(-1), 1, generator=...)` returns a 0-dim long tensor on the logits device. |

`ModelRunner` (`model_runner.py`):

| Lines | Behavior |
| --- | --- |
| `82` | `self.sampler = Sampler(self.device)` |
| `425` | CODA calls `_sample_tensor` once per request row and `torch.stack`s the results. |
| `601-606` | Delegate: reads `request.generator`, calls `Sampler.sample`, stores the result back. |

`worker/sampling.py` (unchanged; kept for the speculative path):

| Lines | Behavior |
| --- | --- |
| `6-19` | `probabilities`: the same temperature/top-k/top-p statements as `Sampler.sample`, returning the normalized distribution. |
| `22-25` | `generator_for`: lazily creates `request.generator` from `params.seed`. |
| `28-29` | `draw`: one `torch.multinomial` draw followed by `squeeze(0)`. |
| `32-37` | `sample_logits`: greedy short circuit or `draw`. This was `_sample_tensor`'s body before this increment, and it has no caller now. |
| `40-56` | `rejection_sample`: exact speculative rejection with residual resampling. |

### Documented project difference

`sampler.py:33` promotes logits to FP32 before dividing by temperature. The pinned
model file (`modeling_ouro.py` at revision `574fa66`) inherits from `GenerationMixin`
and delegates sampling to the Transformers generation stack; it contains no
`multinomial`, `top_k`, `top_p`, `temperature`, or logits-processor code of its own.
This project does not pin that generation stack, so there is no pinned reference
sampling path to compare against. The FP32 promotion is therefore a project choice
awaiting comparison with a real reference sampling path, not a difference from a
pinned sampling implementation. This is the "record the sampling implementation
separately" item required by [precision policy](precision-policy.md). `M7 1/3`
preserves it unchanged; the speculative path repeats the same promotion in
`sampling.py:9`.

### Known limitation

`model_runner.py:424-426` samples each request row in a Python loop, so every row pays
its own `sort`/`topk` over the model vocabulary (`vocab_size = 49152` in `config.py`).
vLLM instead batches the multinomial across the batch. This is recorded as a known
limitation and is deliberately not optimized in M7; any batched rewrite must keep the
arithmetic order `float()/temperature → top_k → top_p → multinomial` and the
no-host-sync guarantee.

## 5. Evidence-based findings

**Dead code: `_sample` had no callers (cleanup).** Repository-wide search found
only the definition, no call site; tests monkeypatch `_sample_tensor`
(`tests/test_async_pipeline.py:138`). Removed in `M7 1/3`.

**Sampling exists in two modules after PR #44 (structural finding).**
`worker/sampling.py` arrived with the speculative decoding change, and its
`probabilities()`/`draw()` carry the same statements this increment moved into
`Sampler.sample`; `_sample_tensor` was a one-line delegate to `sampling.sample_logits()`
before it. This increment deliberately keeps both: `Sampler` becomes the M7 boundary for
the CODA path and is pinned by the new tests, while `sampling.py` keeps serving the
speculative path untouched. The cost is one duplicated distribution construction for one
increment, and `sample_logits()` loses its only caller, so its import is dropped here and
the function itself is removed in `M7 2/3` (section 6).

**RNG ownership disagrees with the RFC state table.** The RFC assigns "RNG objects
and their execution/snapshot state" to the worker, while the generator currently lives
on the scheduler-side `Request` dataclass (`request.py:44`). Behavior is correct today
because `Request` is a shared handle and preemption/resumption happens to retain it,
but ownership is implicit: any change that treats `release()` as "termination releases
everything" can silently reset the RNG.

**Preemption and termination share `model_runner.release()` (the key design
constraint).** `preemption.py:102` calls `release()` when suspending a victim;
`llm_engine.py:191` (abort) and `llm_engine.py:332` (finish) call the same method, and
`pd/worker.py:107` (removal) and `pd/worker.py:305` (prefill compute release) do as
well. Moving the RNG into the runner and deleting it inside `release()` would reset it
on preemption, changing the token sequence after restoration. That would violate the
`preemption.py:3` contract and the RFC acceptance criterion for fixed-seed sampling
across preemption/restoration. RNG migration must therefore distinguish "termination
release" from "preemption suspend" (see section 6).

**Sampling had almost no unit coverage (test gap).** Before this increment the
only coverage was batch invariance (`tests/test_engine.py:147`), a monkeypatched
EOS path (`tests/test_async_pipeline.py:138`), and an indirect preemption check:
`tests/test_prefix_growth.py:85` compared token sequences after suspend/resume with
`temperature=0.8, seed=45`, which would fail if the RNG were reseeded, but did not
assert generator identity or state directly. Uncovered: greedy never creating a
generator, `top_k=1` matching greedy, `top_k` tie behavior, `top_p` retaining one
token, same/different seed reproducibility, direct generator identity and state
assertions across preemption, and RNG isolation after a request ID is reused.
`tests/test_sampler.py`, `tests/test_engine.py` and `tests/test_prefix_growth.py`
now pin these.

**The sampling implementation was not documented separately (documentation gap).**
`docs/serving.md:55-56` lists only field defaults, and `docs/precision-policy.md:19`
requires recording the sampling implementation separately. This walkthrough is that
record.

**Sampling is on the CODA hot path (performance constraint).** The async path
stacks the sampled results and scatters them with `tokens=True`
(`model_runner.py:424-428`). Extraction must not add `.item()`, `.cpu()`,
`synchronize()`, or an extra kernel, and must not reorder the arithmetic.

**The CODA per-row Python loop is a known limitation.** See section 4.

## 6. Target design and migration

`M7 1/3` (this increment) only moves the algorithm into `Sampler` and pins behavior.
RNG ownership is unchanged: `request.generator` is still read and written by the
`_sample_tensor` delegate and by `sampling.generator_for` on the speculative path, and
`release()` still does not touch it.

`M7 2/3` will, in order:

1. Collapse `worker/sampling.py` into `Sampler`: move `probabilities()`, `draw()` and
   `generator_for()` onto the class, retarget the `speculative.py` call sites, and delete
   `sample_logits()`.
2. Add a runner-side RNG registry (`dict[str, torch.Generator]`) and have both sampling
   paths read and write it instead of `request.generator`.
3. Split the release path: `suspend(request_id)` synchronizes the event and frees the
   state slot but keeps the RNG (called by preemption); `release(request_id)` does the
   same and additionally drops the RNG (called by abort, finish and PD removal).
4. Remove `Request.generator` and the `scheduler.finish` assignment, then delete the
   temporary adapters.

The unification must keep each path's RNG call order: the CODA path draws one
`multinomial` per token, while the speculative path additionally draws draft candidates,
`torch.rand` acceptance uniforms and residual resamples. The two paths therefore produce
different sequences from the same seed today, and only their distributions agree; `M7 2/3`
must preserve that rather than align the streams.

The target RNG contract:

| Event | Target behavior |
| --- | --- |
| First random sample | Create the generator from `params.seed` in the request's RNG slot |
| Later samples | Reuse and advance the same generator (ordering semantics unchanged) |
| Greedy request | Never create or advance a generator |
| Speculative request | Same slot; draft draws, acceptance uniforms and residual draws all come from it |
| Preemption | Keep the generator (or snapshot `get_state()`) and continue from it on resume |
| Termination (stop/length/abort/PD removal) | Drop the RNG slot so a new request with the same ID starts fresh |
| Late async result | Must not write into a new request's RNG; the existing object-identity check covers results, and "release on termination" covers the RNG |

### Removal condition for the delegate

`ModelRunner._sample_tensor` exists so that `tests/test_async_pipeline.py:138` keeps
working. It can be removed once that test patches `Sampler` instead, which is planned
for `M7 2/3` when the runner-side registry lands and both sampling paths read it.

## 7. Validation

`M7 1/3` CPU evidence (macOS, Python 3.12, PyTorch 2.14.0):

| Suite | Result |
| --- | --- |
| `upstream/main @ 3314c1b` baseline (`test_engine.py test_prefix_growth.py test_async_pipeline.py`) | 47 passed, 15 skipped |
| Same three files after this increment (baseline 47 + 2 new engine tests) | 49 passed, 15 skipped |
| With `test_sampler.py` and `test_serving.py` added | 59 passed, 16 skipped |
| `upstream/main @ 3314c1b` full CPU suite | 240 passed, 121 skipped |
| Full CPU suite (`tests/`) | 252 passed, 121 skipped |

```bash
# upstream/main @ 3314c1b baseline (checkout 3314c1b first)
OMP_NUM_THREADS=1 python -m pytest -q tests/test_engine.py tests/test_prefix_growth.py tests/test_async_pipeline.py
# same three files after this increment
OMP_NUM_THREADS=1 python -m pytest -q tests/test_engine.py tests/test_prefix_growth.py tests/test_async_pipeline.py
# with new test files
OMP_NUM_THREADS=1 python -m pytest -q tests/test_sampler.py tests/test_engine.py \
    tests/test_prefix_growth.py tests/test_async_pipeline.py tests/test_serving.py
# full CPU suite
OMP_NUM_THREADS=1 python -m pytest -q tests/
# lint
python -m ruff check .
python -m ruff format --check vllm_rlt/worker/sampler.py vllm_rlt/worker/model_runner.py tests/test_sampler.py
```

The full suite grows by 12 tests over the baseline: 10 in `tests/test_sampler.py` and the
2 new engine tests. Ruff: `ruff check .` passes and the touched Python files are already
formatted. Three pre-existing markdown files are not `ruff format` clean on the baseline
either, so the documentation check stays scoped to the touched Python files.

The extraction changes no arithmetic, so no new precision or performance evidence is
required. Sampling sequences depend on `torch.multinomial`; the numbers above are with
PyTorch 2.14.0.

`M7 1/3` and `M7 2/3` do **not** include GPU validation. Device-side behavior under
async scheduling and CUDA streams is covered by `M7 3/3`:

```bash
python -m pytest -q tests/ -m gpu --run-gpu
```

Until then, both increments state "GPU validation pending in M7 3/3" in their pull
request descriptions.
