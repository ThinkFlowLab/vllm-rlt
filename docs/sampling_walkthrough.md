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
input and does not re-check it. The rollout fields `logprobs`, `stop_token_ids` and
`seed=None` were added later; see [section 8](#8-rollout-outputs).

### `Sampler`

`Sampler` owns only the device used to create a new generator. It holds no
per-request state: `sample()` receives the generator and returns the possibly created
one, so "where the RNG lives" stays visible at the call site. This is what lets
`M7 2/3` move RNG storage without touching the algorithm or its tests.

### `worker/sampling.py` (unchanged by this increment)

Since the rollout-outputs change, the temperature/top-k/top-p statements live once in
`processed_logits()`, shared by `Sampler.sample`, `probabilities()` and processed
logprobs (section 8). Before that, the same arithmetic also existed as module-level
helpers, introduced by PR #44:
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

## 8. Rollout outputs

RL trainers (verl, vime) consume, for each generated token, its ID, the loop depth
that produced it and its log-probability under the policy that sampled it. This
section is the engine contract from RFC
[#70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70), PR 1. Names follow vLLM.

### Parameters

| Option | Default | Contract |
| --- | --- | --- |
| `SamplingParams.logprobs` | `None` | `0` returns the log-probability of each sampled token, as vLLM's `logprobs=0`. Positive values (top-k alternatives) raise `ValueError` for now. |
| `SamplingParams.seed` | `0` | `None` asks the engine for a fresh 63-bit seed from OS entropy (`secrets`), consuming neither the torch nor the Python global RNG. |
| `SamplingParams.stop_token_ids` | `()` | A list or tuple of IDs, normalized to a tuple. Emitting one finishes the request with `stop`, even with `ignore_eos=True` (vLLM semantics). IDs outside the vocabulary are rejected by `add_request`. |
| `LLMEngine` / `LLM` / `PDEngine` `logprobs_mode` | `"raw_logprobs"` | Engine-level, as in vLLM. `"processed_logprobs"` is the other accepted value; `raw_logits`, `processed_logits` and anything else raise `ValueError`. No CLI flag yet. |

`LLMEngine.add_request` and `PDEngine.add_request` resolve `seed=None` (`resolve_seed`)
before creating the `Request`, so P and D workers receive identical parameters and
`RequestOutput.sampling_params.seed` reports the seed actually used. Replaying with it
reproduces tokens and logprobs bitwise only with the same batch composition, scheduling,
engine configuration, dtype and hardware, as with vLLM seeds. In a different batch, the
GEMM and attention reductions can change the logits in the last bits. Logprobs then
agree only numerically, and near a sampling or exit-threshold boundary the tokens and
exit depths can diverge. `Sampler.sample` and `generator_for` raise if they ever see an
unresolved seed.

### Output contract

`RequestOutput` appends `logprobs: list[float] | None` and `sampling_params` (the
effective parameters). When a request sets `logprobs=0`, every output it emits,
including streamed, `stop`, `length` and `abort` outputs, satisfies
`len(logprobs) == len(token_ids) == len(exit_depths)`; otherwise `logprobs` is `None`.
A stop token (EOS or a `stop_token_ids` entry) stays in `token_ids` with its depth and
logprob. `LLMEngine._append_output` is the single place that appends a token, its depth
and its logprob and decides `stop` (stop IDs, then EOS unless `ignore_eos`) before
`length`; synchronous CODA, asynchronous delivery and speculative rounds all use it.
`LLM.generate` omits the text of a matched `stop_token_ids` entry, as vLLM does; the
token stays in `token_ids`, `exit_depths` and `logprobs`. EOS text handling is unchanged.
The HTTP API does not accept `logprobs`, `stop_token_ids` or `seed: null` until the
token-in/token-out API of #70.

### Definitions

Values are FP32, computed on the device from the CODA logits row of the emitted token.

| Mode | Value |
| --- | --- |
| `raw_logprobs` | `log_softmax(coda_logits.float())[token]`: the model distribution at the depth where the token exited. |
| `processed_logprobs` | `log_softmax(processed_logits(logits, params))[token]`: the distribution actually sampled from (temperature, then top-k, then top-p, excluded tokens `-inf`). Greedy requests report `log_softmax(logits.float())[token]` without temperature, as vLLM does. |

`processed_logits` is the one function `Sampler.sample` and `probabilities()` use, so
sampling arithmetic, op order and RNG consumption are unchanged. Logprobs are computed
from `(logits row, params, token, mode)` after sampling and never consume RNG, so
requesting them does not change sampled tokens.

### Looped-model probability convention

- `logprobs[i]` is conditional on `exit_depths[i]`: the token's policy is the model
  truncated at that loop depth, not the full-depth model.
- `token_ids[0]` always comes from full-depth prefill (`exit_depths[0]` is the model
  depth). Per-request `min_loops`, `max_loops` and `exit_threshold` affect decode only.
- Under `last_exited` KV, a trainer that recomputes logprobs with an ordinary full-depth
  forward matches only rollouts whose `exit_depths` all equal the model depth:
  `exit_threshold=1` and `max_loops` equal to the model depth, in a non-`trace` exit
  mode. `trace` takes its depths from `depths_by_request`.
- Under `shared` KV, prompt and decode positions read earlier tokens' final retained KV
  at every loop depth (see [KV layout examples](kv_layout_computation.md)). No ordinary
  forward reproduces these logprobs, even at full depth. The recompute must implement
  shared-layout attention.
- Per input position, a teacher-forced recompute runs prompt positions at the model
  depth and the input `token_ids[i]` (at position `len(prompt) + i`) for
  `exit_depths[i + 1]` loops: `[depth] * len(prompt) + exit_depths[1:]`. The last
  generated token is never an input.
- With early exit, the recompute must replay `exit_depths`. Under `last_exited` KV, a
  token that exited at depth `d` has its depth-`d` KV copied into the deeper loop
  planes, and later tokens attend to those copies; the recompute must reproduce that.

### Data flow and cost

| Path | Behavior |
| --- | --- |
| Synchronous | `ModelRunner._execute` returns `(token_ids, logprobs)` device tensors; `execute()` reads both back and `_update` appends them. |
| Asynchronous | Token IDs stay on the device for the next prelude and `routing.scatter`, as before. Logprobs get a non-blocking FP32 copy into a pinned slot recorded on the same CUDA event; `_deliver_coda` appends them on delivery. |
| PD | D samples and computes logprobs. The coordinator mirrors each output's `logprobs` into its `Request`, so `abort_request` stays aligned. P never samples. |
| Speculative | `SpeculativeResult.logprobs` is appended per emitted token and sliced with the tokens on EOS, stop IDs and length. The runner does not compute it yet, so `add_request` rejects `logprobs` with `speculative_config`. |
| Preemption, prefix caching | Snapshots retain the `Request`, so `request.logprobs` is neither reset nor duplicated; a prefix hit or a resumption matches the cold logprobs numerically (the tests use `abs=1e-6`), not bitwise on GPU, because the prefill chunking changes. |

When no request in a CODA batch asks for logprobs there are no extra kernels, syncs,
readbacks or allocations. Otherwise raw mode adds one batched
`log_softmax(dtype=float32)` and `gather` over the batch (an FP32 `[rows, vocab]`
temporary), and processed mode recomputes `processed_logits` for each requesting row,
which repeats the top-k/top-p work of sampling. The synchronous path adds one small
device-to-host copy. These costs have not been measured on GPU yet.

### Readback slot sizing

Asynchronous CUDA execution leases pinned host slots from preallocated pools. FP32
exit-score slots are held by recurrent tickets until their scores are collected: at most
one ticket per request through `_pending_exit_signals`, plus the in-flight submissions
bounded by `submission_events`. A request whose output is still pending can already run
its next token's recurrent loops, so it can hold an exit-score lease and a logprob lease
at once. Logprobs therefore use a dedicated pool of `max_num_seqs + 4` FP32 slots with
`max_num_seqs` rows each. A CODA ticket keeps its lease until `_deliver_coda` collects
it. Each request has at most one undelivered output, so live CODA tickets never exceed
`max_num_seqs`; tickets of aborted requests have completed events (`release()`
synchronizes) and are collected at the next step before any new submission. The
exit-score pool keeps its previous sizing argument unchanged.

### Limitations

- `logprobs > 0` (top-k alternatives) is not supported.
- The HTTP API is unchanged: `logprobs` stays null-only, and `stop_token_ids` and
  `seed: null` are rejected until the token-in/token-out API (RFC #70, PR 3).
- Speculative decoding logprobs are pending in the speculative runner.
- `finish_reason` is `stop` for both EOS and stop IDs; the matching ID is not reported.
- GPU paths (CUDA readback, CUDA Graphs, PD) are covered by `@pytest.mark.gpu` tests
  in `tests/test_rollout_outputs.py` that have not yet run in this change.
