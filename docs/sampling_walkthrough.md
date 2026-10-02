# Sampling Module Walkthrough

This note documents the M7 (Sampling) work of RFC
[#32](https://github.com/hsliuustc0106/vllm-rlt/issues/32). It covers
[`worker/sampler.py`](../vllm_rlt/worker/sampler.py) (`Sampler`, `new_generator` and
`RngRegistry`), the speculative call sites in
[`worker/speculative.py`](../vllm_rlt/worker/speculative.py), the sampling call site and RNG
hooks in [`worker/model_runner.py`](../vllm_rlt/worker/model_runner.py) and
[`engine/llm_engine.py`](../vllm_rlt/engine/llm_engine.py), and the preemption contract in
[`engine/preemption.py`](../vllm_rlt/engine/preemption.py).

M7 has three increments. `M7 1/3` moved the algorithm into `Sampler` and pinned it with CPU
contract tests (merged as PR #45, `ea680f7`). `M7 2/3` — the revision this note describes —
collapses `worker/sampling.py` into `Sampler`, moves the sampling RNG into an engine-owned
registry, splits the runner release path into `suspend`/`release`, removes
`Request.generator`, and fixes the device the generator is created on. `M7 3/3` adds GPU
validation.

## 1. Baseline and boundary

`M7 1/3` was reviewed against `upstream/main @ 3314c1b` (the merge of PR #44) and merged as
`ea680f7`. `M7 2/3` starts from `upstream/main @ ecb1f8b`. File and line references below
describe the `M7 2/3` tree; cross-module call sites are also named by symbol, because line
numbers drift as `main` moves.

### Call chain

```text
Scheduler.schedule() -> CODA batch
  -> ModelRunner.execute()                      # sync path
     -> _execute()
        -> model.coda(hidden)                   # lm_head logits rows
        -> _sample_tensor(row, request)         # model_runner.py:429
           -> Sampler.sample(logits, params, rng.peek(request_id))
           -> rng.store(request_id, generator)
        -> result.cpu().tolist()                # the only host readback, in execute()
  -> ModelRunner.submit()                       # async path
     -> _execute() samples on the device
     -> routing.scatter(result, tokens=True)    # sampled IDs stay on the device
  -> SpeculativeRunner.execute()                # speculative path: sync, eager or graphs
     -> Sampler.probabilities(logits, params)   # sampler.py:22
     -> Sampler.draw(probs, rng.acquire(...))   # sampler.py:44
     -> rejection_sample(candidate, p, q, ...)  # speculative.py
```

The async path requires the sampled value to remain a 0-dim device tensor: it is
`torch.stack`ed and scattered into the next prelude, so sampling must not call `.item()`,
`.cpu()`, `synchronize()` or add host work. The sync path's `.cpu().tolist()` happens in
`execute()`, outside the sampling function.

### In scope

| File | Role |
| --- | --- |
| `vllm_rlt/sampling_params.py` | Parameter contract and validation |
| `vllm_rlt/worker/sampler.py` | `Sampler` (algorithm), `new_generator` (the single seed-to-generator definition), `RngRegistry` (per-request RNG owner) |
| `vllm_rlt/worker/speculative.py` | Draft/verify loop plus the speculative-only `rejection_sample` |
| `vllm_rlt/worker/model_runner.py` | CODA sampling delegate, and the `suspend`/`release` RNG hooks |
| `vllm_rlt/engine/llm_engine.py` | Creates the registry and injects it into both runners |
| `vllm_rlt/engine/preemption.py` | Preemption contract: suspension keeps the RNG |
| `vllm_rlt/request.py` | Scheduling-side request state; no longer holds the RNG |
| `vllm_rlt/core/scheduler.py` | Termination; no longer clears an RNG field |
| `tests/test_sampler.py` | Algorithm, registry, seed and creation-device contracts |
| `tests/test_engine.py`, `tests/test_prefix_growth.py` | RNG across termination, abort, request-ID reuse and preemption |
| `tests/test_speculative.py` | Speculative distribution, replay, rejection sampling, registry sharing and advancement |

### Out of scope

Exit policy and stage transitions (M3), KV allocation and prepared metadata (M4),
attention backends (M8), the PD connector protocol (M9), new sampling features (`min_p`,
`repetition_penalty`, `logprobs`, `n > 1`), any change to the sampling arithmetic or to
user-visible defaults, and the batched rewrite of the per-row CODA loop (see "Known
limitation").

## 2. Classes and state

### `SamplingParams` (frozen dataclass)

Defaults are `max_tokens=16`, `temperature=0.0`, `top_p=1.0`, `top_k=-1`, `seed=0`, plus
the Ouro extensions `min_loops`, `max_loops`, `exit_threshold`, `ignore_eos` and `priority`.
`__post_init__` already validates everything sampling depends on: `temperature` is finite
and nonnegative, `top_p ∈ (0, 1]`, `top_k` is `-1` or a positive integer, and
`seed ∈ [0, 2**63)`. `Sampler` and `RngRegistry` therefore assume validated input and do
not re-check it.

### `Sampler`

`Sampler` is stateless: no device, no per-request data. `probabilities` and `draw` are
static helpers, and `sample` takes the generator in and returns the possibly created one.
The generator is created on **`logits.device`** — the device the distribution is sampled on
— rather than on the device the model was loaded on. That fixes the choice at first
creation only: a generator that already exists is reused as-is, and `RngRegistry.acquire`
neither checks nor migrates its device, so callers must still keep a request's logits and
its existing generator on the same device.

### `RngRegistry`

The registry owns the sampling RNG state, which the RFC state table assigns to the worker
side. The engine creates one instance and injects it into `ModelRunner` and
`SpeculativeRunner`, so the CODA and speculative paths advance the same per-request slot
and a single owner decides when it is released.

| Method | Behavior |
| --- | --- |
| `peek` | Return the request's generator, or `None` before its first random draw |
| `store` | Keep a sampler-created generator; `None` (greedy) leaves the slot absent |
| `acquire` | Lazily create on the given device and keep it; idempotent for later calls |
| `drop` | Forget exactly that request's slot; termination only, never suspension |

### RNG lifecycle

| Event | Behavior | Location |
| --- | --- | --- |
| First random sample (CODA) | `Sampler.sample` creates the generator from `params.seed` on `logits.device`; the delegate stores it | `sampler.py:48-66`, `ModelRunner._sample_tensor` |
| First random sample (speculative) | `RngRegistry.acquire` creates the same kind of generator on the logits row's device | `SpeculativeRunner._generator_for` |
| Greedy sample | `argmax`; no slot is ever created | `sampler.py:59-60` |
| Request preempted | `ModelRunner.suspend` frees the runner slot but keeps the RNG slot | `engine/preemption.py:104` |
| Request resumed | Sampling continues on the same generator instead of reseeding | `PreemptionManager.resume` |
| Request terminates | `ModelRunner.release` drops the RNG slot (stop/length/abort/PD removal) | `model_runner.py:608-615` |
| Request ID reused | Starts from a fresh generator, so tokens replay instead of continuing an old stream | `RngRegistry.drop` |
| PD prefill compute release | Also `release()`, which is harmless: the prefill role never samples | `PDWorker.progress` |

`Request` is scheduling-side state again. Speculative decoding rejects asynchronous
scheduling, and requires refill scheduling without preemption
(`engine/llm_engine.py:40-52`). CUDA Graphs are allowed: `SpeculativeRunner` builds its own
recurrent and CODA graphs, and only the sampling step stays eager outside them.
Speculative decoding never preempts, so only the CODA path exercises the preemption rows
above.

## 3. Functions and parameters

```python
class Sampler:
    # logits: [vocab] 1-dim device tensor, any float dtype; caller guarantees ndim == 1
    # params: validated by SamplingParams.__post_init__
    # generator: None on the greedy path

    @staticmethod
    def probabilities(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor: ...

    @staticmethod
    def draw(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor: ...

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Generator | None]:
        """Return a 0-dim long device tensor and the possibly created generator."""


def new_generator(device: torch.device, params: SamplingParams) -> torch.Generator: ...


class RngRegistry:
    def __init__(self): ...
    def peek(self, request_id: str) -> torch.Generator | None: ...
    def store(self, request_id: str, generator: torch.Generator | None) -> None: ...
    def acquire(self, request_id: str, params: SamplingParams, device) -> torch.Generator: ...
    def drop(self, request_id: str) -> None: ...
```

`ModelRunner` keeps a thin delegate so the async path and the existing monkeypatch target
stay valid, and separates the two RNG lifecycles:

```python
def _sample_tensor(self, logits: torch.Tensor, request: Request):
    request_id = request.request_id
    token, generator = self.sampler.sample(
        logits, request.sampling_params, self.rng.peek(request_id)
    )
    self.rng.store(request_id, generator)
    return token


def suspend(self, request_id):  # preemption: free the runner slot, keep the RNG
    self._free_runner_slot(request_id)


def release(self, request_id):  # termination: free the runner slot and drop the RNG
    self._free_runner_slot(request_id)
    self.rng.drop(request_id)
```

`LLMEngine.__init__` creates `self.rng = RngRegistry()` before constructing either runner
and passes that same instance to both, because `SpeculativeRunner` is built before
`ModelRunner` and receives no reference to it.

## 4. Line-by-line coverage

`worker/sampler.py`:

| Lines | Behavior |
| --- | --- |
| `22-42` | `Sampler.probabilities`: promote, temperature-scale, top-k mask, top-p cumulative-shift mask, normalize |
| `29-30` | Greedy has no temperature-scaled distribution, so it is a caller error, not a silent `argmax` |
| `31` | Non-greedy logits are promoted with `float()` and divided by temperature |
| `32-34` | `top_k`: threshold is the k-th largest value and the mask is `logits < threshold`, so tokens tied at the threshold all survive and the candidate set can exceed `k` |
| `35-42` | `top_p`: sort descending, cumulative softmax, then shift the mask and force `remove[0] = False`, so at least one token always survives |
| `44-46` | `Sampler.draw`: one `torch.multinomial`, squeezed to a 0-dim long tensor |
| `48-66` | `Sampler.sample`: greedy short circuit first, then the distribution, then generator creation only when absent, then the draw |
| `59-60` | The greedy short circuit runs before any RNG work |
| `69-71` | `new_generator`: the only place a seed becomes a generator |
| `74-105` | `RngRegistry.peek/store/acquire/drop` |

`worker/model_runner.py`:

| Lines | Behavior |
| --- | --- |
| `83-86` | `self.sampler = Sampler()` and the injected registry (a private one when none is injected) |
| `426-430` | CODA samples one row per request and `torch.stack`s the results |
| `587-598` | `_free_runner_slot`: wait for the request's events, then recycle its device state slot |
| `600-606` | `suspend`: keeps the RNG slot, so a resumed sequence stays lossless |
| `608-615` | `release`: drops the RNG slot, so a reused ID starts from a fresh stream |
| `622-629` | Delegate: peek, sample, store |

`engine/llm_engine.py`:

| Lines | Behavior |
| --- | --- |
| `104` | One `RngRegistry` per engine |
| `105-115` | Injected into `SpeculativeRunner` |
| `116-123` | Injected into `ModelRunner` |

`worker/speculative.py`:

| Lines | Behavior |
| --- | --- |
| `66-71` | The runner takes the shared registry (private fallback) |
| `104-108` | `_generator_for` resolves the request's generator from the registry, on the logits row's device |
| `153`, `198`, `207` | The draft draw, rejection sampling and the bonus draw all draw from that generator |

### Documented project difference

`sampler.py:31` promotes logits to FP32 before dividing by temperature. The pinned model
file (`modeling_ouro.py` at revision `574fa66`) inherits from `GenerationMixin` and
delegates sampling to the Transformers generation stack; it contains no `multinomial`,
`top_k`, `top_p`, `temperature`, or logits-processor code of its own. This project does not
pin that generation stack, so there is no pinned reference sampling path to compare
against. The FP32 promotion is therefore a project choice awaiting comparison with a real
reference sampling path, not a difference from a pinned sampling implementation. This is
the "record the sampling implementation separately" item required by
[precision policy](precision-policy.md). `M7 2/3` preserves it unchanged, and both paths
now share the one helper that applies it.

### Known limitation

`model_runner.py:426-430` samples each request row in a Python loop, so every row pays its
own `sort`/`topk` over the model vocabulary (`vocab_size = 49152` in `config.py`). vLLM
instead batches the multinomial across the batch. This is recorded as a known limitation
and is deliberately not optimized in M7; any batched rewrite must keep the arithmetic order
`float()/temperature → top_k → top_p → multinomial` and the no-host-sync guarantee.

## 5. Evidence-based findings

**Dead code: `_sample` had no callers (cleanup).** Repository-wide search found only the
definition, no call site; tests monkeypatch `_sample_tensor`
(`tests/test_async_pipeline.py:138`). Removed in `M7 1/3`.

**Sampling existed in two modules after PR #44 (structural finding).** `worker/sampling.py`
arrived with the speculative decoding change, and its `probabilities()`/`draw()` carried
the same statements `M7 1/3` had moved into `Sampler.sample`, so a fix to the arithmetic
could silently miss one path. Resolved in `M7 2/3`: `probabilities()` and `draw()` are now
`Sampler` static methods, the speculative-only `rejection_sample()` lives in
`worker/speculative.py`, the unused `sample_logits()` is gone, and `worker/sampling.py` was
emptied and then deleted in its own commit.

**RNG ownership disagreed with the RFC state table.** The RFC assigns "RNG objects and
their execution/snapshot state" to the worker, while the generator lived on the
scheduler-side `Request`. Resolved in `M7 2/3`: an engine-owned `RngRegistry` holds the
generators, and `Request` no longer has an RNG field.

**Preemption and termination shared `model_runner.release()` (the key design constraint).**
`PreemptionManager.preempt` used to call the same method as `LLMEngine.abort_request`,
`LLMEngine._finish`, `PDWorker.remove` and the prefill compute release in
`PDWorker.progress`. Dropping the RNG inside that shared method would have reset it on
preemption and changed the token sequence after restoration, violating the
`engine/preemption.py` contract and the RFC requirement for fixed-seed sampling across
preemption/restoration. Resolved in `M7 2/3` by splitting the path: `suspend` keeps the
RNG slot and preemption calls it, `release` drops the slot and every termination path
keeps calling it.

The prefill-side compute release needs no separate path for RNG purposes: the prefill role
owns only prompt KV and never samples (`pd/worker.py`: "P owns only prompt KV and never
samples. D owns all sampling/RNG."), and only the decode role reaches `Stage.CODA`.
`release()` there therefore cannot drop a generator that is still in use, and `drop` on an
absent slot is a no-op. The RNG-bearing termination sites are the decode worker's
`remove()` and the engine's abort/finish.

**Sampling had almost no unit coverage (test gap).** Before `M7 1/3` the only coverage was
batch invariance (`tests/test_engine.py`), a monkeypatched EOS path
(`tests/test_async_pipeline.py:138`) and an indirect preemption check. `M7 1/3` added
algorithm, seed and generator tests; `M7 2/3` retargeted the RNG lifecycle assertions to
the registry and added registry-contract tests.

That review also exposed a gap in `M7 2/3`'s own tests: asserting that both runners hold
the same `RngRegistry` object does not prove the speculative path draws from it. Rebuilding
the generator per call inside `_generator_for` left the whole suite green while the stored
slot stopped advancing and sampled output diverged after the first round. Closed by
`test_speculative_path_advances_the_registry_generator`, which requires a later speculative
round to reuse the same generator and advance its state, and by
`test_abort_drops_the_rng_slot_so_id_reuse_starts_over`, which requires abort to drop the
slot and a reused ID to replay a same-seed run. Both fail under the corresponding
mutations.

**Generator device coupling (found during `M7 2/3` review).** The generator was created on
the model's device while `multinomial` ran on the distribution's device. With both on one
device this never triggers, but the two are separate values and nothing enforced the
equality. Fixed by creating the generator on `logits.device` in both paths; pinned by
`test_generator_is_created_on_the_logits_device`. Actually executing the two paths on
different devices is a `M7 3/3` GPU item.

**The sampling implementation was not documented separately (documentation gap).**
`docs/serving.md:55-56` lists only field defaults, and `docs/precision-policy.md:19`
requires recording the sampling implementation separately. This walkthrough is that record.

**Sampling is on the CODA hot path (performance constraint).** The async path stacks the
sampled results and scatters them with `tokens=True` (`model_runner.py:426-432`). `M7 1/3`
and `M7 2/3` add no `.item()`, `.cpu()`, `synchronize()` or kernel; the delegate adds two
dictionary operations per token.

**The CODA per-row Python loop is a known limitation.** See section 4.

## 6. Target design and migration

`M7 2/3` lands the target boundary:

- one sampling module: `Sampler` (algorithm), `new_generator` (seed to generator),
  `RngRegistry` (per-request RNG state), with the speculative-only `rejection_sample` left
  next to its only consumer;
- RNG state is engine-owned and shared by both runners; `suspend` keeps a request's slot
  and `release` drops it, so preemption stays lossless and termination keeps request-ID
  reuse isolated;
- greedy requests never acquire a slot;
- the arithmetic order `float()/temperature → top_k → top_p → multinomial` is unchanged,
  and the generator is created on the distribution's device.

Each path keeps its own RNG call order: the CODA path draws one `multinomial` per token,
while the speculative path additionally draws draft candidates, `torch.rand` acceptance
uniforms and residual resamples. The two paths therefore produce different sequences from
the same seed, and that is preserved rather than aligned.

### Removal condition for the delegate

`ModelRunner._sample_tensor` exists so that `tests/test_async_pipeline.py:138` keeps
working. It can be removed once that test patches `Sampler` (or the registry) instead.
This is not required for `M7 3/3`, which validates the current shape.

### What `M7 3/3` must validate

GPU sampling correctness on CUDA; no host synchronization on the async CODA path and
correct `device_values` hand-off to the next prelude; CUDA Graph replay plus eager sampling;
PD decode-worker sampling with P never sampling, including RNG isolation after NIXL
transfer; GPU preemption RNG continuity; and, newly relevant here, generator creation on
the distribution's device when model and logits devices are exercised separately.

## 7. Validation

`M7 2/3` CPU evidence, reproduced independently on two environments:

- macOS, Python 3.12, PyTorch 2.14.0;
- Windows, Python 3.10.20, pytest 9.1.1.

| Command | Result |
| --- | --- |
| `pytest -m "not gpu" -q tests/test_sampler.py tests/test_speculative.py tests/test_engine.py tests/test_prefix_growth.py` | 102 passed, 13 deselected |
| `pytest -m "not gpu" -q` (all collected tests) | 424 passed, 29 skipped, 112 deselected |
| `pytest -q` (the same, with the GPU tests reported as skipped) | 424 passed, 141 skipped |
| `pytest -q tests/test_sampler.py tests/test_engine.py tests/test_prefix_growth.py tests/test_async_pipeline.py tests/test_serving.py tests/test_speculative.py tests/test_pd.py` | 169 passed, 33 skipped |
| `pytest -m gpu -q` | 112 selected, all skipped without `--run-gpu`: **no GPU test was executed** |
| `ruff format --check` over tracked Python files | 81 files already formatted |
| `ruff check .` | One pre-existing `F821` in `tests/test_speculative.py` that this increment does not introduce and that a separate fix PR (#79) removes; no new findings |

Both environments report the same numbers for every command above. The collected total
still depends on installed optional dependencies (`aiohttp`, `lm_eval`, `nixl`, locally
cached checkpoints): an environment missing `aiohttp`, for example, does not collect
`tests/test_serving.py` at all, so its total is lower even though nothing fails.

`M7 2/3` and `M7 1/3` are CPU-only. Device-side behavior under async scheduling, CUDA
streams and CUDA Graphs, and sampling on the PD decode worker, are **not** validated here
and are deferred to `M7 3/3`; `pytest -m gpu` selects 112 tests and skips every one of them
without a reserved device.

### Behavior fingerprint

Behavior preservation is checked by running one script on two trees and comparing its
digests byte for byte:

- before: `ecb1f8b`, the `M7 2/3` base (the docs-only commit above it changed no code);
- after: the `refactor/m7-sampling-module` head that this note ships with.

The script was a local scratch file, not a repository file; save it as `fingerprint.py`
and run it from the repository root on either tree with
`PYTHONPATH=. OMP_NUM_THREADS=1 python fingerprint.py`. It handles both trees, where
`Sampler` still required a device, `probabilities` lived in `worker/sampling.py`, and
`rejection_sample` had not moved yet.

```python
"""Scratch fingerprint for the M7 2/3 migration; not a repository file."""

import hashlib

import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import LLM, SamplingParams, SpeculativeConfig
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.worker.sampler import Sampler


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def new_sampler():
    try:
        return Sampler(torch.device("cpu"))  # ecb1f8b requires the device argument
    except TypeError:
        return Sampler()


try:  # rejection_sample moved to worker/speculative.py in M7 2/3
    from vllm_rlt.worker.speculative import rejection_sample
except ImportError:
    from vllm_rlt.worker.sampling import rejection_sample

try:  # probabilities moved onto Sampler in M7 2/3
    from vllm_rlt.worker import sampling
except ImportError:
    sampling = None

probabilities = getattr(Sampler, "probabilities", None) or sampling.probabilities

logits = torch.linspace(3.0, -3.0, 64)
sampler_cases = [
    SamplingParams(temperature=0.0, top_k=5, top_p=0.5),
    SamplingParams(temperature=0.7, seed=42),
    SamplingParams(temperature=0.7, seed=42, top_k=8),
    SamplingParams(temperature=0.7, seed=42, top_k=8, top_p=0.9),
    SamplingParams(temperature=1.3, seed=7, top_k=1),
    SamplingParams(temperature=0.5, seed=99, top_p=0.3),
    SamplingParams(temperature=2.0, seed=0, top_k=64),
]
for index, params in enumerate(sampler_cases):
    sampler, generator, tokens = new_sampler(), None, []
    for _ in range(16):
        token, generator = sampler.sample(logits, params, generator)
        tokens.append(int(token))
    print("SAMPLER_SEQ", index, digest(tokens))

distribution_cases = [
    SamplingParams(temperature=0.7, top_k=8),
    SamplingParams(temperature=0.7, top_k=8, top_p=0.9),
    SamplingParams(temperature=1.3, top_p=0.3),
    SamplingParams(temperature=0.5, top_k=1),
    SamplingParams(temperature=1.0, top_p=1.0),
]
for index, params in enumerate(distribution_cases):
    print("PROBS", index, digest(probabilities(logits, params).tolist()))

target, proposal = torch.tensor([0.1, 0.6, 0.3]), torch.tensor([0.7, 0.2, 0.1])
generator, rejected = torch.Generator().manual_seed(431), []
for _ in range(64):
    candidate = int(torch.multinomial(proposal, 1, generator=generator))
    rejected.append(rejection_sample(candidate, target, proposal, generator))
print("REJECTION", digest(rejected))

params = SamplingParams(
    max_tokens=10, temperature=0.7, top_k=8, top_p=0.9, seed=42, ignore_eos=True
)
prompts = [[2, 3], [7, 8, 9]]
for label, speculative in (("ENGINE_VANILLA", False), ("ENGINE_SPEC", True)):
    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config())
    config = SpeculativeConfig(3) if speculative else None
    outputs = LLM(model, speculative_config=config).generate(prompts, params)
    print(label, digest([output.token_ids for output in outputs]))

torch.manual_seed(123)
model = OuroForCausalLM(tiny_ouro_config())
greedy = [SamplingParams(max_tokens=6, ignore_eos=True)] * len(prompts)
outputs = LLM(model, speculative_config=SpeculativeConfig(3)).generate(prompts, greedy)
print("ENGINE_SPEC_GREEDY", digest([output.token_ids for output in outputs]))
```

Observed digests, identical on both trees (PyTorch 2.14.0):

| Fingerprint | Digests |
| --- | --- |
| `SAMPLER_SEQ` 0-6 | `90b44c0fdcab6576`, `140d37838f2431a3`, `3c8c3439c566e448`, `162a16fe79f387eb`, `90b44c0fdcab6576`, `0a073fcd7059c19d`, `722b2d8426397507` |
| `PROBS` 0-4 | `3143cb4b2b4b5c78`, `041065bd5ee56109`, `d39c85ffe528c39b`, `103c0c89409f2ae3`, `f3fa4dada4ca115c` |
| `REJECTION` | `8177115ae6cc98d5` |
| `ENGINE_VANILLA` | `c259f68e7af2bcf9` |
| `ENGINE_SPEC` | `a11b6338d3050bee` |
| `ENGINE_SPEC_GREEDY` | `34f83d80322576cf` |

`SAMPLER_SEQ` 0 and 4 coincide by construction: case 0 is greedy and case 4 is `top_k=1`
on the same logits, which is the equivalence `tests/test_sampler.py` asserts without
hard-coding a stream.

Mutation checks confirm the RNG lifecycle tests have teeth: adding a drop to `suspend()`
fails the preemption-continuity test, removing the drop from `release()` fails the
finish-reuse and abort-reuse tests, and rebuilding the generator per call fails the
speculative-advancement test.

```bash
# sampling surface, CPU only (identical on both validation environments)
OMP_NUM_THREADS=1 python -m pytest -m "not gpu" -q tests/test_sampler.py \
    tests/test_speculative.py tests/test_engine.py tests/test_prefix_growth.py
# this increment's affected suites
OMP_NUM_THREADS=1 python -m pytest -q tests/test_sampler.py tests/test_engine.py \
    tests/test_prefix_growth.py tests/test_async_pipeline.py tests/test_serving.py \
    tests/test_speculative.py tests/test_pd.py
# full CPU suite
OMP_NUM_THREADS=1 python -m pytest -m "not gpu" -q
OMP_NUM_THREADS=1 python -m pytest -q tests/
# lint
python -m ruff check .
python -m ruff format --check $(git ls-files '*.py')
```

The migration changes no arithmetic and adds no kernel, so no new precision or performance
evidence is required. Sampling sequences depend on `torch.multinomial`; the fingerprints
above were captured and re-run on one environment (PyTorch 2.14.0), which is what makes the
before/after comparison meaningful.

`M7 1/3` and `M7 2/3` do **not** include GPU validation. Device-side behavior under async
scheduling and CUDA streams is covered by `M7 3/3`:

```bash
python -m pytest -q tests/ -m gpu --run-gpu
```

Until then, both increments state "GPU validation pending in M7 3/3" in their pull request
descriptions.