---
name: rlt-refactor
description: Review and refactor inference-runtime code using concrete rules for responsibility boundaries, state ownership, interfaces, asynchronous lifetimes, KV management, and maintainability. Use for code-quality reviews and behavior-preserving refactoring of vllm-rlt.
---

# Code Quality and Refactoring

Apply the following rules to the changed code and its callers. Explain each applicable finding with its location, triggering condition, impact, and correction. For review-only requests, report findings without modifying code. When implementation is requested, keep changes within scope and validate affected contracts.

Examples illustrate contracts, not mandatory APIs or complete implementations. Introduce a class, enum, helper, or abstraction only when it resolves a concrete problem. Preserve stable rule IDs as this library evolves.

## Responsibilities and State

### R01 — Separate responsibilities before splitting files

Separate policy, admission, allocation, execution, and result application where they have different responsibilities. Moving code between files without clarifying its dependencies is not enough. Prefer small, cohesive operations over speculative frameworks.

**Example:** make the scheduling path read as `choose_stage() → admit_requests() → build_batch()`. Do not hide all three responsibilities inside a renamed `SchedulerHelper.run()`.

### R02 — Give mutable state one authoritative owner

Identify who initializes, advances, resets, and releases each state. Distinguish configuration from runtime progress. Other modules should request transitions or provide feedback rather than independently mutate the owner's bookkeeping.

**Example:** a policy owns its fairness counter and no-refill phase; Scheduler owns queues and logical request progression; ModelRunner owns device tensors and execution events. These owners may use internal helpers without creating additional authoritative copies.

### R03 — Match state lifetime to the operation it describes

Keep temporary selections and exclusions local to the operation whose validity they describe. Pass the context explicitly. Check repeated calls, alternate entry points, cancellation, and request-ID reuse; do not depend on a reset performed by only one caller.

**Example:** a batch builder creates a local `selected` set and passes `excluded=frozenset(selected)` when requesting preemption. Storing it on Scheduler can accidentally protect requests selected by an earlier batch.

### R04 — Keep policy decisions separate from resource operations

Keep eligibility, ranking, victim selection, and queue transitions with scheduling. Snapshot, copy, restore, and release mechanisms should execute a defined operation and report its outcome. Agree on the transition protocol so failed resource operations do not leave inconsistent scheduling state.

**Example:** Scheduler chooses a preemption victim and coordinates its suspension. A state-preservation component saves the victim's device state; it does not independently choose a different victim or edit scheduler queues.

## Interfaces, Types, and Readability

### R05 — Use named contracts for related data

Name related values crossing module boundaries. Define units, ownership, lifetime, and mutation rights. Specify tensor shape, dtype, device, layout, and borrowing scope where relevant. Do not expose mutable internal domain objects merely for convenience.

**Example:** `AdmissionPlan(required_blocks=8, reclaimable_blocks=3)` communicates more than `(8, 3)`. An execution result should identify whether it contains gate scores, sampled tokens, or completed prefill ranges rather than requiring the caller to infer an untyped list's meaning.

### R06 — Represent finite states and distinct outcomes explicitly

Use an enum or named outcome when strings or overloaded booleans obscure valid states. Preserve external serialization at the boundary. Ordinary two-way conditions do not automatically need enums.

**Example:** use `ResumeResult.NOT_FOUND`, `BLOCKED`, and `RESTORED` for three restoration outcomes. An internal `FinishReason.LENGTH` can still serialize to the existing public string `"length"`.

### R07 — Make arguments, units, side effects, and failure outcomes explicit

Expose meaningful dependencies instead of reading unrelated mutable state implicitly. Prefer keyword-only switches where positional booleans obscure meaning. Distinguish valid zero values from failure and define inclusive/exclusive boundaries.

**Example:**

```python
available_count = ensure_active_slot(request, active_count)
if available_count is None:
    return  # Zero is a valid count, not failure.

reserved_blocks = reserved_growth_blocks(reserve_outputs=True)
```

A `frontier` for token range `[4, 8)` is the exclusive token extent `8`, not the loop depth. A planning function that updates prefix LRU state must disclose that side effect rather than appear pure.

### R08 — Make control flow read as meaningful operations

Extract cohesive steps that reveal intent and keep successful, blocked, retry, and failure paths understandable. Preserve ordering and side effects. Avoid helpers that add indirection without clarifying a responsibility.

**Example:** admission can read as queue ordering, slot availability, restoration, budget planning, and allocation/queue commit. Extracting those steps must not silently introduce another-stage fallback when the selected stage produces an empty batch.

### R09 — Name purpose, scope, and measurement units

Choose names that distinguish quantities and lifecycle roles. Separate supplied input from derived usable extent. Preserve externally defined model fields unless changing them is an intentional compatibility decision.

**Example:** prefer `_pending_exit_signals` to `_signals`, `request_id` to `rid`, and `watermark_ratio` / `watermark_blocks` to one ambiguous `watermark`. Keep `publishable_tokens` separate from the supplied prefilled extent.

### R10 — Keep dependencies visible and initialization invariants intact

Initialize required components through their defined lifecycle. Do not weaken production invariants solely to support tests that bypass constructors. Keep ordinary imports visible; use lazy imports when justified by optional dependencies or initialization constraints.

**Example:** if every initialized engine owns a preemption manager, call `self.preemption.discard_snapshot(request_id)` directly. A test using `object.__new__` should supply that required component rather than require production `hasattr` guards.

## Boundaries and Capabilities

### R11 — Replace private-state access with an owner-defined operation

Use narrow public operations or read-only views to cross component boundaries. Define borrowing and mutation rights. A getter exposing unrestricted mutable internals does not resolve ownership. Let the responsible module define its contract rather than creating a competing interface in each caller.

**Example:** call `preemption.discard_snapshot(request_id)` instead of `preemption.snapshots.pop(request_id, None)`. Serving should call engine request/statistics methods instead of requiring fake scheduler/cache objects on a PD facade.

### R12 — Separate logical KV resources from device operations

Logical KV management owns allocation, references, block mappings, depth validity, leases, and release conditions. Execution owns tensor storage, writes, copies, and buffer/event lifetimes. Attention owns its capability and metadata semantics. Cross-boundary operations need explicit completion feedback.

**Example:** a scheduler asks for the resource cost of growth without inspecting private reference counts. An early-exit finalization plan describes affected depths and positions; execution performs the copies, and logical validity advances only after the required completion evidence.

### R13 — Declare capabilities instead of inferring them from versions

Let each backend or adapter declare the operations it supports. Route through those capabilities. A version identity can inform the adapter's implementation but should not substitute for the caller-facing contract.

**Example:**

```python
# Coupled to one implementation's version numbering.
packed_prefill = backend.generation == 4

# Describes the operation the caller actually requires.
packed_prefill = backend.capabilities.supports_packed_prefill
```

### R14 — Select backends against actual execution requirements

Automatic selection should consider hardware, available implementations, dtype, head dimension, block size, and required execution capabilities. Keep explicit selection available, fail clearly on incompatibility, and report the selected implementation and reason.

**Example:** an installed package is not enough to establish CUDA Graph or packed-prefill support. An `auto` request may select a compatible alternative with a diagnostic; an explicit incompatible choice should not silently switch to another backend.

### R15 — Share contracts without forcing identical execution timing

Share scheduling, execution, and result-application semantics where they agree. Preserve differences between submission, device completion, and output delivery. Share recurrent computation while keeping prefill/decode state handling explicit at suitable boundaries.

**Example:** sync and async runners can consume the same task/result contracts while async execution retains multiple in-flight submissions. Requiring both to wait and return at identical points can destroy overlap. A local executor boundary does not, by itself, justify adding generic RPC or unused distributed implementations.

## Lifecycle and Runtime Semantics

### R16 — Centralize cleanup without losing release ordering

Share cleanup through owner-defined operations across completion, abort, cancellation, and failure. Distinguish immediately releasable state from resources still referenced by asynchronous work. Preserve distinct termination outcomes and required ordering.

**Example:** abort can delegate common queue/accounting cleanup to `finish(reason=ABORT)`. Recycling a runner slot still waits for the event that covers its last use; consolidating cleanup must not bypass that dependency.

### R17 — Reject stale results using execution identity

Associate results with the generation or execution that produced them. Validate that identity before updating current state. An external request ID can be reused and is not sufficient on its own.

**Example:**

```python
if requests.get(result.request_id) is not submitted_request:
    return  # Cancelled request or a different request reusing the ID.
```

Object identity is appropriate only within a shared process. Cross-process results need an equivalent generation/sequence contract.

### R18 — Distinguish submission, completion, validity, and success

Define the evidence required for each lifecycle transition. A submitted operation is not completed; completed device work does not prove every required range is valid; finished processing does not necessarily mean success.

**Examples:**

- Prefix publication needs completed writes and contiguous coverage at every required layer/depth. If only KV is cached, the final prompt token may need recomputation to recover its hidden state.
- A PD commit marks the end of submission. Destination activation also requires receiving the expected data and satisfying execution dependencies.
- An artifact job can be complete with an error. Delete recoverable raw files only after successful output verification and publication.

### R19 — Preserve loop, exit, restoration, and sampling semantics

Keep token position distinct from loop depth. Preserve gate-score production/consumption timing, cumulative exit state, depth-aware KV validity, and subsequent-token behavior. Restoration must preserve relevant KV, hidden state, logical progress, exit state, and RNG state.

**Example:** moving sampling into a separate component must preserve request-local generator creation and advancement; resuming a request must not reseed it. Recomputing at full depth is not automatically equivalent to restoring a request that previously exited early.

### R20 — Specify handoff and failure protocols across workers

Define valid KV ranges, destination mappings, final prompt hidden state, generation/chunk identities, commit, receive completion, activation, and acknowledgment. Specify how remote writes terminate before memory can be reused after cancellation, timeout, or worker failure.

Preserve interactions with prefix hits, in-flight transfers, and preemption. Do not alter transfer direction or chunk timing merely to fit a new interface.

**Example:** the P worker sends the required KV and final prompt hidden state; D performs coda and sampling. Cancelling D cannot immediately release destination pages while P may still write to them.

### R21 — Preserve device residency and resource/thread ownership

Do not add CPU round trips, `.item()`, synchronization, or repeated metadata allocation merely to simplify an interface. Keep buffers, streams, events, graphs, and asynchronous state under coherent lifetime contracts. Offload only operations that are safe on the destination thread.

**Example:** keep sampled token IDs on the GPU when the next prelude consumes them there. A profiler may need finalization/export on its owner thread while parsing and compressing already exported files runs in the background; wrapping everything in a future does not remove that constraint.

## Change Discipline and Validation

### R22 — Make behavioral fixes explicit within refactoring work

Separate structural changes from intentional behavior changes in the explanation and validation. Describe the old trigger, old outcome, new behavior, and evidence. They may share a change when appropriate, but a behavior fix must not masquerade as a pure extraction.

**Example:** changing fairness accounting so empty batches no longer consume an allowance is a behavior fix. Specify that the counter tracks nonempty scheduled batches, not GPU completions, and exercise both empty and nonempty cases.

### R23 — Migrate incrementally and preserve explicit compatibility contracts

Keep increments runnable. Use temporary adapters only where needed, migrate callers, and define when old paths can be removed. Avoid maintaining two authoritative state stores. Distinguish Python keywords, CLI flags, output serialization, and defaults when assessing compatibility.

**Example:** renaming a Python keyword from `watermark` to `watermark_ratio` requires caller migration even if `--kv-watermark` stays unchanged. Renaming `publish_prefix(rid=...)` changes keyword callers even when positional calls still work.

### R24 — Explain rationale, contracts, and blocked paths

Document why ordering and lifecycle constraints exist, what state they protect, and what happens when progress is blocked. Use worked examples for budgets/layouts and diagrams where timing is otherwise hard to follow. Keep comments tied to actual invariants rather than narrating obvious syntax.

Update usage documentation when interfaces change. Place detailed evidence where it helps review; do not automatically add a permanent walkthrough or benchmark directory for every extraction.

**Example:** explain that the final prompt token is excluded from prefix reuse because its hidden state is needed for coda. Show a concrete admission-budget calculation that distinguishes physical blocks, reserved growth, and watermark headroom.

### R25 — Validate affected invariants and state the limits of evidence

Validate the changed contract and its callers, including applicable blocked, cancellation, late-result, ID-reuse, failure, and restoration paths. Reuse existing checks where suitable. For a behavioral fix, reproduce the old failure where practical; do not write tests that merely mirror helper structure.

Cover affected exit policies, KV layouts, sampling paths, and feature combinations. Use justified numerical tolerances; do not assume cross-batch bitwise equivalence. For hot-path changes, compare performance under matched conditions and report the measured tradeoff.

Tie evidence to the tested revision and environment. Mocks, CPU tests, skipped GPU tests, and results from an earlier revision do not establish current device behavior or performance. Mark unverified paths explicitly.

**Example:** test that cancelled work cannot mutate a new request with the same ID, not merely that a helper was called. A CPU test of preemption logic does not establish safe CUDA stream ordering. Documentation-only changes do not require GPU benchmarks.
