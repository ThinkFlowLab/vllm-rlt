# Core and user-interface review

For changes to core behavior in scheduling, engine/request ownership, KV,
model execution or kernels, or to user-facing interfaces (Python API, HTTP API,
CLI, configuration, outputs or an interactive UI), explicitly assess the design
choice before the implementation details. Keep the explanation proportional
to the change:

- **Why change:** name the concrete failure, measured bottleneck or required
  capability. Cite the reproducer, profile or requirement; distinguish evidence
  from a hypothesis. Explain why the existing design cannot adequately handle it
  and why the affected core module or public interface must change instead of
  solving the problem within the existing contract.
- **Chosen approach:** connect the proposed mechanism to that problem and state
  the invariants, complexity and compatibility costs it introduces.
- **Plan B:** compare at least one credible alternative, such as a narrower fix,
  reuse of an existing mechanism, or retaining the current implementation while
  gathering evidence. Explain its correctness, performance and maintenance
  tradeoffs, why it was not selected, and what evidence would favor it instead.
  If no viable alternative is apparent, explain the constraints; do not invent one.
  Discuss an existing fallback or rollback where relevant, without requiring a
  second implementation merely to demonstrate an alternative.

Use the PR's rationale when supported, and label reviewer-proposed alternatives
as such. Missing rationale or unresolved tradeoffs are design questions, not
automatically proven bugs. Surface them prominently when they affect whether
the core or interface change is justified.

For user-facing changes, show a concrete before/after call, command or user flow.
Check existing callers/clients, defaults, accepted inputs, response/output schemas,
errors and streaming behavior where affected. Identify breaking changes and the
migration or compatibility path; verify help text, documentation and examples
match the implementation. Compare a compatible extension or opt-in behavior as
Plan B when viable, and explain any added complexity. For an interactive UI,
also check the affected navigation, loading/error states and accessibility.
Use a targeted client test or interaction check for the changed contract rather
than treating internal unit tests alone as proof of compatibility.

## Follow the changed behavior

Read the changed code with its callers, ownership and failure paths. Use the
reviewed checkout's `docs/design.md`, `docs/serving.md`, `docs/accuracy.md` and
`docs/benchmarks.md` where relevant. Treat archived reports as dated evidence.
If `docs/ab-tests.md` and `benchmarks/ab.py` exist in the reviewed snapshot,
inspect their actual contract and validators; do not assume a local proposed
workflow has merged. Resolve stale documentation against code and explicit
current requirements rather than treating every historical statement as a gate.

| Changed area | Review focus |
| --- | --- |
| `request.py`, `core/scheduler.py`, `engine/llm_engine.py` | Stage/progress ownership, one in-flight decode token per request, full-depth chunked prefill, exactly one first output from final prefill, refill/cohort routing, bounded progress, admission and cancellation. |
| `core/kv_cache_manager.py`, attention kernels | Per-request/depth block tables and causal lengths; populated versus merely reserved KV; partial pages, mixed depths, skipped-depth propagation and reuse. |
| `models/`, `worker/model_runner.py`, sampling | Shared-core recurrence and normalization, position/RoPE unchanged across a token's loops, cumulative gate updates even before minimum exit depth, forced maximum-depth exit, coda/logit selection and request-local RNG progression. |
| `serving/`, entrypoints | Worker ownership, real readiness, queue/admission bounds, disconnects, timeout/error propagation, streaming token/usage accounting and shutdown. |
| `benchmarks/`, `vllm_rlt/benchmarks/`, result documentation | Frozen controls, source/protocol provenance, complete coverage, timing/accounting, paired gates and honest handling of failed evidence. |

Specific invariants to trace when touched:

- Last-exited KV copies each layer's final computed token K/V to skipped deeper
  depths, preserving shallower state and neighboring tokens. A whole-page alias
  is not equivalent when adjacent tokens exited at different depths.
- Reservation covers executable positions, excluding the last sampled token
  when it has no forward pass. Full-depth prefill still needs all model depths
  even when decode has a smaller loop limit. Incremental allocation needs an
  explicit progress/ownership argument; release must wait for dependent work.
- Inactive/padded rows must not read uninitialized KV or mutate live KV, RNG,
  outputs or hidden state. Persistent buffers/graphs additionally need stable
  addresses, slot lifetime, synchronization, valid fallback and bounded memory.
- Streaming counts actual generated IDs, including empty-text token events;
  usage and termination markers are not additional tokens. An interrupted
  stream must not be accepted as a successful short response by the pinned
  benchmark client. Slow-client/error cleanup must preserve other requests.
- Readiness means model, tokenizer and engine are initialized. Validate the
  first real request; compilation/warmup and process-to-readiness are separate
  measurements. Refill alone does not imply asynchronous host/device overlap.

## Report the design assessment

Open the review with a short assessment of why the change is needed and its
Plan B, including any unresolved design question. Then give actionable findings
in priority order and the validation/snapshot note. Keep the design assessment
proportional to the change; do not turn a missing explanation into a fabricated
correctness defect.
