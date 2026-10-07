# Engine / Scheduler result boundary

R1 implements the M2 boundary in [RFC #32](https://github.com/ThinkFlowLab/vllm-rlt/issues/32).
Scheduler applies request progress through one result contract; `LLMEngine`
keeps separate sync and async drivers with their existing timing.

## Ownership

| Component | Responsibility |
| --- | --- |
| `Scheduler` | Select work, advance stages, decide exits and termination, publish prefixes, truncate speculative KV |
| `LLMEngine` | Execute/submit work, collect host results, retain readbacks, apply runner cleanup, construct public outputs |
| `ModelRunner` / `SpeculativeRunner` | Run the model, manage device dependencies and sampling state |

`Scheduler.update_from_output(batch, result)` applies host progress without
runner calls or device waits. It returns `SchedulerUpdate` actions, which Engine
applies through `_apply_scheduler_update`.

## Batch identity and snapshots

`Scheduler.add_request` assigns a monotonic generation. Abort/finish followed by
ID reuse gets a new generation; preemption/resume preserves the current one.
Results apply only while `(request_id, generation)` is live.

Each nonempty batch gets a unique `seq`. `ScheduledItem` snapshots its request
identity, position, loop count and CODA output index before execution. Runner
still reads hidden, token and RNG state through `ScheduledItem.request`.
Filtering a CODA batch preserves its sequence number.

`position` is a token position. `loops_done` counts host recurrence progress:
completed loops in sync, submitted loops in async. Before RECURRENT execution
it selects the next zero-based model depth.
An exit signal's `signal_depth` counts loops completed when that score was produced.

## Runner results

The adapters in `engine/output_adapter.py` wrap existing calls and readbacks
into `worker/output.py` types without adding host reads.

| Progress | Meaning |
| --- | --- |
| `COMPLETED` | Sync execution returned; its host results are available |
| `SUBMITTED` | Async work was submitted; recurrent results may include the preceding loop's score |
| `DELIVERED` | An earlier async CODA token reached the CPU |

GPU completion remains a runner/stream dependency. It is not inferred from
`Progress`; prefill completion events pass through to prefix publication.
Results carry prefill ranges, identified exit signals, sampled host tokens or
speculative tokens with accepted counts, depending on the stage.

## Driver timing

Sync applies one result after `execute`, using `SpeculativeRunner` for the
SPECULATIVE stage:

```text
schedule → execute → adapt → update_from_output → _apply_scheduler_update
```

Async first delivers ready CODA tickets, then selects and submits work:

```text
collect CODA → schedule → submit → collect prior score → update_from_output
                                                   → _apply_scheduler_update
```

CODA delivery and submission each call the same result application entry.
Before another CODA sample, Engine forces the preceding token's CPU delivery
and filters requests that finished. PRELUDE/RECURRENT may already have run
using that preceding token on the device.

## Preserved invariants

- Full-depth, chunked prefill produces exactly one initial output token.
- Async reads delayed scores after the next core submission. With active delayed
  exits and forced depth `M ≥ 2`, sync consumes `1..M−1`; async consumes `1..M−2`.
- Async retains only scores the next submission will consume. Handles survive
  preemption; old generations are discarded before collecting their scores.
- CODA allows one outstanding sample per request. Delivery uses submission-time
  depth/output-index snapshots and checks output order and placeholder count.
- `_inflight` retains readback buffers until DMA finishes, including discarded
  scores. Finish orders runner release, prefix polling, signal removal, then
  `Scheduler.finish` before constructing the public output.
- Speculative commit observes EOS/length and truncates the uncommitted KV suffix.
  The final emitted correction/bonus token has not been forwarded yet. Runner
  owns draft statistics; Engine records committed/accepted counts.
- A failed sync step aborts its selected live requests. An async failure drains
  owned streams before aborting live requests and clearing pending work.
- Requests waiting for remote KV yield without turning RECEIVING into an error.

## Validation

Run the repository checks with `pre-commit run --all-files`. The contract tests
cover stale results, ID reuse, preemption, delayed scores, delivery corruption
and failure cleanup. CPU differential probes also compare populated KV, RNG,
tokens, exit depths and scheduling against the same upstream base.

`benchmarks/r1_runtime.py` records source hashes, controls, output histories,
timing and memory. Its profiling and synchronization diagnostics run separately
from measured timing. See [runtime configuration](cdb_runtime.md) for execution
modes and [FlashAttention/PD prerequisites](flash_attention.md) for dependencies.

CPU evidence does not qualify CUDA streams, events or graphs. Current-head GPU
correctness, controlled BF16 performance/memory/profiling, paired GSM8K10 quality
and real two/four-GPU PD transfer remain pending.
