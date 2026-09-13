# CUDA graph A/B benchmark

Run from a source checkout; this tooling is excluded from the installed package.
The executor and its buffers remain under `vllm_lt/worker/`. Shared controllers,
metric reconstruction and numerical auditing are reused from `vllm_lt`.

| Module | Purpose |
| --- | --- |
| `runner.py` | CLI and workers, using the shared A/B controller |
| `schema.py` | CPU planning, source hashes and contract validation |
| `runtime.py` | Eager/replay adapter and profiler attribution |
| `report.py` | Offline timing, ownership and profile audits |
| `validation.py` | Numerical projection and graph-specific evidence |

```bash
python -m benchmarks.capture --help
python -m benchmarks.capture source-probe
python -m benchmarks.capture probe --help
python -m benchmarks.capture report --run-dir /absolute/path/to/results
```

`probe` freezes clean source checkouts, a prepared model, explicit physical GPU
and CPU/NUMA affinity without using CUDA. `run` and `worker` require a scheduler
reservation. Keep generated results under ignored `artifacts/` or external storage.

The single input contract in `fixtures/` defines **v2: 93 executions**: 15 model
cases with 31 numerical comparisons, plus 78 benchmark runs including 28 measured
AB/BA observations. Performance uses buckets 4/8/16 and a 1 MiB tensor budget.
Setup, feasibility, warmup and profiles stay outside timing.
Standalone kernel and synthetic lifecycle reruns from earlier milestones have
been removed; graph setup, scratch restoration, ownership, transactions, full-KV
numerical comparisons and teardown are still audited on actual model cases.
Runtime regression tests retain the failure, fallback and pool-lifetime coverage.

Historical 101-execution plans and the owned-pool contract require their recorded
source snapshots; v2 rejects them. Freeze a new plan for new experiments. See the
[graph design](../../docs/m3-cuda-graphs.md) and [historical evidence archive](../../docs/benchmarks/m3-capture.md).

Plans are trusted executable inputs: their interpreter, checkout roots and import paths control worker processes. Never accept plans from third parties. The digest proves self-consistency, not authenticity. Generate plans only from your own verified checkouts and environment. Local worker interrupts retain a manifest and cleanup evidence, then propagate to the caller.
