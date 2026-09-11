# Resolved M4 attention experiment — 2026-09-11

**Pre-run contract; no M4 GPU result is claimed.** This freezes one candidate and one finite experiment for issue #7. The [design](m4-attention-design.md) defines the selected scope, numerical policies and acceptance gates. No retry, additional tile, replacement execution or extra measured/profile sample is allowed.

A uses the accepted compact synchronous attention tile 32; B changes only that launch constant to 64, with four warps unchanged. Both sources contain the same qualification harness. Private persistent execution is disabled in benchmarks; separate held correctness visits cover that existing path. M3 graph capture is not adopted.

## Frozen identity

| Input | Frozen identity |
| --- | --- |
| A source | `55f93ce51eaa4e0f490d32fc455d0bbbf6ff3851` at `/data/hsliu2/tmp/vllm-lt-m4-attention-a` |
| B source | `576c468d79d971d3585d7cf85d159bb763a5b543` at `/data/hsliu2/tmp/vllm-lt-m4-attention-b` |
| Common harness | `fa0de547af7ca09c9160c0f72333afc6167bc481504d6ebf586bf5eb2d657ced` |
| Canonical plan | `5c0e9a4b3e2d3546d849cf3cd4cc941353f249de494f671718a9a148cce0825b` |
| Exact plan file | `11baf97a9d1b9a9618b189dec150d8f51e45153890bab023b3d13e19b01ccd77`; 867,339 bytes |
| Numerical plan | `89f2fcd48e595f0c53de7d7e64a5220eb705fe9286065b3cde146574a89c2e2d` |
| Held plan | `7ef99c36a756b7e584ab77cb3a254452670ea3296e8b39cc8a0bbf637e3b2662` |
| Host record | `ff711f2126a14a4043292a97a58d1bf8fe816725eb684ff931dd251e096d18e8` at 2026-09-10T23:35:22.280161+00:00 |

Plan path: `/data/hsliu2/tmp/vllm-lt-m4-attention/artifacts/m4-plan-20260911/plan.json`. Full source manifests, imports, dependency versions, checkpoint/config/tokenizer byte hashes, fixtures and arithmetic flags are embedded in the plan. The only production-file difference is `vllm_lt/kernels/triton_attention.py`; its baseline SHA is `0861c57c25d7e5b284cfe4c259ae324a2a420a42c72fa3d3124d0fd13e63098e`. Both execution checkouts must remain clean and detached throughout the attempt.

The real checkpoint is ByteDance/Ouro-1.4B at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. Use the existing prepared files/environment; no model download or environment rebuild is included. All inference is FP32, fixed four loops or the explicitly frozen Q1 live/forced policy. BF16 remains unqualified.

Full pre-freeze CPU validation: **770 passed, 17 skipped in 215.53 seconds**; all 135 source files remained unchanged during the test. Log SHA-256 `7a405ffebdc91f21fa14ce798d474db83552b66ece78b7a8569463c1682a97f1`. Ruff check/format and diff checks passed. After testing, one documentation sentence dropped an approximate line count; no executable or test source changed before the candidate commit. The A control then restored only the accepted attention literal.

## Controls and finite budget

Host `dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw`, account `hsliu2`. Physical GPU 0: `0, NVIDIA L20X, GPU-006eb78c-23cb-f37d-eb7c-0ccb578b8f11, 0 MiB`. CPU IDs 56–63, memory node 1, OMP/intra-op/inter-op threads 1, seed 0. Actual visibility is assigned by the scheduler and must remain physical 0 / logical cuda:0. The same device UUID and controls apply to every worker. Refresh all four scheduler status views immediately before launch; the host selection record is not a reservation.

Interpreter `/home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python`. Arithmetic, dependency and runtime-variable identities are frozen in the plan; TF32 and reduced-precision reductions are disabled. Sources/models remain on their prepared paths. Raw output is `/tmp/hsliu2-vllm-lt-m4-attention-20260911/run`, under the verified local overlay filesystem; this applies equally to both sides.

Exactly 139 rows/eight workers/eight model loads: N-A40, N-B35, A1/B1/B2/A2 each14, P-A/P-B each4. Counts are 27 numerical cases/51 streams, 30 direct kernels, four lifecycle cases, 14 benchmark feasibility, 28 warmups, 28 measured, four diagnostic warmups and four profiles. A feasibility/correctness finishes before B; both raw prerequisite audits pass before A1. N-A and N-B begin inference with W1 feasibility; the four timing workers begin with W1 warmup, and profile workers with W1 diagnostic warmup. Downloads, preparation, warmup and profiles are excluded from measured comparisons.

Each whole case has a 600-second watchdog and the controller 7,200 seconds, including preparation/export/cleanup. Scheduler timeout is 130 minutes to allow owned-process termination; it does not expand the 7,200-second experiment. GPU work occurs only through the exact reserved command below. Each worker must release its model/workspace and report zero final allocated/reserved CUDA memory before its successor. Clean up only this task’s resources.

| Artifact component | Upper bound, bytes |
| --- | ---: |
| comparison_json_bytes | 498,677,760 |
| dump_index_bytes | 498,677,760 |
| dump_tensor_bytes | 268,435,456 |
| held_artifact_bytes | 327,155,712 |
| other_artifact_bytes | 1,073,741,824 |
| profile_trace_bytes | 4,294,967,296 |
| retained_index_bytes | 107,943,936 |
| retained_tensor_bytes | 4,860,873,504 |
| Total conservative bound | 11,930,473,248 |
| Hard cap | 12,884,901,888 |

Numerical limits also retain 8 GiB cumulative tensor writes, 2 GiB per group, 256 MiB diagnostic dumps and 64 MiB transient matching. Held evidence is capped at 312 MiB; each profile at 2 GiB and all profiles at 4 GiB; auxiliary artifacts at 1 GiB. Any cap violation ends the attempt and preserves its prefix.

## 验收标准 / acceptance

Require Q1 FP32 final-logit atol=0.001/rtol=0.0001, exact actual top-1/token/exit histories and finite layer/gate/KV observations. Unchanged Torch A/B comparisons are exact; Triton A/B intermediate deltas are diagnostic. Held FP64 dense attention uses atol=rtol=2e-5, while same-tile compact/physical active outputs, inactive zeros and whole-pool guards are exact. All 34 held rows and their ownership/callback lifetimes must pass.

For every measured pair: W1 throughput B/A ≥1.01; each of six control cells ≥0.95; both peak allocated and reserved B−A ≤64 MiB; engine setup B−A ≤100 ms. W1 additionally requires min(B)>max(A). Report all four values and their observed ranges; two observations do not establish a confidence interval. Finite gate probabilities are counted and retained; their float deltas are diagnostic, while actual token/depth/work histories remain exact.

Both W1 and W4-refill require matching actual compact-prepared work windows and CUDA kernel/copy attribution: W1 steps2–97 with64 recurrent traversals; W4 steps2–35 with640 interleaved prefill tokens and7 decode traversals (27 recurrent-body traversals including prefill; 648 physical-layer invocations). Each window includes16 subsequent outputs. CPU user annotations and GPU annotations are separate categories. Generic kernel names alone cannot prove a tile; frozen source/import identities bind the launch policy.

AC-M4-04/05/06 are N/A for this selected attention-only PR, with profiling/scope reasons in the design. The fixed 6-GiB benchmark pool, request-reserved/populated pages, logical copy bytes and completed work remain separately reported; no admission-capacity or physical-memory saving is claimed. Device memory-access checkers are unavailable, so there are zero sanitizer executions.

Corrupt evidence is invalid/unqualified; independently verified required failures remain failures even when later coverage is missing. Missing-only evidence or unresolved variation is inconclusive. No outcome authorizes additional trials under this contract. Preserve all raw inputs, failures, commands and output bytes; publish compact results and a separately checksummed evidence bundle after the attempt.

## Commands and sequencing

The source snapshots and CUDA-blocked CPU probe precede this generated document. Review and DCO-commit this document before the first device execution. The probe destination already exists afterward and must not be overwritten. The commands below document the exact preparation and subsequent finite run/report invocation.

```bash
cd /data/hsliu2/tmp/vllm-lt-m4-attention-b
numactl --physcpubind=56-63 --membind=1 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-m4-attention-b /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-m4-attention/artifacts/m4_cpu_cli.py probe --baseline-root /data/hsliu2/tmp/vllm-lt-m4-attention-a --candidate-root /data/hsliu2/tmp/vllm-lt-m4-attention-b --contract /data/hsliu2/tmp/vllm-lt-m4-attention-b/benchmarks/fixtures/ouro-m4-attention-contract.json --model-path /data/hsliu2/tmp/vllm-lt-models/ouro-1.4b --gpu-id 0 --output /data/hsliu2/tmp/vllm-lt-m4-attention/artifacts/m4-plan-20260911
gpu run --gpu-ids 0 --nonblock --timeout 130m --note 'M4 attention 576c468d79d971d3585d7cf85d159bb763a5b543' -- numactl --physcpubind=56-63 --membind=1 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-m4-attention-b /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python -m vllm_lt.benchmarks.m4_attention run --plan /data/hsliu2/tmp/vllm-lt-m4-attention/artifacts/m4-plan-20260911/plan.json --output /tmp/hsliu2-vllm-lt-m4-attention-20260911/run
numactl --physcpubind=56-63 --membind=1 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-m4-attention-b /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-m4-attention/artifacts/m4_cpu_cli.py report --run-dir /tmp/hsliu2-vllm-lt-m4-attention-20260911/run
```

Producer SHA-256: `1eb7b9a488bbc3881a854ae9d45c72920b1dd0017f05a59b87601de4a11a4df2`. CPU-only wrapper: `077f05a7cb7fcefd3b40da8d9259d8faa055876b1f959ee370f6006fb6d895bb`. CPU checks and independent preparation reviews are retained under the preparation root’s `artifacts/`; none qualifies device performance.
