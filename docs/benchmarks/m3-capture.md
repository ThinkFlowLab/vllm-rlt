# M3 CUDA graph evidence

Status: draft; current source is not device-qualified. Historical evidence below is bound to its recorded source.

## Latest device attempt — 2026-09-12

Measured performance gates passed; original experiment incomplete after profiler interruption.

Source `af3327d5a30219bcc92c57deb43a10cc7be3040f`; plan `7bf363e4611bafd1a26c1c3fb051e4e7b60f16273f6e638ce14320013c4a7e15`. Model ByteDance/Ouro-1.4B at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. FP32, Triton, GPU 2, NVIDIA L20X, CPU cores 56–63, memory node 1.

C: compact eager. E: padded eager. G: the same padded executor with graph replay. G/E isolates replay; G/C includes padding, metadata, staging, actual gate readback, routing and output publication.

Each range contains two measured observations, not a confidence interval. Model loading, cache/executor setup, warmups, profiles and cleanup are excluded from delivery throughput and recorded separately. Synchronized throughput, every request’s TTFT/TPOT/completion latency and paired memory deltas are retained in report.json.

| Workload | C tok/s | E tok/s | G tok/s | G/C paired ratio | G/E paired ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 request, 128 input / 64 output | 19.77–20.65 | 20.04–21.16 | 42.06–42.12 | 2.04–2.13× | 1.99–2.10× |
| 4 requests, 128 / 64 each | 73.35–76.01 | 75.53–78.97 | 106.21–106.70 | 1.40–1.45× | 1.34–1.41× |
| 8 mixed-depth requests, 64–128 / 8–22 | 99.72–105.49 | 96.75–103.45 | 129.07–133.01 | 1.22–1.33× | 1.29–1.33× |
| 8 requests, 128 / 64 each | 135.50–143.83 | 123.62–133.53 | 176.75–177.74 | 1.24–1.30× | 1.32–1.44× |
| 1 request, 512 / 32 | 17.40–18.37 | 17.68–18.62 | 17.60–18.06 | 0.96–1.04× | 0.97–1.00× |

| Workload | C/G TPOT ms | C/G mean TTFT ms | G replay/fallback calls per run | Inactive physical rows | Paired extra peak reserved MiB |
| --- | --- | --- | --- | ---: | ---: |
| single | 48.25–50.44 / 23.20–23.24 | 59.30–59.54 / 57.35–58.17 | 252/0, 252/0 | 75.00–75.00% | 36.00–36.00 |
| batch4 | 50.87–52.92 / 35.81–35.94 | 156.17–163.02 / 143.04–146.35 | 252/0, 252/0 | 50.00–50.00% | 36.00–36.00 |
| mixed8 | 45.74–49.01 / 36.32–36.83 | 232.26–240.52 / 229.51–252.11 | 63/0, 63/0 | 56.18–56.18% | 36.00–36.00 |
| batch8 | 52.35–55.50 / 41.44–41.74 | 261.74–282.31 / 267.07–269.84 | 252/0, 252/0 | 50.00–50.00% | 36.00–36.00 |
| prefill512 | 48.12–51.17 / 48.95–50.41 | 249.92–252.71 / 254.93–255.33 | 0/124, 0/124 | compact fallback | 36.00–36.00 |

The 512-input control requires 34 table columns and takes compact fallback; the other four workloads replay. The default scheduler’s eight live requests now use physical bucket 16. All input copies, gate readbacks, CPU routing and fallbacks remain inside measured delivery time.

| Workload | C setup ms | G setup ms | Estimated setup break-even output tokens, two pairs |
| --- | ---: | ---: | --- |
| single | 0.89–0.96 | 222.73–223.34 | 9, 10 |
| batch4 | 0.97–1.05 | 219.54–317.81 | 76, 58 |
| mixed8 | 0.93–0.99 | 229.92–431.67 | 172, 133 |
| batch8 | 0.89–2.00 | 229.17–234.73 | 132, 177 |
| prefill512 | 0.89–0.91 | 238.88–263.29 | 115, no break-even |

Setup estimates use the excluded setup overhead and observed throughput difference; they are workload-specific projections, not measured resident-server amortization. All these engines are fresh per execution; model loading is once per worker, and allocator caches remain intact between executions.

The separate diagnostic CPU plan predicts buckets 4/8/16. The separately planned matched diagnostic profiles complete their 14-output window. G records 9 correlated graph launches and 11031 linked GPU kernels across all three buckets. E records zero graph launches. These are diagnostic traces, excluded from throughput.

All 15 feasibility and 30 measured runs completed, with matching token/depth histories and work counts; all requests released KV. The timing plan did not complete its original diagnostic profiles. Feasibility checked finite loop/gate/logit outputs, actual coda argmax histories and first/last coda logits (atol 0.001, rtol 0.0001). Mixed8 is a shortened eight-request derivative of W4 using forced trace prefixes, not the original W4 qualification case or a live adaptive-quality benchmark.

Predeclared gates: both single-request G/C ratios ≥1.10, single TTFT ratio ≤1.05, nonoverlapping single-request throughput ranges, all four controls’ G/C ratios ≥0.95, and paired peak allocated/reserved increases ≤512 MiB.

- single: compact_throughput: PASS, memory_vs_compact: PASS, ttft: PASS, range_separation: PASS
- batch4: compact_throughput: PASS, memory_vs_compact: PASS
- mixed8: compact_throughput: PASS, memory_vs_compact: PASS
- batch8: compact_throughput: PASS, memory_vs_compact: PASS
- prefill512: compact_throughput: PASS, memory_vs_compact: PASS

The planned 79-run experiment attempted 77 executions: 15 feasibility, 30 measurements and 31 warmups completed; the first profile was interrupted after exceeding its 180-second budget in PyTorch CPU aggregation. The remaining graph-profile worker was not started. Its interrupted predecessor retained traceback-owned tensors during its cleanup check, then exited; scheduler status showed GPU 2 free at zero memory. Nine ordinary workers completed cleanly. A separate four-run diagnostic (two feasibility, two profiles) completed all four executions with both workers reaching zero memory, but its 588,043,520 bytes of artifacts exceeded the 536,870,912-byte cap (560.8 versus 512 MiB). Trace attribution is valid evidence; the diagnostic failed its resource gate. It adds no measured throughput samples and does not turn the original experiment into a complete one. No further GPU run was added.

This screen does not qualify the complete M3 milestone, BF16, quality, arbitrary context lengths or increasing concurrent-engine count. The original benchmark launch stopped at source-manifest validation before model loading because GPU pytest had written cache files into that archive. A second attempt passed 15 feasibility runs, then stopped before its first measured worker because eight packages appeared in the shared environment. The timing attempt uses a task-owned copy of that prepared environment and a clean source archive; its CPU-predicted 112-output profile still exceeded the processing budget described above. Neither stopped attempt contributes measured samples; both remain preserved.

Raw evidence: `/tmp/hsliu2-vllm-lt-graph-review-ab-v3-20260912/results`. Plan, commands, environment, all results and the original audit script are retained in the external evidence bundle.


## Attempt log

Earlier details are preserved as committed blobs and in `/home/hsliu2/tmp/vllm-lt-pr15-review-cleanup/historical-records.tar.gz` (SHA-256 `ae1b484e57770b5410d217c038d16fe135db7a43998e9e16821a89ce4f9a5527`); all four documents were read back and checked byte-for-byte. Raw device evidence remains in the archives linked by those documents. No samples are pooled across attempts.

| Date / attempt | Source SHA | Outcome and cause | Detail and checksum pointer |
| --- | --- | --- | --- |
| September 11 / stopped | `0d9284a849c3ecdc2bbb5fbc4103729c98ce998d` | Failed memory gate (532 versus 512 MiB); incomplete after controller I/O error | [Original record](https://github.com/hsliuustc0106/vllm-lt/blob/8f3590f4248752cc44690569c7a4e0edd323ba56/docs/benchmarks/m3-capture-stopped-20260911.md), including original and reassessment SHA-256 |
| September 11 / owned pools | `0e4516a011ccea2e96900d2092dbe92573b3fd4a` | Inconclusive: W4 profile contained only fallback; fresh setup streams retained 34 MiB per cycle | [Original record](https://github.com/hsliuustc0106/vllm-lt/blob/8f3590f4248752cc44690569c7a4e0edd323ba56/docs/benchmarks/m3-capture-owned-20260911.md), including report hashes and published evidence release |
| September 12 / post-review | `af3327d5a30219bcc92c57deb43a10cc7be3040f` | Measured gates passed; profiler processing and separate diagnostic artifact caps failed | Latest attempt above; full evidence archive checksum below |

Generated JSON, logs and archived driver copies are kept outside the repository.
The original published summaries and plans are preserved under
`/home/hsliu2/tmp/vllm-lt-graph-review-20260912/published-evidence/`.
They include lifecycle byte counts and output histories, frozen plans, audited
timing/latency metrics, and every stopped-attempt record. No failed attempt is
relabeled as complete. The maintained benchmark and validation code lives in
[`benchmarks/capture/`](../../benchmarks/capture/README.md).

Full local evidence archive: `/home/hsliu2/tmp/vllm-lt-graph-review-20260912/evidence.tar.gz`.
It contains both source snapshots, all attempts, profiles, raw results, logs,
plans, diagnostics and the archive manifest. Checkpoint weights and virtual
environment packages are excluded. The source snapshots and stopped attempts
remain separately named. Raw traces have not been uploaded to GitHub.

SHA-256: `70309cc90bbd55db79c211dc16916f6e5e6eaba3f139f883afb3194e7f5ccf30`. Archive size: 156,479,663 bytes.
All 846 payload files were read back from the archive and checked
against their recorded sizes and SHA-256 hashes. Compression does not change the
short-profile experiment's failed **raw** artifact-budget decision.

## Lifecycle and live-gate validation

At the implementation snapshot above, one feasibility cycle and 16 measured
create/close cycles reused one resident model/cache without dropping caches.
Post-close allocated memory stayed at 12,181,062,656 bytes and reserved memory
at 12,218,007,552 bytes; no graph-private segments survived close. This is a
fixed-concurrency plateau. The controlled predecessor diagnostic isolated the
34 MiB/cycle growth to inactive default-pool allocations on fresh setup streams;
reusing a safely released stream held reservation flat.

Live-gate checks at exit threshold 0.5 ran two eight-request waves per compact
and graph configuration. Tokens and depths (3 and 4) agreed, returned tensors
were finite, and graph waves reached buckets 4/8/16. Closing a second concurrent
executor with its own pool and stream did not affect the resident executor's
second wave. Lifecycle and diagnostic workers reached zero terminal allocation
and reservation.

Historical [owner-review notes](https://github.com/hsliuustc0106/vllm-lt/blob/27f9e80c998a9658c6ffbad28431e8a1d211a8cc/docs/reviews/pr15-20260912.md)
and [compatibility-review notes](https://github.com/hsliuustc0106/vllm-lt/blob/27f9e80c998a9658c6ffbad28431e8a1d211a8cc/docs/reviews/pr15.md)
remain available at their committed versions. The latter records Python 3.10
exception-note guards tested under Python 3.12, not a native Python 3.10 run.
The current [runtime design](../m3-cuda-graphs.md) owns the implementation
contract; these historical results do not qualify later source changes.
