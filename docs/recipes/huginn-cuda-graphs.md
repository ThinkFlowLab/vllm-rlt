# Huginn recurrent-core CUDA Graphs

The native Huginn adapter loads
[`tomg-group-umd/huginn-0125`](https://huggingface.co/tomg-group-umd/huginn-0125)
without executing Hub model code. It preserves the checkpoint's adjacent-pair
RoPE, sandwich RMSNorm, QK bias, linear injection, and tied embedding/head.
The runner carries the recurrent state and fixed prelude injection together.
Only the four recurrent layers use per-depth KV; prelude and coda use boundary
KV at depth zero.

Huginn currently supports synchronous execution, fixed recurrent depth, and
LAST_EXITED KV. SHARED KV, prefix caching, asynchronous execution, and speculative
decoding are rejected. CUDA Graphs capture the recurrent decode core; prelude,
coda, prefill, and sampling execute eagerly.

## Reproduce the checkpoint E2E

Use a CUDA-enabled PyTorch environment with Triton and the repository's
dependencies installed. Download the public checkpoint at revision
`bb6621b65e90b6a4b9b29ef88dc83866d450470c` into a local directory:

```bash
hf download tomg-group-umd/huginn-0125 \
  --revision bb6621b65e90b6a4b9b29ef88dc83866d450470c \
  --include '*.json' '*.safetensors' \
  --local-dir /path/to/huginn-0125

python -m benchmarks.huginn_cuda_graphs \
  --model /path/to/huginn-0125 \
  --revision bb6621b65e90b6a4b9b29ef88dc83866d450470c \
  --output /path/to/huginn-e2e.json
```

The revision argument records provenance; the benchmark reads local files.
Verify that the download used the specified revision. The measurements below
verified all four original FP32 shard sizes and SHA256 values against the pinned
official LFS metadata, plus config SHA256
`e9fe79df06a783ca33a76038c59a715b6a62f15c4a5b17b9681e697fea46c79c`.
The native loader checks every parameter name and shape and loads the shards
sequentially into BF16 GPU parameters.

The benchmark first compares eager and Graph recurrent states, logits, and all
four core-layer KV values at two positions across 32 depths. It then generates
16 tokens per request for one and four requests, using five warmups per arm and
five measured pairs per case. Each measured pair alternates execution order,
resets the initial-state RNG identically, and checks exact token IDs and exit
depths. Request IDs are reused across bursts, and every burst must reclaim all
KV blocks. Capture and fallback counters are checked for each measured arm.

| Setting | Value |
| --- | --- |
| Model dtype | BF16; original checkpoint is FP32 |
| Recurrent depth | Fixed 32 loops |
| Attention / KV | Triton / LAST_EXITED; prefix caching and preemption disabled |
| Sampling | Greedy; EOS ignored; 16 output tokens per request |
| Scheduler | Refill; max sequences 4; max batched tokens 32; prefill chunk 8 |
| KV allocation | 1,024 blocks, block size 16 |
| Prompt lengths | 5 tokens for B1; 5, 12, 14, 22 for B4 |
| Timing | CUDA-synchronized admission through finished outputs and KV reclamation |
| Excluded from timing | Model loading, Graph capture, warmups, tokenization and HTTP |

The script refuses to start another timing pair after its 100-minute budget;
it does not interrupt an active step.

## Regression checks

```bash
pre-commit run --all-files
python -m pytest -q tests/test_huginn.py -m gpu --run-gpu
```

The CPU tests use an independent dense reference for chunked prefill,
multi-request KV, strict checkpoint loading, and unsupported execution options.
The three BF16 GPU tests use head dimension 96 and check exact state/logit/KV
replay, heterogeneous requests, ID reuse, and bounded-capture eager fallback.

The checkpoint validation also compared FP32 paged execution with the
[unmodified official model at the same revision](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py),
using explicit identical initial states and 3+2 token chunks at 32 loops.
This reference comparison is separate from GPU BF16 Graph/eager parity.

## A800 results (2026-10-02)

The completed GPU campaign used source
`5a93776c2f210915d82d3715ee36d6a593f4f4ca`, based on
`ea7a8a12c1216631dcd4233ec953ac6bd9b76fca`. That base's runtime matches
upstream `299bf14b117f42a38d15852886d673f89e123307`; its extra Nanbeige
files were documentation and tests. Those additions are excluded from this PR.

The publication implementation is
`d3329305bef0b7bee9eb819b2e054e63da85c88d`, based on actual upstream
`ecb1f8b505b7e831815b40aec3b4598619cca23a`. Source
`4cfddab37e1f80c3410c1b5ada24073973a003de` also declares the repository's
test package, preventing an installed `tests` package from shadowing fixtures.
Its full pre-commit checks passed (425 CPU tests passed, 144 skipped), including
Ruff lint and format checks. The Huginn adapter and benchmark are unchanged
from the original GPU campaign. The separate A100 recheck below passed with
upstream's newer shared KV metadata path. Thor GPU admission remains pending;
the A800 numbers below belong to the original campaign.

| Validation | Completed result |
| --- | --- |
| Related CPU regressions, campaign source | 179 passed; the initial 178-pass/1-failure run and corrected fixture rerun were retained |
| Official tiny FP32 reference, 1/3/32 loops | Exact argmax; max absolute logit error 7.152557e-6 |
| Official real FP32 checkpoint, 32 loops, 3+2-token chunks | Exact five-token argmax; max absolute logit error 6.675720e-6 |
| BF16 GPU regressions | Three passed: state/logit/KV parity, request reuse, and bounded-capture fallback |
| Real BF16 checkpoint, 2 positions × 32 depths | State, logits and all four core-layer KV tensors bit-exact; 1 capture, 64 replays, 0 fallback |
| All generation warmups and measured pairs | Exact token IDs and 32-loop exit depths; all KV reclaimed; no measured capture/fallback |

Hardware: NVIDIA A800 80GB PCIe, SM80, driver 580.105.08. Runtime:
Python 3.12.3, Torch `2.12.1+cu130`, Triton `3.7.1`, Transformers `4.54.1`,
safetensors `0.6.2`. Torch/CUDA/Triton were inherited without modification;
other dependencies were task-local. The GPU queue held a shared whole-device
lock, with no other active GPU compute at admission.

Model load (5.213392 s) and capture/warmup were excluded.
Peak allocated CUDA memory was 11.832 GiB.

| Requests | Eager mean (s) | Graph mean (s) | Mean time ratio | Eager median (s) | Graph median (s) | Median time ratio | Median pair ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.931644 | 1.559152 | 1.2389× | 1.930065 | 1.557612 | 1.2391× | 1.2362× |
| 4 | 2.430247 | 2.075210 | 1.1711× | 2.429707 | 2.071351 | 1.1730× | 1.1677× |

Ratios are eager time / Graph time. A ratio below 1 means Graph is slower.

| Requests | Pair | Order | Eager (s) | Graph (s) | Ratio | Graph replays |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | eager → graph | 1.932082 | 1.557489 | 1.2405× | 480 |
| 1 | 2 | graph → eager | 1.928758 | 1.561800 | 1.2350× | 480 |
| 1 | 3 | eager → graph | 1.930065 | 1.561259 | 1.2362× | 480 |
| 1 | 4 | graph → eager | 1.925021 | 1.557612 | 1.2359× | 480 |
| 1 | 5 | eager → graph | 1.942294 | 1.557598 | 1.2470× | 480 |
| 4 | 1 | eager → graph | 2.401109 | 2.067514 | 1.1614× | 482 |
| 4 | 2 | graph → eager | 2.429707 | 2.068293 | 1.1747× | 482 |
| 4 | 3 | eager → graph | 2.410810 | 2.071351 | 1.1639× | 482 |
| 4 | 4 | graph → eager | 2.474119 | 2.083232 | 1.1876× | 482 |
| 4 | 5 | eager → graph | 2.435488 | 2.085661 | 1.1677× | 482 |

| Requests | Total captures including warmup | Total replays including warmup | Total fallbacks |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 4800 | 0 |
| 4 | 3 | 4820 | 0 |

Raw result JSON SHA256: `c69ba1361192f44c80731b252a947f406340918a73b781c3abb891785db4a0ba`.
Original logs, configs, checkpoint hashes and result JSON were retained in a
verified archive before the completed campaign's task weights were removed.

## A100 final-source recheck (2026-10-02)

The recheck used source `4cfddab37e1f80c3410c1b5ada24073973a003de`,
implementation `d3329305bef0b7bee9eb819b2e054e63da85c88d`, and actual upstream
parent `ecb1f8b505b7e831815b40aec3b4598619cca23a`. The model pin, settings,
five warmups per arm, and five alternating pairs per case match the recipe above.

| Validation | Result |
| --- | --- |
| Full pre-commit | Ruff lint/format PASS; 425 CPU tests passed, 144 skipped |
| Official tiny FP32 reference, 1/3/32 loops | Exact argmax; max absolute logit error 7.152557e-6; CUDA uninitialized |
| Official real FP32 checkpoint, 32 loops, 3+2-token chunks | Exact five-token argmax; max absolute logit error 6.198883e-6; CUDA uninitialized |
| BF16 GPU regressions | Three passed: state/logit/KV parity, heterogeneous requests/reuse, bounded-capture fallback |
| Upstream speculative fixture | One passed |
| Real BF16 checkpoint, 2 positions × 32 depths | State, logits and all four core-layer KV tensors bit-exact; 1 capture, 64 replays, 0 fallback; KV reclaimed |
| All generation warmups and measured pairs | Exact token IDs and 32-loop depths; all KV reclaimed; no measured capture/fallback |

Hardware: NVIDIA A100-SXM4-40GB, SM80, driver 595.71.05. Runtime:
Python 3.12.3, Torch `2.13.0+cu130`, Triton `3.7.1`, Transformers `4.54.1`,
safetensors `0.6.2`. Torch/CUDA/Triton were inherited read-only; other
dependencies were task-local. This Torch version and the upstream KV metadata
differ from the original A800 campaign, so the two campaigns do not isolate
the effect of hardware.

Model load (7.842400 s), capture and warmup were excluded. Peak allocated CUDA
memory was 12,704,182,784 bytes (11.832 GiB).

The whole-queue lock was held, with both GPUs reporting zero compute/memory at
admission. The preceding CPU CI finished before admission. A later read-only
NVML observer recorded 2,796 MiB of GPU1 compute outside the visible container
namespace during the later/B4 timing window; Huginn ran on GPU0. All 18 samples
from 16:53:57 through 16:54:32 UTC show that GPU1 occupancy. B1 had completed
before the observer began, so its timing window has no continuous NVML coverage.
These are observed paired timings on a shared host; physical two-GPU pool
isolation is not established. Correctness assertions passed independently of
this timing limitation. No other process was stopped.

| Requests | Eager mean (s) | Graph mean (s) | Mean time ratio | Eager median (s) | Graph median (s) | Median time ratio | Median pair ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.799333 | 2.022894 | 1.8782× | 3.805527 | 2.022825 | 1.8813× | 1.8814× |
| 4 | 4.593691 | 2.849109 | 1.6123× | 4.593514 | 2.854773 | 1.6091× | 1.6110× |

Ratios are eager time / Graph time. A ratio below 1 means Graph is slower.

| Requests | Pair | Order | Eager (s) | Graph (s) | Ratio | Graph replays |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | eager → graph | 3.786784 | 2.024100 | 1.8708× | 480 |
| 1 | 2 | graph → eager | 3.781354 | 2.022825 | 1.8693× | 480 |
| 1 | 3 | eager → graph | 3.805527 | 2.022075 | 1.8820× | 480 |
| 1 | 4 | graph → eager | 3.808277 | 2.024190 | 1.8814× | 480 |
| 1 | 5 | eager → graph | 3.814721 | 2.021281 | 1.8873× | 480 |
| 4 | 1 | eager → graph | 4.600710 | 2.836094 | 1.6222× | 482 |
| 4 | 2 | graph → eager | 4.573389 | 2.832792 | 1.6144× | 482 |
| 4 | 3 | eager → graph | 4.607586 | 2.860049 | 1.6110× | 482 |
| 4 | 4 | graph → eager | 4.593258 | 2.861840 | 1.6050× | 482 |
| 4 | 5 | eager → graph | 4.593514 | 2.854773 | 1.6091× | 482 |

| Requests | Total captures including warmup | Total replays including warmup | Total fallbacks |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 4800 | 0 |
| 4 | 3 | 4820 | 0 |

Raw result JSON SHA256: `27ad9db09d7f028e72e4e98af3daf9fde8aa159febd952119c167480666ccd5d`.
The report, original failure logs, final source manifest, model hashes, CPU/GPU
logs and timing observations were backed up with 198 individually verified files.
Archive SHA256: `0f3641daa6554fbd1cfd3b43223b19efdae7a9bd61c2fc0c6a486aa60bad707b`.
The finite queue exited naturally and its whole-queue lock was released.
Thor has the same source prepared; its healthy existing workload retains GPU
priority, and no Thor Huginn GPU result is claimed.

## Measurement scope

These are short fixed-depth correctness and performance measurements.
They do not measure task accuracy, adaptive halting, long contexts, HTTP
serving, or sustained throughput. The model's initial recurrent state is
random; paired runs reset its RNG identically.
