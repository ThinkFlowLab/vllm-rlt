# Q2 FP32 quality — resolved pre-run contract

**No quality generation has run and no accuracy or paired-quality claim is made.** This is the F32-only sub-deliverable of [Q2 #8](https://github.com/hsliuustc0106/vllm-lt/issues/8), implemented in PR18. See [design](q2-fp32-quality-design.md) for exact semantics and acceptance. BF16/adaptive comparisons remain blocked by Q1, so their A/B gate is unexecuted, not passed.

## Frozen identities

| Input | Identity |
| --- | --- |
| Execution source | `6229db722bae99d6a6ad2886676f376a0521735a`, clean133 tracked files, `/data/hsliu2/tmp/vllm-lt-q2-quality-exec` |
| Canonical plan | `932d6c454685941da7b5f8c1cc1e29e8ba1d30fe5de888a9fbc54e150d666547` |
| Exact plan file | `151e82f4306c1fa251b8ed81e2c9d0a5f2c31a086e3c44a165424df865af8378`,1672366 bytes, `/data/hsliu2/tmp/vllm-lt-q2-quality/artifacts/q2-quality-plan-20260911/plan.json` |
| Controls | `56727758e3b82da55e91c1faf89ded1ce1b07d783f11feddaa5bb90899521374` |
| Fixed contract | `6efc37eadf80526901afb75e9b76597c306d97cc08aefe5e68aa4cac99d968dd` |
| Selection | `9664c9bfdee1b48111771784b7710058efbda2e8f658741e1ebd600c4944e70b`; exact selected IDs and prompt IDs embedded in plan |
| Normalized dataset | `756fb196608685c6eaaf95578499989c23ce847fe25419379f0d462ef55cb45f` |
| Q1 original qualification | `e1b77c625962aa21b9a68f89dfe29ced9c20def0844fa2f6737043b2e539586f` |

The source retains accepted PR14 production bytes. It uses real `ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1` weights, compact FP32 Triton tile32, fixed four loops and natural EOS. No graph, persistent-table opt-in, M4 tile64 or numerical fix is included. Checkpoint/tokenizer/import/dependency hashes and disabled TF32/reduced-precision controls are in the plan.

Physical GPU2 is NVIDIA L20X, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`. CPU0–7, memory node0, OMP/intra-op/inter-op1 and seed0 are fixed. The scheduler sets visibility; refresh all four status forms and use only its exact reservation. This run starts after the separate external GPU experiment has settled. Queue time is at most30 minutes, outside the7200-second experiment. The130-minute scheduler timeout supplies outer termination grace; it does not enlarge the experiment.

## Data, outputs and scoring

The immutable GSM8K main/test source has1319 rows; all1319 satisfy the predeclared1–512-token prompt limit, with0 exclusions. The fixed seeded ranking selects64 evaluation and two disjoint feasibility questions. These66 selected prompts are61–180 tokens long. The question-only template, exact parser grammar and ranking policy were committed before source download or selection. Preserve the raw419088-byte Parquet file, normalized strings, README/license declaration, conversion producer and all population records; no source question was truncated or silently removed.

Execute F-feas-00 and F-feas-01 first, then F-eval-00 through F-eval-63, exactly once. Both feasibility examples are excluded from the denominator. Each example generates its own actual greedy history, with natural EOS0 and at most256 outputs. EOS as output256 still means stop; non-EOS output256 means length. The final prediction is not forwarded. Every output depth is four, with the first excluded from decode-depth averages.

Full recurrent/coda finite hooks run on the two feasibility examples. A per-output lm_head finite hook runs before argmax on all66 rows; finite-logit checks must equal actual output count. Evaluation does not claim intermediate-value validation. This observer supports no speed claim.

Raw text keeps special tokens. Scoring removes only an actual terminal EOS0 on stop, then uses the last fully matching number line and exact Decimal equality. Missing/unparseable answers count as wrong; valid final answers at the length cap are scored normally. Timeouts, nonfinite logits, device failures or missing generations are incomplete execution, not ordinary wrong answers. The standalone scorer re-decodes and re-parses from preserved local tokenizer/data bytes without Torch, GPU or model weights. No loss margin, confidence interval, accuracy floor or paired BF16/adaptive qualification is inferred.

## Finite execution and resource envelope

One worker, one model load, one retained192-page pool: **1207959552 bytes (1152MiB)**. Admission reserves a whole request lifetime; actual selected requests require at most112 pages. Their maximum step bound is1533; the generic512-prompt contract allows1535 steps. Actual expected steps are ceil(P/128)+1+6*(O-1). Reset request/queue/page ownership to zero before the next example while retaining model, pool and allocator state.

The global cap is7200 seconds and each full case is capped at600 seconds. The first case also bounds worker model setup. There are no extra warmups, repeated samples, replacement examples or automatic retries. Stop on the first source/input/control, finite, prefix/depth, ownership, device, deadline, byte-cap or cleanup failure. Preserve started/result/completion/ACK files and failed partial evidence; a late failure does not rewrite completed bytes or invent a valid acknowledgment.

Artifacts are capped at2MiB per case and256MiB total, including preparation and the retained plan. The case allowances total138412032 bytes, leaving130023424 bytes for auxiliary data. Prepared input files occupy6187834 bytes. The pinned vocabulary maximum is162 UTF-8 bytes per token; the conservative decoded-output bound is124416 bytes per example. No tensors or profiles are exported. Downloads, normalization and CPU preparation are separate from device execution.

Raw destination: `/tmp/hsliu2-vllm-lt-q2-quality-20260911/run`. Final teardown must synchronize and leave zero task-owned allocated/reserved bytes; independently record reservation release. Cleanup is limited to this job's requests, subprocess group and resources.

## Verification and acceptance

All128 combined feature CPU tests passed in23.27 seconds with CUDA discovery/init forbidden and unchanged source. Ruff/format and independent integration review passed. The prepared CPU selection blocked Torch imports. The resolved plan probe forbade CUDA and checkpoint tensor loading; it verified metadata/content hashes and reproduced the frozen selection. These checks do not claim successful model generation.

Require exactly two completed feasibility plus64 unique completed evaluation records, correct natural-EOS/depth/work/finite/ownership accounting, and zero final cleanup. Offline scoring must reproduce all answers and the fixed64-example denominator from raw records; missing coverage cannot pass. Preserve all commands, source/plan/data identities, starts/completions/ACK hashes, raw IDs/text and failures. A complete F32 baseline is separate from model qualification and full Q2 closure. There is no paired configuration or performance measurement in this run.

The separate external experiment's original automated report passes but its manual timing decision is inconclusive because this quality CPU probe overlapped an external measured case. That is a performance-control failure retained in PR17; it is not a quality result. This quality run does not repair or repeat those timing measurements.

Review and DCO-commit this resolved document before queueing the single quality run. The source snapshot above stays fixed; the documentation commit is later and separate.

```bash
cd /data/hsliu2/tmp/vllm-lt-q2-quality-exec
gpu run --gpu-ids 2 --wait 30m --timeout 130m --note 'Q2 FP32 quality 6229db722bae99d6a6ad2886676f376a0521735a' -- numactl --physcpubind=0-7 --membind=0 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-q2-quality-exec /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python -m vllm_lt.benchmarks.q2_quality run-quality --plan /data/hsliu2/tmp/vllm-lt-q2-quality/artifacts/q2-quality-plan-20260911/plan.json --output /tmp/hsliu2-vllm-lt-q2-quality-20260911/run
/home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-q2-quality-exec/vllm_lt/benchmarks/q2_quality_score.py --run-dir /tmp/hsliu2-vllm-lt-q2-quality-20260911/run --output /data/hsliu2/tmp/vllm-lt-q2-quality/artifacts/q2-quality-report-20260911.json
```
