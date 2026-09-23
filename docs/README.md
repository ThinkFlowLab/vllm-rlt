# Documentation

New users should follow the [first-run user guide](launching.md), from creating
an environment to receiving the first generated response. The
[README](../README.md#getting-started) provides a brief installation reference.

| Guide | Topics |
| --- | --- |
| [First-run user guide](launching.md) | Environment setup, model download, server startup, first request, CLI, Python API, and troubleshooting |
| [Engine design](design.md) | Model stages, scheduling, and KV ownership |
| [Scheduler walkthrough](scheduler_walkthrough.md) | Responsibilities, admission cases, control flow, and refactoring checklist |
| [KV layout examples](kv_layout_computation.md) | SHARED and LAST_EXITED semantics and worked attention examples |
| [Runtime configuration](cdb_runtime.md) | Exit policies, KV layouts, execution options, and CUDA Graphs |
| [Asynchronous scheduling](https://github.com/hsliuustc0106/vllm-rlt/pull/30) | CPU/GPU pipelining and single-stream or multi-stream execution |
| [FlashAttention](https://github.com/hsliuustc0106/vllm-rlt/pull/30) | FA2/FA3/FA4 installation, hardware selection, and constraints |
| [Cache and scheduling features](https://github.com/hsliuustc0106/vllm-rlt/pull/31) | Prefix reuse, incremental KV, priorities, and preemption |
| [Prefill/decode disaggregation](https://github.com/hsliuustc0106/vllm-rlt/pull/31) | Single-host GPU worker pools and NIXL transfer |
| [HTTP serving](serving.md) | Completions API, streaming, and service lifecycle |
| [Accuracy evaluation](accuracy.md) | GSM8K regression setup and comparison methodology |
| [Synchronous speculation evaluation](speculative_evaluation.md) | Reproducible BF16/FA4 accuracy, decode, and serving measurements for RFC #43 |

## Support and Runtime Notes

Current model support is limited to Ouro-1.4B. CPU execution provides a Torch
reference backend. FlashAttention hardware validation is currently documented
for FA4 on B300; FA2/FA3 require validation on their target devices. Disaggregated
serving currently targets multiple GPUs on a single host.

`ouro_delayed` reuses the trained Ouro gate with a one-loop delay; it changes
the exit policy. The `random_lookahead` mode uses an untrained head for runtime
experiments. Neither is a distilled lookahead predictor.
