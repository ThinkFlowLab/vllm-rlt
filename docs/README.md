# Documentation

New users should follow the [first-run user guide](launching.md), from creating
an environment to receiving the first generated response. The
[README](../README.md#getting-started) provides a brief installation reference.

| Guide | Topics |
| --- | --- |
| [First-run user guide](launching.md) | Environment setup, model download, server startup, first request, CLI, Python API, and troubleshooting |
| [Engine design](design.md) | Model stages, scheduling, and KV ownership |
| [Scheduler walkthrough](scheduler_walkthrough.md) | Responsibilities, admission cases, control flow, and refactoring checklist |
| [Sampling walkthrough](sampling_walkthrough.md) | Sampling algorithm, RNG lifecycle across preemption and termination, and known limits |
| [KV layout examples](kv_layout_computation.md) | SHARED and LAST_EXITED semantics and worked attention examples |
| [Runtime configuration](cdb_runtime.md) | Exit policies, KV layouts, execution options, and CUDA Graphs |
| [Asynchronous scheduling](https://github.com/hsliuustc0106/vllm-rlt/pull/30) | CPU/GPU pipelining and single-stream or multi-stream execution |
| [FlashAttention](https://github.com/hsliuustc0106/vllm-rlt/pull/30) | FA2/FA3/FA4 installation, hardware selection, and constraints |
| [Cache and scheduling features](https://github.com/hsliuustc0106/vllm-rlt/pull/31) | Prefix reuse, incremental KV, priorities, and preemption |
| [Prefill/decode disaggregation](https://github.com/hsliuustc0106/vllm-rlt/pull/31) | Single-host GPU worker pools and NIXL transfer |
| [HTTP serving](serving.md) | Completions API, streaming, and service lifecycle |
| [Profiling](profiling.md) | Core profiler options, Python/HTTP controls, PD collection, and per-rank archives |
| [Accuracy evaluation](accuracy.md) | GSM8K regression setup and comparison methodology |

## Support and Runtime Notes

Current model support includes:
- **Ouro-1.4B/2.6B**: Full feature support including adaptive exit, PD, and speculative decoding
- **Nanbeige4.2-3B**: Basic implementation with fixed 2-loop execution; PD support available (1P1D configuration)

CPU execution provides a Torch reference backend. FlashAttention hardware validation is currently documented
for FA4 on B300; FA2/FA3 require validation on their target devices. Disaggregated
serving currently targets multiple GPUs on a single host.

### Nanbeige4.2 PD Configuration

Nanbeige4.2 supports Prefill/Decode disaggregation with the following configuration:
- **Workers**: 1 Prefill + 1 Decode worker (1P1D)
- **Loop semantics**: Fixed 2-loop execution (total_ut_steps=2)
- **Requirements**: 2 GPUs with NIXL transport support
- **Limitations**: Multi-worker configurations not yet supported; adaptive exit not available in PD mode

Test validation: `pytest tests/test_pd.py::test_pd_nanbeige_1p1d_matches_single_engine -v` (requires GPU+NIXL)

`ouro_delayed` reuses the trained Ouro gate with a one-loop delay; it changes
the exit policy. The `random_lookahead` mode uses an untrained head for runtime
experiments. Neither is a distilled lookahead predictor.
