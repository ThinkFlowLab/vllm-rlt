# Q2 FP32 quality baseline

This change implements the native FP32 baseline portion of [Q2 #8](https://github.com/hsliuustc0106/vllm-lt/issues/8): one deterministic GSM8K screen, with 64 evaluation examples and two disjoint excluded feasibility examples. It does not qualify BF16, adaptive decoding, speed, or the whole Q2 milestone. Q1 still blocks the B16/A16 comparisons; no paired margin or accuracy threshold is invented for the baseline.

The engine is accepted PR14 `630a8fdc0dd47b6da68a3d30db6b851fc08af5c5`, using compact FP32 Triton attention, tile32, fixed four loops and one request at a time. Its production files remain unchanged. M4 tile64 and M3 graph capture are not part of this configuration.

## Inputs frozen before selection and generation

The fixture [ouro-q2-fp32-quality-contract.json](../../benchmarks/fixtures/ouro-q2-fp32-quality-contract.json) fixes the prompt, parser, sampling and selection policy. Its initial version was committed before dataset download or selection.

- Model/tokenizer: `ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, using the existing verified local checkpoint and prepared environment.
- Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866), `main/test`, immutable revision `740312add88f781978c0658806c59bc2815b9866`. The 419,088-byte Parquet object's SHA256 is `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`. Its metadata/card declares MIT. Preserve the raw file, card and exact conversion command independently of the normalized input.
- Normalization preserves all question and answer strings. IDs are `{revision}/main/test/{zero_based_row_index}`; no silent deduplication or row filtering occurs during conversion.
- Encode the complete question-only prompt with pinned `tokenizers==0.21.4`, no added special tokens, truncation, padding or chat template. Only prompts with 1–512 tokens are eligible.
- Rank eligible IDs by SHA256 of canonical compact JSON `["q2-fp32-quality-v1",0,source_id]`, breaking ties by ID. The first 64 are evaluation and the next two are feasibility; execute feasibility first. Preserve every eligible/excluded ID, token length and selection key.
- The numerical prerequisite binds the original Q1 report: complete evidence, 93/93 required FP32 trajectories passing (64 main, 16 live, four official, nine original). This does not establish accuracy or cover every longer history used here. BF16's failed decision remains recorded.

The resolved execution plan must additionally freeze source/import/dependency hashes, checkpoint/tokenizer bytes, all selected IDs and prompt IDs, parser identity, GPU ID/UUID, CPU/NUMA binding, arithmetic flags, controls and resource limits before any generated answer is inspected.

## Generation and bounds

Use actual greedy generation with `temperature=0`, `top_p=1`, `top_k=-1`, `min_loops=max_loops=4`, threshold1 and `ignore_eos=False`. Preserve all sampled IDs, including actual EOS0. EOS takes precedence over the token limit, so EOS as output256 finishes with `stop`; a non-EOS output256 finishes with `length`. There is no parser-driven stopping, forced continuation, replacement example or output replay.

The scheduler uses refill, one live request and 128-token prefill chunks. With P prompt tokens and O actual outputs, expected steps are `ceil(P/128)+1+6*(O-1)`; the worst case is 1,535. Every output depth is four. Exclude the first output from decode-depth averages; an EOS-first example has no decode outputs.

The final prediction is not forwarded. At P≤512 and O≤256, at most767 positions enter the cache. Lifetime admission reserves `4*ceil((P+255)/16)` physical pages, at most192. The fixed192-page pool occupies **1,207,959,552 bytes (1,152 MiB)** for this FP32 model. Private persistent/graph options remain disabled because the maximum table width is48.

Use full recurrent/coda finite checks for the two feasibility examples. Check each actual lm_head output for finite logits before greedy sampling; the existing greedy path otherwise uses argmax without a NaN guard. Evaluation checks only logits, not every intermediate. Record exactly O logit checks. This observer adds synchronization and supports no performance claim.

One worker loads the model once and keeps the model/pool/allocator resident for all66 executions. Before every next example, require no request, queued ID or reserved page from the previous example. Record final synchronized teardown with zero task-owned allocated/reserved bytes and then the scheduler release independently.

The quality run has its own **7,200-second global / 600-second full-example deadline**, two feasibility examples and one pass over64 evaluation examples, with no extra warmup or retry. The global deadline wins. Limit each case to2MiB and total artifacts to256MiB; no tensor dumps or profiles. Queue/download/preparation are reported separately. Persist started/result/completion/ACK records incrementally, enforce deadlines through ACK writing, and preserve immutable completed bytes plus a separate failure record if a later acknowledgment step fails. Missing execution or cleanup remains incomplete evidence.

## Text and exact scoring

Preserve raw text decoded with `skip_special_tokens=False`. For scoring only, remove a final actual EOS0 when finish reason is `stop`, then decode with the same policy. Do not remove other special tokens, repair numbers or add markers. The standalone CPU scorer re-decodes from local pinned tokenizer bytes and requires neither GPU nor model weights.

Use the last fully matching number line according to the fixture's ASCII grammar: `####`, whitespace, a signed integer or decimal with optional correctly grouped thousands commas, and optional horizontal whitespace. LF separates lines; CRLF and an end-of-generation line without LF are accepted. Remove grouping commas and compare exact Decimal values; never use float rounding. Exponents, fractions, currency, malformed grouping and trailing prose are invalid. If a malformed marker follows a valid matching line, the last matching line remains the chosen answer; preserve malformed-marker diagnostics.

Missing or unparseable answers from completed generations count as incorrect. A valid final line at the token limit is scored normally with `length` recorded separately. Timeouts, nonfinite logits, device failures, corrupted evidence or missing generations affect execution coverage and cannot become ordinary wrong answers or a smaller denominator.

Raw results preserve every example's actual IDs, raw/scoring text, reference provenance, parser decision, correctness, output length, finish reason and depths. The scorer verifies these fields and summarizes the outcomes; it does not duplicate every raw token/text payload in the aggregate report. Recompute all derived fields offline. A completed result reports correct/64, parse failures, length-limited outputs and descriptive depth/length statistics. Partial evidence may report known outcomes and coverage but cannot claim a completed64-example accuracy. Set paired comparisons to empty and paired qualification to `not_applicable: F32_baseline_only`.

## Interfaces and acceptance

The data module owns deterministic selection, tokenization, decoding and parsing. The schema module binds the frozen inputs and computes capacity. The worker drives the existing engine and owns only its subprocess/request lifecycle. The CPU scorer validates raw evidence and computes the bounded baseline report. These are private evaluation interfaces, not a new serving API.

Acceptance requires:

1. Immutable dataset/model/tokenizer/source and applicable Q1 prerequisites; exact selection and parser frozen before generation.
2. Two excluded feasibility plus64 unique completed evaluation records, each actual natural-EOS history within its declared bound, finite logits and fixed depth.
3. Offline reproduction of IDs-to-text, parsing, reference comparison, all64 outcomes and execution/cleanup coverage; meaningful CPU edge cases for EOS, parser, duplicate/missing/corrupt records and deadlines.
4. Preserved commands, raw results, start/completion/ACK hashes, observed resource use, limits and cleanup. No extra runs or samples after a failure.
5. A scoped decision: the FP32 baseline can be complete without qualifying a model policy. AC-Q2-01–03/06 are fulfilled only for this sub-deliverable; BF16/adaptive comparisons and full issue closure remain open. The separate cached external experiment supplies AC-Q2-04/05; its timing cannot establish task quality.

This document is a design contract, not a generated-answer or accuracy result. Commit a resolved source/device/selection contract before the finite quality execution and report its actual outcome separately.
