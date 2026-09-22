# Ouro GSM8K-87 accuracy regression case

The default case uses **87 fixed questions from GSM8K `main`/`test`**. Its measured
original Hugging Face Transformers baseline is **59/87 (67.82%)**. Native vllm-rlt
also scored **59/87**, with identical per-question correctness on this case.

The [checked-in fixture](../benchmarks/fixtures/gsm8k-87.json) pins the exact source
row IDs, dataset revision, HF score and recipe fingerprint. These are the 87
questions from the original seed-0 100-question case that had completed on both
backends when the case was adopted on September 12, 2026. The choice was made
after inspecting results, so this is a regression case for future changes, not
a new held-out estimate of model quality. The observed native score on the
original 100-question case was 69/100; its protocol and results are preserved.
Adopting the 87-question case required no new GPU inference.

## Protocol and baseline

- Pinned `ByteDance/Ouro-1.4B` checkpoint and official release code; BF16,
  fixed four loops, greedy decoding, one request at a time.
- lm-eval-harness 0.4.9.2's `gsm8k_cot`, first three demonstrations and strict
  answer extraction/scoring, following the Ouro paper's stated 3-shot CoT setup.
- No chat template or added special tokens. Natural EOS and identical harness
  stop sequences; at most 1,024 new tokens within a 2,048-token total context.
- Exact-answer accuracy with unparseable answers counted as incorrect. Save
  raw text, token IDs, extracted answers, correctness and stopping reason.
- Default regression threshold: native may lose at most one percentage point
  against the measured HF baseline. One question out of 87 is about 1.15 points,
  so the default gate requires **at least 59 correct**. This is a regression
  screen, not proof of population equivalence.

The reference calls `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` and the released model's `generate()`. It uses eager
attention and the standard Transformers `DynamicCache` with 96 depth/layer slots,
plus `exit_at_step=3` for fourth-loop logits. This cache setup accommodates the
pinned release's older cache interface; the model source is not patched.
The candidate uses the existing native engine and Triton attention. Both use
BF16 weights and activations, retaining each backend's existing FP32 reductions
and other operations that need higher precision. The evaluator changes no
inference arithmetic.

The recorded environment used torch 2.13.0, Transformers 4.55.0, lm-eval 0.4.9.2,
datasets 3.6.0, tokenizers 0.21.4 and Triton 3.7.1. The fixture records these
versions and hashes of the source HF protocol and selected raw outputs. The
baseline fingerprint covers model files, prompts/token IDs, questions, task
configuration, generation controls and package versions. Changing the checkout
SHA or checkpoint directory is allowed; changing the recipe requires a fresh
HF comparison using a custom protocol.

## Prepare and run the default case

Reuse the prepared accuracy environment when available. For a new environment,
install the project and evaluation requirements, using the package versions
recorded above to reuse the stored baseline:

```bash
pip install -e .
pip install -r benchmarks/requirements-accuracy.txt
hf download ByteDance/Ouro-1.4B \
  --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /path/to/ouro
python -m benchmarks.gsm8k prepare --model /path/to/ouro \
  --output /path/to/gsm8k-87-protocol.json
```

Preparation runs without CUDA. It verifies the local checkpoint against the
pinned Hub release, loads the pinned dataset, and freezes the questions, prompts,
token IDs, controls and package/model hashes. Oversized prompts fail preparation;
questions are never replaced. The default also verifies the recipe fingerprint
and embeds the stored HF baseline. Keep generated protocols and results outside
Git.

Before inference, declare the GPU/CPU affinity and run budget. Use two disjoint
training questions (`prepare --split train --limit 2`) for feasibility, then one
greedy pass over the 87 test questions for this regression screen. This run budget
does not measure repeat-run variability or throughput. On a host with the GPU
scheduler, use an available exact device ID and run the command below through
`gpu run --gpu-ids <available-id> --timeout 2h --note "GSM8K-87 accuracy" --`.

```bash
python -m benchmarks.gsm8k run --backend native \
  --protocol /path/to/gsm8k-87-protocol.json --output /path/to/native
```

The native summary includes `baseline_comparison`, with HF accuracy, native-minus-HF
percentage points on the fixed cohort and the gate result. The command saves the
summary and exits nonzero if the gate fails. Exceptions and non-finite logits
stop execution while preserving completed records; do not reduce the denominator.
An optional `--min-reference-accuracy-pct` floor and `--max-regression-pp` changes
must be declared during preparation. There is no default paper-score floor.

## Fresh comparisons and custom cases

To remeasure HF on the same default case, run both backends from the same protocol
and compare their saved results. Keep the exact GPU, CPU/NUMA affinity and controls
fixed across backends. Budget one feasibility pass and one measured greedy pass
per backend before starting; this budget does not establish repeat-run variability.

```bash
python -m benchmarks.gsm8k run --backend transformers \
  --protocol /path/to/gsm8k-87-protocol.json --output /path/to/transformers
python -m benchmarks.gsm8k compare --transformers /path/to/transformers \
  --native /path/to/native --output /path/to/comparison.json
```

The comparison reports correct counts, accuracies, percentage-point difference,
paired losses/gains and answer disagreements. It rejects incomplete or mismatched
runs. Loading and generation durations are recorded separately.

Explicit `--limit 100 --seed 0` recreates the original 100-question selection;
`--all` uses the complete split. Custom subsets use the SHA-256 ranking of dataset
revision, split, seed and source row ID, independent of answers. In particular,
`--limit 87` samples a different set from the fixed default case. Custom protocols
do not reuse the stored 59/87 baseline: run HF and native, then use `compare`.

Dataset: [GSM8K](https://huggingface.co/datasets/openai/gsm8k).
The [Ouro evaluation settings](https://arxiv.org/html/2510.25741v5#A3.T16) do not
pin the exact harness revision, demonstrations or token limits, so the settings
above are explicit project choices rather than an exact paper reproduction.

The [full 1,319-question comparison](https://github.com/hsliuustc0106/vllm-rlt/blob/d2f7db20593cccfe666b3c04c3ee50abc8c39af7/docs/benchmarks/gsm8k-bf16-20260912.md) recorded
**62.02% native versus 61.64% Transformers** with strict matching. It includes
per-question audits and a separate extraction diagnostic. That experiment's
frozen 75.92% reference floor failed; the current default remains the measured
GSM8K-87 baseline described above.
