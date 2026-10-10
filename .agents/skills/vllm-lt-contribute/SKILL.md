---
name: vllm-lt-contribute
description: "Plan and implement contributions to hsliuustc0106/vllm-lt, from a concrete issue or evidence-backed opportunity through validation and PR preparation. Use for fixes, features and continuing implementation in this project; use vllm-lt-review-pr for standalone reviews. Not for vLLM-Omni or unrelated repositories."
---

# Contribute to vllm-lt

Carry the requested contribution through a focused implementation, appropriate
validation and a reviewable handoff. Honor planning-only or experiment-only
scope; neither authorizes source changes or publishing a PR.

## Establish the task and base

- Verify the checkout, remotes, status and applicable AGENTS.md. This project is
  `hsliuustc0106/vllm-lt`; accept its verified forks. It is a standalone Ouro
  engine, not vLLM or vLLM-Omni. Do not import their contribution gates or assume
  their runtime features exist here.
- Continue an explicitly identified task branch. For new development, fetch
  the verified upstream main and create a task branch in a separate worktree,
  following the applicable branch and path conventions. Preserve unrelated
  changes; disclose a failed fetch rather than calling the base current.
- Read the requested issue/PR and current discussion when supplied. Record the
  base SHA, HEAD and existing in-scope modifications. For an explicit stack,
  identify the immediate prerequisite base. Verify reported merges against
  fetched Git history when metadata disagrees; do not continue a stale stack
  merely because its old PR description says a prerequisite is pending.
- For open-ended contribution requests, inspect current code, tests and open
  work for a concrete defect, measured bottleneck or requested capability.
  Present a bounded opportunity with evidence before implementing when the
  user asked only for selection or planning. Historical milestones and archived
  experiments are context, not an active backlog or measurements of current main.

## Design and implement

Read the affected callers and the checkout's [engine design](../../../docs/design.md).
The scheduler owns queue/progress decisions, the runner owns tensor execution
and persistent hidden state, and the KV manager owns physical allocation.
Keep a change within its responsible module unless the required behavior needs
an ownership or public-contract change.

For core or public-interface changes, read the shared
[core and interface criteria](../vllm-lt-review-pr/references/core-and-interface.md)
before choosing an implementation. Explain the concrete need, evidence, affected
invariants and why this module or interface must change. Compare a credible
Plan B and its tradeoffs; an alternative need not be implemented. For API, CLI
or HTTP changes, show before/after usage, account for compatibility and update
the affected help, examples or [serving guide](../../../docs/serving.md).

Make the smallest change that addresses the requested behavior. Avoid unrelated
cleanup, speculative abstractions and new benchmark infrastructure when existing
paths suffice. Preserve relevant request/KV isolation, full-depth prefill,
last-exited KV propagation and request-local RNG behavior. For numeric changes,
read the [precision policy](../../../docs/precision-policy.md): BF16 is the
inference target, with justified FP32 intermediates retained.

## Validate the changed behavior

For production or test changes, read the shared
[test quality and execution criteria](../vllm-lt-review-pr/references/test-quality-and-execution.md).
Use a regression check that exercises the affected production path and protects
its observable contract. For a behavior fix, demonstrate the original failure
on the frozen base and success on the candidate where feasible; explain any
adaptation or execution gap. Existing coverage can suffice when it catches the
defect. Documentation-only changes need content/link checks, not new runtime tests.

Verify the host and reuse a suitable environment before installing dependencies.
Read [pyproject.toml](../../../pyproject.toml) and the relevant guide for extras;
serving tests can skip without `aiohttp`/tokenizer dependencies, and accuracy
tests can skip without `lm_eval`. Run from the intended worktree and verify
imports resolve there, especially when reusing an editable install.

Select affected tests rather than running every group unconditionally:

| Changed area | Starting points, relative to the repository root |
| --- | --- |
| Scheduling, request lifecycle, public LLM API | `tests/test_engine.py` and affected callers |
| KV allocation, prepared metadata, attention | `tests/test_kv_cache.py`, `tests/test_prepared_kv.py`, `tests/test_attention.py` |
| Ouro execution, checkpoint loading, numerics | `tests/test_ouro.py`, `tests/test_engine.py` |
| HTTP/streaming/worker behavior | `tests/test_serving.py`, `tests/test_engine.py`; follow `docs/serving.md` for real-model checks |
| Accuracy preparation, scoring, backends | `tests/test_gsm8k_accuracy.py`, `tests/test_gsm8k_backends.py`; follow `docs/accuracy.md` |

Use `OMP_NUM_THREADS=1 python -m pytest -q <selected-test-paths>` for CPU tests.
For Python edits, run `python -m ruff check <changed-python-paths>` and
`python -m ruff format --check <changed-python-paths>`. Broaden testing when
cross-module impact or unresolved failures warrant it. Record commands, source
identity, environment, results and skips; CPU passes do not qualify CUDA kernels.
`--run-gpu` opts into CUDA tests and requires scheduler-provided visibility.
Every device operation must follow the applicable host/reservation policy;
never set visibility by hand to satisfy the test guard.

## Gather measurement evidence when applicable

For inference, numerics, memory, measurement changes or accuracy/speed claims,
read the shared [A/B evidence criteria](../vllm-lt-review-pr/references/ab-evidence.md)
and select checks by actual impact. Unaffected documentation/interfaces need no
A/B; correctness fixes need relevant regression evidence, not a speedup.

Before experiments, freeze the hypothesis, isolated variable, sources/patches,
BF16 configuration, controls, success criteria, run budget and stop condition.
Start with preparation without accelerator use and a feasibility run. Follow
the applicable experiment policy for measured repetitions, reservations and
cleanup; keep downloads, warmups and profiling separate from measured execution.
Report failures, variability and inconclusive results at the budget limit.

Use the current [accuracy guide](../../../docs/accuracy.md) for executable
protocols and the shared criteria for paired A/B interpretation. A fixed
ten-question smoke subset does not qualify the full case or inherit its stored
HF floor. For concurrency-one speed comparisons, include profiles of both A
and B, separate prefill/decode analysis and the required annotated timeline
figure on matching scales. Confirm the intended fast path actually executes.
Read the [illustrative contributor report](../vllm-lt-review-pr/references/ab-example.md)
when assembling applicable evidence; its synthetic values are not results or
acceptance thresholds. Verify available harnesses in the checkout rather than
assuming an unmerged A/B tool exists.

## Prepare the handoff

Inspect the complete in-scope diff for correctness, unnecessary machinery and
accidental files. Apply the selected shared criteria to the final source; do not
present earlier checks as validation of later changes that invalidate them.
Keep raw experiment artifacts under ignored `artifacts/` or the task's chosen
artifact location, with durable links when reporting externally.

Prepare a PR title/body when requested, leading with the problem and resulting
behavior. Include relevant rationale/Plan B, compatibility, validation and
evidence links, plus precise unrun checks or limitations. Before committing,
verify the author name/email and include a DCO trailer with `git commit -s`.
Push, open/update a PR or post comments only within the user's authorized scope;
reuse existing authorization. A requested self-review uses
[$vllm-lt-review-pr](../vllm-lt-review-pr/SKILL.md).

Report the branch/worktree, changed behavior, checks and remaining qualification
gaps. Distinguish implemented behavior from measured accuracy or performance.
