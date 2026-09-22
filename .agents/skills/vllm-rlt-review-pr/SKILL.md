---
name: vllm-rlt-review-pr
description: "Review PRs and local changes for hsliuustc0106/vllm-rlt: Ouro engine and KV correctness, serving behavior, and BF16 accuracy/speed evidence. Use for maintainer reviews, repeat reviews, and author self-reviews of this project; not for vLLM-Omni or unrelated repositories."
---

# Review vllm-rlt changes

Review the requested change against its intended base. Return actionable,
evidence-backed defects; zero findings is valid. Distinguish implementation
correctness, reference fidelity, accuracy, speed and memory qualification.

## Freeze the review

- Verify the repository/remotes and applicable AGENTS.md. The project is
  `hsliuustc0106/vllm-rlt`; accept its verified forks. Do not inherit vLLM-Omni's
  architecture, reviewers, test requirements or milestone gates.
- For a PR, record base/head SHAs, merge base, state, description, full diff,
  relevant review threads and checks. Read code from the frozen head, not a
  moving local branch. A stacked PR's prerequisite stack is its immediate
  comparison base; distinguish a cumulative comparison against main.
- For local changes, record HEAD, intended base, status and the relevant staged,
  unstaged and untracked diff. Preserve user changes. A review does not authorize
  source fixes, commits, benchmark campaigns or posting a GitHub review.
- Before delivery, check whether the PR head changed. Label the reviewed SHA;
  recheck affected findings if continuing against the newer head. Do not present
  old CI or experiment evidence as validation of an untested newer revision.
- When a prerequisite is reported merged, fetch the verified upstream main
  before selecting a new experiment base. If PR metadata disagrees with Git,
  verify the merge's parents/ancestry or integrated diff rather than relying on
  a cached API state or commit subject alone. Report the discrepancy and the
  verified Git SHA; do not keep treating an integrated prerequisite as open.
  Preserve the requested snapshot for an ongoing review.

## Load relevant references

Read the diff and its callers to select references. Load only those relevant to
the changed behavior or claims, and apply their checks proportionally.

| Reference | Read when |
| --- | --- |
| [Test quality and review execution](references/test-quality-and-execution.md) | Production or test changes, executable validation, or repeat reviews. Covers regression evidence, source identity, simplification and revalidation. |
| [Core and interface review](references/core-and-interface.md) | Core behavior or public API, CLI, configuration, outputs or UI changes. Covers why the change is necessary, Plan B, invariants and compatibility. |
| [A/B evidence](references/ab-evidence.md) | Inference, numerics, memory, measurement logic or accuracy/performance claims are affected. Covers applicable BF16 checks and frozen experiment gates. |

No A/B is required for unaffected runtime/measurement paths. Correctness-only
changes need relevant regression checks, not a speedup.

## Validate and deliver

Use targeted CPU tests or a small reproducer to resolve a concrete uncertainty.
Check test dependencies and skips; a green suite with GPU cases skipped does not
validate kernels or real serving. GPU validation requires the user's applicable
experiment scope and verified scheduler reservation; never use an unreserved
device or silently expand a benchmark's exhausted run budget. If unavailable,
finish the code/evidence review and identify the remaining validation precisely.

For each finding, give priority, a short defect title, the smallest relevant
changed-line location, a reachable trigger, observable impact and supporting
evidence. Trace the failure into unchanged callers if needed, but attribute it
to this change. Exclude speculative risks, unrelated backlog and style-only
feedback. In repeat reviews, verify fixes at the new snapshot and avoid reposting
resolved findings.

Follow the applicable reference's reporting emphasis, then give findings in
priority order and a brief validation/scope note with the reviewed SHA. If there
are no findings, say so without implying unrun gates passed. Use verified PR
diff links or absolute local file links. Draft/post comments only within the
user's explicit authorization; select reviewers from actual ownership evidence.
