# Test quality and review execution

Apply the relevant sections to the changed behavior. These checks complement
the core/interface and A/B references; they do not require benchmarks or a full
test-suite run for every PR.

## Check what the tests prove

- Ask whether reverting or breaking the changed behavior would fail the test.
  Assertions must protect the contract, not merely non-null output, a log line
  or process survival.
- Trace tests to the production dispatcher and affected consumer. Identify
  mocks that bypass the change, unrealistic fakes, and successful fallback
  paths that could hide a broken intended implementation.
- For behavior fixes, seek an automated regression test that reproduces the
  defect on the frozen base and passes on the head. Record exact commands and
  results when executed. If the base cannot run the test unchanged, explain the
  adaptation or limitation; an unrelated import failure is not reproduction.
  Existing coverage may suffice if it demonstrably catches the original defect.
- Check affected boundary, invalid-input, feature-off and failure/cancellation
  paths. For shared buffers or KV state, exercise a relevant transition such as
  request replacement, row reorder or buffer reuse, not only identical repeats.
- Check seeds, synchronization, numeric tolerances and hardware/test markers.
  Skipped device tests leave a gap. Classify failures as code, test, environment
  or flaky; a passing rerun does not erase an unexplained failure.

Report concrete test weaknesses only when they leave changed behavior materially
unprotected. Missing evidence is not itself proof of a runtime defect. Avoid
tests that merely duplicate implementation details or enforce review wording.

## Bind validation to the reviewed source

Use a pinned isolated checkout for PR validation. For local changes, preserve
the intended base, HEAD, staged/unstaged patches and contents of in-scope
untracked files; names and a clean status alone do not establish source identity.
Record each check's command, outcome, source identity, Python and relevant
dependency versions. Note ignored/generated files if they affect imports or
execution rather than hashing unrelated caches.

Before validation and delivery, check that the relevant source contents and
remote head still match the reviewed snapshot. If they changed, mark affected
evidence stale and revalidate the changed paths; preserve unrelated user edits.
Ordinary test caches do not invalidate unchanged source by themselves.

An isolated worktree does not sandbox executable PR code. For untrusted code,
use an appropriately isolated environment without credentials or sensitive host
access; otherwise use static inspection and existing CI evidence and state the
execution gap. Follow the user's resource policy for every device operation.

## Look for a smaller change

After correctness checks, inspect added helpers, classes, state, copies and
fallback/compatibility branches for a distinct live caller, invariant or support
need. Compare reuse or a narrower implementation where evidence supports it.
Keep this pass within the diff and the callers needed to assess it; do not turn
style preferences or unrelated backlog into findings. Zero candidates is valid.

## Re-review only what changed

Compare the previous reviewed SHA with the new head, including rebase/conflict
resolutions. Recheck unresolved findings against current code and line numbers,
inspect fixes for new regressions, and rerun checks invalidated by the delta.
Do not repeat resolved comments or present old results as testing the new head.

## When profiling instrumentation changes

Check collection is opt-in and has bounded start/stop and cleanup behavior.
Exercise disabled operation, repeated start/stop and partial failure; verify
trace attribution identifies the relevant request/thread/device without user
payloads or secrets. Check instrumentation preserves outputs, scheduling and
resource ownership apart from disclosed collection overhead. Measure disabled
and enabled overhead when making claims about instrumentation performance.
These implementation checks are separate from the before/after figure required
for single-request speed evidence.
