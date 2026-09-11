"""Offline evidence and scope decision for persistent eager storage."""

from pathlib import Path

from vllm_lt.benchmarks.schema import write_json

from .m3_inactive_report import _build_correctness_report
from .m3_persistent import audit_model_rows
from .m3_persistent_schema import validate_plan


def build_report(output_dir):
    from .m3_persistent_lifecycle import audit_lifecycle_outputs

    return _build_correctness_report(
        output_dir,
        validate=validate_plan,
        audit_models=audit_model_rows,
        audit_checks=audit_lifecycle_outputs,
        artifact_prefix="m3_persistent",
        checks_key="lifecycle",
        check_kind="lifecycle",
        plan_hash_key="lifecycle_plan_sha256",
        scope="eager persistent-storage prerequisite; not graph capture or performance",
        limitations=[
            "One deterministic pass per declared case; no timing or throughput claim.",
            "One opt-in 8-row/32-column bucket; at most four live rows in odd physical slots.",
            "Core intermediates allocate eagerly; only owned boundary storage is persistent.",
            "Model intermediate/KV deltas are diagnostic; held-input checks require exactness.",
            "Memory-access checker unavailable; masking and lifecycle guards do not replace it.",
            "No device fault injected; CPU failure tests cannot establish hardware fault recovery.",
            "Capture, alternating captured buckets, BF16 and end-to-end performance remain open.",
        ],
    )


def write_report(output_dir):
    output_dir = Path(output_dir).resolve()
    result = build_report(output_dir)
    write_json(output_dir / "persistent-report.json", result)
    text = [
        "# M3 persistent eager-storage correctness evidence",
        "",
        f"Evidence: **{result['evidence_status']}**. Correctness: **{result['decision']}**.",
        "",
        result["scope"],
        "",
        *["- " + line for line in result["limitations"]],
        "",
    ]
    if "counts" in result:
        counts = result["counts"]
        text.extend(
            [
                f"Verified executions: {counts['completed_executions']}/17.",
                f"Verified model cases: {counts['verified_cases']}/13 "
                "(11 qualification and two excluded feasibility cases).",
                f"Verified lifecycle evaluations: {counts['verified_lifecycle_evaluations']}/4.",
            ]
        )
    text.extend("- " + str(error) for error in result["errors"])
    (output_dir / "persistent-report.md").write_text("\n".join(text) + "\n")
    return result
