"""Offline coverage and chronology for the inactive-row prerequisite only."""

from pathlib import Path

from vllm_lt.benchmarks import ab
from vllm_lt.benchmarks.ab_schema import equal, require
from vllm_lt.benchmarks.schema import _file_record, read_json, write_json

from .m3_inactive import audit_model_rows
from .m3_inactive_run import controls_view
from .m3_inactive_schema import validate_plan


def build_report(output_dir):
    from .m3_inactive_kernels import audit_kernel_outputs

    return _build_correctness_report(
        output_dir,
        validate=validate_plan,
        audit_models=audit_model_rows,
        audit_checks=audit_kernel_outputs,
        artifact_prefix="m3_inactive",
        checks_key="kernels",
        check_kind="kernel",
        plan_hash_key="kernel_plan_sha256",
        scope="inactive-row correctness prerequisite; not graph capture or performance",
        limitations=[
            "One deterministic pass per case; no timing or throughput claim.",
            "Memory-access checker unavailable; guard/source evidence does not replace it.",
            "Only eager decode uses eight physical rows; prefill and coda remain compact.",
            "Model intermediate/KV deltas are diagnostic; held-input checks require exactness.",
        ],
    )


def _build_correctness_report(
    output_dir,
    *,
    validate,
    audit_models,
    audit_checks,
    artifact_prefix,
    checks_key,
    check_kind,
    plan_hash_key,
    scope,
    limitations,
):
    """Audit shared source/control/chronology gates for bounded eager prerequisites."""
    output_dir = Path(output_dir).resolve()
    result = {
        "schema_version": 1,
        "artifact_type": artifact_prefix + "_report",
        "evidence_status": "incomplete",
        "decision": "inconclusive",
        "errors": [],
        "hashes": {},
        "scope": scope,
        "limitations": limitations,
    }
    try:
        plan = read_json(output_dir / "plan.json")
        validate(plan)
        manifest = read_json(output_dir / "manifest.json")
        equal(
            [manifest["schema_version"], manifest["artifact_type"], manifest["plan_sha256"]],
            [1, artifact_prefix + "_manifest", plan["plan_sha256"]],
            "manifest identity",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["evidence_status"] = "invalid"
        result["errors"].append({"scope": "plan/manifest", "message": str(exc)})
        return result
    result["plan_sha256"] = plan["plan_sha256"]
    result["manifest_status"] = manifest["status"]
    result["execution_failures"] = manifest["failures"]
    for name in ("plan.json", "manifest.json"):
        result["hashes"][name] = _file_record(output_dir / name)["sha256"]
    result["numerical"] = audit_models(output_dir, plan)
    result[checks_key] = audit_checks(output_dir / checks_key, plan[checks_key])
    children = {}
    try:
        start, end, deadline = (manifest[key] for key in ("started_ns", "ended_ns", "deadline_ns"))
        require(
            all(type(value) is int and value > 0 for value in (start, end, deadline))
            and start <= end < deadline
            and deadline - start == 3600 * 10**9,
            "invalid parent deadline/lifetime",
        )
        workers = plan["workers"]
        ids = [row["execution_id"] for row in plan["execution_order"]]
        equal(
            manifest["completed_executions"],
            ids[: len(manifest["completed_executions"])],
            "completed prefix",
        )
        equal(
            manifest["completed_workers"],
            [w["worker_id"] for w in workers][: len(manifest["completed_workers"])],
            "completed worker prefix",
        )
        launches = manifest["workers"]
        equal(
            [row["worker_id"] for row in launches],
            [row["worker_id"] for row in workers][: len(launches)],
            "actual worker launch order",
        )
        actual = {p.name for p in (output_dir / "workers").iterdir() if p.is_dir()}
        require(actual <= {row["worker_id"] for row in workers}, "unplanned worker directories")
        previous = None
        completed_from_workers = []
        for index, launch in enumerate(launches):
            worker = workers[index]
            path = output_dir / "workers" / worker["worker_id"] / "manifest.json"
            child = read_json(path)
            children[worker["worker_id"]] = child
            result["hashes"][str(path.relative_to(output_dir))] = _file_record(path)["sha256"]
            equal(
                [
                    child["schema_version"],
                    child["artifact_type"],
                    child["worker_id"],
                    child["implementation_id"],
                ],
                [
                    1,
                    artifact_prefix + "_worker_manifest",
                    worker["worker_id"],
                    worker["implementation_id"],
                ],
                "worker identity",
            )
            successful = child["status"] == "complete"
            if successful:
                require(
                    child["passed"] is True and not child["failures"] and launch["exit_code"] == 0,
                    "inconsistent completed worker",
                )
                equal(child["model_loads"], 1, "one model load per worker")
                equal(
                    child["completed_executions"], worker["execution_ids"], "worker full coverage"
                )
                require(
                    worker["worker_id"] in manifest["completed_workers"], "missing completed worker"
                )
            else:
                require(
                    child["status"] in ("failed", "incomplete")
                    and child["passed"] is False
                    and bool(child["failures"])
                    and index == len(launches) - 1
                    and manifest["status"] != "complete"
                    and worker["worker_id"] not in manifest["completed_workers"],
                    "failed worker must be the final launch with a recorded stop",
                )
                require(
                    type(child["model_loads"]) is int and child["model_loads"] in (0, 1),
                    "invalid model load count",
                )
            equal(
                child["completed_executions"],
                worker["execution_ids"][: len(child["completed_executions"])],
                "worker completed execution prefix",
            )
            completed_from_workers.extend(child["completed_executions"])
            equal(child["deadline_ns"], deadline, "worker deadline")
            timestamps = [
                launch["launched_ns"],
                child["started_ns"],
                child["ended_ns"],
                launch["returned_ns"],
            ]
            require(
                all(type(value) is int for value in timestamps)
                and start <= timestamps[0]
                and timestamps == sorted(timestamps)
                and timestamps[-1] <= end,
                "worker lifetime outside parent",
            )
            if index:
                require(
                    launches[index - 1]["returned_ns"] <= launch["launched_ns"],
                    "worker processes overlap",
                )
            if successful:
                ab.audit_worker_controls(controls_view(plan), child, previous)
            else:
                equal(child["plan_sha256"], plan["plan_sha256"], "stopped worker plan")
                equal(
                    child["source"],
                    plan["implementations"][worker["implementation_id"]]["source"],
                    "stopped worker source",
                )
                equal(child["harness_sha256"], plan["harness"]["sha256"], "stopped worker harness")
                equal(
                    child["affinity"],
                    plan["contract"]["controls"]["affinity"],
                    "stopped worker affinity",
                )
                equal(
                    child["runtime_environment"],
                    plan["runtime_environment"],
                    "stopped worker environment",
                )
            previous = child
        equal(
            completed_from_workers,
            manifest["completed_executions"],
            "controller/worker completed prefix",
        )
        if manifest["status"] == "complete":
            equal(len(launches), 2, "two actual workers required")
        result["artifact_usage"] = ab.artifact_usage(output_dir, plan["contract"]["limits"])
        ledger = result["numerical"].get("ledger", {})
        numerical_times = ledger.get("case_lifetimes", {})
        check_times = {row["evaluation_id"]: row for row in result[checks_key]["evaluations"]}
        started = set()
        actual_model = {p.name for p in (output_dir / "numerical/cases").glob("*") if p.is_dir()}
        actual_checks = {
            p.name for p in (output_dir / checks_key / "evaluations").glob("*") if p.is_dir()
        }
        require(
            actual_model
            <= {r["execution_id"] for r in plan["execution_order"] if r["kind"] == "model"}
            and actual_checks
            <= {r["execution_id"] for r in plan["execution_order"] if r["kind"] == check_kind},
            "unplanned model/held-input directories",
        )
        for row in plan["execution_order"]:
            if row["kind"] == "model":
                marker = output_dir / "numerical/cases" / row["execution_id"] / "started.json"
            else:
                marker = (
                    output_dir / checks_key / "evaluations" / row["execution_id"] / "started.json"
                )
            if marker.exists():
                started.add(row["execution_id"])
        require(
            started == set(ids[: len(started)])
            and len(manifest["completed_executions"])
            <= len(started)
            <= len(manifest["completed_executions"]) + 1,
            "started executions do not form the frozen prefix",
        )
        previous_end = {}
        for row in plan["execution_order"][: len(manifest["completed_executions"])]:
            worker = children[row["worker_id"]]
            if row["kind"] == check_kind and row["execution_id"] not in check_times:
                raw = read_json(
                    output_dir / checks_key / "evaluations" / row["execution_id"] / "result.json"
                )
                equal(
                    raw[plan_hash_key],
                    plan[checks_key][plan_hash_key],
                    "partial held-input identity",
                )
                require(
                    raw["evaluation"]["evaluation_id"] == row["execution_id"]
                    and raw["status"] == "complete"
                    and raw["passed"] is True,
                    "completed partial held-input result is inconsistent",
                )
                check_times[row["execution_id"]] = raw
            timing = (
                numerical_times[row["execution_id"]]
                if row["kind"] == "model"
                else check_times[row["execution_id"]]
            )
            first, last = timing["started_ns"], timing["finished_ns"]
            limit = timing["deadline_ns"]
            require(
                type(first) is int
                and type(last) is int
                and type(limit) is int
                and worker["started_ns"] <= first <= last <= limit <= deadline
                and last <= worker["ended_ns"]
                and limit == min(deadline, first + 600 * 10**9),
                "execution lifetime differs from assigned worker/case cap",
            )
            require(
                previous_end.get(row["worker_id"], first) <= first,
                "model/held-input ordering overlaps or differs from plan",
            )
            previous_end[row["worker_id"]] = last
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append({"scope": "controls/order", "message": str(exc)})
    numerical, checks = result["numerical"], result[checks_key]
    complete = (
        manifest["status"] == "complete"
        and not manifest["failures"]
        and len(manifest["completed_executions"]) == len(plan["execution_order"])
        and len(manifest["completed_workers"]) == 2
        and numerical["complete"]
        and checks["complete"]
    )
    result["evidence_status"] = (
        "invalid" if result["errors"] else "complete" if complete else "incomplete"
    )
    if result["evidence_status"] == "complete":
        result["decision"] = "passed" if numerical["passed"] and checks["passed"] else "failed"
    elif manifest["status"] == "failed" and manifest["failures"]:
        result["decision"] = "failed"
        result["failure_basis"] = (
            "Recorded execution or required-gate failure stopped the protocol; "
            "remaining coverage is incomplete and cannot qualify the implementation."
        )
    result["counts"] = {
        "planned_executions": len(plan["execution_order"]),
        "completed_executions": len(manifest["completed_executions"]),
        **numerical["counts"],
        f"planned_{check_kind}_evaluations": len(plan[checks_key]["execution_order"]),
        f"verified_{check_kind}_evaluations": len(checks["completed_evaluations"]),
        "sanitizer_evaluations": 0,
    }
    return result


def write_report(output_dir):
    output_dir = Path(output_dir).resolve()
    result = build_report(output_dir)
    write_json(output_dir / "inactive-report.json", result)
    text = [
        "# M3 inactive-row correctness evidence",
        "",
        f"Evidence: **{result['evidence_status']}**. Correctness: **{result['decision']}**.",
        "",
        result["scope"],
        "",
        *["- " + line for line in result["limitations"]],
        "",
    ]
    if "counts" in result:
        text.append(
            f"Verified model cases: {result['counts']['verified_cases']}/13 "
            "(11 qualification and two excluded feasibility cases)."
        )
        text.append(
            f"Verified kernel evaluations: {result['counts']['verified_kernel_evaluations']}/52."
        )
    text.extend("- " + str(error) for error in result["errors"])
    (output_dir / "inactive-report.md").write_text("\n".join(text) + "\n")
    return result
