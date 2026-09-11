"""Offline M4 evidence: existing event/pair audits plus explicit outer ownership."""

from pathlib import Path

from . import m4_attention_schema as schema
from .ab_report import _memory, pair_results
from .ab_schema import equal, require
from .report import _profiles, _run_record
from .schema import _file_record, read_json, write_json

ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)


def _note(report, kind, scope, message):
    report[kind].append({"scope": scope, "message": str(message)})


def _clock(value):
    require(type(value) is int and value >= 0, "timestamp must be a nonnegative integer")
    return value


def _read(root, relative, report):
    path = root / relative
    value = read_json(path)
    report["hashes"][relative] = _file_record(path)["sha256"]
    return value


def _workers(root, plan, parent, report):
    from .ab import audit_worker_controls

    expected = [w["worker_id"] for w in plan["workers"]]
    ids = [r["execution_id"] for r in plan["execution_order"]]
    done = parent["completed_executions"]
    equal(done, ids[: len(done)], "controller execution acknowledgment prefix")
    equal(
        parent["completed_workers"],
        expected[: len(parent["completed_workers"])],
        "controller worker acknowledgment prefix",
    )
    launches = parent["workers"]
    equal([r["worker_id"] for r in launches], expected[: len(launches)], "worker launch prefix")
    require(
        parent["status"] in ("complete", "failed", "incomplete", "running"), "controller status"
    )
    start, deadline = _clock(parent["started_ns"]), _clock(parent["deadline_ns"])
    equal(
        deadline - start,
        plan["contract"]["limits"]["total_timeout_s"] * 10**9,
        "controller deadline",
    )
    end = parent.get("ended_ns")
    if end is None:
        _note(report, "missing", "controller", "terminal controller timestamp unavailable")
    else:
        require(start <= _clock(end), "controller ends before start")
        if end >= deadline:
            _note(report, "required_failures", "controller", "global deadline exceeded")
    actual = {p.name for p in (root / "workers").glob("*") if p.is_dir()}
    require(actual <= {r["worker_id"] for r in launches}, "unlaunched worker artifacts")
    result, previous, acknowledged = {}, None, []
    for index, launch in enumerate(launches):
        specification = plan["workers"][index]
        wid = specification["worker_id"]
        try:
            launched = _clock(launch["launched_ns"])
            require(start <= launched < deadline, "worker launch outside controller lifetime")
            if index:
                require(
                    _clock(launches[index - 1]["returned_ns"]) <= launched,
                    "worker processes overlap",
                )
                require(
                    previous is not None
                    and previous["status"] == "complete"
                    and previous["passed"] is True
                    and not previous["failures"],
                    "worker launched after unsuccessful predecessor",
                )
            path = root / "workers" / wid / "manifest.json"
            if not path.exists():
                require(index == len(launches) - 1, "missing nonfinal worker")
                _note(report, "missing", wid, "worker manifest unavailable")
                break
            child = _read(root, str(path.relative_to(root)), report)
            for key, value in specification.items():
                equal(child[key], value, f"worker {key}")
            equal(
                [child["schema_version"], child["artifact_type"]],
                [1, "m4_attention_worker_manifest"],
                "worker artifact identity",
            )
            equal(child["deadline_ns"], deadline, "worker deadline")
            completed = child["completed_executions"]
            equal(completed, specification["execution_ids"][: len(completed)], "worker row prefix")
            acknowledged.extend(completed)
            require(launched <= _clock(child["started_ns"]) < deadline, "worker start ordering")
            require(
                child["status"] in ("complete", "failed", "incomplete", "running"), "worker status"
            )
            if child["status"] == "running" or child.get("ended_ns") is None:
                require(
                    index == len(launches) - 1 and parent["status"] != "complete",
                    "nonterminal worker before completed execution",
                )
                _note(report, "missing", wid, "interrupted worker; cleanup is unavailable")
                report["interrupted_workers"].append(wid)
                break
            ended = _clock(child["ended_ns"])
            returned = launch.get("returned_ns")
            require(child["started_ns"] <= ended, "worker ends before it starts")
            if returned is None:
                require(index == len(launches) - 1, "nonfinal worker lacks return timestamp")
                _note(report, "missing", wid, "controller did not record worker return")
            else:
                require(
                    ended <= _clock(returned) and (end is None or returned <= end),
                    "worker return ordering",
                )
            if ended >= deadline:
                _note(report, "required_failures", wid, "worker deadline exceeded")
            if "environment" not in child or child.get("model_loads") != 1:
                require(not completed, "acknowledged work without environment/model")
                _note(report, "missing", wid, "worker setup did not complete")
                break
            audit_worker_controls(plan, child, previous, require_cleanup=False)
            cleanup = child.get("teardown_after_workspace_release")
            clean = False
            if cleanup is None:
                _note(report, "missing", wid, "final allocator cleanup measurement unavailable")
            else:
                require(
                    set(cleanup) == {"allocated_bytes", "reserved_bytes"}
                    and all(type(v) is int and v >= 0 for v in cleanup.values())
                    and cleanup["reserved_bytes"] >= cleanup["allocated_bytes"],
                    "invalid worker cleanup measurements",
                )
                clean = all(v == 0 for v in cleanup.values())
                if not clean:
                    _note(report, "required_failures", wid, "final allocator cleanup is nonzero")
            if child["status"] == "complete":
                require(
                    child["passed"] is True and not child["failures"] and launch["exit_code"] == 0,
                    "inconsistent complete worker outcome",
                )
                equal(completed, specification["execution_ids"], "complete worker row order")
            else:
                require(index == len(launches) - 1, "failed worker has a successor")
                _note(report, "missing", wid, "worker stopped before successful completion")
            result[wid] = {
                "manifest": child,
                "eligible": clean and returned is not None and ended < deadline,
                "acknowledged": completed,
            }
            previous = child
        except ERRORS as error:
            _note(report, "errors", wid, error)
            break
    equal(done, acknowledged[: len(done)], "controller acknowledgments exceed worker prefix")
    for wid in parent["completed_workers"]:
        require(
            wid in result and result[wid]["manifest"]["status"] == "complete",
            "controller claims an unsuccessful completed worker",
        )
    if parent["status"] == "complete":
        equal(done, ids, "complete controller execution count")
        equal(parent["completed_workers"], expected, "complete controller worker count")
        require(not parent["failures"], "complete controller records failures")
    report["worker_records"] = result
    return result


def _outer(root, row, worker, report):
    relative = f"workers/{row['worker_id']}/{row['execution_id']}.lifetime.json"
    if not (root / relative).exists():
        _note(report, "missing", row["execution_id"], "outer lifetime marker unavailable")
        return None
    value = _read(root, relative, report)
    require(
        set(value) == {"execution_id", "started_ns", "deadline_ns", "case_completed_ns"},
        "outer lifetime fields",
    )
    equal(value["execution_id"], row["execution_id"], "outer lifetime identity")
    start, end, deadline = (
        _clock(value[k]) for k in ("started_ns", "case_completed_ns", "deadline_ns")
    )
    equal(deadline, min(worker["deadline_ns"], start + 600 * 10**9), "outer deadline")
    require(worker["started_ns"] <= start <= end <= worker["ended_ns"], "outer worker lifetime")
    if end >= deadline:
        _note(report, "required_failures", row["execution_id"], "outer case deadline exceeded")
    return value


def _benchmark(root, plan, row, workers, acknowledged, report):
    record = _run_record(root, row, schema.execution_view(plan), report["hashes"])
    value = record["result"]
    if value is None:
        _note(report, "missing", row["execution_id"], "benchmark result unavailable")
        return record
    errors = record["validation_errors"]
    # A missing zero is missing evidence; an observed nonzero is a required failure.
    for key in ("active_requests", "used_kv_blocks"):
        actual = value.get("cleanup", {}).get(key)
        message = f"cleanup.{key} must be zero"
        if message in errors and type(actual) is int and actual > 0:
            errors.remove(message)
            _note(report, "required_failures", row["execution_id"], message)
    try:
        for key, expected in row.items():
            equal(value.get(key), expected, f"frozen benchmark {key}")
        equal(
            value.get("comparison_eligible"),
            row["phase"] == "measured" and value["status"] == "complete",
            "benchmark phase eligibility",
        )
        context = workers.get(row["worker_id"])
        if context is None:
            _note(report, "missing", row["execution_id"], "no terminal audited worker")
            record["comparison_eligible"] = False
            return record
        worker = context["manifest"]
        outer = _outer(root, row, worker, report)
        record["outer_lifetime"] = outer
        if outer is None:
            record["comparison_eligible"] = False
            return record
        _memory(value, plan["workload_stats"][row["workload_id"]]["pool_bytes"])
        require("after_engine_release" in value["memory"], "engine release accounting unavailable")
        start, end = (_clock(value[k]) for k in ("case_started_ns", "case_completed_ns"))
        marker = _read(root, f"runs/{row['run_id']}/started.json", report)
        equal(
            marker,
            {
                "schema_version": 1,
                **row,
                "started_ns": start,
                "deadline_ns": min(outer["deadline_ns"], start + 600 * 10**9),
            },
            "inner benchmark watchdog marker",
        )
        require(
            outer["started_ns"]
            <= start
            <= _clock(value["arrival_ns"])
            <= _clock(value["synchronized_ns"])
            <= end
            <= outer["case_completed_ns"],
            "benchmark escape from outer lifetime",
        )
        record["comparison_eligible"] &= (
            context["eligible"]
            and row["execution_id"] in acknowledged
            and row["execution_id"] in context["acknowledged"]
            and outer["case_completed_ns"] < outer["deadline_ns"]
        )
        if row["execution_id"] not in acknowledged:
            _note(report, "missing", row["execution_id"], "result lacks controller acknowledgment")
    except ERRORS as error:
        errors.append(str(error))
    if errors:
        target = "errors" if value.get("status") == "complete" else "missing"
        for error in errors:
            _note(report, target, row["execution_id"], error)
        record["comparison_eligible"] = False
    return record


def _prerequisites(root, plan, report):
    from vllm_lt.validation.m4_attention import audit_numerical
    from vllm_lt.validation.m4_attention_held import audit_held

    for key, function, location in (
        ("numerical", audit_numerical, root),
        ("held", audit_held, root / "held"),
    ):
        if not (root / key).exists() or (
            key == "held" and not (root / "held" / "evaluations").exists()
        ):
            value = {
                "complete": False,
                "passed": False,
                "errors": [],
                "missing": [f"{key} execution unavailable"],
                "known_required_failure": False,
            }
        else:
            value = function(location, plan)
        report[key] = value
        for error in value.get("errors", []):
            _note(report, "errors", key, error)
        for missing in value.get("missing", []):
            _note(report, "missing", key, missing)
        if value.get("known_required_failure") and not value.get("errors"):
            _note(
                report, "required_failures", key, "independently reconstructed prerequisite failure"
            )
        if key == "held":
            for name, entry in value.get("artifacts", {}).items():
                report["hashes"]["held/" + name] = entry["sha256"]


def _chronology(root, plan, workers, parent, report):
    numerical = report["numerical"]
    cases = numerical.get("validated_prefix", {}).get(
        "case_lifetimes", numerical.get("ledger", {}).get("case_lifetimes", {})
    )
    held = {r["execution_id"]: r for r in report["held"].get("evaluations", [])}
    benchmark = {r["planned"]["execution_id"]: r for r in report["records"]}
    for wid, context in workers.items():
        worker, previous = context["manifest"], context["manifest"]["started_ns"]
        rows = [r for r in plan["execution_order"] if r["worker_id"] == wid]
        for row in rows:
            identity = row["execution_id"]
            if identity not in context["acknowledged"]:
                continue
            try:
                if row["kind"] == "numerical":
                    inner = cases.get(identity)
                    marker_path = root / f"numerical/cases/{identity}/m4-completed.json"
                    if inner is None or not marker_path.exists():
                        _note(
                            report,
                            "missing",
                            identity,
                            "audited numerical callback lifetime unavailable",
                        )
                        continue
                    marker = _read(root, f"numerical/cases/{identity}/m4-completed.json", report)
                    start, end = inner["started_ns"], marker["case_completed_ns"]
                    require(
                        inner["finished_ns"] <= end < inner["deadline_ns"],
                        "numerical callback deadline",
                    )
                    equal(
                        marker,
                        {
                            "case_id": identity,
                            "started_ns": start,
                            "deadline_ns": inner["deadline_ns"],
                            "case_completed_ns": end,
                        },
                        "numerical callback marker",
                    )
                else:
                    outer = (
                        benchmark[identity].get("outer_lifetime")
                        if row["kind"] == "benchmark"
                        else _outer(root, row, worker, report)
                    )
                    if outer is None:
                        continue
                    start, end = outer["started_ns"], outer["case_completed_ns"]
                    if row["kind"] != "benchmark":
                        inner = held.get(identity)
                        if inner is None:
                            _note(
                                report, "missing", identity, "acknowledged held audit unavailable"
                            )
                            continue
                        require(
                            start
                            <= inner["started_ns"]
                            <= inner["ended_ns"]
                            <= inner["case_completed_ns"]
                            <= end,
                            "held lifetime escapes controller callback",
                        )
                require(
                    previous <= _clock(start) <= _clock(end) <= worker["ended_ns"],
                    "within-worker mixed execution chronology",
                )
                previous = end
            except ERRORS as error:
                _note(report, "errors", identity, error)
    if any(w["worker_id"] == "A1" for w in parent["workers"]):
        gate = _read(root, "numerical-gate.json", report)
        equal(gate, parent["numerical_gate"], "stored prerequisite gate")
        require(
            gate["complete"]
            and gate["passed"]
            and all(report[k]["complete"] and report[k]["passed"] for k in ("numerical", "held")),
            "timing launched without independently qualified prerequisites",
        )
        a1 = next(r for r in parent["workers"] if r["worker_id"] == "A1")
        nb = next(r for r in parent["workers"] if r["worker_id"] == "N-B")
        require(
            nb["returned_ns"] <= _clock(parent["numerical_gate_ns"]) < a1["launched_ns"],
            "prerequisite gate chronology",
        )


def _profile_structure(profile, row, record, plan):
    require(
        not profile["validation_errors"] and profile["gpu_trace_available"],
        "profile lacks valid actual GPU attribution",
    )
    equal(profile["metadata"]["kv_metadata_path"], "prepared", "both profile metadata paths")
    scopes = {r["name"]: r["count"] for r in profile["cpu_inclusive_scopes"]}
    traversals, prefill = (64, 0) if row["workload_id"] == "W1" else (27, 640)
    for name, count in (
        ("recurrent", traversals),
        ("kv_prepare", traversals),
        ("kv_write", traversals * 24),
        ("attention", traversals * 24),
    ):
        equal(scopes.get("vllm_lt::" + name, 0), count, "prepared compact profile " + name)
    equal(profile["interleaved_prefill_tokens"], prefill, "frozen profile prefill work")
    require(profile["gpu_kernels"] and profile["gpu_memcpy"], "actual kernels and copies required")
    attention = sum(
        r["count"] for r in profile["gpu_kernels"] if "_paged_attention_kernel" in r["name"]
    )
    equal(attention, traversals * 24, "actual attention kernel count")
    metadata = profile["metadata"]
    history = [
        r["batch"]
        for r in record["result"]["counts"]["snapshots"]
        if metadata["start_step"] <= r["batch"]["step_id"] <= metadata["end_step"]
    ]
    require(history, "actual profile batch history unavailable")
    equal(
        [metadata["start_step"], metadata["end_step"]],
        [2, 97 if row["workload_id"] == "W1" else 35],
        "frozen profile step window",
    )
    equal(
        sum(r["stage"] == "recurrent" for r in history),
        64 if row["workload_id"] == "W1" else 7,
        "profile decode dispatch count",
    )
    equal(
        sum(r["stage"] == "prefill" for r in history),
        0 if row["workload_id"] == "W1" else 5,
        "profile prefill dispatch count",
    )
    profile["matched_batch_history"] = history
    profile["source_commit"] = plan["implementations"][row["implementation_id"]]["source"]["commit"]
    profile["launch_policy"] = {
        "BLOCK_T": schema.TILE_POLICY[row["implementation_id"]],
        "num_warps": 4,
    }


def build_report(output_dir):
    from .m4_attention import artifact_usage

    root = Path(output_dir).resolve()
    report = {
        "schema_version": 1,
        "artifact_type": "m4_attention_report",
        "evidence_status": "incomplete",
        "decision": "inconclusive",
        "milestone_status": "unqualified_attention_candidate",
        "errors": [],
        "missing": [],
        "required_failures": [],
        "records": [],
        "cells": [],
        "profiles": [],
        "hashes": {},
        "worker_records": {},
        "interrupted_workers": [],
        "limitations": [
            "Two observations per side show observed ranges, not statistical confidence.",
            "M4 permits diagnostic floating gate deltas; finite counts and actual "
            "token/depth/work histories remain exact.",
            "Both sides use compact prepared metadata; no graph performance or proof is claimed.",
            "Profile event sums are diagnostic, not end-to-end savings; "
            "observer scans are benchmark overhead.",
            "BF16, task quality and unavailable device memory-access checker remain unqualified.",
            "Legacy lifecycle failures remain unqualified audit errors unless the held auditor "
            "can independently reconstruct a required numerical failure.",
        ],
    }
    try:
        plan = _read(root, "plan.json", report)
        schema.validate_plan(plan)
        parent = _read(root, "manifest.json", report)
        equal(
            [parent["schema_version"], parent["artifact_type"], parent["plan_sha256"]],
            [1, "m4_attention_manifest", plan["plan_sha256"]],
            "M4 controller identity",
        )
        report["plan_sha256"] = plan["plan_sha256"]
    except ERRORS as error:
        _note(report, "errors", "plan/manifest", error)
        report["evidence_status"] = "invalid"
        return report
    report["execution_failures"] = parent["failures"]
    workers = {}
    try:
        workers = _workers(root, plan, parent, report)
    except ERRORS as error:
        _note(report, "errors", "workers/prefix", error)
    try:
        report["artifact_usage"] = artifact_usage(root, plan)
    except ValueError as error:
        _note(report, "required_failures", "artifact resources", error)
    except OSError as error:
        _note(report, "missing", "artifact resources", error)
    try:
        _prerequisites(root, plan, report)
    except ERRORS as error:
        _note(report, "errors", "prerequisites", error)
        report.setdefault("numerical", {"complete": False, "passed": False})
        report.setdefault("held", {"complete": False, "passed": False})
    rows = [r for r in plan["execution_order"] if r["kind"] == "benchmark"]
    expected = {r["run_id"] for r in rows}
    actual = {p.name for p in (root / "runs").glob("*") if p.is_dir()}
    for name in actual - expected:
        _note(report, "errors", name, "unplanned benchmark artifacts")
    launched_workers = {worker["worker_id"] for worker in parent["workers"]}
    for row in rows:
        if row["run_id"] in actual and row["worker_id"] not in launched_workers:
            _note(report, "errors", row["run_id"], "benchmark artifacts from unlaunched worker")
    for row in rows:
        report["records"].append(
            _benchmark(root, plan, row, workers, parent["completed_executions"], report)
        )
    try:
        _chronology(root, plan, workers, parent, report)
    except ERRORS as error:
        _note(report, "errors", "mixed chronology/gate", error)
    report["cells"] = pair_results(
        plan, report["records"], pair_prefix="M4", compare_gate_values=False
    )
    for cell in report["cells"]:
        for pair in cell["pairs"]:
            if pair["status"] == "failed":
                _note(report, "required_failures", pair["pair_id"], pair["gates"])
            elif pair["status"] == "invalid":
                members = [
                    r for r in report["records"] if r["planned"].get("pair_id") == pair["pair_id"]
                ]
                if len(members) == 2 and all(r["comparison_eligible"] for r in members):
                    _note(report, "errors", pair["pair_id"], pair["errors"])
    if "artifact_usage" in report:
        profiles = _profiles(root, report["hashes"], report["records"], schema.execution_view(plan))
        expected_profiles = {r["run_id"]: r for r in rows if r["phase"] == "profile"}
        by_id = {p["capture_id"]: p for p in profiles}
        for name in set(by_id) - set(expected_profiles):
            _note(report, "errors", name, "unplanned profile")
        for name, row in expected_profiles.items():
            if name not in by_id or not all(
                (root / "profiles" / name / f).exists() for f in ("trace.json", "metadata.json")
            ):
                _note(report, "missing", name, "profile unavailable")
                continue
            try:
                record = next(r for r in report["records"] if r["run_id"] == name)
                require(
                    record["status"] == "complete" and not record["validation_errors"],
                    "profile run is invalid",
                )
                _profile_structure(by_id[name], row, record, plan)
            except ERRORS as error:
                _note(report, "errors", name, error)
        for workload in ("W1", "W4"):
            matched = [
                p
                for p in profiles
                if p.get("matched_batch_history")
                and expected_profiles[p["capture_id"]]["workload_id"] == workload
            ]
            if len(matched) == 2:
                try:
                    equal(
                        matched[0]["matched_batch_history"],
                        matched[1]["matched_batch_history"],
                        "paired profile batch work",
                    )
                except ValueError as error:
                    _note(report, "errors", workload, error)
        report["profiles"] = profiles
    else:
        _note(report, "missing", "profiles", "resource accounting prevents profile parsing")
    complete = (
        parent["status"] == "complete"
        and len(parent["completed_executions"]) == 139
        and not report["missing"]
        and report["numerical"]["complete"]
        and report["held"]["complete"]
        and all(r["status"] == "complete" and not r["validation_errors"] for r in report["records"])
    )
    report["evidence_status"] = (
        "invalid" if report["errors"] else "complete" if complete else "incomplete"
    )
    report["decision"] = (
        "inconclusive"
        if report["errors"]
        else "failed"
        if report["required_failures"]
        else "passed"
        if complete and all(c["status"] == "passed" for c in report["cells"])
        else "inconclusive"
    )
    if report["decision"] == "passed":
        report["milestone_status"] = "candidate_passed_pending_manual_evidence_review"
    report["counts"] = {
        "planned_executions": 139,
        "completed_executions": len(parent["completed_executions"]),
        "benchmark_runs": len(report["records"]),
        "eligible_timing_runs": sum(r["comparison_eligible"] for r in report["records"]),
        "profiles": len(report["profiles"]),
        "workers": len(workers),
    }
    return report


def write_report(output_dir):
    root = Path(output_dir).resolve()
    report = build_report(root)
    write_json(root / "m4-attention-report.json", report)
    lines = [
        "# M4 attention evidence",
        "",
        f"Evidence: **{report['evidence_status']}**; decision: **{report['decision']}**.",
        "",
        "| Cell | Decision | A tokens/s | B tokens/s |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c['cell_id']} | {c['status']} | {c['baseline_values']} | {c['candidate_values']} |"
        for c in report["cells"]
    ]
    lines += ["", *report["limitations"], ""]
    for kind in ("required_failures", "errors", "missing"):
        lines += [f"- {kind}: {r['scope']}: {r['message']}" for r in report[kind]]
    (root / "m4-attention-report.md").write_text("\n".join(lines) + "\n")
    return report
