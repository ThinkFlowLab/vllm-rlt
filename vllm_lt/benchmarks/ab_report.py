"""Offline M2 pairing; M1 owns event/metric reconstruction and actual profile parsing."""

import math
from pathlib import Path

from .ab_schema import CELLS, WORKERS, equal, execution_view, require, validate_ab_plan
from .report import _profiles, _run_record
from .schema import _file_record, read_json, write_json


def audit_worker_launches(manifest):
    launches = manifest["workers"]
    equal(
        [row["worker_id"] for row in launches],
        list(WORKERS)[: len(launches)],
        "worker launch order",
    )
    if manifest["status"] == "complete":
        equal(len(launches), len(WORKERS), "eight actual launches")


def audit_worker_lifetime(manifest, index, child, *, interrupted=False):
    launches = manifest["workers"]
    launch = launches[index]
    times = [manifest["started_ns"], launch["launched_ns"], child["started_ns"]]
    if not interrupted:
        times.append(child["ended_ns"])
    times.extend([launch["returned_ns"], manifest["ended_ns"]])
    require(all(a <= b for a, b in zip(times, times[1:])), "worker process lifetime")
    if index:
        require(launches[index - 1]["returned_ns"] <= launch["launched_ns"], "overlapping workers")
    if child["worker_id"] == "A1":
        require(
            launches[index - 1]["returned_ns"]
            <= manifest["numerical_gate_ns"]
            < launch["launched_ns"],
            "correctness gate must precede timed workers",
        )
        require(
            manifest["numerical_gate"]["complete"] is True
            and manifest["numerical_gate"]["passed"] is True,
            "failed pre-timing gate",
        )


def _finite(value, label, *, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"invalid {label}",
    )
    return value


def _memory(result, expected_pool):
    memory = result["memory"]
    equal(memory["pool_bytes"], expected_pool, "frozen KV pool bytes")
    for key, value in memory.items():
        if isinstance(value, dict):
            values = value.values()
            require(
                value["reserved_bytes"] >= value["allocated_bytes"],
                "reserved memory must cover allocated memory",
            )
        else:
            values = (value,)
        require(all(type(item) is int and item >= 0 for item in values), "integer memory bytes")
    require(
        memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"],
        "peak reserved memory must cover peak allocated memory",
    )
    require(type(result["setup_ns"]) is int and result["setup_ns"] >= 0, "integer setup time")


def _work_identity(result):
    return {
        "requests": [
            {
                key: row[key]
                for key in ("request_id", "token_ids", "exit_depths", "finished", "finish_reason")
            }
            for row in result["requests"]
        ],
        "counts": {
            key: result["counts"][key]
            for key in (
                "gate_probabilities",
                "logical_copy_bytes",
                "nonfinite_gate_probabilities",
                "recurrent_depth_counts",
                "recurrent_occupancy",
                "request_work",
                "stage_counts",
                "stage_tokens",
            )
        },
    }


def pair_results(plan, records, *, pair_prefix="M2", pair_extra=None):
    """Both observations must meet their own limits; never average away a failure."""
    acceptance = plan["contract"]["acceptance"]
    measured = [row for row in records if row["planned"]["phase"] == "measured"]
    cells = []
    for cell in CELLS:
        pairs, avalues, bvalues = [], [], []
        for repetition in (1, 2):
            pair_id = f"{pair_prefix}-{cell}-{repetition}"
            members = [row for row in measured if row["planned"]["pair_id"] == pair_id]
            item = {"pair_id": pair_id, "status": "invalid", "errors": []}
            try:
                require(len(members) == 2, "matched pair needs exactly two observations")
                by_impl = {row["planned"]["implementation_id"]: row for row in members}
                require(set(by_impl) == {"A", "B"}, "pair must contain one A and one B")
                arow, brow = by_impl["A"], by_impl["B"]
                for row in members:
                    require(row["comparison_eligible"], "pair has missing/failed/invalid evidence")
                    equal(
                        [row["planned"]["cell_id"], row["planned"]["repetition"]],
                        [cell, repetition],
                        "pair stratum",
                    )
                for field in ("controls_sha256", "workload_sha256", "instrumentation"):
                    equal(arow["planned"][field], brow["planned"][field], f"paired {field}")
                a, b = arow["result"], brow["result"]
                for result in (a, b):
                    _memory(
                        result, plan["workload_stats"][arow["planned"]["workload_id"]]["pool_bytes"]
                    )
                equal(
                    _work_identity(a), _work_identity(b), "actual A/B token/depth/gate/work history"
                )
                equal(a["memory"]["pool_bytes"], b["memory"]["pool_bytes"], "fixed KV pool bytes")
                av = _finite(
                    arow["recomputed_metrics"]["generated_tokens_per_second"],
                    "A throughput",
                    positive=True,
                )
                bv = _finite(
                    brow["recomputed_metrics"]["generated_tokens_per_second"],
                    "B throughput",
                    positive=True,
                )
                target = (
                    acceptance["target_ratio_min"]
                    if cell == acceptance["target_cell"]
                    else acceptance["control_ratio_min"]
                )
                increases = {}
                for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
                    increases[name] = _finite(b["memory"][name], name) - _finite(
                        a["memory"][name], name
                    )
                setup = _finite(b["setup_ns"], "B setup") - _finite(a["setup_ns"], "A setup")
                gates = {
                    "throughput": bv / av >= target,
                    **{
                        name: value <= acceptance["peak_increase_bytes_max"]
                        for name, value in increases.items()
                    },
                }
                item.update(
                    status="passed" if all(gates.values()) else "failed",
                    gates=gates,
                    baseline_tokens_per_s=av,
                    candidate_tokens_per_s=bv,
                    candidate_over_baseline=bv / av,
                    required_ratio_min=target,
                    throughput_improvement_percent=100 * (bv / av - 1),
                    peak_increases_bytes=increases,
                    setup_increase_ns=setup,
                    baseline_run_id=arow["run_id"],
                    candidate_run_id=brow["run_id"],
                    baseline_metrics=arow["recomputed_metrics"],
                    candidate_metrics=brow["recomputed_metrics"],
                )
                if pair_extra is None:
                    gates["setup"] = setup <= acceptance["setup_increase_ns_max"]
                else:
                    pair_extra(arow, brow, item, acceptance)
                item["status"] = "passed" if all(gates.values()) else "failed"
                avalues.append(av)
                bvalues.append(bv)
            except (ValueError, KeyError, TypeError) as exc:
                item["status"] = "invalid"
                item["errors"].append(str(exc))
            pairs.append(item)
        complete = len(avalues) == len(bvalues) == 2
        separated = (min(bvalues) > max(avalues)) if complete else None
        status = (
            "failed"
            if any(pair["status"] == "failed" for pair in pairs)
            else "inconclusive"
            if not complete
            else "inconclusive"
            if (cell == acceptance["target_cell"] and not separated)
            else "passed"
        )
        cells.append(
            {
                "cell_id": cell,
                "status": status,
                "pairs": pairs,
                "baseline_values": avalues,
                "candidate_values": bvalues,
                "baseline_range": [min(avalues), max(avalues)] if avalues else None,
                "candidate_range": [min(bvalues), max(bvalues)] if bvalues else None,
                "strict_range_separation": separated,
                "variation_gate_required": cell == acceptance["target_cell"],
            }
        )
    return cells


def _audit_profile_structure(profile, implementation):
    metadata = profile["metadata"]
    equal(
        metadata["kv_metadata_path"],
        "public" if implementation == "A" else "prepared",
        "profile metadata implementation path",
    )
    require(
        not profile["validation_errors"] and profile["gpu_trace_available"],
        "profile lacks valid GPU attribution",
    )
    scopes = {row["name"]: row["count"] for row in profile["cpu_inclusive_scopes"]}
    recurrent = scopes.get("vllm_lt::recurrent", 0)
    require(type(recurrent) is int and recurrent > 0, "profile lacks recurrent traversals")
    equal(scopes.get("vllm_lt::kv_write"), recurrent * 24, "24 writes per traversal")
    equal(scopes.get("vllm_lt::attention"), recurrent * 24, "24 attention calls per traversal")
    equal(
        scopes.get("vllm_lt::kv_prepare", 0),
        recurrent if implementation == "B" else 0,
        "one candidate metadata preparation per traversal",
    )
    require(
        bool(profile["gpu_kernels"]) and bool(profile["gpu_memcpy"]),
        "GPU kernel and memcpy events are both required",
    )


def build_report(output_dir):
    from vllm_lt.validation.m2 import audit_numerical

    from .ab import artifact_usage, audit_worker_controls

    output_dir = Path(output_dir).resolve()
    report = {
        "schema_version": 1,
        "artifact_type": "m2_ab_report",
        "evidence_status": "incomplete",
        "decision": "inconclusive",
        "milestone_status": "manual_profile_attribution_review_required",
        "errors": [],
        "records": [],
        "cells": [],
        "profiles": [],
        "hashes": {},
        "limitations": [
            "Two observations per implementation give a range, not statistical confidence.",
            "Feasibility, warmup, numerical checks and profiles are excluded from timing.",
            "Profiler scope times overlap and include instrumentation overhead.",
            "Metadata copy/synchronization attribution needs reviewed correlation-based analysis "
            "across preparation, write, attention and position construction before M2 promotion.",
        ],
    }
    try:
        plan = read_json(output_dir / "plan.json")
        validate_ab_plan(plan)
        report["plan_sha256"] = plan["plan_sha256"]
        manifest = read_json(output_dir / "manifest.json")
        equal(
            [manifest["schema_version"], manifest["artifact_type"], manifest["plan_sha256"]],
            [1, "m2_ab_manifest", plan["plan_sha256"]],
            "M2 manifest identity",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["evidence_status"] = "invalid"
        report["errors"].append({"scope": "plan/manifest", "message": str(exc)})
        return report
    report["manifest_status"] = manifest["status"]
    report["execution_failures"] = manifest["failures"]
    for path in (output_dir / "plan.json", output_dir / "manifest.json"):
        report["hashes"][str(path.relative_to(output_dir))] = _file_record(path)["sha256"]
    worker_results = {}
    try:
        equal(
            manifest["completed_workers"],
            list(WORKERS)[: len(manifest["completed_workers"])],
            "completed worker prefix",
        )
        expected_ids = [row["execution_id"] for row in plan["execution_order"]]
        completed = manifest["completed_executions"]
        equal(completed, expected_ids[: len(completed)], "completed execution prefix")
        require(
            manifest["deadline_ns"] - manifest["started_ns"] == 7200 * 10**9,
            "global deadline contract differs",
        )
        require(
            manifest["started_ns"] <= manifest["ended_ns"] < manifest["deadline_ns"],
            "global execution time is invalid or over budget",
        )
        previous = None
        launches = manifest["workers"]
        audit_worker_launches(manifest)
        actual_workers = {path.name for path in (output_dir / "workers").glob("*") if path.is_dir()}
        require(actual_workers <= set(WORKERS), "unplanned worker artifacts")
        for index, launch in enumerate(launches):
            worker = plan["workers"][index]
            path = output_dir / "workers" / worker["worker_id"] / "manifest.json"
            child = read_json(path)
            worker_results[worker["worker_id"]] = child
            report["hashes"][str(path.relative_to(output_dir))] = _file_record(path)["sha256"]
            equal(
                [
                    child["schema_version"],
                    child["artifact_type"],
                    child["worker_id"],
                    child["implementation_id"],
                ],
                [1, "m2_worker_manifest", worker["worker_id"], worker["implementation_id"]],
                "worker identity",
            )
            require(
                child["status"] == "complete"
                and child["passed"] is True
                and not child["failures"]
                and launch["exit_code"] == 0,
                "worker was unsuccessful",
            )
            equal(
                child["completed_executions"], worker["execution_ids"], "worker completed row order"
            )
            equal(child["model_loads"], 1, "one model load per worker")
            equal(child["deadline_ns"], manifest["deadline_ns"], "worker global deadline")
            audit_worker_lifetime(manifest, index, child)
            audit_worker_controls(plan, child, previous)
            previous = child
        report["artifact_usage"] = artifact_usage(output_dir, plan["contract"]["limits"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append({"scope": "workers/controls/order", "message": str(exc)})
    rows = [row for row in plan["execution_order"] if row["kind"] == "benchmark"]
    expected_runs = {row["run_id"] for row in rows}
    actual_runs = {path.name for path in (output_dir / "runs").glob("*") if path.is_dir()}
    for name in sorted(actual_runs - expected_runs):
        report["errors"].append({"scope": name, "message": "unplanned benchmark execution"})
    view = execution_view(plan)
    previous_case_end = {}
    for row in rows:
        record = _run_record(output_dir, row, view, report["hashes"])
        if record["result"] is not None:
            for key, value in row.items():
                if record["result"].get(key) != value:
                    record["validation_errors"].append(f"{key} differs from frozen M2 row")
            if record["result"].get("comparison_eligible") != (row["phase"] == "measured"):
                record["validation_errors"].append("phase exclusion/eligibility is inconsistent")
            started, ended = (
                record["result"].get(key) for key in ("case_started_ns", "case_completed_ns")
            )
            if not (
                type(started) is int and type(ended) is int and 0 <= ended - started <= 600 * 10**9
            ):
                record["validation_errors"].append(
                    "whole-case lifetime is absent or exceeds 600 seconds"
                )
            worker = worker_results.get(row["worker_id"])
            try:
                require(worker is not None, "run has no audited worker")
                _memory(record["result"], plan["workload_stats"][row["workload_id"]]["pool_bytes"])
                marker = read_json(output_dir / "runs" / row["run_id"] / "started.json")
                equal(
                    marker,
                    {
                        "schema_version": 1,
                        **row,
                        "started_ns": started,
                        "deadline_ns": min(worker["deadline_ns"], started + 600 * 10**9),
                    },
                    "case start/watchdog marker",
                )
                require(
                    worker["started_ns"]
                    <= started
                    <= record["result"]["arrival_ns"]
                    <= record["result"]["synchronized_ns"]
                    <= ended
                    <= worker["ended_ns"],
                    "run timestamps fall outside their assigned worker",
                )
                require(
                    previous_case_end.get(row["worker_id"], started) <= started,
                    "within-worker execution order overlaps or reverses",
                )
                previous_case_end[row["worker_id"]] = ended
            except (OSError, ValueError, TypeError, KeyError) as exc:
                record["validation_errors"].append(str(exc))
            if "after_engine_release" not in record["result"].get("memory", {}):
                record["validation_errors"].append("engine release accounting is missing")
        if record["validation_errors"]:
            record["comparison_eligible"] = False
        report["records"].append(record)
    report["cells"] = pair_results(plan, report["records"])
    report["numerical"] = audit_numerical(output_dir, plan)
    try:
        lifetimes = report["numerical"]["ledger"]["case_lifetimes"]
        for case in plan["numerical"]["execution_order"]:
            worker_id = "N-" + case["implementation_id"]
            worker = worker_results[worker_id]
            lifetime = lifetimes[case["case_id"]]
            require(
                previous_case_end[worker_id]
                <= lifetime["started_ns"]
                <= lifetime["finished_ns"]
                <= worker["ended_ns"],
                "numerical execution must follow feasibility within its assigned worker",
            )
            previous_case_end[worker_id] = lifetime["finished_ns"]
    except (KeyError, TypeError, ValueError) as exc:
        report["errors"].append({"scope": "numerical/order", "message": str(exc)})
    profiles = _profiles(output_dir, report["hashes"], report["records"], view)
    expected_profiles = {row["run_id"]: row for row in rows if row["phase"] == "profile"}
    try:
        equal(
            sorted(item["capture_id"] for item in profiles),
            sorted(expected_profiles),
            "four declared profile captures",
        )
        for profile in profiles:
            _audit_profile_structure(
                profile, expected_profiles[profile["capture_id"]]["implementation_id"]
            )
    except (ValueError, KeyError, TypeError) as exc:
        report["errors"].append({"scope": "profiles", "message": str(exc)})
    report["profiles"] = profiles
    complete = (
        manifest["status"] == "complete"
        and not manifest["failures"]
        and manifest["completed_workers"] == list(WORKERS)
        and len(manifest["completed_executions"]) == 105
        and report["numerical"]["complete"]
        and all(
            row["status"] == "complete" and not row["validation_errors"]
            for row in report["records"]
        )
    )
    report["evidence_status"] = (
        "invalid" if report["errors"] else "complete" if complete else "incomplete"
    )
    if report["evidence_status"] == "complete":
        report["decision"] = (
            "failed"
            if not report["numerical"]["passed"]
            or any(cell["status"] == "failed" for cell in report["cells"])
            else "inconclusive"
            if any(cell["status"] == "inconclusive" for cell in report["cells"])
            else "passed"
        )
    report["counts"] = {
        "planned_executions": 105,
        "completed_executions": len(manifest["completed_executions"]),
        "benchmark_runs": len(report["records"]),
        "eligible_timing_runs": sum(row["comparison_eligible"] for row in report["records"]),
    }
    return report


def write_report(output_dir):
    output_dir = Path(output_dir).resolve()
    report = build_report(output_dir)
    write_json(output_dir / "ab-report.json", report)
    lines = [
        "# M2 A/B evidence",
        "",
        f"Evidence: **{report['evidence_status']}**. "
        f"Performance/required numerical decision: **{report['decision']}**.",
        "",
        "| Cell | Decision | A values (tokens/s) | B values (tokens/s) |",
        "| --- | --- | --- | --- |",
    ]
    for cell in report["cells"]:
        lines.append(
            f"| {cell['cell_id']} | {cell['status']} | "
            f"{cell['baseline_values']} | {cell['candidate_values']} |"
        )
    lines.extend(["", *report["limitations"], ""])
    for error in report["errors"]:
        lines.append(f"- {error['scope']}: {error['message']}")
    (output_dir / "ab-report.md").write_text("\n".join(lines) + "\n")
    return report
