"""Offline acceptance for the frozen eager-versus-replay experiment."""

import math
import re
from pathlib import Path

from benchmarks.capture.schema import PAIR_PREFIX
from vllm_lt.benchmarks.ab_report import (
    _finite,
    _memory,
    audit_worker_launches,
    audit_worker_lifetime,
)
from vllm_lt.benchmarks.ab_schema import WORKERS, equal, require
from vllm_lt.benchmarks.report import _event_problems, _profiles, _read_events, _run_record
from vllm_lt.benchmarks.schema import _file_record, read_json, write_json


def _ttft(record):
    requests = record["recomputed_metrics"]["per_request"]
    require(len(requests) == 1, "W1 TTFT requires its one frozen request")
    return _finite(next(iter(requests.values()))["ttft_ns"], "W1 TTFT", positive=True)


def _capture_pair(arow, brow, pair, acceptance):
    repetition = arow["planned"]["repetition"]
    equal(
        [arow["planned"]["worker_id"], brow["planned"]["worker_id"]],
        [f"A{repetition}", f"B{repetition}"],
        "AB/BA worker assignment",
    )
    if arow["planned"]["cell_id"] == acceptance["target_cell"]:
        at, bt = _ttft(arow), _ttft(brow)
        pair["gates"]["ttft"] = bt <= at * acceptance["target_ttft_ratio_max"]
        pair["ttft"] = {"baseline_ns": at, "candidate_ns": bt, "candidate_over_baseline": bt / at}
    else:
        pair["ttft"] = None
    saving = 1e9 * (1 / pair["baseline_tokens_per_s"] - 1 / pair["candidate_tokens_per_s"])
    pair.update(
        baseline_setup_ns=arow["result"]["setup_ns"],
        candidate_setup_ns=brow["result"]["setup_ns"],
        estimated_setup_break_even_tokens=(
            math.ceil(max(0, pair["setup_increase_ns"]) / saving) if saving > 0 else None
        ),
    )


def pair_results(plan, records):
    from vllm_lt.benchmarks.ab_report import pair_results as shared_pairs

    return shared_pairs(plan, records, pair_prefix=PAIR_PREFIX, pair_extra=_capture_pair)


def audit_capture_record(
    capture, planned, limits, *, model_config, block_size, expected_device="cuda:0"
):
    """Audit setup inventories and measured counters, without per-step tensor reads."""
    from benchmarks.capture.validation import (
        _audit_graph_close,
        _audit_graph_setup,
        _audit_graph_snapshot,
    )

    equal(capture["schema_version"], 1, "capture schema")
    side = planned["implementation_id"]
    equal(capture["implementation_id"], side, "capture implementation")
    equal(capture["use_graphs"], side == "B", "capture mode")
    initial, final, closed = (capture[key] for key in ("initial", "final", "closed"))
    equal(initial["backend"], "triton", "frozen performance attention backend")
    equal(final["backend"], initial["backend"], "unchanged backend")
    _audit_graph_setup(
        initial,
        model_config,
        replay=side == "B",
        block_size=block_size,
        expected_device=expected_device,
        limits=limits,
    )
    equal(initial["use_graphs"], side == "B", "actual executor mode")
    equal(initial["limits"], limits, "actual graph limits")
    _audit_graph_snapshot(initial, final)
    setup = initial["setup"]
    require(setup["finished_ns"] < setup["deadline_ns"], "setup time cap")
    equal(
        initial["fallback_counts"],
        dict.fromkeys(("backend", "live_count", "table_width"), 0),
        "initial fallback counters",
    )
    count, fallbacks = final["counters"], final["fallback_counts"]
    equal(sorted(count), sorted(initial["counters"]), "counter inventory")
    equal(sorted(fallbacks), sorted(initial["fallback_counts"]), "fallback counter inventory")
    require(
        all(type(v) is int and v >= 0 for v in (*count.values(), *fallbacks.values())),
        "nonnegative actual counters",
    )
    equal(count["prepared"], count["eager"] + count["replays"], "one launch per preparation")
    equal(count["committed"], count["prepared"], "all submitted transactions committed")
    equal(
        count["completed"], count["prepared"] + sum(fallbacks.values()), "completed dispatch count"
    )
    equal(count["calls"], count["completed"] + count["empty"], "total dispatch accounting")
    equal(count["eager" if side == "B" else "replays"], 0, "actual selected execution path")
    for name in ("prepared", "eager", "replays", "committed", "completed"):
        equal(
            sum(b["counters"][name] for b in final["buckets"].values()),
            count["prepared"] if name == "completed" else count[name],
            "bucket/total counters",
        )
    for key, before in initial["buckets"].items():
        after = final["buckets"][key]
        require(after["in_use"] is False and after["failed"] is False, "unfinished/failed lease")
        equal(before["setup_generation"], 1 if side == "B" else 0, "postsetup generation baseline")
        equal(
            after["generation"],
            before["generation"] + after["counters"]["prepared"],
            "one generation per steady traversal",
        )
        require(
            all(type(v) is int and v >= 0 for v in after["counters"].values()), "bucket counters"
        )
    _audit_graph_close(final, closed)
    equal(capture["cleanup"], {"closed": True}, "graph cleanup marker")
    if planned["phase"] != "profile":
        equal(capture["profile_dispatches"], [], "no profile instrumentation in timing")
    return {
        "setup_ns": setup["finished_ns"] - setup["started_ns"],
        "memory_deltas": setup["memory_deltas"],
        "fallback_counts": fallbacks,
        "counters": count,
        "buckets": {k: v["counters"] for k, v in final["buckets"].items()},
    }


_DISPATCH = re.compile(
    r"^vllm_lt::graph_dispatch::(\d+)::bucket::(\d+)::(eager|replay|compact|empty)$"
)
_REPLAY = re.compile(r"^vllm_lt::graph_replay::(\d+)::exec::(\d+)$")


def _contains(outer, inner):
    return (
        outer.get("pid") == inner.get("pid")
        and outer.get("tid") == inner.get("tid")
        and outer["ts"] <= inner["ts"]
        and inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"]
    )


def audit_graph_profile(trace, capture):
    """Link user dispatches to CUDA launch events and actual graph kernels offline."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    events = [event for event in events if event.get("ph") == "X"]
    for event in events:
        _finite(event["ts"], "trace timestamp")
        _finite(event["dur"], "trace duration")
    # Kineto may mirror record_function scopes as gpu_user_annotation events.
    # Only the original host scopes contain the Python/CUDA launch boundary.
    host_scopes = [e for e in events if e.get("cat") == "user_annotation"]
    dispatches = [(e, _DISPATCH.fullmatch(e.get("name", ""))) for e in host_scopes]
    dispatches = sorted(
        ((e, m) for e, m in dispatches if m is not None), key=lambda pair: pair[0]["ts"]
    )
    records = capture["profile_dispatches"]
    inventories, final_buckets = capture["initial"]["buckets"], capture["final"]["buckets"]
    last_generation = {}
    equal(len(dispatches), len(records), "one trace scope per observed profile dispatch")
    require(bool(records), "profile has no recurrent dispatch evidence")
    record_by_id = {r["dispatch_id"]: r for r in records}
    equal(len(record_by_id), len(records), "unique observed dispatch IDs")
    launches = [
        e
        for e in events
        if e.get("name", "").startswith("cudaGraphLaunch")
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
    ]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    replay_scopes = [(e, _REPLAY.fullmatch(e.get("name", ""))) for e in host_scopes]
    replay_scopes = [(e, m) for e, m in replay_scopes if m is not None]
    used_launches, used_scopes, seen_dispatches = set(), set(), set()
    graph_buckets, rows = {}, []
    used_correlations, linked_kernel_ids = set(), set()
    for scope, match in dispatches:
        dispatch_id, bucket, kind = match.groups()
        dispatch_id = int(dispatch_id)
        bucket = None if bucket == "0" else int(bucket)
        require(dispatch_id not in seen_dispatches, "duplicate dispatch trace scope")
        seen_dispatches.add(dispatch_id)
        record = record_by_id[dispatch_id]
        equal([record["bucket_id"], record["kind"]], [bucket, kind], "actual dispatch scope")
        if kind in ("eager", "replay"):
            require(
                str(bucket) in inventories and str(bucket) in final_buckets,
                "supported dispatch bucket must belong to the executor inventory",
            )
            owned = inventories[str(bucket)]
            generation = record["generation"]
            require(
                type(generation) is int
                and owned["generation"] < generation <= final_buckets[str(bucket)]["generation"],
                "profile generation outside actual bucket lifetime",
            )
            if bucket in last_generation:
                equal(
                    generation, last_generation[bucket] + 1, "consecutive profile bucket generation"
                )
            last_generation[bucket] = generation
            equal(record["graph_exec_id"], owned["graph_exec_id"], "profile executable inventory")
            equal(
                kind,
                "replay" if capture["use_graphs"] else "eager",
                "actual configured profile path",
            )
        else:
            equal(
                [bucket, record["generation"], record["graph_exec_id"]],
                [None, None, None],
                "compact/empty graph metadata",
            )
        nested = [(e, m) for e, m in replay_scopes if _contains(scope, e)]
        own_launches = [e for e in launches if _contains(scope, e)]
        if kind != "replay":
            require(not nested and not own_launches, "non-replay dispatch launched a graph")
            rows.append(
                {
                    "dispatch_id": dispatch_id,
                    "bucket_id": bucket,
                    "kind": kind,
                    "graph_launches": 0,
                    "graph_kernel_count": 0,
                }
            )
            continue
        require(
            capture["use_graphs"] is True and len(nested) == len(own_launches) == 1,
            "replay requires one actual graph scope/launch",
        )
        replay, replay_match = nested[0]
        equal(
            [int(v) for v in replay_match.groups()],
            [bucket, record["graph_exec_id"]],
            "actual replay executable and bucket",
        )
        launch = own_launches[0]
        require(_contains(replay, launch), "CUDA graph launch outside actual replay scope")
        require(
            id(launch) not in used_launches and id(replay) not in used_scopes,
            "graph launch ambiguously assigned",
        )
        used_launches.add(id(launch))
        used_scopes.add(id(replay))
        correlation = launch.get("args", {}).get("correlation")
        require(type(correlation) is int and correlation >= 0, "actual CUDA launch correlation")
        require(correlation not in used_correlations, "duplicate actual graph launch correlation")
        used_correlations.add(correlation)
        linked = [e for e in kernels if e.get("args", {}).get("correlation") == correlation]
        linked_kernel_ids.update(id(e) for e in linked)
        require(bool(linked), "replay lacks correlation-linked actual GPU kernels")
        graph_ids = {
            e.get("args", {}).get("graph id", e.get("args", {}).get("graphId")) for e in linked
        }
        graph_ids.discard(None)
        require(len(graph_ids) == 1, "replay lacks one actual CUDA graph identity")
        graph_id = next(iter(graph_ids))
        require(type(graph_id) is int and graph_id > 0, "invalid actual CUDA graph ID")
        if bucket in graph_buckets:
            equal(graph_buckets[bucket], graph_id, "bucket reused its captured graph")
        graph_buckets[bucket] = graph_id
        rows.append(
            {
                "dispatch_id": dispatch_id,
                "bucket_id": bucket,
                "kind": kind,
                "graph_exec_id": record["graph_exec_id"],
                "graph_id": graph_id,
                "correlation": correlation,
                "graph_launches": 1,
                "graph_kernel_count": len(linked),
            }
        )
    equal(sorted(seen_dispatches), sorted(record_by_id), "exact profile dispatch coverage")
    equal(len(used_launches), len(launches), "no unassigned CUDA graph launch")
    equal(len(used_scopes), len(replay_scopes), "no unassigned replay scope")
    require(
        len(set(graph_buckets.values())) == len(graph_buckets), "buckets share a graph identity"
    )
    for kernel in kernels:
        args = kernel.get("args", {})
        if args.get("graph id", args.get("graphId", 0)):
            require(id(kernel) in linked_kernel_ids, "unassigned actual graph kernel")
    replay_covered = capture["use_graphs"] is False or bool(used_launches)
    return {
        "complete": replay_covered,
        "missing": [] if replay_covered else ["candidate profile has no replay"],
        "graph_launches": len(used_launches),
        "graph_kernel_count": sum(r["graph_kernel_count"] for r in rows),
        "graph_ids_by_bucket": {str(k): v for k, v in graph_buckets.items()},
        "dispatches": rows,
    }


def _anchor(root, path, hashes):
    hashes[str(path.relative_to(root))] = _file_record(path)["sha256"]


def _audit_workers(root, plan, manifest, report, *, workers=None):
    from benchmarks.capture.runner import audit_worker_controls

    limits = plan["contract"]["limits"]
    expected = [row["execution_id"] for row in plan["execution_order"]]
    equal(
        manifest["completed_workers"],
        list(WORKERS)[: len(manifest["completed_workers"])],
        "completed worker prefix",
    )
    equal(
        manifest["completed_executions"],
        expected[: len(manifest["completed_executions"])],
        "completed execution prefix",
    )
    equal(
        manifest["deadline_ns"] - manifest["started_ns"],
        limits["total_timeout_s"] * 10**9,
        "global deadline",
    )
    require(manifest["started_ns"] <= manifest["ended_ns"], "global execution lifetime")
    if manifest["ended_ns"] >= manifest["deadline_ns"]:
        report["hard_failures"].append({"scope": "global", "reason": "whole experiment deadline"})
    launches = manifest["workers"]
    audit_worker_launches(manifest)
    actual = {p.name for p in (root / "workers").glob("*") if p.is_dir()}
    require(actual <= {r["worker_id"] for r in launches}, "unplanned worker artifacts")
    previous, completed = None, []
    workers = {} if workers is None else workers
    for index, launch in enumerate(launches):
        worker = plan["workers"][index]
        path = root / "workers" / worker["worker_id"] / "manifest.json"
        if not path.exists():
            require(
                index == len(launches) - 1 and manifest["status"] != "complete",
                "missing worker before later launch",
            )
            report["missing"].append(str(path.relative_to(root)))
            break
        child = read_json(path)
        _anchor(root, path, report["hashes"])
        equal(
            [child["schema_version"], child["artifact_type"]],
            [1, "m3_capture_worker_manifest"],
            "worker manifest type",
        )
        for field, value in worker.items():
            equal(child[field], value, f"actual worker {field}")
        done = child["completed_executions"]
        equal(done, worker["execution_ids"][: len(done)], "worker completed prefix")
        completed.extend(done)
        if child["status"] == "running":
            require(
                index == len(launches) - 1
                and manifest["status"] in ("failed", "incomplete")
                and worker["worker_id"] not in manifest["completed_workers"],
                "nonterminal worker is not the interrupted tail",
            )
            require(child["passed"] is False and child["failures"] == [], "running worker outcome")
            require(
                "ended_ns" not in child and "teardown_after_workspace_release" not in child,
                "running worker contradicts terminal fields",
            )
            equal(child["deadline_ns"], manifest["deadline_ns"], "interrupted worker deadline")
            audit_worker_lifetime(manifest, index, child, interrupted=True)
            # Validate retained controls when setup reached them, but never supply
            # a missing process-completion or CUDA-cleanup boundary from a result.
            if child.get("environment") is not None:
                audit_worker_controls(plan, child, previous, require_cleanup=False)
            else:
                report["missing"].append(f"workers/{worker['worker_id']}/runtime controls")
            report["interrupted_workers"][worker["worker_id"]] = {
                "status": child["status"],
                "recorded_completed_executions": done,
                "controller_returned_ns": launch["returned_ns"],
                "terminal_cleanup_available": False,
            }
            report["missing"].extend(
                [
                    f"workers/{worker['worker_id']}/terminal manifest",
                    f"workers/{worker['worker_id']}/final CUDA measurement",
                ]
            )
            break
        require(child["status"] in ("complete", "failed", "incomplete"), "worker terminal status")
        require(
            type(child["passed"]) is bool and isinstance(child["failures"], list), "worker outcome"
        )
        success = child["status"] == "complete" and child["passed"] and not child["failures"]
        if success:
            equal(done, worker["execution_ids"], "successful worker exact completed rows")
            equal(launch["exit_code"], 0, "successful worker process exit")
        else:
            require(
                index == len(launches) - 1 and manifest["status"] != "complete",
                "worker launched after failure",
            )
            require(
                child["failures"] or child["status"] == "incomplete", "unexplained worker failure"
            )
        require(
            type(child["model_loads"]) is int and child["model_loads"] in (0, 1),
            "at most one model load per worker",
        )
        require(not success or child["model_loads"] == 1, "successful worker model load")
        equal(child["deadline_ns"], manifest["deadline_ns"], "worker deadline")
        audit_worker_lifetime(manifest, index, child)
        if worker["worker_id"] == "A1":
            gate_path = root / "numerical-gate.json"
            equal(read_json(gate_path), manifest["numerical_gate"], "persisted pre-timing gate")
            _anchor(root, gate_path, report["hashes"])
        audit_worker_controls(plan, child, previous, require_cleanup=False)
        cleanup = child.get("teardown_after_workspace_release")
        if cleanup is None:
            report["missing"].append(f"workers/{worker['worker_id']}/final CUDA measurement")
        else:
            equal(sorted(cleanup), ["allocated_bytes", "reserved_bytes"], "CUDA cleanup fields")
            require(
                all(type(v) is int and v >= 0 for v in cleanup.values())
                and cleanup["reserved_bytes"] >= cleanup["allocated_bytes"],
                "CUDA cleanup bytes",
            )
            if any(cleanup.values()):
                report["hard_failures"].append(
                    {"scope": worker["worker_id"], "reason": "CUDA cleanup", "observed": cleanup}
                )
                require(not success, "successful worker contradicts retained CUDA memory")
        for failure in child["failures"]:
            require(
                isinstance(failure, dict)
                and isinstance(failure.get("type"), str)
                and isinstance(failure.get("message"), str),
                "typed worker failure",
            )
            # Model comparison failures are independently reconstructed below.
            # Actual runtime/cleanup exceptions are failed executions in their own right.
            if failure["type"] in ("cleanup", "MemoryError", "TimeoutError"):
                report["hard_failures"].append({"scope": worker["worker_id"], "reason": failure})
        previous = child
        workers[worker["worker_id"]] = child
    equal(completed, expected[: len(completed)], "actual completed worker-row prefix")
    equal(
        manifest["completed_executions"],
        completed[: len(manifest["completed_executions"])],
        "parent acknowledgments match actual completed prefix",
    )
    return workers


def _audit_chronology(root, plan, workers, reports, records, hashes):
    """Bind every planned kind into one serial timeline, including excluded cases."""
    case_timeout_ns = plan["contract"]["limits"]["case_timeout_s"] * 10**9
    by_id = {row["run_id"]: row for row in records}
    previous = {}
    for row in plan["execution_order"]:
        if row["worker_id"] not in workers:
            continue
        worker = workers[row["worker_id"]]
        eid = row["execution_id"]
        if eid not in worker["completed_executions"]:
            continue
        if row["kind"] == "benchmark":
            result = by_id[eid]["result"]
            require(result is not None, "missing benchmark lifetime")
            start, end = result["case_started_ns"], result["case_completed_ns"]
            deadline = min(worker["deadline_ns"], start + case_timeout_ns)
            marker = root / "runs" / eid / "started.json"
            equal(
                read_json(marker),
                {"schema_version": 1, **row, "started_ns": start, "deadline_ns": deadline},
                "benchmark start/watchdog marker",
            )
            require(
                start
                <= result["capture"]["initial"]["setup"]["started_ns"]
                <= result["capture"]["initial"]["setup"]["finished_ns"]
                <= result["arrival_ns"]
                <= result["synchronized_ns"]
                <= end,
                "setup and timed timestamps inside case",
            )
        elif row["kind"] == "model":
            lifetime = reports["numerical"]["ledger"]["case_lifetimes"][eid]
            start, end = lifetime["started_ns"], lifetime["finished_ns"]
            deadline = lifetime["deadline_ns"]
            marker = root / "numerical" / "cases" / eid / "started.json"
        for timestamp in (start, end, deadline):
            require(type(timestamp) is int and timestamp >= 0, "integer execution timestamps")
        require(
            worker["started_ns"]
            <= previous.get(row["worker_id"], start)
            <= start
            <= end
            <= worker["ended_ns"]
            and end < deadline,
            "mixed execution chronology",
        )
        equal(deadline, min(worker["deadline_ns"], start + case_timeout_ns), "whole case deadline")
        previous[row["worker_id"]] = end
        _anchor(root, marker, hashes)


def _audit_failed_benchmark(root, row, result, report, workers, *, case_timeout_ns):
    """Validate a producer-recorded failed execution without inventing final metrics."""
    for key, value in row.items():
        equal(result.get(key), value, f"failed benchmark identity {key}")
    equal(result["plan_sha256"], report["plan_sha256"], "failed benchmark plan")
    require(
        result["status"] in ("failed", "incomplete") and result["comparison_eligible"] is False,
        "failed benchmark status",
    )
    failures = result["failures"]
    require(
        isinstance(failures, list) and bool(failures), "failed benchmark has no recorded reason"
    )
    for failure in failures:
        require(
            isinstance(failure, dict)
            and isinstance(failure.get("type"), str)
            and isinstance(failure.get("message"), str),
            "typed benchmark failure",
        )
    folder = root / "runs" / row["run_id"]
    marker = read_json(folder / "started.json")
    for key, value in row.items():
        equal(marker.get(key), value, f"failed start marker {key}")
    start, deadline = marker["started_ns"], marker["deadline_ns"]
    worker = workers[row["worker_id"]]
    done = worker["completed_executions"]
    require(
        len(done) < len(worker["execution_ids"])
        and worker["execution_ids"][len(done)] == row["execution_id"],
        "failed run is not the worker's next planned execution",
    )
    require(worker["started_ns"] <= start <= worker["ended_ns"], "failed run outside worker")
    equal(
        deadline, min(worker["deadline_ns"], start + case_timeout_ns), "failed run global deadline"
    )
    require(
        type(start) is int and type(deadline) is int and 0 < deadline - start <= case_timeout_ns,
        "failed benchmark start/deadline",
    )
    events = folder / "events.jsonl"
    if events.exists() and result.get("arrival_ns") is not None:
        require(
            not _event_problems(_read_events(events), result), "partial emitted history differs"
        )
    else:
        require(not result["requests"], "partial requests lack their raw emitted events")
    cleanup = result.get("cleanup")
    if cleanup is not None:
        equal(sorted(cleanup), ["active_requests", "used_kv_blocks"], "partial cleanup fields")
        require(all(type(v) is int and v >= 0 for v in cleanup.values()), "partial cleanup counts")
    for path in folder.iterdir():
        if path.is_file():
            _anchor(root, path, report["hashes"])
    report["hard_failures"].append(
        {"scope": row["run_id"], "reason": "failed execution", "observed": failures}
    )


def _decision(*, invalid, complete, hard_failure, variation):
    if invalid:
        return "invalid", "inconclusive"
    status = "complete" if complete else "incomplete"
    if hard_failure:
        return status, "failed"
    if not complete or variation:
        return status, "inconclusive"
    return status, "passed"


def build_report(output_dir):
    from benchmarks.capture.runner import artifact_usage, audit_correctness
    from benchmarks.capture.schema import execution_view, validate_plan

    root = Path(output_dir).resolve()
    report = {
        "schema_version": 1,
        "artifact_type": "m3_capture_report",
        "evidence_status": "incomplete",
        "decision": "inconclusive",
        "milestone_status": "unqualified_optional_path",
        "errors": [],
        "records": [],
        "cells": [],
        "profiles": [],
        "hashes": {},
        "hard_failures": [],
        "missing": [],
        "interrupted_workers": {},
        "limitations": [
            "A is the common padded tensor body; this comparison does not measure gain over "
            "default compact execution.",
            "Two observations per side describe a range, not statistical confidence or "
            "serving-tail stability.",
            "Feasibility, warmup, correctness, graph setup and profiles are excluded from timed "
            "comparisons.",
            "Profile scope/kernel durations are diagnostic and never used as speedup evidence.",
            "Setup break-even tokens extrapolate measured throughput and setup cost; they are "
            "not another measurement.",
            "Missing graph launch correlation or identities leave profile evidence unqualified; "
            "no diagnostic reruns are implied.",
            "The protocol has no memory-checker executions; CUDA memory-checker qualification "
            "remains open.",
        ],
    }
    try:
        plan = read_json(root / "plan.json")
        validate_plan(plan)
        manifest = read_json(root / "manifest.json")
        equal(
            [manifest["schema_version"], manifest["artifact_type"], manifest["plan_sha256"]],
            [1, "m3_capture_manifest", plan["plan_sha256"]],
            "parent manifest identity",
        )
        report["plan_sha256"] = plan["plan_sha256"]
        report["manifest_status"] = manifest["status"]
        report["execution_failures"] = manifest["failures"]
        for path in (root / "plan.json", root / "manifest.json"):
            _anchor(root, path, report["hashes"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["evidence_status"] = "invalid"
        report["errors"].append({"scope": "plan/manifest", "message": str(error)})
        return report
    workers = {}
    try:
        _audit_workers(root, plan, manifest, report, workers=workers)
        report["artifact_usage"] = artifact_usage(
            root,
            plan["contract"]["limits"],
            failures=report["hard_failures"],
            exclude=("capture-report.json", "capture-report.md"),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append({"scope": "workers/controls", "message": str(error)})
    rows = [r for r in plan["execution_order"] if r["kind"] == "benchmark"]
    view = execution_view(plan)
    actual = {p.name for p in (root / "runs").glob("*") if p.is_dir()}
    expected = {r["run_id"] for r in rows}
    for name in sorted(actual - expected):
        report["errors"].append({"scope": name, "message": "unplanned benchmark directory"})
    for row in rows:
        record = _run_record(root, row, view, report["hashes"])
        result = record["result"]
        interrupted = row["worker_id"] in report["interrupted_workers"]
        if result is not None and interrupted:
            record["status"] = "incomplete"
            record["comparison_eligible"] = False
            record["worker_completion_available"] = False
            report["missing"].append(f"{row['run_id']}: no terminal worker completion/cleanup")
        if result is not None and result.get("status") in ("failed", "incomplete"):
            if interrupted:
                report["records"].append(record)
                continue
            try:
                _audit_failed_benchmark(
                    root,
                    row,
                    result,
                    report,
                    workers,
                    case_timeout_ns=plan["contract"]["limits"]["case_timeout_s"] * 10**9,
                )
                record["valid_failed_execution"] = True
                report["missing"].append(f"{row['run_id']}: no complete timing observation")
            except (OSError, ValueError, KeyError, TypeError) as error:
                report["errors"].append({"scope": row["run_id"], "message": str(error)})
        elif result is not None:
            try:
                if not interrupted:
                    require(
                        row["worker_id"] in workers
                        and row["run_id"] in workers[row["worker_id"]]["completed_executions"],
                        "complete run has no corresponding audited terminal worker execution",
                    )
                for key, value in row.items():
                    equal(result.get(key), value, f"frozen benchmark row {key}")
                equal(
                    result["comparison_eligible"], row["phase"] == "measured", "phase eligibility"
                )
                _memory(result, plan["workload_stats"][row["workload_id"]]["pool_bytes"])
                require("after_engine_release" in result["memory"], "missing engine release memory")
                capture = result["capture"]
                setup = capture["initial"]["setup"]
                executor_ns = setup["finished_ns"] - setup["started_ns"]
                require(
                    0
                    <= executor_ns
                    <= result["setup_ns"]
                    <= result["arrival_ns"] - result["case_started_ns"],
                    "executor setup must fit excluded engine setup and pre-arrival lifetime",
                )
                record["capture_summary"] = audit_capture_record(
                    capture,
                    row,
                    plan["graph_limits"],
                    model_config=plan["model_config"],
                    block_size=plan["benchmark_contract"]["engine"]["cache"]["block_size"],
                )
                equal(
                    capture["final"]["counters"]["calls"],
                    result["counts"]["stage_counts"]["recurrent"],
                    "independent observed recurrent dispatch count",
                )
                path = root / "runs" / row["run_id"] / "graph-setup.json"
                equal(read_json(path), capture["initial"], "persisted setup snapshot")
                _anchor(root, path, report["hashes"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                record["validation_errors"].append(str(error))
        if record["validation_errors"]:
            record["comparison_eligible"] = False
            if result is not None and result.get("status") == "complete":
                report["errors"].append(
                    {"scope": row["run_id"], "message": record["validation_errors"]}
                )
            elif result is None:
                if (
                    record["started"]
                    or not (root / "runs" / row["run_id"] / "result.json").exists()
                ):
                    report["missing"].append(row["run_id"])
                else:
                    report["errors"].append(
                        {"scope": row["run_id"], "message": record["validation_errors"]}
                    )
        report["records"].append(record)
    report["cells"] = pair_results(plan, report["records"])
    try:
        correctness = audit_correctness(root, plan)
        if not correctness["numerical"]["complete"]:
            from benchmarks.capture.validation import audit_completed_prefix

            prefix = audit_completed_prefix(root, plan)
            if prefix["errors"]:
                report["errors"].append({"scope": "numerical/prefix", "message": prefix["errors"]})
            correctness["numerical"] = prefix
        numerical = correctness["numerical"]
        if not numerical["complete"]:
            report["missing"].append("numerical: incomplete planned evidence")
        correctness["complete"] = numerical["complete"]
        correctness["known_required_failure"] = numerical.get("known_required_failure", False) or (
            numerical["complete"] and not numerical["passed"]
        )
        report["numerical"] = numerical
        if manifest.get("numerical_gate") is not None:
            audited_gate = {key: correctness[key] for key in ("complete", "passed", "numerical")}
            equal(
                audited_gate, manifest["numerical_gate"], "recomputed pre-timing correctness gate"
            )
        _audit_chronology(root, plan, workers, correctness, report["records"], report["hashes"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        correctness = {"complete": False, "passed": False}
        report["errors"].append({"scope": "correctness/order", "message": str(error)})
    profiles = _profiles(root, report["hashes"], report["records"], view)
    expected_profiles = {
        r["run_id"]: r for r in report["records"] if r["planned"]["phase"] == "profile"
    }
    actual_profiles = {profile["capture_id"] for profile in profiles}
    if actual_profiles - set(expected_profiles):
        report["errors"].append({"scope": "profiles", "message": "unplanned profile artifacts"})
    report["missing"].extend(
        f"profiles/{name}" for name in sorted(set(expected_profiles) - actual_profiles)
    )
    for profile in profiles:
        try:
            record = expected_profiles[profile["capture_id"]]
            if record["status"] != "complete":
                report["missing"].append(f"profiles/{profile['capture_id']}: incomplete capture")
                continue
            require(
                not profile["validation_errors"] and profile["gpu_trace_available"],
                "invalid GPU profile",
            )
            path = root / "profiles" / profile["capture_id"] / "trace.json"
            profile["graph_attribution"] = audit_graph_profile(
                read_json(path), record["result"]["capture"]
            )
            require(profile.get("stage_dispatches") is not None, "missing profile stage dispatches")
            equal(
                len(profile["graph_attribution"]["dispatches"]),
                profile["stage_dispatches"].get("recurrent", 0),
                "profile scheduled dispatch count",
            )
            if not profile["graph_attribution"]["complete"]:
                report["missing"].extend(
                    f"profiles/{profile['capture_id']}: {reason}"
                    for reason in profile["graph_attribution"]["missing"]
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["errors"].append(
                {"scope": f"profiles/{profile['capture_id']}", "message": str(error)}
            )
    report["profiles"] = profiles
    complete = (
        manifest["status"] == "complete"
        and not manifest["failures"]
        and manifest["completed_workers"] == list(WORKERS)
        and len(manifest["completed_executions"]) == len(plan["execution_order"])
        and correctness["complete"]
        and all(r["status"] == "complete" and not r["validation_errors"] for r in report["records"])
    )
    hard_failure = (
        bool(report["hard_failures"])
        or correctness.get("known_required_failure", False)
        or (correctness["complete"] and not correctness["passed"])
        or any(c["status"] == "failed" for c in report["cells"])
    )
    report["evidence_status"], report["decision"] = _decision(
        invalid=bool(report["errors"]),
        complete=complete and not report["missing"],
        hard_failure=hard_failure,
        variation=any(c["status"] == "inconclusive" for c in report["cells"]),
    )
    if report["decision"] == "passed":
        report["milestone_status"] = "accepted_optional_path_pending_manual_evidence_review"
    report["counts"] = {
        "planned_executions": len(plan["execution_order"]),
        "completed_executions": len(manifest["completed_executions"]),
        "benchmark_runs": len(report["records"]),
        "eligible_timing_runs": sum(r["comparison_eligible"] for r in report["records"]),
        "profiles": len(profiles),
        "workers": len(workers),
    }
    return report


def write_report(output_dir):
    root = Path(output_dir).resolve()
    report = build_report(root)
    write_json(root / "capture-report.json", report)
    lines = [
        "# Recurrent graph A/B evidence",
        "",
        f"Evidence: **{report['evidence_status']}**. Decision: **{report['decision']}**.",
        "",
        "| Cell | Decision | A tokens/s | B tokens/s |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {c['cell_id']} | {c['status']} | {c['baseline_values']} | {c['candidate_values']} |"
        for c in report["cells"]
    )
    lines.extend(["", *report["limitations"], ""])
    lines.extend(f"- {e['scope']}: {e['message']}" for e in report["errors"])
    (root / "capture-report.md").write_text("\n".join(lines) + "\n")
    return report
