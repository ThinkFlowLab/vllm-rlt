"""Offline evidence checks; synthetic timings and CPU fakes are not GPU measurements."""

from copy import deepcopy

import pytest
import test_benchmark_capture as fixtures
import torch
from test_benchmark_ab_report import records_for as shared_records
from test_benchmark_ab_report import target as pick

from benchmarks.capture import report
from vllm_lt.benchmarks.schema import write_json

capture_plan = fixtures.capture_plan
no_cuda_or_weights = fixtures.no_cuda_or_weights


def records_for(plan):
    records = shared_records(plan)
    for row in records:
        row["result"]["setup_ns"] = (2 if row["planned"]["use_graphs"] else 1) * 100_000_000
        row["recomputed_metrics"]["per_request"] = {"q": {"ttft_ns": 100}}
    return records


def test_pairs_exclude_warmups_and_keep_setup_separate(capture_plan):
    records = records_for(capture_plan)
    for row in records:
        if row["planned"]["phase"] != "measured":
            row["recomputed_metrics"] = {"generated_tokens_per_second": float("nan")}
    cells = report.pair_results(capture_plan, records)
    assert len(cells) == 7 and all(c["status"] == "passed" for c in cells)
    assert sum(len(c["pairs"]) for c in cells) == 14
    pair = cells[0]["pairs"][0]
    assert pair["setup_increase_ns"] == 100_000_000
    assert pair["estimated_setup_break_even_tokens"] == 77
    assert pair["ttft"]["candidate_over_baseline"] == 1


def test_failed_or_invalid_ttft_cannot_qualify(capture_plan):
    records = records_for(capture_plan)
    row = pick(records)
    row["recomputed_metrics"]["per_request"]["q"]["ttft_ns"] = 106
    assert report.pair_results(capture_plan, records)[0]["status"] == "failed"
    row["recomputed_metrics"]["per_request"]["q"]["ttft_ns"] = float("nan")
    assert report.pair_results(capture_plan, records)[0]["pairs"][0]["status"] == "invalid"


def profile_fixture(*, replay=True, large_bucket=16):
    def event(name, category, timestamp, duration, args=None):
        return dict(
            name=name,
            cat=category,
            ph="X",
            ts=timestamp,
            dur=duration,
            pid=1,
            tid=7,
            args=args or {},
        )

    kind = "replay" if replay else "eager"
    events, dispatches = [], []
    generations = dict.fromkeys((4, large_bucket), int(replay))
    initial = {
        "buckets": {
            str(k): {"generation": v, "graph_exec_id": k + 100 if replay else None}
            for k, v in generations.items()
        }
    }
    for i, bucket in enumerate((large_bucket, 4, large_bucket), 1):
        generations[bucket] += 1
        dispatches.append(
            dict(
                dispatch_id=i,
                bucket_id=bucket,
                kind=kind,
                generation=generations[bucket],
                graph_exec_id=bucket + 100 if replay else None,
            )
        )
        events.append(
            event(
                f"vllm_lt::graph_dispatch::{i}::bucket::{bucket}::{kind}",
                "user_annotation",
                i * 100,
                30,
            )
        )
        if replay:
            events.extend(
                [
                    event(
                        f"vllm_lt::graph_replay::{bucket}::exec::{bucket + 100}",
                        "user_annotation",
                        i * 100 + 1,
                        10,
                    ),
                    event("cudaGraphLaunch", "cuda_runtime", i * 100 + 2, 5, {"correlation": i}),
                    event(
                        "actual_kernel",
                        "kernel",
                        i * 100 + 9,
                        10,
                        {"correlation": i, "graph id": bucket + 200},
                    ),
                ]
            )
    return {"traceEvents": events}, {
        "use_graphs": replay,
        "profile_dispatches": dispatches,
        "initial": initial,
        "final": {"buckets": {str(k): {"generation": v} for k, v in generations.items()}},
    }


@pytest.mark.parametrize("replay,bucket", [(True, 8), (True, 16), (True, 32), (False, 8)])
def test_profile_matches_host_scopes_to_actual_launches(replay, bucket):
    trace, capture = profile_fixture(replay=replay, large_bucket=bucket)
    evidence = report.audit_graph_profile(trace, capture)
    assert evidence["complete"] and evidence["graph_launches"] == (3 if replay else 0)
    if replay:
        assert evidence["graph_ids_by_bucket"] == {"4": 204, str(bucket): bucket + 200}
        assert evidence["graph_kernel_count"] == 3
    assert not any("time" in key or "speed" in key for key in evidence)
    trace["traceEvents"].extend(
        [
            {**e, "cat": "gpu_user_annotation"}
            for e in trace["traceEvents"]
            if e["cat"] == "user_annotation"
        ]
    )
    assert report.audit_graph_profile(trace, capture) == evidence


@pytest.mark.parametrize(
    "change", ["host", "launch", "correlation", "identity", "generation", "unassigned", "duplicate"]
)
def test_profile_rejects_missing_or_ambiguous_evidence(change):
    trace, capture = profile_fixture()
    events = trace["traceEvents"]
    if change == "host":
        events[0]["cat"] = "gpu_user_annotation"
    elif change == "launch":
        events.pop(2)
    elif change == "correlation":
        events[6]["args"]["correlation"] = 1
    elif change == "identity":
        capture["profile_dispatches"][0]["graph_exec_id"] += 99
        events[1]["name"] = "vllm_lt::graph_replay::16::exec::215"
    elif change == "generation":
        capture["profile_dispatches"][0]["generation"] += 10
    elif change == "unassigned":
        events.append({**events[3], "args": {"correlation": 999, "graph id": 216}})
    else:
        events.append(deepcopy(events[0]))
    with pytest.raises(ValueError):
        report.audit_graph_profile(trace, capture)


def test_compact_profile_cannot_qualify_replay_or_claim_graph_kernels():
    trace, capture = profile_fixture()
    kernel = trace["traceEvents"][3]
    trace["traceEvents"] = [
        {**trace["traceEvents"][0], "name": "vllm_lt::graph_dispatch::1::bucket::0::compact"}
    ]
    capture["profile_dispatches"] = [
        dict(dispatch_id=1, bucket_id=None, kind="compact", generation=None, graph_exec_id=None)
    ]
    assert report.audit_graph_profile(trace, capture)["missing"] == [
        "candidate profile has no replay"
    ]
    capture["use_graphs"] = False
    assert report.audit_graph_profile(trace, capture)["complete"]
    trace["traceEvents"].append(kernel)
    with pytest.raises(ValueError, match="unassigned actual graph kernel"):
        report.audit_graph_profile(trace, capture)


@pytest.fixture
def cpu_capture(monkeypatch):
    from test_recurrent_graph import fake_runtime, make_cache

    from vllm_lt.models import OuroConfig, OuroForCausalLM
    from vllm_lt.worker.model_runner import ModelRunner

    model = OuroForCausalLM(OuroConfig.tiny())
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    initial = runner._graph_snapshot()
    cache.allocate("a", 2)
    runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    final = runner._graph_snapshot()
    cache.free("a")
    runner._close_recurrent_graph()
    capture = {
        "schema_version": 1,
        "implementation_id": "B",
        "use_graphs": True,
        "initial": initial,
        "final": final,
        "closed": runner._graph_snapshot(),
        "cleanup": {"closed": True},
        "profile_dispatches": [],
    }
    return capture, model.config.to_dict()


def audit_cpu_capture(value):
    capture, config = value
    return report.audit_capture_record(
        capture,
        {"implementation_id": "B", "phase": "measured"},
        capture["initial"]["limits"],
        model_config=config,
        block_size=16,
        expected_device="cpu",
    )


def test_actual_cpu_standin_setup_and_counters_audit_only_with_explicit_cpu_override(cpu_capture):
    assert audit_cpu_capture(cpu_capture)["counters"]["replays"] == 1
    changed = deepcopy(cpu_capture)
    changed[0]["final"]["counters"]["committed"] = 0
    with pytest.raises(ValueError):
        audit_cpu_capture(changed)
    capture, config = cpu_capture
    with pytest.raises(ValueError, match="device"):
        report.audit_capture_record(
            capture,
            {"implementation_id": "B", "phase": "measured"},
            capture["initial"]["limits"],
            model_config=config,
            block_size=16,
        )


@pytest.mark.parametrize(
    "invalid,complete,failure,variation,expected",
    [
        (False, False, True, True, ("incomplete", "failed")),
        (False, False, False, False, ("incomplete", "inconclusive")),
        (True, False, True, False, ("invalid", "inconclusive")),
        (False, True, True, True, ("complete", "failed")),
        (False, True, False, True, ("complete", "inconclusive")),
        (False, True, False, False, ("complete", "passed")),
    ],
)
def test_decision_requires_complete_valid_evidence(invalid, complete, failure, variation, expected):
    assert (
        report._decision(
            invalid=invalid, complete=complete, hard_failure=failure, variation=variation
        )
        == expected
    )


def test_interrupted_worker_cannot_claim_terminal_cleanup(capture_plan, tmp_path):
    worker = capture_plan["workers"][0]
    deadline = 1 + capture_plan["contract"]["limits"]["total_timeout_s"] * 10**9
    manifest = dict(
        status="failed",
        started_ns=1,
        ended_ns=5,
        deadline_ns=deadline,
        completed_workers=[],
        completed_executions=[],
        workers=[dict(worker_id=worker["worker_id"], launched_ns=2, returned_ns=4)],
    )
    child = {
        **worker,
        "schema_version": 1,
        "artifact_type": "m3_capture_worker_manifest",
        "status": "running",
        "passed": False,
        "failures": [],
        "started_ns": 3,
        "deadline_ns": deadline,
        "completed_executions": worker["execution_ids"][:1],
    }
    path = tmp_path / "workers" / worker["worker_id"] / "manifest.json"
    path.parent.mkdir(parents=True)
    write_json(path, child)
    output = dict(hashes={}, missing=[], hard_failures=[], interrupted_workers={})
    assert report._audit_workers(tmp_path, capture_plan, manifest, output) == {}
    interrupted = output["interrupted_workers"][worker["worker_id"]]
    assert interrupted["recorded_completed_executions"] == child["completed_executions"]
    assert not interrupted["terminal_cleanup_available"]
    assert any("final CUDA measurement" in value for value in output["missing"])
    child["teardown_after_workspace_release"] = {"allocated_bytes": 0, "reserved_bytes": 0}
    write_json(path, child)
    with pytest.raises(ValueError, match="terminal fields"):
        report._audit_workers(tmp_path, capture_plan, manifest, output)


def test_missing_parent_artifacts_return_invalid_without_runtime_imports(tmp_path):
    value = report.build_report(tmp_path)
    assert value["evidence_status"] == "invalid" and value["decision"] == "inconclusive"
    assert value["errors"][0]["scope"] == "plan/manifest"
