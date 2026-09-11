"""CPU-only reconstructed controls, acknowledgments and lifetime counterexamples."""

from copy import deepcopy

import pytest
import test_benchmark_m4_attention_schema as fixtures
import torch
from test_benchmark_ab_report import records_for, target

from vllm_lt.benchmarks import ab_report
from vllm_lt.benchmarks import m4_attention_report as report
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.validation import m4_attention as numerical
from vllm_lt.validation import m4_attention_held as held

m4_plan = fixtures.m4_plan


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline report touched CUDA or loaded weights")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)


def paired(plan):
    records = records_for(plan)
    for row in records:
        row["result"]["counts"]["stage_tokens"] = {"recurrent": 1}
        row["result"]["counts"]["gate_probabilities"] = [
            0.4 if row["planned"]["implementation_id"] == "A" else 0.40000001
        ]
    return records


def test_m4_pair_namespace_and_only_gate_values_are_diagnostic(m4_plan):
    rows = paired(m4_plan)
    assert all(
        c["status"] == "passed"
        for c in ab_report.pair_results(m4_plan, rows, pair_prefix="M4", compare_gate_values=False)
    )
    assert all(c["status"] == "inconclusive" for c in ab_report.pair_results(m4_plan, rows))
    assert (
        ab_report.pair_results(m4_plan, rows, pair_prefix="M4")[0]["pairs"][0]["status"]
        == "invalid"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("throughput", 100.9),
        ("setup_ns", 100_000_101),
        ("peak_allocated_bytes", 67_109_065),
        ("peak_reserved_bytes", 67_109_065),
    ],
)
def test_one_failed_pair_is_not_averaged_away(m4_plan, field, value):
    rows = paired(m4_plan)
    row = target(rows)
    if field == "throughput":
        row["recomputed_metrics"]["generated_tokens_per_second"] = value
    elif field == "setup_ns":
        row["result"][field] = value
    else:
        row["result"]["memory"][field] = value
        row["result"]["memory"]["peak_reserved_bytes"] = max(
            value, row["result"]["memory"]["peak_reserved_bytes"]
        )
    cell = ab_report.pair_results(m4_plan, rows, pair_prefix="M4", compare_gate_values=False)[0]
    assert cell["status"] == "failed" and cell["pairs"][1]["status"] == "passed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_gate_diagnostic_values_still_require_finiteness_and_range(m4_plan, value):
    rows = paired(m4_plan)
    target(rows)["result"]["counts"]["gate_probabilities"] = [value]
    assert (
        ab_report.pair_results(m4_plan, rows, pair_prefix="M4", compare_gate_values=False)[0][
            "pairs"
        ][0]["status"]
        == "invalid"
    )


def test_range_overlap_only_after_all_pair_gates_pass(m4_plan):
    rows = paired(m4_plan)
    target(rows, "A", 2)["recomputed_metrics"]["generated_tokens_per_second"] = 102
    target(rows, "B", 1)["recomputed_metrics"]["generated_tokens_per_second"] = 102
    target(rows, "B", 2)["recomputed_metrics"]["generated_tokens_per_second"] = 104
    cell = ab_report.pair_results(m4_plan, rows, pair_prefix="M4", compare_gate_values=False)[0]
    assert all(p["status"] == "passed" for p in cell["pairs"])
    assert cell["status"] == "inconclusive" and not cell["strict_range_separation"]


@pytest.fixture
def evidence(m4_plan, tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    write_json(root / "plan.json", m4_plan)
    clock, deadline = 10**12, 10**12 + 7200 * 10**9
    rows = {r["run_id"]: r for r in paired(m4_plan)}
    workers, ids, launches, lifetimes, completed_ns, evaluations = [], [], [], {}, {}, []
    for worker in m4_plan["workers"]:
        wid = worker["worker_id"]
        clock += 1000
        launch = clock
        clock += 10
        start = clock
        folder = root / "workers" / wid
        folder.mkdir(parents=True)
        for row in [r for r in m4_plan["execution_order"] if r["worker_id"] == wid]:
            clock += 10
            s = clock
            e = s + 7
            clock = e
            identity = row["execution_id"]
            outer = {
                "execution_id": identity,
                "started_ns": s,
                "case_completed_ns": e,
                "deadline_ns": min(deadline, s + 600 * 10**9),
            }
            if row["kind"] != "numerical":
                write_json(folder / (identity + ".lifetime.json"), outer)
            if row["kind"] == "benchmark":
                value = rows[row["run_id"]]["result"]
                value.update(
                    schema_version=1,
                    artifact_type="run_result",
                    status="complete",
                    failures=[],
                    case_started_ns=s + 1,
                    arrival_ns=s + 2,
                    synchronized_ns=s + 3,
                    case_completed_ns=s + 4,
                    cleanup={"active_requests": 0, "used_kv_blocks": 0},
                    test_throughput=115 if row["implementation_id"] == "B" else 100,
                )
                stages = (
                    [
                        stage
                        for _ in range(16)
                        for stage in (
                            "prelude",
                            "recurrent",
                            "recurrent",
                            "recurrent",
                            "recurrent",
                            "coda",
                        )
                    ]
                    if row["workload_id"] == "W1"
                    else ["prefill"] * 5 + ["prelude"] * 11 + ["recurrent"] * 7 + ["coda"] * 11
                )
                value["counts"]["snapshots"] = [
                    {
                        "batch": {
                            "step_id": i,
                            "stage": stage,
                            "rows": [{"request_id": "same", "depth": 1, "position": 1}],
                        }
                    }
                    for i, stage in enumerate(stages, 2)
                ]
                output = root / "runs" / row["run_id"]
                output.mkdir(parents=True)
                write_json(output / "result.json", value)
                write_json(
                    output / "started.json",
                    {
                        "schema_version": 1,
                        **row,
                        "started_ns": s + 1,
                        "deadline_ns": outer["deadline_ns"],
                    },
                )
            elif row["kind"] == "numerical":
                lifetimes[identity] = {
                    "started_ns": s + 1,
                    "finished_ns": s + 4,
                    "deadline_ns": outer["deadline_ns"],
                }
                completed_ns[identity] = s + 6
                p = root / "numerical" / "cases" / identity
                p.mkdir(parents=True)
                write_json(
                    p / "m4-completed.json",
                    {
                        "case_id": identity,
                        "started_ns": s + 1,
                        "deadline_ns": outer["deadline_ns"],
                        "case_completed_ns": s + 6,
                    },
                )
            else:
                (root / "held" / "evaluations").mkdir(parents=True, exist_ok=True)
                evaluations.append(
                    {
                        "execution_id": identity,
                        "started_ns": s + 1,
                        "ended_ns": s + 4,
                        "case_completed_ns": s + 5,
                        "deadline_ns": outer["deadline_ns"],
                    }
                )
            ids.append(identity)
        env = {
            "cuda_visible_devices": "7",
            "logical_device": "cuda:0",
            "cpu_affinity": fixtures.AFFINITY["cpu_ids"],
            "numa_status": fixtures.AFFINITY["numa_status"],
            "actual_torch_threads": {"intraop": 1, "interop": 1},
            "python": m4_plan["dependencies"]["python"],
            "torch_cuda_version": m4_plan["dependencies"]["torch_cuda_build"],
            "software": {},
            "arithmetic": {
                **m4_plan["benchmark_contract"]["arithmetic"],
                "cudnn_allow_tf32": False,
            },
            "scheduler": [{"type": "RUN", "gpu_id": 7, "user": "test"}],
            "account": "test",
            "host": "cpu-fixture",
            "gpu_uuid": "fixed",
            "gpu_name": "fixture",
            "total_device_bytes": 1,
            "compute_capability": [0, 0],
            "reservation_environment": {},
        }
        clock += 10
        child = {
            "schema_version": 1,
            "artifact_type": "m4_attention_worker_manifest",
            **worker,
            "plan_sha256": m4_plan["plan_sha256"],
            "source": m4_plan["implementations"][worker["implementation_id"]]["source"],
            "harness_sha256": m4_plan["harness"]["sha256"],
            "affinity": fixtures.AFFINITY,
            "runtime_environment": m4_plan["runtime_environment"],
            "environment": env,
            "status": "complete",
            "passed": True,
            "failures": [],
            "completed_executions": worker["execution_ids"],
            "started_ns": start,
            "ended_ns": clock,
            "deadline_ns": deadline,
            "model_loads": 1,
            "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
        }
        write_json(folder / "manifest.json", child)
        clock += 10
        launches.append(
            {"worker_id": wid, "exit_code": 0, "launched_ns": launch, "returned_ns": clock}
        )
        workers.append(wid)
    audits = {
        "numerical": {
            "complete": True,
            "passed": True,
            "errors": [],
            "missing": [],
            "known_required_failure": False,
            "ledger": {"case_lifetimes": lifetimes},
            "case_completed_ns": completed_ns,
        },
        "held": {
            "complete": True,
            "passed": True,
            "errors": [],
            "missing": [],
            "known_required_failure": False,
            "evaluations": evaluations,
            "artifacts": {},
        },
    }
    gate = {"complete": True, "passed": True, "checks": deepcopy(audits)}
    parent = {
        "schema_version": 1,
        "artifact_type": "m4_attention_manifest",
        "plan_sha256": m4_plan["plan_sha256"],
        "status": "complete",
        "failures": [],
        "started_ns": 10**12,
        "ended_ns": clock + 10,
        "deadline_ns": deadline,
        "workers": launches,
        "completed_workers": workers,
        "completed_executions": ids,
        "numerical_gate": gate,
        "numerical_gate_ns": launches[1]["returned_ns"] + 1,
    }
    write_json(root / "manifest.json", parent)
    write_json(root / "numerical-gate.json", gate)
    monkeypatch.setattr(numerical, "audit_numerical", lambda *a: deepcopy(audits["numerical"]))
    monkeypatch.setattr(held, "audit_held", lambda *a: deepcopy(audits["held"]))

    def record(folder, row, view, hashes):
        path = folder / "runs" / row["run_id"] / "result.json"
        if not path.exists():
            return {
                "run_id": row["run_id"],
                "planned": row,
                "result": None,
                "status": "incomplete",
                "comparison_eligible": False,
                "validation_errors": ["missing"],
            }
        value = read_json(path)
        return {
            "run_id": row["run_id"],
            "planned": row,
            "result": value,
            "status": value["status"],
            "comparison_eligible": row["phase"] == "measured",
            "validation_errors": [],
            "recomputed_metrics": {"generated_tokens_per_second": value["test_throughput"]},
        }

    monkeypatch.setattr(report, "_run_record", record)
    profiles = []
    for row in m4_plan["execution_order"]:
        if row.get("phase") != "profile":
            continue
        p = root / "profiles" / row["run_id"]
        p.mkdir(parents=True)
        write_json(p / "trace.json", {})
        write_json(p / "metadata.json", {})
        n, prefill = (64, 0) if row["workload_id"] == "W1" else (27, 640)
        profiles.append(
            {
                "capture_id": row["run_id"],
                "metadata": {
                    "kv_metadata_path": "prepared",
                    "start_step": 2,
                    "end_step": 97 if row["workload_id"] == "W1" else 35,
                },
                "validation_errors": [],
                "gpu_trace_available": True,
                "cpu_inclusive_scopes": [
                    {"name": "vllm_lt::" + name, "count": count}
                    for name, count in (
                        ("recurrent", n),
                        ("kv_prepare", n),
                        ("kv_write", n * 24),
                        ("attention", n * 24),
                    )
                ],
                "gpu_kernels": [{"name": "_paged_attention_kernel", "count": n * 24}],
                "gpu_memcpy": [{"name": "HtoD", "count": 1}],
                "interleaved_prefill_tokens": prefill,
            }
        )
    monkeypatch.setattr(report, "_profiles", lambda *a: deepcopy(profiles))
    return root, m4_plan, audits, profiles


def change(path, mutate):
    data = read_json(path)
    mutate(data)
    write_json(path, data)


def test_complete139_row_matrix_reuses_pair_rules_and_separate_outer_lifetimes(evidence):
    root, plan, _, _ = evidence
    value = report.build_report(root)
    assert value["errors"] == [] and value["missing"] == []
    assert value["decision"] == "passed" and value["evidence_status"] == "complete"
    assert value["counts"] == {
        "planned_executions": 139,
        "completed_executions": 139,
        "benchmark_runs": 78,
        "eligible_timing_runs": 28,
        "profiles": 4,
        "workers": 8,
    }
    assert value["profiles"][0]["launch_policy"] == {"BLOCK_T": 32, "num_warps": 4}


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "affinity",
        "gpu",
        "overlap",
        "ack",
        "outer_identity",
        "outer_clock",
        "inner_escape",
    ],
)
def test_corrupt_controls_or_boundaries_are_invalid(evidence, mutation):
    root, plan, _, _ = evidence
    worker = root / "workers" / "B1" / "manifest.json"
    row = next(
        r
        for r in plan["execution_order"]
        if r["worker_id"] == "B1" and r.get("phase") == "measured"
    )
    outer = root / "workers" / "B1" / (row["execution_id"] + ".lifetime.json")
    if mutation == "source":
        change(worker, lambda x: x["source"].update(commit="c" * 40))
    elif mutation == "affinity":
        change(worker, lambda x: x["environment"].update(cpu_affinity=[99]))
    elif mutation == "gpu":
        change(worker, lambda x: x["environment"].update(gpu_uuid="other"))
    elif mutation == "overlap":
        change(
            root / "manifest.json",
            lambda x: x["workers"][3].update(launched_ns=x["workers"][2]["returned_ns"] - 1),
        )
    elif mutation == "ack":
        change(root / "manifest.json", lambda x: x["completed_executions"].reverse())
    elif mutation == "outer_identity":
        change(outer, lambda x: x.update(execution_id="wrong"))
    elif mutation == "outer_clock":
        change(outer, lambda x: x.update(case_completed_ns=x["started_ns"] - 1))
    else:
        change(
            root / "runs" / row["run_id"] / "result.json",
            lambda x: x.update(case_completed_ns=10**16),
        )
    result = report.build_report(root)
    assert result["evidence_status"] == "invalid" and result["decision"] == "inconclusive"


def stop_after_b1(root, plan):
    # Preserve a terminal clean prefix and an interrupted final worker; later outputs are absent.
    import shutil

    parent = read_json(root / "manifest.json")
    parent.update(status="failed", failures=[{"type": "OSError", "errno": 5}])
    parent["workers"] = parent["workers"][:5]
    parent["completed_workers"] = parent["completed_workers"][:4]
    parent["completed_executions"] = [
        r["execution_id"]
        for r in plan["execution_order"]
        if r["worker_id"] in ("N-A", "N-B", "A1", "B1")
    ]
    write_json(root / "manifest.json", parent)
    change(
        root / "workers" / "B2" / "manifest.json",
        lambda x: (
            x.update(status="running", passed=False, completed_executions=[]),
            x.pop("ended_ns"),
            x.pop("teardown_after_workspace_release"),
        ),
    )
    for wid in ("A2", "P-A", "P-B"):
        shutil.rmtree(root / "workers" / wid)
    for row in plan["execution_order"]:
        if row["worker_id"] in ("B2", "A2", "P-A", "P-B") and row.get("kind") == "benchmark":
            shutil.rmtree(root / "runs" / row["run_id"])
    shutil.rmtree(root / "profiles")


def test_known_first_pair_failure_survives_later_interrupted_worker(evidence):
    root, plan, _, profiles = evidence
    stop_after_b1(root, plan)
    profiles.clear()
    row = next(
        r
        for r in plan["execution_order"]
        if r["worker_id"] == "B1" and r.get("phase") == "measured" and r["cell_id"] == "W1-refill"
    )
    change(root / "runs" / row["run_id"] / "result.json", lambda x: x.update(test_throughput=100.5))
    value = report.build_report(root)
    assert value["errors"] == []
    assert value["decision"] == "failed" and value["evidence_status"] == "incomplete"
    assert value["counts"]["eligible_timing_runs"] == 14 and value["interrupted_workers"] == ["B2"]
    assert value["cells"][0]["pairs"][0]["status"] == "failed"


def test_missing_only_prefix_stays_inconclusive(evidence):
    root, plan, _, profiles = evidence
    stop_after_b1(root, plan)
    profiles.clear()
    value = report.build_report(root)
    assert value["errors"] == [] and not value["required_failures"]
    assert value["decision"] == "inconclusive" and value["evidence_status"] == "incomplete"


def test_planned_benchmark_artifact_without_worker_launch_is_invalid(evidence):
    root, plan, _, profiles = evidence
    row = next(r for r in plan["execution_order"] if r["worker_id"] == "A2")
    saved = read_json(root / "runs" / row["run_id"] / "result.json")
    stop_after_b1(root, plan)
    profiles.clear()
    output = root / "runs" / row["run_id"]
    output.mkdir()
    write_json(output / "result.json", saved)
    result = report.build_report(root)
    assert result["evidence_status"] == "invalid" and result["decision"] == "inconclusive"
    assert any("unlaunched worker" in error["message"] for error in result["errors"])


def test_completed_pair_with_corrupt_gate_count_is_invalid(evidence):
    root, plan, _, _ = evidence
    row = next(
        r
        for r in plan["execution_order"]
        if r["worker_id"] == "B1" and r.get("phase") == "measured"
    )
    change(
        root / "runs" / row["run_id"] / "result.json",
        lambda value: value["counts"].update(gate_probabilities=[]),
    )
    result = report.build_report(root)
    assert result["evidence_status"] == "invalid" and result["decision"] == "inconclusive"


def test_observed_nonzero_cleanup_is_required_failure_missing_measurement_is_not(evidence):
    root, plan, _, profiles = evidence
    stop_after_b1(root, plan)
    profiles.clear()
    p = root / "workers" / "B1" / "manifest.json"
    change(
        p,
        lambda x: x.update(
            teardown_after_workspace_release={"allocated_bytes": 1, "reserved_bytes": 1}
        ),
    )
    result = report.build_report(root)
    assert result["decision"] == "failed" and not result["errors"]
    change(p, lambda x: x.pop("teardown_after_workspace_release"))
    result = report.build_report(root)
    assert result["decision"] == "inconclusive" and not result["required_failures"]


@pytest.mark.parametrize(
    "mutation", ["public", "prepare_count", "no_gpu", "attention_count", "prefill", "batch_history"]
)
def test_profiles_require_prepared_compact_matched_actual_work(evidence, mutation):
    root, plan, _, profiles = evidence
    profile = profiles[-1]
    if mutation == "public":
        profile["metadata"]["kv_metadata_path"] = "public"
    elif mutation == "prepare_count":
        profile["cpu_inclusive_scopes"][1]["count"] -= 1
    elif mutation == "no_gpu":
        profile["gpu_kernels"] = []
    elif mutation == "attention_count":
        profile["gpu_kernels"][0]["count"] -= 1
    elif mutation == "prefill":
        profile["interleaved_prefill_tokens"] -= 1
    else:
        p = root / "runs" / profile["capture_id"] / "result.json"
        change(p, lambda x: x["counts"]["snapshots"][0]["batch"]["rows"][0].update(position=9))
    result = report.build_report(root)
    assert result["evidence_status"] == "invalid" and result["decision"] == "inconclusive"


def test_actual_profile_parser_separates_cpu_and_gpu_annotations(tmp_path, m4_plan):
    from vllm_lt.benchmarks.report import _profiles

    row = next(r for r in m4_plan["execution_order"] if r.get("phase") == "profile")
    folder = tmp_path / "profiles" / row["run_id"]
    folder.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "artifact_type": "profile_metadata",
        "capture_id": row["run_id"],
        "complete_window": True,
        "start_step": 2,
        "end_step": 97,
        **{
            k: row[k]
            for k in ("cell_id", "workload_id", "mode", "controls_sha256", "workload_sha256")
        },
    }
    write_json(folder / "metadata.json", metadata)
    events = [
        {"ph": "X", "cat": cat, "name": "vllm_lt::kv_prepare", "ts": 1, "dur": 2}
        for cat in ("user_annotation", "gpu_user_annotation")
    ]
    events += [
        {"ph": "X", "cat": "kernel", "name": "_paged_attention_kernel", "ts": 3, "dur": 4},
        {"ph": "X", "cat": "gpu_memcpy", "name": "HtoD", "ts": 5, "dur": 1},
    ]
    write_json(folder / "trace.json", {"traceEvents": events})
    value = _profiles(
        tmp_path, {}, [{"run_id": row["run_id"], "planned": row, "result": None}], {}
    )[0]
    assert value["cpu_inclusive_scopes"] == [
        {"name": "vllm_lt::kv_prepare", "count": 1, "total_us": 2}
    ]
    assert value["gpu_trace_available"] and value["gpu_kernels"][0]["count"] == 1
