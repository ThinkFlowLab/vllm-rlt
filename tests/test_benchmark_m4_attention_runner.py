"""CPU orchestration tests exercise the actual M4 plan and finite stop behavior."""

import signal
from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_benchmark_m4_attention_schema as schema_fixtures
import torch
from test_benchmark_m4_attention_schema import AFFINITY

from vllm_lt.benchmarks import ab, runner
from vllm_lt.benchmarks import m4_attention as controller
from vllm_lt.benchmarks import m4_attention_schema as schema
from vllm_lt.benchmarks.schema import write_json
from vllm_lt.validation import m4_attention as numerical
from vllm_lt.validation import m4_attention_held as held

m4_plan = schema_fixtures.m4_plan
no_device_or_weights = schema_fixtures.no_device_or_weights


@pytest.fixture
def worker(m4_plan, monkeypatch, tmp_path):
    clock = [10**12]

    def now():
        clock[0] += 1000
        return clock[0]

    monkeypatch.setattr(controller.time, "perf_counter_ns", now)
    monkeypatch.setattr(schema, "verify_plan", lambda *a, **kw: None)
    monkeypatch.setattr(controller, "affinity_snapshot", lambda: deepcopy(AFFINITY))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(runner, "configure_process", lambda *a: None)
    monkeypatch.setattr(runner, "environment", lambda: {})
    calls, state = [], {"fail": None, "delay_callback": False}

    def load(plan, manifest):
        calls.append("load")
        return object()

    monkeypatch.setattr(runner, "load_model", load)
    monkeypatch.setattr(runner, "release_device", lambda manifest: calls.append("cleanup"))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    original_usage = controller.artifact_usage

    def usage(root, plan):
        result = original_usage(root, plan)
        if state["delay_callback"]:
            clock[0] += 601 * 10**9
        return result

    monkeypatch.setattr(controller, "artifact_usage", usage)

    def benchmark(model, plan, rows, *, output_dir, deadline, record_completed):
        row = rows[0]
        calls.append(row["execution_id"])
        assert (
            controller.active_case_deadline(
                output_dir,
                next(w for w in m4_plan["workers"] if w["worker_id"] == row["worker_id"]),
            )
            == deadline
        )
        if state["fail"] == "benchmark":
            raise OSError(5, "injected write error", "owned-result.json")
        record_completed(row["execution_id"], {"status": "complete"})

    monkeypatch.setattr(runner, "run_loaded_rows", benchmark)

    def direct(model, plan, row, output_dir, deadline):
        assert row in plan["held"]["execution_order"]
        calls.append(row["execution_id"])
        if state["fail"] == "held":
            return {"status": "failed", "passed": False}
        return {"status": "complete", "passed": True}

    monkeypatch.setattr(held, "run_held_row", direct)

    def model_cases(model, plan, side, output, deadline, *, after_case):
        for row in plan["numerical"]["execution_order"]:
            if row["implementation_id"] == side:
                calls.append(row["case_id"])
                after_case(row, {"status": "complete"})
        return {"complete": True, "passed": True}

    monkeypatch.setattr(numerical, "run_numerical_rows", model_cases)
    return SimpleNamespace(
        plan=m4_plan, output=tmp_path / "run", clock=clock, calls=calls, state=state, now=now
    )


def test_numerical_worker_binds_full_held_rows_and_acknowledges_all40(worker):
    original = signal.getsignal(signal.SIGTERM)
    result = controller.run_worker(
        worker.plan, worker_id="N-A", output_dir=worker.output, deadline_ns=10**15
    )
    assert result["status"] == "complete", result["failures"]
    assert result["completed_executions"] == worker.plan["workers"][0]["execution_ids"]
    assert worker.calls == ["load", *result["completed_executions"], "cleanup"]
    assert len(list((worker.output / "workers/N-A").glob("*.lifetime.json"))) == 24
    assert signal.getsignal(signal.SIGTERM) == original


@pytest.mark.parametrize("failure,completed", [("benchmark", 0), ("held", 7)])
def test_worker_preserves_prefix_stops_and_cleans_up(worker, failure, completed):
    worker.state["fail"] = failure
    result = controller.run_worker(
        worker.plan, worker_id="N-A", output_dir=worker.output, deadline_ns=10**15
    )
    assert result["status"] == "failed"
    assert len(result["completed_executions"]) == completed
    assert worker.calls[-1] == "cleanup"
    assert controller.active_case_deadline(worker.output, worker.plan["workers"][0]) is not None
    if failure == "benchmark":
        assert result["failures"][0]["errno"] == 5
        assert result["failures"][0]["filename"] == "owned-result.json"


def test_callback_time_is_included_in_full_case_limit(worker):
    worker.state["delay_callback"] = True
    result = controller.run_worker(
        worker.plan, worker_id="A1", output_dir=worker.output, deadline_ns=10**15
    )
    assert result["status"] == "failed"
    assert "complete case lifetime exceeded" in result["failures"][0]["message"]
    assert len(result["completed_executions"]) == 1
    assert not list((worker.output / "workers/A1").glob("*.lifetime.json"))
    assert worker.calls[-1] == "cleanup"


def test_expired_source_verification_does_not_touch_device(worker, monkeypatch):
    def verify(*args, **kwargs):
        worker.clock[0] = 10**15

    monkeypatch.setattr(schema, "verify_plan", verify)
    with pytest.raises(ValueError, match="before device use"):
        controller.run_worker(
            worker.plan, worker_id="N-A", output_dir=worker.output, deadline_ns=10**15
        )
    assert worker.calls == []


def test_partial_environment_failure_cleans_initialized_device(worker, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def fail():
        raise RuntimeError("partial discovery")

    monkeypatch.setattr(runner, "environment", fail)
    value = controller.run_worker(
        worker.plan, worker_id="N-A", output_dir=worker.output, deadline_ns=10**15
    )
    assert value["status"] == "failed" and worker.calls == ["cleanup"]


@pytest.fixture
def parent(worker, monkeypatch):
    state = {"bad_gate": False, "fail_worker": None, "gate_delay": False}
    launches = []

    def gate(*args, **kwargs):
        if state["gate_delay"]:
            worker.clock[0] += 7201 * 10**9
        return {"complete": True, "passed": not state["bad_gate"]}

    monkeypatch.setattr(numerical, "audit_numerical", gate)
    monkeypatch.setattr(held, "audit_held", gate)
    monkeypatch.setattr(ab, "audit_worker_controls", lambda *a, **kw: None)

    def launch(plan, row, root, deadline, **kwargs):
        assert kwargs["module"] == "vllm_lt.benchmarks.m4_attention"
        assert kwargs["active_deadline"] is controller.active_case_deadline
        launches.append(row["worker_id"])
        failed = row["worker_id"] == state["fail_worker"]
        value = {
            "completed_executions": row["execution_ids"][:1] if failed else row["execution_ids"],
            "status": "failed" if failed else "complete",
            "passed": not failed,
            "started_ns": worker.now(),
            "ended_ns": worker.now(),
            "model_loads": 1,
        }
        folder = root / "workers" / row["worker_id"]
        folder.mkdir()
        write_json(folder / "manifest.json", value)
        return 1 if failed else 0

    monkeypatch.setattr(ab, "_launch_worker", launch)
    return SimpleNamespace(**vars(worker), parent_state=state, launches=launches)


def test_parent_launches_exact139_finite_sequence_and_gate_precedes_timing(parent):
    value = controller.run_ab(parent.plan, output_dir=parent.output)
    assert value["status"] == "complete", value["failures"]
    assert len(value["completed_executions"]) == 139
    assert parent.launches == ["N-A", "N-B", "A1", "B1", "B2", "A2", "P-A", "P-B"]
    assert value["numerical_gate_ns"] < value["workers"][2]["launched_ns"]


@pytest.mark.parametrize("reason", ["bad_gate", "gate_delay"])
def test_parent_cannot_start_timing_after_gate_failure_or_deadline(parent, reason):
    parent.parent_state[reason] = True
    value = controller.run_ab(parent.plan, output_dir=parent.output)
    assert value["status"] != "complete"
    assert parent.launches == ["N-A", "N-B"]


def test_parent_retains_child_acknowledged_prefix_without_advancing(parent):
    parent.parent_state["fail_worker"] = "B1"
    value = controller.run_ab(parent.plan, output_dir=parent.output)
    assert value["status"] == "failed"
    assert value["completed_workers"] == ["N-A", "N-B", "A1"]
    assert len(value["completed_executions"]) == 40 + 35 + 14 + 1
    assert parent.launches[-1] == "B1"


def test_artifact_partition_and_auxiliary_cap_are_actual_bytes(m4_plan, tmp_path):
    root = tmp_path / "usage"
    for name in (
        "numerical/spools/a/f.bin",
        "numerical/dumps/a.bin",
        "numerical/comparisons/a.jsonl",
        "held/a.bin",
        "profiles/P/trace.json",
        "numerical/ledger.json",
        "manifest.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"123")
    result = controller.artifact_usage(root, m4_plan)
    assert result == {
        "total_bytes": 21,
        "profile_trace_bytes": 3,
        "numerical_bytes": 9,
        "held_bytes": 3,
        "auxiliary_bytes": 6,
    }
    m4_plan["contract"]["limits"]["auxiliary_artifact_bytes_max"] = 5
    with pytest.raises(ValueError, match="auxiliary"):
        controller.artifact_usage(root, m4_plan)


def test_failure_diagnostic_is_bounded_and_keeps_io_operation():
    result = controller.failure(OSError(5, "x" * 10000, "task-result.json"))
    assert result["errno"] == 5 and result["filename"] == "task-result.json"
    assert len(result["message"]) <= 2048 and len(result["traceback"]) <= 8192
