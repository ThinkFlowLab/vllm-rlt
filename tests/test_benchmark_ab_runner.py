"""CPU-only subprocess/control fixtures exercise M2 lifecycle and offline ordering."""

import signal
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_benchmark_ab_schema as schema_fixtures
import torch
from test_benchmark_ab_report import records_for

from vllm_lt.benchmarks import ab, ab_report, runner
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.validation import m2

ab_plan = schema_fixtures.ab_plan
AFFINITY = schema_fixtures.AFFINITY


@pytest.fixture
def controller(ab_plan, monkeypatch, tmp_path):
    clock = [10**12]

    def now():
        clock[0] += 1000
        return clock[0]

    monkeypatch.setattr(ab.time, "perf_counter_ns", now)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(ab, "verify_ab_plan", lambda *a, **kw: None)
    rows = {r["run_id"]: r for r in records_for(ab_plan)}
    lifetimes, launches = {}, []
    state = {"fail_worker": None, "bad_gate": False}

    def numerical(*args):
        complete = len(lifetimes) == 27
        return {
            "complete": complete,
            "passed": complete and not state["bad_gate"],
            "errors": [],
            "ledger": {"case_lifetimes": deepcopy(lifetimes)},
        }

    monkeypatch.setattr(m2, "audit_numerical", numerical)

    def launch(plan, worker, output, deadline):
        launches.append(worker["worker_id"])
        started = now()
        completed = []
        for execution_id in worker["execution_ids"]:
            if execution_id in rows:
                record = deepcopy(rows[execution_id])
                row, result = record["planned"], record["result"]
                case_start, arrival, synchronized, end = [now() for _ in range(4)]
                result.update(
                    case_started_ns=case_start,
                    case_completed_ns=end,
                    arrival_ns=arrival,
                    synchronized_ns=synchronized,
                )
                folder = output / "runs" / execution_id
                folder.mkdir(parents=True)
                write_json(folder / "result.json", result)
                write_json(
                    folder / "started.json",
                    {
                        "schema_version": 1,
                        **row,
                        "started_ns": case_start,
                        "deadline_ns": min(deadline, case_start + 600 * 10**9),
                    },
                )
            else:
                case_start, case_end = now(), now()
                lifetimes[execution_id] = {
                    "started_ns": case_start,
                    "finished_ns": case_end,
                    "deadline_ns": min(deadline, case_start + 600 * 10**9),
                }
            completed.append(execution_id)
            if state["fail_worker"] == worker["worker_id"]:
                break
        failed = state["fail_worker"] == worker["worker_id"]
        env = {
            "cuda_visible_devices": "7",
            "logical_device": "cuda:0",
            "cpu_affinity": AFFINITY["cpu_ids"],
            "numa_status": AFFINITY["numa_status"],
            "actual_torch_threads": {"intraop": 1, "interop": 1},
            "python": plan["dependencies"]["python"],
            "torch_cuda_version": plan["dependencies"]["torch_cuda_build"],
            "software": {},
            "arithmetic": {**plan["benchmark_contract"]["arithmetic"], "cudnn_allow_tf32": False},
            "scheduler": [{"type": "RUN", "gpu_id": 7, "user": "cpu-test-user"}],
            "account": "cpu-test-user",
            "host": "synthetic-cpu-only",
            "gpu_uuid": "synthetic",
            "gpu_name": "synthetic",
            "total_device_bytes": 0,
            "compute_capability": [0, 0],
            "reservation_environment": {"test": "owned"},
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": "m2_worker_manifest",
            **worker,
            "plan_sha256": plan["plan_sha256"],
            "source": plan["implementations"][worker["implementation_id"]]["source"],
            "harness_sha256": plan["harness"]["sha256"],
            "affinity": deepcopy(AFFINITY),
            "runtime_environment": plan["runtime_environment"],
            "environment": env,
            "status": "failed" if failed else "complete",
            "passed": not failed,
            "failures": [{"message": "injected"}] if failed else [],
            "completed_executions": completed,
            "started_ns": started,
            "ended_ns": now(),
            "deadline_ns": deadline,
            "model_loads": 1,
            "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
        }
        folder = output / "workers" / worker["worker_id"]
        folder.mkdir()
        write_json(folder / "manifest.json", manifest)
        return 1 if failed else 0

    monkeypatch.setattr(ab, "_launch_worker", launch)

    def run_record(output, row, view, hashes):
        record = deepcopy(rows[row["run_id"]])
        path = output / "runs" / row["run_id"] / "result.json"
        if path.exists():
            record["result"] = read_json(path)
        else:
            record.update(
                result=None,
                status="missing",
                comparison_eligible=False,
                validation_errors=["missing raw run"],
            )
        return record

    monkeypatch.setattr(ab_report, "_run_record", run_record)

    def profiles(output, hashes, records, view):
        result = []
        for row in ab_plan["execution_order"]:
            if row.get("phase") != "profile":
                continue
            candidate = row["implementation_id"] == "B"
            result.append(
                {
                    "capture_id": row["run_id"],
                    "validation_errors": [],
                    "gpu_trace_available": True,
                    "gpu_kernels": ["synthetic"],
                    "gpu_memcpy": ["synthetic"],
                    "metadata": {"kv_metadata_path": "prepared" if candidate else "public"},
                    "cpu_inclusive_scopes": [
                        {"name": "vllm_lt::recurrent", "count": 4},
                        {"name": "vllm_lt::kv_write", "count": 96},
                        {"name": "vllm_lt::attention", "count": 96},
                        {"name": "vllm_lt::kv_prepare", "count": 4 if candidate else 0},
                    ],
                }
            )
        return result

    monkeypatch.setattr(ab_report, "_profiles", profiles)
    return SimpleNamespace(
        plan=ab_plan,
        output=tmp_path / "experiment",
        state=state,
        launches=launches,
        lifetimes=lifetimes,
        clock=clock,
    )


def test_controller_exact_eight_workers_and_offline_cross_evidence_pass(controller):
    old_handler = signal.getsignal(signal.SIGTERM)
    manifest = ab.run_ab(controller.plan, output_dir=controller.output)
    assert manifest["status"] == "complete", manifest["failures"]
    assert controller.launches == ["N-A", "N-B", "A1", "B1", "B2", "A2", "P-A", "P-B"]
    assert len(manifest["completed_executions"]) == 105
    assert signal.getsignal(signal.SIGTERM) == old_handler
    report = ab_report.build_report(controller.output)
    assert report["evidence_status"] == "complete", report["errors"]
    assert report["decision"] == "passed"
    assert report["counts"]["eligible_timing_runs"] == 28


def test_worker_failure_preserves_prefix_and_never_launches_next(controller):
    controller.state["fail_worker"] = "N-B"
    manifest = ab.run_ab(controller.plan, output_dir=controller.output)
    assert manifest["status"] == "failed"
    assert controller.launches == ["N-A", "N-B"]
    assert len(manifest["completed_executions"]) == 24  # 23 N-A, one failed-worker completed row.
    report = ab_report.build_report(controller.output)
    assert report["decision"] == "inconclusive"
    assert report["evidence_status"] != "complete"


def test_failed_offline_numerical_gate_prevents_first_timed_worker(controller):
    controller.state["bad_gate"] = True
    manifest = ab.run_ab(controller.plan, output_dir=controller.output)
    assert manifest["status"] == "failed"
    assert controller.launches == ["N-A", "N-B"]
    assert manifest["numerical_gate"]["passed"] is False


@pytest.mark.parametrize(
    "change",
    [
        "outside_worker",
        "over600",
        "reversed",
        "marker",
        "pool",
        "worker_deadline",
        "numerical_time",
        "gate_time",
        "parent_interval",
        "missing_case",
        "missing_marker",
    ],
)
def test_offline_report_rejects_cross_evidence_tampering(controller, change):
    ab.run_ab(controller.plan, output_dir=controller.output)
    output = controller.output
    manifest = read_json(output / "manifest.json")
    planned = next(
        r
        for r in controller.plan["execution_order"]
        if r.get("worker_id") == "A1" and r.get("phase") == "measured"
    )
    path = output / "runs" / planned["run_id"] / "result.json"
    result = read_json(path)
    if change == "outside_worker":
        result["arrival_ns"] = 0
    elif change == "over600":
        result["case_completed_ns"] = result["case_started_ns"] + 601 * 10**9
    elif change == "reversed":
        result["case_started_ns"] -= 10**8
    elif change == "pool":
        result["memory"]["pool_bytes"] = 0
    elif change == "missing_marker":
        path.with_name("started.json").unlink()
    elif change == "marker":
        marker_path = path.with_name("started.json")
        marker = read_json(marker_path)
        marker["deadline_ns"] += 1
        write_json(marker_path, marker)
    elif change == "worker_deadline":
        child_path = output / "workers/A1/manifest.json"
        child = read_json(child_path)
        child["deadline_ns"] += 1
        write_json(child_path, child)
    elif change == "numerical_time":
        case = controller.plan["numerical"]["execution_order"][0]["case_id"]
        controller.lifetimes[case]["started_ns"] = 1
    elif change == "gate_time":
        manifest["numerical_gate_ns"] = manifest["started_ns"]
    elif change == "parent_interval":
        manifest["ended_ns"] = manifest["started_ns"]
    if change == "missing_case":
        path.unlink()
    else:
        write_json(path, result)
    write_json(output / "manifest.json", manifest)
    report = ab_report.build_report(output)
    assert report["evidence_status"] != "complete"
    assert report["decision"] == "inconclusive"


def test_watchdog_stays_live_until_post_result_cleanup(tmp_path):
    folder = tmp_path / "runs/run"
    folder.mkdir(parents=True)
    write_json(folder / "started.json", {"deadline_ns": 600})
    write_json(folder / "result.json", {"status": "complete"})
    worker = {"worker_id": "A1", "execution_ids": ["run"]}
    assert ab.active_case_deadline(tmp_path, worker) == 600
    write_json(folder / "result.json", {"status": "complete", "case_completed_ns": 599})
    assert ab.active_case_deadline(tmp_path, worker) is None


def test_final_cleanup_attempts_all_stages_after_sync_failure(monkeypatch):
    calls = []

    def broken():
        calls.append("sync")
        raise RuntimeError("injected synchronization failure")

    monkeypatch.setattr(torch.cuda, "synchronize", broken)
    monkeypatch.setattr(
        torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append("workspace"), raising=False
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("allocator"))
    monkeypatch.setattr(runner, "memory", lambda: {"allocated_bytes": 0, "reserved_bytes": 0})
    manifest = {}
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.release_device(manifest)
    assert calls == ["sync", "workspace", "allocator"]
    assert manifest["cleanup_errors"][0]["stage"] == "synchronize"
    assert manifest["teardown_after_workspace_release"] == {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }


def test_worker_partial_environment_failure_still_releases_device(ab_plan, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(ab, "verify_ab_plan", lambda *a, **kw: None)
    monkeypatch.setattr(ab, "affinity_snapshot", lambda: deepcopy(AFFINITY))
    monkeypatch.setattr(runner, "configure_process", lambda *a: None)

    def fail():
        raise RuntimeError("partial environment discovery")

    monkeypatch.setattr(runner, "environment", fail)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(runner, "release_device", lambda manifest: calls.append("release"))
    value = ab.run_worker(ab_plan, worker_id="N-A", output_dir=tmp_path / "out", deadline_ns=10**30)
    assert value["status"] == "failed"
    assert value["model_loads"] == 0 and calls == ["release"]
    assert value["failures"][0]["message"] == "partial environment discovery"


@pytest.mark.parametrize("reason", ["interrupt", "artifact_cap"])
def test_owned_child_group_is_terminated_on_controller_stop(ab_plan, tmp_path, monkeypatch, reason):
    calls = []

    class Child:
        pid = 424242

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                if reason == "interrupt":
                    raise KeyboardInterrupt("owned controller stop")
                raise subprocess.TimeoutExpired("owned worker", timeout)
            return -15

        def poll(self):
            return None

    child = Child()

    def launch(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == ab_plan["implementations"]["A"]["root"]
        return child

    monkeypatch.setattr(ab.subprocess, "Popen", launch)
    monkeypatch.setattr(ab.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    if reason == "artifact_cap":

        def over_cap(*args):
            raise ValueError("artifact cap exceeded")

        monkeypatch.setattr(ab, "artifact_usage", over_cap)
    (tmp_path / "workers").mkdir()
    with pytest.raises(KeyboardInterrupt if reason == "interrupt" else ValueError):
        ab._launch_worker(ab_plan, ab_plan["workers"][0], tmp_path, deadline_ns=10**30)
    assert (424242, signal.SIGTERM) in calls
    assert calls[-1] == ("wait", 2)


def test_controller_sigterm_records_stop_and_restores_handler(controller, monkeypatch):
    old = signal.getsignal(signal.SIGTERM)

    def interrupted(*args):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(ab, "_launch_worker", interrupted)
    result = ab.run_ab(controller.plan, output_dir=controller.output)
    assert result["status"] == "incomplete"
    assert result["completed_workers"] == []
    assert len(result["workers"]) == 1 and result["workers"][0]["returned_ns"] is not None
    assert signal.getsignal(signal.SIGTERM) == old


def test_active_case_deadline_stops_owned_child(ab_plan, tmp_path, monkeypatch):
    class Child:
        pid = 424242

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -15

    calls = []
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(ab.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(ab, "active_case_deadline", lambda *a: 1)
    (tmp_path / "workers").mkdir()
    with pytest.raises(TimeoutError, match="case lifetime"):
        ab._launch_worker(ab_plan, ab_plan["workers"][0], tmp_path, deadline_ns=10**30)
    assert calls == [(424242, signal.SIGTERM)]


def test_shared_executor_rejects_final_step_that_crosses_deadline(tmp_path, monkeypatch):
    from test_benchmark_runner import case, cuda_boundaries

    from vllm_lt.engine.llm_engine import LLMEngine
    from vllm_lt.models import OuroConfig, OuroForCausalLM

    cuda_boundaries.__wrapped__(monkeypatch)
    plan, row, workload = case()
    plan["contract"]["limits"]["workload_timeout_s"] = 1
    clock = [10**12]

    def now():
        clock[0] += 1000
        return clock[0]

    monkeypatch.setattr(runner.time, "perf_counter_ns", now)

    def engine(*args, **kwargs):
        instance = LLMEngine(*args, **kwargs)
        original = instance.step

        def step():
            outputs = original()
            if not instance.has_unfinished_requests():
                clock[0] += 2 * 10**9
            return outputs

        instance.step = step
        return instance

    monkeypatch.setattr(runner, "LLMEngine", engine)
    result = runner._execute(
        OuroForCausalLM(OuroConfig.tiny()), plan, row, workload, tmp_path, deadline=10**30
    )
    assert result["status"] == "incomplete" and not result["comparison_eligible"]
    assert result["failures"][0]["type"] == "TimeoutError"
    assert len(result["requests"][0]["token_ids"]) == 3  # Completed tokens remain raw evidence.
    assert result["cleanup"] == {"active_requests": 0, "used_kv_blocks": 0}
