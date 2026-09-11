"""CPU controller integration: recorded controls, stop prefixes, and owned workers."""

import signal
from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_validation_m3_persistent_schema as fixtures

from vllm_lt.benchmarks import ab
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.validation import m3_inactive_run as common
from vllm_lt.validation import m3_persistent_report as report
from vllm_lt.validation import m3_persistent_run as driver

ab_plan = fixtures.ab_plan
lifecycle_template = fixtures.lifecycle_template
persistent_plan = fixtures.persistent_plan


def environment(plan):
    """Synthetic recorded facts passed through the real CPU controls auditor."""
    affinity = plan["contract"]["controls"]["affinity"]
    return {
        "cuda_visible_devices": "7",
        "logical_device": "cuda:0",
        "cpu_affinity": affinity["cpu_ids"],
        "numa_status": affinity["numa_status"],
        "actual_torch_threads": {"intraop": 1, "interop": 1},
        "python": plan["dependencies"]["python"],
        "torch_cuda_version": plan["dependencies"]["torch_cuda_build"],
        "software": {},
        "arithmetic": {**plan["numerical"]["contract"]["arithmetic"], "cudnn_allow_tf32": False},
        "scheduler": [{"type": "RUN", "gpu_id": 7, "user": "fixture"}],
        "account": "fixture",
        "host": "synthetic-cpu-fixture",
        "gpu_uuid": "fixture-only",
        "gpu_name": "fixture-only",
        "total_device_bytes": 80 * 1024**3,
        "compute_capability": [9, 0],
        "reservation_environment": {"fixture": "same"},
    }


@pytest.fixture
def controller(persistent_plan, monkeypatch, tmp_path):
    from vllm_lt.validation import m3_persistent_lifecycle as lifecycle

    clock = [10**12]

    def now():
        clock[0] += 1000
        return clock[0]

    monkeypatch.setattr(common.time, "perf_counter_ns", now)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(driver, "verify_plan", lambda *a, **kw: None)
    model_times = {}
    state = {"fail": False, "launches": [], "pointer_missing": False}

    def launch(plan, worker, output, deadline, *, module, active_deadline):
        assert module == "vllm_lt.validation.m3_persistent_run"
        assert active_deadline is driver.active_deadline
        state["launches"].append(worker["worker_id"])
        started, completed = now(), []
        for row in plan["execution_order"]:
            if row["worker_id"] != worker["worker_id"]:
                continue
            first, last = now(), now()
            timing = {
                "started_ns": first,
                "finished_ns": last,
                "deadline_ns": min(deadline, first + 600 * 10**9),
            }
            if row["kind"] == "model":
                folder = output / "numerical/cases" / row["execution_id"]
                folder.mkdir(parents=True)
                write_json(folder / "started.json", {"case_id": row["execution_id"], **timing})
                if state["fail"]:
                    break
                model_times[row["execution_id"]] = timing
            else:
                folder = output / "lifecycle/evaluations" / row["execution_id"]
                folder.mkdir(parents=True)
                marker = {
                    "evaluation": {"evaluation_id": row["execution_id"]},
                    **timing,
                    "lifecycle_plan_sha256": plan["lifecycle"]["lifecycle_plan_sha256"],
                }
                write_json(folder / "started.json", marker)
                write_json(folder / "result.json", {**marker, "status": "complete", "passed": True})
            completed.append(row["execution_id"])
        failed = state["fail"]
        value = {
            "schema_version": 1,
            "artifact_type": "m3_persistent_worker_manifest",
            **worker,
            "plan_sha256": plan["plan_sha256"],
            "source": plan["implementations"][worker["implementation_id"]]["source"],
            "harness_sha256": plan["harness"]["sha256"],
            "affinity": plan["contract"]["controls"]["affinity"],
            "runtime_environment": plan["runtime_environment"],
            "environment": environment(plan),
            "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
            "model_loads": 1,
            "started_ns": started,
            "ended_ns": now(),
            "deadline_ns": deadline,
            "status": "failed" if failed else "complete",
            "passed": not failed,
            "failures": [{"message": "injected feasibility failure"}] if failed else [],
            "completed_executions": completed,
        }
        folder = output / "workers" / worker["worker_id"]
        folder.mkdir()
        write_json(folder / "manifest.json", value)
        return int(failed)

    monkeypatch.setattr(ab, "_launch_worker", launch)

    def numerical(*args):
        complete = len(model_times) == 13 and not state["pointer_missing"]
        return {
            "complete": complete,
            "passed": complete,
            "completed_cases": list(model_times),
            "comparisons": [],
            "errors": [] if complete else [{"message": "missing pointer proof"}],
            "ledger": {"case_lifetimes": deepcopy(model_times)},
            "counts": {
                "planned_cases": 13,
                "verified_cases": len(model_times),
                "planned_comparisons": 28,
                "verified_comparisons": 28 if complete else 0,
            },
        }

    def checks(output, plan):
        rows = []
        for evaluation in plan["execution_order"]:
            path = output / "evaluations" / evaluation["evaluation_id"] / "result.json"
            if path.exists():
                raw = read_json(path)
                rows.append(
                    {
                        "evaluation_id": evaluation["evaluation_id"],
                        **{key: raw[key] for key in ("started_ns", "finished_ns", "deadline_ns")},
                    }
                )
        complete = len(rows) == 4
        return {
            "complete": complete,
            "passed": complete,
            "errors": [],
            "evaluations": rows,
            "completed_evaluations": [r["evaluation_id"] for r in rows],
        }

    monkeypatch.setattr(report, "audit_model_rows", numerical)
    monkeypatch.setattr(lifecycle, "audit_lifecycle_outputs", checks, raising=False)
    return SimpleNamespace(
        plan=persistent_plan, output=tmp_path / "run", state=state, model_times=model_times
    )


def test_full17_controller_and_combined_audit_use_real_control_checks(controller):
    result = driver.run(controller.plan, output_dir=controller.output)
    assert result["status"] == "complete", result["failures"]
    assert controller.state["launches"] == ["N-A", "N-B"]
    assert len(result["completed_executions"]) == 17
    audited = report.build_report(controller.output)
    assert audited["evidence_status"] == "complete", audited["errors"]
    assert (
        audited["decision"] == "passed" and audited["counts"]["verified_lifecycle_evaluations"] == 4
    )
    assert "not graph capture or performance" in audited["scope"]


def test_failed_feasibility_preserves_partial_stop_and_never_launches_b(controller):
    controller.state["fail"] = True
    result = driver.run(controller.plan, output_dir=controller.output)
    assert result["status"] == "failed" and controller.state["launches"] == ["N-A"]
    audited = report.build_report(controller.output)
    assert audited["evidence_status"] == "incomplete", audited["errors"]
    assert audited["decision"] == "failed" and not audited["errors"]
    assert audited["counts"]["completed_executions"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "lifecycle_before_feasibility",
        "model_outside_worker",
        "case_deadline",
        "overlap",
        "nonzero_cleanup",
        "different_gpu",
        "missing_lifecycle",
        "missing_pointer_proof",
        "extra_case",
    ],
)
def test_combined_audit_never_promotes_missing_evidence_or_drift(controller, change):
    driver.run(controller.plan, output_dir=controller.output)
    output = controller.output
    if change == "missing_pointer_proof":
        # A real subaudit rejection must dominate successful parent/worker flags.
        # Detailed corrupted-record tests live with the actual pointer auditor.
        controller.state["pointer_missing"] = True
    elif change == "model_outside_worker":
        first = controller.plan["numerical"]["execution_order"][0]["case_id"]
        controller.model_times[first]["started_ns"] = 1
    elif change == "extra_case":
        (output / "numerical/cases/unbudgeted").mkdir()
    elif change == "overlap":
        path = output / "manifest.json"
        value = read_json(path)
        value["workers"][1]["launched_ns"] = value["workers"][0]["returned_ns"] - 1
        write_json(path, value)
    elif change in ("nonzero_cleanup", "different_gpu"):
        path = output / "workers/N-B/manifest.json"
        value = read_json(path)
        if change == "nonzero_cleanup":
            value["teardown_after_workspace_release"]["reserved_bytes"] = 1
        else:
            value["environment"]["gpu_uuid"] = "different-device"
        write_json(path, value)
    else:
        path = output / "lifecycle/evaluations/L-A-torch/result.json"
        if change == "missing_lifecycle":
            path.unlink()
        else:
            value = read_json(path)
            value["started_ns" if change == "lifecycle_before_feasibility" else "deadline_ns"] += (
                -100000 if change == "lifecycle_before_feasibility" else 1
            )
            write_json(path, value)
    audited = report.build_report(output)
    assert audited["decision"] != "passed"
    assert audited["evidence_status"] != "complete"


def test_real_shared_worker_releases_after_partial_environment_initialization(
    persistent_plan, tmp_path, monkeypatch
):
    events = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(common, "affinity_snapshot", lambda: fixtures.fixtures.AFFINITY)
    monkeypatch.setattr(common.runner, "configure_process", lambda c: events.append("configure"))

    def partial():
        events.append("partial_environment")
        raise RuntimeError("metadata failed after context creation")

    monkeypatch.setattr(common.runner, "environment", partial)
    monkeypatch.setattr(common.runner.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(common.runner, "release_device", lambda m: events.append("release"))
    result = common._run_worker(
        persistent_plan,
        worker_id="N-A",
        output_dir=tmp_path,
        deadline_ns=10**30,
        validate=lambda p: None,
        verify=lambda *a, **k: None,
        model_rows=lambda *a, **k: pytest.fail("model executed after preparation failed"),
        checks=lambda *a, **k: pytest.fail("lifecycle executed after preparation failed"),
        artifact_type="m3_persistent_worker_manifest",
    )
    assert result["status"] == "failed" and result["model_loads"] == 0
    assert result["completed_executions"] == []
    assert events == ["configure", "partial_environment", "release"]
    assert result["failures"][0]["message"] == "metadata failed after context creation"


@pytest.mark.parametrize("bad", ["identity", "reversed", "over_cap", "bool"])
def test_lifecycle_watchdog_rejects_bad_started_markers(tmp_path, bad):
    folder = tmp_path / "lifecycle/evaluations/L-A-torch"
    folder.mkdir(parents=True)
    marker = {
        "evaluation": {"evaluation_id": "L-A-torch"},
        "started_ns": 1000,
        "deadline_ns": 1000 + 600 * 10**9,
    }
    if bad == "identity":
        marker["evaluation"]["evaluation_id"] = "../other"
    elif bad == "reversed":
        marker["deadline_ns"] = 1
    elif bad == "over_cap":
        marker["deadline_ns"] += 1
    else:
        marker["started_ns"] = True
    write_json(folder / "started.json", marker)
    worker = {"worker_id": "N-A", "execution_ids": ["m3-persistent-feas", "L-A-torch"]}
    with pytest.raises(ValueError):
        driver.active_deadline(tmp_path, worker)


def test_lifecycle_watchdog_stays_active_until_result_and_combines_numerical_deadline(tmp_path):
    folder = tmp_path / "lifecycle/evaluations/L-A-torch"
    folder.mkdir(parents=True)
    marker = {"evaluation": {"evaluation_id": "L-A-torch"}, "started_ns": 1000, "deadline_ns": 3000}
    write_json(folder / "started.json", marker)
    worker = {"worker_id": "N-A", "execution_ids": ["m3-persistent-feas", "L-A-torch"]}
    assert driver.active_deadline(tmp_path, worker) == 3000
    (tmp_path / "numerical").mkdir()
    write_json(
        tmp_path / "numerical/ledger.json",
        {"active_case": {"case_id": "m3-persistent-feas", "started_ns": 1000, "deadline_ns": 2000}},
    )
    assert driver.active_deadline(tmp_path, worker) == 2000
    write_json(tmp_path / "numerical/ledger.json", {"active_case": None})
    assert driver.active_deadline(tmp_path, worker) == 3000
    write_json(folder / "result.json", {**marker, "status": "complete"})
    assert driver.active_deadline(tmp_path, worker) is None


@pytest.mark.parametrize("cause", ["case_deadline", "artifact_cap", "sigterm"])
def test_actual_launcher_terminates_only_owned_child_on_lifecycle_stop(
    persistent_plan, tmp_path, monkeypatch, cause
):
    events = []

    class Child:
        pid = 424242
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if self.stopped:
                return -signal.SIGTERM
            raise ab.subprocess.TimeoutExpired("fixture child", timeout)

    child = Child()

    def start(command, **kwargs):
        assert command[2] == "vllm_lt.validation.m3_persistent_run"
        assert kwargs["cwd"] == persistent_plan["implementations"]["A"]["root"]
        assert kwargs["env"]["PYTHONPATH"] == kwargs["cwd"]
        assert kwargs["start_new_session"] is True
        return child

    def kill(pid, signum):
        assert pid == child.pid and signum == signal.SIGTERM
        events.append(("kill", pid))
        child.stopped = True

    monkeypatch.setattr(ab.subprocess, "Popen", start)
    monkeypatch.setattr(ab.os, "killpg", kill)
    monkeypatch.setattr(ab.time, "perf_counter_ns", lambda: 1000)
    (tmp_path / "workers").mkdir()
    if cause == "artifact_cap":
        (tmp_path / "forbidden").mkdir()
        (tmp_path / "forbidden/trace.json").write_bytes(b"unexpected profile")
    folder = tmp_path / "lifecycle/evaluations/L-A-torch"
    folder.mkdir(parents=True)
    write_json(
        folder / "started.json",
        {
            "evaluation": {"evaluation_id": "L-A-torch"},
            "started_ns": 1,
            "deadline_ns": 999 if cause == "case_deadline" else 10**9,
        },
    )

    def watchdog(output, worker):
        if cause == "sigterm":
            raise KeyboardInterrupt("owned controller received SIGTERM")
        return driver.active_deadline(output, worker)

    with pytest.raises((TimeoutError, ValueError, KeyboardInterrupt)):
        ab._launch_worker(
            persistent_plan,
            persistent_plan["workers"][0],
            tmp_path,
            10**30,
            module="vllm_lt.validation.m3_persistent_run",
            active_deadline=watchdog,
        )
    assert child.stopped and [event for event in events if event[0] == "kill"] == [
        ("kill", child.pid)
    ]


def test_controller_sigterm_records_incomplete_and_restores_handler(controller, monkeypatch):
    original = signal.getsignal(signal.SIGTERM)

    def interrupted(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(ab, "_launch_worker", interrupted)
    result = driver.run(controller.plan, output_dir=controller.output)
    assert result["status"] == "incomplete" and result["completed_executions"] == []
    assert len(result["workers"]) == 1 and result["workers"][0]["returned_ns"] is not None
    assert signal.getsignal(signal.SIGTERM) == original


@pytest.mark.parametrize("failure", ["numerical", "lifecycle"])
def test_shared_worker_keeps_failed_feasibility_prefix_and_releases_model(
    persistent_plan, tmp_path, monkeypatch, failure
):
    import weakref

    events, references = [], []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(common, "affinity_snapshot", lambda: fixtures.fixtures.AFFINITY)
    monkeypatch.setattr(common.runner, "configure_process", lambda c: None)
    monkeypatch.setattr(common.runner, "environment", lambda: environment(persistent_plan))

    class Model:
        pass

    def load(view, manifest):
        value = Model()
        references.append(weakref.ref(value))
        events.append("load")
        return value

    def release(manifest):
        assert references[0]() is None, "worker retained its model into final device release"
        events.append("release")
        manifest["teardown_after_workspace_release"] = {"allocated_bytes": 0, "reserved_bytes": 0}

    def numerical(model, plan, implementation, output, deadline, *, after_case):
        case = next(
            row
            for row in plan["numerical"]["execution_order"]
            if row["implementation_id"] == implementation
        )
        events.append("feasibility")
        if failure == "lifecycle":
            after_case(case, {"status": "complete"})
            pytest.fail("a failed first lifecycle evaluation allowed more model cases")
        return {
            "complete": False,
            "passed": False,
            "completed_cases": [case["case_id"]],
            "errors": [{"message": "required feasibility comparison failed"}],
        }

    def checks(plan, implementation, output, deadline):
        events.append("lifecycle")
        assert failure == "lifecycle", "failed numerical feasibility must exclude lifecycle work"
        first = next(
            row
            for row in plan["lifecycle"]["execution_order"]
            if row["implementation_id"] == implementation
        )
        yield first["evaluation_id"], {"status": "complete", "passed": False}
        pytest.fail("failed held-input gate allowed the next evaluation")

    monkeypatch.setattr(common.runner, "load_model", load)
    monkeypatch.setattr(common.runner, "release_device", release)
    result = common._run_worker(
        persistent_plan,
        worker_id="N-A",
        output_dir=tmp_path,
        deadline_ns=10**30,
        validate=lambda p: None,
        verify=lambda *a, **k: None,
        model_rows=numerical,
        checks=checks,
        artifact_type="m3_persistent_worker_manifest",
    )
    assert result["status"] == "failed" and result["passed"] is False
    assert result["completed_executions"] == persistent_plan["workers"][0]["execution_ids"][:1]
    assert result["model_loads"] == 1 and result["failures"]
    assert events == ["load", "feasibility"] + (["lifecycle"] if failure == "lifecycle" else []) + [
        "release"
    ]


def test_controller_refuses_b_when_a_has_nonzero_final_memory(controller, monkeypatch):
    original = ab._launch_worker

    def leaky(plan, worker, output, deadline, **kwargs):
        result = original(plan, worker, output, deadline, **kwargs)
        path = output / "workers" / worker["worker_id"] / "manifest.json"
        value = read_json(path)
        value["teardown_after_workspace_release"]["allocated_bytes"] = 1
        write_json(path, value)
        return result

    monkeypatch.setattr(ab, "_launch_worker", leaky)
    result = driver.run(controller.plan, output_dir=controller.output)
    assert result["status"] == "failed" and controller.state["launches"] == ["N-A"]
    assert "cleanup" in result["failures"][0]["message"]
    assert len(result["completed_executions"]) == 11 and result["completed_workers"] == []
