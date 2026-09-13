"""The benchmark adapter preserves setup failures and confirmed resource ownership."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import test_recurrent_graph as fixtures
import torch
from test_recurrent_graph import fake_runtime, make_cache

from benchmarks.capture import runtime
from vllm_lt.benchmarks import runner
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.worker.model_runner import ModelRunner
from vllm_lt.worker.recurrent_graph import GraphLimits

model = fixtures.model
pytestmark = pytest.mark.usefixtures("forbid_cuda")


def test_feasibility_checks_replay_return_without_module_hooks():
    class Model:
        def recurrent(self, value):
            return value, value

        def coda(self, value):
            return value

    class Runner:
        def _recurrent(self, value):
            return value, value

    engine = SimpleNamespace(model=Model(), model_runner=Runner())
    with runtime.finite_checks(engine, True) as counts:
        engine.model.recurrent(torch.ones(2))
        engine.model_runner._recurrent(torch.ones(2))
        engine.model.coda(torch.ones(2))
        with pytest.raises(ValueError, match="nonfinite recurrent"):
            engine.model_runner._recurrent(torch.tensor([float("nan")]))
    assert counts == {"recurrent": 2, "coda": 1}
    assert vars(engine.model) == vars(engine.model_runner) == {}
    with runtime.finite_checks(engine, False) as counts:
        assert vars(engine.model) == vars(engine.model_runner) == {}
        engine.model_runner._recurrent(torch.tensor([float("nan")]))
    assert counts == {"recurrent": 0, "coda": 0}


@pytest.mark.parametrize("failure", ["capture", "export", "unconfirmed", "budget"])
def test_setup_failure_preserves_primary_and_resource_ownership(
    model, tmp_path, monkeypatch, failure
):
    cache = make_cache(model)
    device = fake_runtime(monkeypatch, cache)
    model_runner = ModelRunner(model, cache)
    engine = SimpleNamespace(model_runner=model_runner, scheduler=SimpleNamespace(requests={}))
    primary = RuntimeError("capture failed")

    def fail():
        if failure == "unconfirmed":
            device.main.failure = RuntimeError("completion uncertain")
        raise primary

    def disk_full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(device, "new_graph", fail)
    if failure == "export":
        monkeypatch.setattr(runtime, "write_json", disk_full)
    limits = asdict(GraphLimits())
    if failure == "budget":
        limits["common_payload_bytes"] = 1
    adapter = runtime.ExecutionAdapter(implementation_id="B", graph_limits=limits)
    with pytest.raises(ValueError if failure == "budget" else RuntimeError) as caught:
        adapter.prepare(engine, {"implementation_id": "B", "use_graphs": True}, tmp_path)
    adapter.abort(caught.value, tmp_path)
    if failure == "budget":
        assert "declined its budget" in str(caught.value)
    else:
        assert caught.value is primary
    assert (adapter.engine is engine) is (failure == "unconfirmed")
    if failure == "export":
        assert all("disk full" in note for note in caught.value.__notes__)
    else:
        assert (tmp_path / "graph-setup-failure.json").is_file()
    device.main.failure = None
    model_runner._close_recurrent_graph()
    assert all(ref() is None for ref in device.pool_refs)


def test_shared_loop_calls_adapter_abort_for_setup_systemexit(tmp_path, monkeypatch):
    events = []

    def execute(*args, **kwargs):
        raise SystemExit("setup interrupted")

    monkeypatch.setattr(runner, "_execute", execute)
    adapter = SimpleNamespace(abort=lambda error, folder: events.append((error, folder)))
    row = {"run_id": "test", "workload_id": "W1", "phase": "feasibility"}
    plan = {"suite": {"workloads": [{"workload_id": "W1"}]}, "plan_sha256": "test"}
    with pytest.raises(SystemExit, match="setup interrupted"):
        runner.run_loaded_rows(
            None,
            plan,
            [row],
            output_dir=tmp_path,
            deadline=10**30,
            record_completed=lambda *args: pytest.fail("failed case marked complete"),
            execution_adapter=adapter,
        )
    assert isinstance(events[0][0], SystemExit)
    assert read_json(tmp_path / "runs/test/result.json")["status"] == "failed"


def test_graph_final_export_deadline_prevents_completion(tmp_path, monkeypatch):
    clock = [10]
    run_dir = tmp_path / "runs/test"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "started.json", {"started_ns": 1, "deadline_ns": 100})
    result = {"status": "complete", "memory": {}, "failures": [], "comparison_eligible": True}
    monkeypatch.setattr(runner, "_execute", lambda *args, **kwargs: result)
    monkeypatch.setattr(runner.gc, "collect", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "memory", lambda: {"allocated_bytes": 0, "reserved_bytes": 0})
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: clock[0])
    exports = []

    def export(path, value):
        write_json(path, value)
        exports.append(path.name)
        if exports.count("result.pending.json") == 2:
            clock[0] = 101

    monkeypatch.setattr(runner, "write_json", export)
    with pytest.raises(TimeoutError, match="final result export"):
        runner.run_loaded_rows(
            None,
            {"suite": {"workloads": [{"workload_id": "W1"}]}},
            [{"run_id": "test", "workload_id": "W1"}],
            output_dir=tmp_path,
            deadline=1000,
            record_completed=lambda *args: pytest.fail("late export marked complete"),
            execution_adapter=object(),
        )
    assert read_json(run_dir / "result.json")["status"] == "incomplete"
