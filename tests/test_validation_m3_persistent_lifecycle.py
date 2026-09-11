"""Finite held-boundary CPU execution and negative evidence audits, never GPU proof."""

import shutil
from copy import deepcopy

import pytest
import torch

from vllm_lt.validation import m3_persistent_lifecycle as lifecycle


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("lifecycle CPU tests attempted CUDA discovery or execution")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """Triton-named rows substitute Torch only to exercise four-row artifact orchestration.

    This fixture is CPU-only, and the default CUDA qualification audit must reject it.
    The actual GPU Triton arithmetic is left to the separately frozen experiment.
    """
    from vllm_lt.core.kv_cache_manager import KVCacheManager

    root = tmp_path_factory.mktemp("lifecycle-complete")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU fixture attempted CUDA")

        for name in (
            "is_available",
            "device_count",
            "current_device",
            "init",
            "_lazy_init",
            "synchronize",
        ):
            patch.setattr(torch.cuda, name, forbidden)
        initialize = KVCacheManager.__init__

        def cpu(self, *args, **kwargs):
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)

        patch.setattr(KVCacheManager, "__init__", cpu)
        plan = lifecycle.build_lifecycle_plan()
        for row in plan["execution_order"]:
            result = lifecycle.run_lifecycle_evaluation(plan, row["evaluation_id"], root, "cpu")
            assert result["passed"], result["errors"]
    yield root, plan
    torch.set_num_threads(previous_threads)


def test_plan_freezes_histories_fragmentation_capacity_and_finite_coverage():
    plan = lifecycle.build_lifecycle_plan()
    lifecycle.validate_lifecycle_plan(plan)
    assert [row["evaluation_id"] for row in plan["execution_order"]] == [
        "L-A-torch",
        "L-A-triton",
        "L-B-torch",
        "L-B-triton",
    ]
    steps = plan["steps"]
    assert [s["step_id"] for s in steps if s["kind"] == "supported"] == [1, 2, 3, 5, 7, 9, 11]
    assert [(s["step_id"], s["fallback_reason"]) for s in steps if s["kind"] == "fallback"] == [
        (8, "table_width"),
        (10, "live_count"),
    ]
    assert [s["used_blocks"] for s in steps] == [156] * 8 + [32] * 3
    table = plan["initial_allocations"]["long"]["block_tables"][0]
    assert table[:8] == [7, 6, 5, 4, 12, 13, 14, 15]
    assert [steps[i]["positions"][0] for i in (0, 1, 7)] == [510, 511, 512]
    for step in steps:
        pages = [
            page
            for row in step["allocations"].values()
            for depth in row["block_tables"]
            for page in depth
        ]
        assert len(pages) == len(set(pages)) == step["used_blocks"]
        assert len(step["guard_chunks"]) == 40
    old, new = steps[4]["allocations"]["r1"], steps[5]["allocations"]["r1"]
    assert old["allocation_id"] != new["allocation_id"]
    assert sorted(sum(old["block_tables"], [])) == sorted(sum(new["block_tables"], []))
    assert plan["resource_estimates"]["cache_bytes_per_evaluation"] == 80 * 1024**2
    assert plan["resource_estimates"]["artifact_bytes_upper_bound"] == 48 * 1024**2
    changed = deepcopy(plan)
    changed["steps"][7]["positions"] = [511]
    changed["lifecycle_plan_sha256"] = lifecycle._digest(
        {k: v for k, v in changed.items() if k != "lifecycle_plan_sha256"}
    )
    with pytest.raises(ValueError, match="frozen exact sequence"):
        lifecycle.validate_lifecycle_plan(changed)
    plan["steps"].clear()
    assert len(lifecycle.build_lifecycle_plan()["steps"]) == 11


def test_complete_cpu_pair_proves_real_cache_publication_and_reuse(completed):
    root, plan = completed
    report = lifecycle.audit_lifecycle_outputs(root, plan, expected_device="cpu")
    assert report["complete"] and report["passed"], report["errors"]
    assert len(report["completed_evaluations"]) == 4
    assert sum(row["guard_chunks"] for row in report["evaluations"]) == 1920
    assert sum(row["tensor_records"] for row in report["evaluations"]) == 692
    assert report["artifact_bytes"] < 48 * 1024**2
    b = lifecycle._read(root / "evaluations/L-B-torch/result.json")
    assert b["persistent_final"]["generation"] == 7
    assert b["persistent_final"]["fallback_counts"] == {"live_count": 1, "table_width": 1}
    assert len(b["held_checks"]) == 10
    assert "core" not in b["steps"][3]
    assert b["steps"][3]["persistent"]["last_dispatch"]["row_count"] == 8
    assert b["steps"][5]["stale_allocation_rejected"] and b["steps"][5]["stale_generation_rejected"]
    assert b["cleanup"] == {
        "completion_confirmed": True,
        "used_blocks": 0,
        "active_requests": 0,
        "cache_references_released": True,
        "buffer_references_released": True,
    }


def test_cpu_evidence_cannot_pass_default_cuda_qualification(completed):
    root, plan = completed
    report = lifecycle.audit_lifecycle_outputs(root, plan)
    assert not report["passed"] and not report["complete"]
    assert report["completed_evaluations"] == []
    assert all("device" in error for error in report["errors"])


@pytest.mark.parametrize(
    "change",
    [
        "guard",
        "pointer",
        "core_alias",
        "gate_alias",
        "held",
        "generation",
        "fallback",
        "buffer",
        "cleanup",
        "step",
        "actual_gate",
        "in_flight",
    ],
)
def test_rehashed_json_omissions_or_false_lifecycle_claims_are_rejected(
    completed, tmp_path, change
):
    root, plan = completed
    destination = tmp_path / "altered"
    shutil.copytree(root, destination)
    path = destination / "evaluations/L-B-torch/result.json"
    result = lifecycle._read(path)
    if change == "guard":
        result["steps"][0]["guard_chunks"][0]["actual_sha256"] = "0" * 64
    elif change == "pointer":
        del result["steps"][0]["persistent"]["tensors"]["hidden_in"]["data_ptr"]
    elif change == "core_alias":
        result["steps"][0]["core"]["metadata"]["active"]["data_ptr"] += 1
    elif change == "gate_alias":
        result["steps"][0]["publication"]["gates"]["storage_ptr"] = result["persistent_initial"][
            "tensors"
        ]["gate_out"]["storage_ptr"]
    elif change == "held":
        result["held_checks"][-1]["state"]["generator_sha256"] = "0" * 64
    elif change == "generation":
        result["steps"][4]["persistent"]["generation"] += 1
    elif change == "fallback":
        result["steps"][7]["persistent"]["fallback_counts"]["table_width"] = 0
    elif change == "buffer":
        result["steps"][0]["buffer_hashes"]["hidden_out"] = "0" * 64
    elif change == "cleanup":
        result["cleanup"]["cache_references_released"] = False
    elif change == "step":
        result["steps"].pop(3)
    elif change == "actual_gate":
        result["steps"][0]["gate_probabilities"][0] += 0.01
    else:
        result["steps"][0]["in_flight"]["last_publication"] = result["steps"][0]["persistent"][
            "last_publication"
        ]
    lifecycle._write(path, result)
    report = lifecycle.audit_lifecycle_outputs(destination, plan, expected_device="cpu")
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize(
    "change",
    ["missing_raw", "extra_file", "extra_evaluation", "raw_byte", "raw_metadata", "symlink"],
)
def test_artifact_coverage_and_payload_are_verified(completed, tmp_path, change):
    root, plan = completed
    destination = tmp_path / "altered"
    shutil.copytree(root, destination)
    folder = destination / "outputs/L-A-torch"
    if change == "missing_raw":
        (folder / "evidence.bin").unlink()
    elif change == "extra_file":
        (folder / "unplanned.txt").write_text("extra")
    elif change == "extra_evaluation":
        (destination / "evaluations/L-C-torch").mkdir()
    elif change == "raw_byte":
        with (folder / "evidence.bin").open("r+b") as stream:
            stream.write(b"\xff")
    elif change == "raw_metadata":
        path = folder / "evidence.index.json"
        index = lifecycle._read(path)
        record = index["records"][0]
        record["metadata"]["evaluation_id"] = "L-B-torch"
        record["record_sha256"] = lifecycle._digest(
            {k: v for k, v in record.items() if k != "record_sha256"}
        )
        lifecycle._write(path, index)
    else:
        path = folder / "evidence.bin"
        path.unlink()
        path.symlink_to(root / "outputs/L-A-torch/evidence.bin")
    report = lifecycle.audit_lifecycle_outputs(destination, plan, expected_device="cpu")
    assert not report["passed"] and report["errors"]


def test_a_does_not_call_persistent_only_apis(tmp_path, monkeypatch):
    from vllm_lt.core.kv_cache_manager import KVCacheManager
    from vllm_lt.worker.model_runner import ModelRunner

    def forbidden(*args, **kwargs):
        raise AssertionError("allocating A used a B-only API")

    for name in ("_enable_persistent_decode", "_persistent_snapshot", "_recurrent_persistent"):
        monkeypatch.setattr(ModelRunner, name, forbidden)
    for name in ("_allocate_metadata_storage", "_prepare_into", "_release_prepared"):
        monkeypatch.setattr(KVCacheManager, name, forbidden)
    result = lifecycle.run_lifecycle_evaluation(
        lifecycle.build_lifecycle_plan(), "L-A-torch", tmp_path, "cpu"
    )
    assert result["passed"], result["errors"]


def test_partial_evidence_and_primary_error_survive_spool_close_failure(tmp_path, monkeypatch):
    from vllm_lt.validation.diagnostics import TensorSpool

    write, close = TensorSpool.write, TensorSpool.close

    def failing_write(self, key, *args, **kwargs):
        if key == "step01/layer0/key":
            raise ValueError("injected raw observation failure")
        return write(self, key, *args, **kwargs)

    def failing_close(self):
        close(self)
        raise OSError("injected close failure")

    monkeypatch.setattr(TensorSpool, "write", failing_write)
    monkeypatch.setattr(TensorSpool, "close", failing_close)
    result = lifecycle.run_lifecycle_evaluation(
        lifecycle.build_lifecycle_plan(), "L-A-torch", tmp_path, "cpu"
    )
    assert not result["passed"]
    assert [error["type"] for error in result["errors"]] == ["ValueError", "evidence_cleanup"]
    assert result["errors"][0]["message"] == "injected raw observation failure"
    assert result["cleanup"]["cache_references_released"]
    assert result["cleanup"]["used_blocks"] == 0
    assert result == lifecycle._read(tmp_path / "evaluations/L-A-torch/result.json")
    index = lifecycle._read(tmp_path / "outputs/L-A-torch/evidence.index.json")
    assert len(index["records"]) == 8
    assert result["steps"][0]["status"] == "incomplete"


def test_surviving_cache_reference_marks_failure_but_persists_result(tmp_path, monkeypatch):
    from vllm_lt.core.kv_cache_manager import KVCacheManager

    retained = []
    initialize = KVCacheManager.__init__

    def capture(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        retained.append(self.key_cache)

    monkeypatch.setattr(KVCacheManager, "__init__", capture)
    result = lifecycle.run_lifecycle_evaluation(
        lifecycle.build_lifecycle_plan(), "L-A-torch", tmp_path, "cpu"
    )
    assert result["status"] == "failed" and not result["passed"]
    assert result["errors"][-1]["type"] == "lifecycle_cleanup"
    assert not result["cleanup"]["cache_references_released"]
    assert result["cleanup"]["used_blocks"] == 0
    assert result == lifecycle._read(tmp_path / "evaluations/L-A-torch/result.json")
    retained.clear()


def test_first_bad_whole_pool_chunk_is_preserved(tmp_path, monkeypatch):
    plan = lifecycle.build_lifecycle_plan()
    original = lifecycle._base_chunk

    def wrong(component, start):
        value = original(component, start)
        if component == start == 0:
            value.flatten()[0] += 1
        return value

    monkeypatch.setattr(lifecycle, "_base_chunk", wrong)
    result = lifecycle.run_lifecycle_evaluation(plan, "L-A-torch", tmp_path, "cpu")
    assert (
        not result["passed"]
        and result["errors"][0]["message"] == "whole-pool write/guard hash differs"
    )
    assert len(result["initial_guard_chunks"]) == 1
    assert (
        result["initial_guard_chunks"][0]["sha256"]
        != result["initial_guard_chunks"][0]["actual_sha256"]
    )
    assert result["cleanup"]["cache_references_released"]


def test_expired_deadline_never_claims_success_and_disallows_retry(tmp_path):
    result = lifecycle.run_lifecycle_evaluation(
        lifecycle.build_lifecycle_plan(), "L-A-torch", tmp_path, "cpu", deadline_ns=1
    )
    assert result["status"] == "incomplete" and not result["passed"]
    assert result["steps"] == []
    with pytest.raises(ValueError, match="cannot be retried"):
        lifecycle.run_lifecycle_evaluation(
            lifecycle.build_lifecycle_plan(), "L-A-torch", tmp_path, "cpu"
        )


@pytest.mark.parametrize("delay", ["export", "cap_scan", "final_export", "publish"])
def test_slow_export_or_cap_cannot_publish_a_passing_terminal_marker(tmp_path, monkeypatch, delay):
    folder, output = tmp_path / "evaluation", tmp_path / "output"
    folder.mkdir()
    output.mkdir()
    clock = [100]
    monkeypatch.setattr(lifecycle.time, "perf_counter_ns", lambda: clock[0])
    result = {"status": "complete", "passed": True, "errors": []}
    original_write, original_usage = lifecycle._write, lifecycle._evidence_bytes
    calls = [0]

    def write(path, value):
        if path.name == "result.pending.json":
            assert not (folder / "result.json").exists()
            calls[0] += 1
            if (delay == "export" and calls[0] == 1) or (delay == "final_export" and calls[0] == 2):
                clock[0] = 701
        original_write(path, value)

    def usage(*args):
        assert not (folder / "result.json").exists()
        if delay == "cap_scan":
            clock[0] = 701
        return original_usage(*args)

    from pathlib import Path

    replace = Path.replace

    def publish(path, target):
        outcome = replace(path, target)
        if path.name == "result.pending.json" and delay == "publish":
            clock[0] = 701
        return outcome

    monkeypatch.setattr(lifecycle, "_write", write)
    monkeypatch.setattr(lifecycle, "_evidence_bytes", usage)
    monkeypatch.setattr(Path, "replace", publish)
    lifecycle._persist_result(folder, output, result, 700)
    assert result["status"] == "incomplete" and not result["passed"]
    assert result["finished_ns"] == 701
    assert result["errors"] == [{"type": "deadline", "message": "lifecycle lifetime exceeded"}]
    assert lifecycle._read(folder / "result.json") == result
    assert not (folder / "result.pending.json").exists()


def test_validation_time_is_inside_lifetime_and_does_not_start_device_work(tmp_path, monkeypatch):
    plan = lifecycle.build_lifecycle_plan()
    clock = [100]
    monkeypatch.setattr(lifecycle.time, "perf_counter_ns", lambda: clock[0])
    validate = lifecycle.validate_lifecycle_plan

    def delayed(plan):
        marker = lifecycle._read(tmp_path / "evaluations/L-A-torch/started.json")
        assert marker["started_ns"] == 100
        validate(plan)
        clock[0] = 700 * 10**9

    monkeypatch.setattr(lifecycle, "validate_lifecycle_plan", delayed)
    result = lifecycle.run_lifecycle_evaluation(plan, "L-A-torch", tmp_path, device="cuda")
    assert result["started_ns"] == 100
    assert result["status"] == "incomplete" and not result["passed"]
    assert result["initial_allocations"] is None
    assert result["errors"][0]["type"] == "TimeoutError"


def test_export_cap_failure_preserves_primary_error_and_partial_result(tmp_path, monkeypatch):
    folder, output = tmp_path / "evaluation", tmp_path / "output"
    folder.mkdir()
    output.mkdir()
    result = {
        "status": "failed",
        "passed": False,
        "errors": [{"type": "ValueError", "message": "primary execution failure"}],
    }
    monkeypatch.setattr(lifecycle, "_evidence_bytes", lambda *args: (9 * 1024**2, 0))
    lifecycle._persist_result(folder, output, result, lifecycle.time.perf_counter_ns() + 10**9)
    assert not result["passed"]
    assert [row["type"] for row in result["errors"]] == ["ValueError", "result_persistence"]
    assert result == lifecycle._read(folder / "result.json")


def test_final_export_error_preserves_primary_failure_before_terminal(tmp_path, monkeypatch):
    folder, output = tmp_path / "evaluation", tmp_path / "output"
    folder.mkdir()
    output.mkdir()
    result = {
        "status": "failed",
        "passed": False,
        "errors": [{"type": "ValueError", "message": "primary execution failure"}],
    }
    original, calls = lifecycle._write, [0]

    def failing_write(path, value):
        calls[0] += 1
        assert not (folder / "result.json").exists()
        if calls[0] == 2:
            raise OSError("final export failed once")
        original(path, value)

    monkeypatch.setattr(lifecycle, "_write", failing_write)
    lifecycle._persist_result(folder, output, result, lifecycle.time.perf_counter_ns() + 10**9)
    assert [error["type"] for error in result["errors"]] == ["ValueError", "result_persistence"]
    assert result == lifecycle._read(folder / "result.json")
