"""Frozen held inputs, genuine storage rows and corrupted evidence on CPU only."""

import time
from copy import deepcopy

import pytest
import torch

from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.validation import m4_attention_held as held
from vllm_lt.validation.schema import read_json


@pytest.fixture(scope="module")
def plan():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield held.build_held_plan()
    torch.set_num_threads(previous)


def test_frozen_layouts_and_true_lifecycle_subset(plan):
    held.validate_held_plan(deepcopy(plan))
    assert len(plan["execution_order"]) == 34
    assert len(plan["layouts"]) == 15
    assert plan["layouts"][13]["lengths"] == [63, 64, 65, 129]
    assert plan["layouts"][14]["lengths"] == list(range(1, 129))
    assert plan["resource_estimates"]["artifact_bytes_upper_bound"] == 312 * 1024**2
    assert plan["resource_estimates"]["attention_device_calls"] == 52
    assert plan["lifecycle_subset"] == ["L-A-triton", "L-B-triton"]
    assert plan["lifecycle_plan"]["artifact_type"] == "m3_persistent_lifecycle_plan"
    assert len(plan["lifecycle_plan"]["execution_order"]) == 4
    for layout in plan["layouts"]:
        assert len(plan["cache_identities"][layout["layout_id"]]) == 40
    altered = deepcopy(plan)
    altered["layouts"][13]["lengths"][0] = 62
    with pytest.raises(ValueError, match="frozen matrix"):
        held.validate_held_plan(altered)


def test_dense_prefix_equations_match_fragmented_physical_history(plan):
    layout = plan["layouts"][12]
    tensors = held._fixture(plan, layout)
    for logical, row in enumerate(layout["live_rows"]):
        for component in ("key", "value"):
            history = held._history(plan, layout, tensors, logical, component)
            for position in (0, layout["lengths"][logical] - 1):
                block = int(tensors["block_tables"][row, position // 16])
                chunk = held._chunk(plan, layout, component, block // 8 * 8)
                assert torch.equal(history[position], chunk[block % 8, 1, position % 16].double())
    layout = plan["layouts"][14]
    tensors = held._fixture(plan, layout)
    assert torch.equal(tensors["block_tables"][0], tensors["block_tables"][-1])
    assert tensors["block_tables"].max() == 15


@pytest.fixture(scope="module")
def direct(tmp_path_factory, plan):
    root = tmp_path_factory.mktemp("m4-direct-cpu")
    values = {}
    for row in plan["execution_order"]:
        if row["kind"] == "kernel" and row["implementation_id"] == "A":
            result = held._kernel_evaluation(
                plan, row, root, device="cpu", attention=torch_paged_attention
            )
            assert result["passed"], result["errors"]
            values[row["execution_id"]] = result
    return root, values


def test_all_fifteen_cpu_layouts_dense_padding_and_guard_evidence(plan, direct):
    root, values = direct
    assert len(values) == 15
    for row in plan["execution_order"]:
        if row["execution_id"] in values:
            audit = held._audit_kernel(root, plan, row, values[row["execution_id"]])
            assert audit["passed"] and audit["guard_chunks"] == 80
            assert audit["tensor_records"] == 2


def test_corrupt_guard_and_omitted_output_cannot_pass(plan, direct):
    root, values = direct
    row = plan["execution_order"][12]
    value = deepcopy(values[row["execution_id"]])
    value["cache_after"]["key:0"] = "0" * 64
    with pytest.raises(ValueError, match="statistics differ"):
        held._audit_kernel(root, plan, row, value)
    raw = root / "outputs" / row["execution_id"] / "attention.bin"
    original = raw.read_bytes()
    try:
        raw.write_bytes(original[:-4])
        with pytest.raises(ValueError):
            held._audit_kernel(root, plan, row, values[row["execution_id"]])
    finally:
        raw.write_bytes(original)


def test_finite_wrong_attention_is_preserved_as_required_failure(plan, tmp_path):
    row = plan["execution_order"][2]

    def wrong(*args):
        return torch_paged_attention(*args) + 0.01

    result = held._kernel_evaluation(plan, row, tmp_path, device="cpu", attention=wrong)
    assert not result["passed"]
    audit = held._audit_kernel(tmp_path, plan, row, result)
    assert not audit["passed"] and audit["errors"]


def test_outer_wrapper_rejects_cpu_without_claiming_cuda_or_clean_execution(plan, tmp_path):
    model = torch.nn.Linear(1, 1)
    parent = {"plan_sha256": "a" * 64, "held": plan}
    row = plan["execution_order"][0]
    result = held.run_held_row(model, parent, row, tmp_path, time.perf_counter_ns() + 600 * 10**9)
    assert not result["passed"] and result["status"] == "failed"
    assert "reserved CUDA" in result["errors"][0]["message"]
    report = held.audit_held(tmp_path, parent, expected_device="cpu")
    assert not report["passed"] and report["errors"]
    assert (
        read_json(tmp_path / "evaluations" / row["execution_id"] / "completed.json")[
            "case_completed_ns"
        ]
        >= result["ended_ns"]
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory, plan):
    """Actual held orchestration with explicitly substituted CPU arithmetic only."""
    from types import SimpleNamespace

    from vllm_lt.core.kv_cache_manager import KVCacheManager

    root = tmp_path_factory.mktemp("m4-all-held-cpu")
    parent = {"plan_sha256": "a" * 64, "held": plan}
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))])
    )
    with pytest.MonkeyPatch.context() as patch:
        initialize = KVCacheManager.__init__

        def cpu(self, *args, **kwargs):
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)

        patch.setattr(KVCacheManager, "__init__", cpu)
        original_kernel = held._kernel_evaluation

        def kernel(plan, row, root, **kwargs):
            return original_kernel(plan, row, root, device="cpu", attention=torch_paged_attention)

        patch.setattr(held, "_kernel_evaluation", kernel)
        original_life = held.lifecycle.run_lifecycle_evaluation

        def life(plan, identity, root, **kwargs):
            return original_life(
                plan, identity, root, device="cpu", deadline_ns=kwargs["deadline_ns"]
            )

        patch.setattr(held.lifecycle, "run_lifecycle_evaluation", life)
        for row in plan["execution_order"]:
            result = held.run_held_row(
                model,
                parent,
                {**row, "worker_id": "N-" + row["implementation_id"]},
                root,
                time.perf_counter_ns() + 600 * 10**9,
            )
            assert result["passed"], result
    return root, parent


def test_full_outer_tile_and_inner_storage_cpu_audit(completed):
    root, parent = completed
    report = held.audit_held(root, parent, expected_device="cpu")
    assert report["complete"] and report["passed"], report["errors"]
    assert len(report["evaluations"]) == 34
    assert sum(row["guard_chunks"] for row in report["evaluations"]) == 4320
    assert sum(row["tensor_records"] for row in report["evaluations"]) == 752
    assert set(report["cross_tile_lifecycle"]) == {"allocating", "persistent"}
    assert held.active_case_deadline(root, "B") is None
    assert not held.audit_held(root, parent)["passed"]  # CPU substitution cannot qualify CUDA.


def test_held_audit_rejects_unplanned_outputs_and_lifetime(completed):
    root, parent = completed
    extra = root / "outputs/unplanned.bin"
    try:
        extra.write_bytes(b"extra")
        report = held.audit_held(root, parent, expected_device="cpu")
        assert not report["passed"] and "inventory" in report["errors"][0]["message"]
    finally:
        extra.unlink()
    path = root / "evaluations/K-A-L00/completed.json"
    original = path.read_bytes()
    try:
        marker = read_json(path)
        marker["case_completed_ns"] = marker["deadline_ns"]
        held._write(path, marker)
        report = held.audit_held(root, parent, expected_device="cpu")
        assert not report["passed"] and "lifetime" in report["errors"][0]["message"]
    finally:
        path.write_bytes(original)


def test_settled_numerical_failure_retained_even_when_suffix_missing(plan, tmp_path, monkeypatch):
    from types import SimpleNamespace

    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))])
    )
    parent = {"plan_sha256": "a" * 64, "held": plan}
    original = held._kernel_evaluation

    def kernel(plan, row, root, **kwargs):
        def wrong(*args):
            result = torch_paged_attention(*args)
            return result + 0.01 if row["layout_id"] == "L02" else result

        return original(plan, row, root, device="cpu", attention=wrong)

    monkeypatch.setattr(held, "_kernel_evaluation", kernel)
    for row in plan["execution_order"][:3]:
        held.run_held_row(model, parent, row, tmp_path, time.perf_counter_ns() + 600 * 10**9)
    report = held.audit_held(tmp_path, parent, expected_device="cpu")
    assert not report["complete"] and not report["passed"]
    assert report["known_required_failure"]
    assert len(report["evaluations"]) == 3 and len(report["missing"]) == 31


def test_direct_export_failure_still_settles_and_releases_cache(plan, tmp_path, monkeypatch):
    original = held.TensorSpool.close

    def failure(self):
        original(self)
        raise OSError("deliberate index close failure")

    monkeypatch.setattr(held.TensorSpool, "close", failure)
    result = held._kernel_evaluation(
        plan, plan["execution_order"][2], tmp_path, device="cpu", attention=torch_paged_attention
    )
    assert not result["passed"]
    assert result["cleanup"] == {"completion_confirmed": True, "cache_references_released": True}
    assert "index close failure" in result["errors"][0]["message"]


def test_full_case_deadline_includes_result_export(plan, tmp_path, monkeypatch):
    from types import SimpleNamespace

    now = [1_000_000_000]
    monkeypatch.setattr(held.time, "perf_counter_ns", lambda: now[0])
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))])
    )
    monkeypatch.setattr(held, "_kernel_evaluation", lambda *a, **k: {"passed": True})
    original = held._write

    def slow(path, value):
        original(path, value)
        if path.name == "result.json":
            now[0] = 601_000_000_001

    monkeypatch.setattr(held, "_write", slow)
    row = plan["execution_order"][0]
    result = held.run_held_row(
        model, {"plan_sha256": "a" * 64, "held": plan}, row, tmp_path, 999_000_000_000
    )
    assert not result["passed"] and result["status"] == "failed"
    assert "deadline" in result["errors"][0]["message"]
    assert not (tmp_path / "evaluations" / row["execution_id"] / "completed.json").exists()
    assert held.active_case_deadline(tmp_path, "A") == 601_000_000_000
