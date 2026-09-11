"""M4 subset identities and real CPU orchestration, without device qualification."""

import math
import time
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import m4_attention as m4
from vllm_lt.validation.schema import _digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (
        read_json(ROOT / "benchmarks/fixtures/ouro-q1.json"),
        read_json(ROOT / "benchmarks/fixtures/ouro-q1-contract.json"),
    )


def test_frozen_tile_subset_retains_oracle_policy_and_separates_exactness():
    suite, contract = inputs()
    plan = m4.build_numerical_plan(suite, contract, OuroConfig().to_dict())
    m4.validate_numerical_plan(plan)
    assert len(plan["execution_order"]) == 27
    comparisons = plan["comparison_order"]
    assert len(comparisons) == 51
    assert sum(row["require_exact"] for row in comparisons) == 4
    assert sum(row["comparison_kind"] == "implementation_fidelity" for row in comparisons) == 13
    assert plan["contract"]["comparison_policy"] == contract["comparison_policy"]
    assert all(row["case_id"].startswith("m4-attention-") for row in plan["execution_order"])
    assert plan["resource_estimates"]["retained_tensor_bytes_upper_bound"] == 4860873504
    changed = deepcopy(plan)
    next(row for row in changed["comparison_order"] if row["require_exact"])["require_exact"] = (
        False
    )
    changed["numerical_plan_sha256"] = _digest(
        {k: v for k, v in changed.items() if k != "numerical_plan_sha256"}
    )
    with pytest.raises(ValueError, match="exact frozen"):
        m4.validate_numerical_plan(changed)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    from vllm_lt.core import kv_cache_manager

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        initialize = kv_cache_manager.KVCacheManager.__init__

        def cpu(self, *args, **kwargs):
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)

        patch.setattr(kv_cache_manager.KVCacheManager, "__init__", cpu)
        config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
        torch.manual_seed(41)
        model = OuroForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        parent = {
            "plan_sha256": "a" * 64,
            "numerical": m4.build_numerical_plan(*inputs(), config.to_dict()),
        }
        root = tmp_path_factory.mktemp("m4-numerical")
        observed = []

        def after_case(case, value):
            assert m4.active_case_deadline(root, case["implementation_id"]) is not None
            observed.append(case["case_id"])

        for source in ("A", "B"):
            result = m4.run_numerical_rows(
                model,
                parent,
                source,
                root,
                time.perf_counter_ns() + 180 * 10**9,
                after_case=after_case,
            )
            assert result["passed"], result["errors"]
        assert len(observed) == 27
    yield root, parent
    torch.set_num_threads(previous)


def test_real_cpu_case_ledger_and_full_callback_lifetimes(completed):
    root, parent = completed
    report = m4.audit_numerical(root, parent)
    assert report["complete"] and report["passed"], report["errors"]
    assert len(report["case_completed_ns"]) == 27
    assert m4.active_case_deadline(root, "A") is None
    assert m4.active_case_deadline(root, "B") is None
    case = parent["numerical"]["execution_order"][0]["case_id"]
    path = root / "numerical/cases" / case / "m4-completed.json"
    original = path.read_bytes()
    try:
        marker = read_json(path)
        marker["case_completed_ns"] = marker["deadline_ns"] + 1
        write_json(path, marker)
        report = m4.audit_numerical(root, parent)
        assert not report["passed"] and "full-case completion" in report["errors"][0]["message"]
    finally:
        path.write_bytes(original)


def test_callback_failure_preserves_active_watchdog(tmp_path, monkeypatch):
    from vllm_lt.validation import runner

    config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
    model = OuroForCausalLM(config)
    parent = {
        "plan_sha256": "a" * 64,
        "numerical": m4.build_numerical_plan(*inputs(), config.to_dict()),
    }

    def fake(*args):
        return {"status": "complete", "comparisons": [], "elapsed_s": 0.001}

    monkeypatch.setattr(runner, "execute_case", fake)

    def after(*args):
        raise RuntimeError("cap check failed")

    result = m4.run_numerical_rows(
        model, parent, "A", tmp_path, time.perf_counter_ns() + 600 * 10**9, after_case=after
    )
    assert not result["passed"]
    assert len(result["completed_cases"]) == 1
    assert m4.active_case_deadline(tmp_path, "A") is not None


def test_full_ledger_corruption_is_not_reclassified_as_missing_suffix(completed):
    root, parent = completed
    path = root / "numerical/ledger.json"
    original = path.read_bytes()
    try:
        ledger = read_json(path)
        ledger["workers"]["B"]["passed"] = False
        write_json(path, ledger)
        report = m4.audit_numerical(root, parent)
        assert not report["complete"] and not report["passed"] and report["errors"]
        assert "validated_prefix" not in report
        assert "full_plan_errors" not in report
    finally:
        path.write_bytes(original)


def test_actual_stopped_numerical_failure_is_audited_not_hidden(tmp_path, monkeypatch):
    from vllm_lt.core.kv_cache_manager import KVCacheManager
    from vllm_lt.validation import runner

    initialize = KVCacheManager.__init__

    def cpu(self, *args, **kwargs):
        kwargs["backend"] = "torch"
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(KVCacheManager, "__init__", cpu)
    config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
    model = OuroForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
    parent = {
        "plan_sha256": "a" * 64,
        "numerical": m4.build_numerical_plan(*inputs(), config.to_dict()),
    }
    original = runner.execute_case

    def divergent(model, view, case, *args):
        with torch.no_grad():
            if case["implementation"] == "native":
                model.lm_head.weight[1, 0] = 10
        return original(model, view, case, *args)

    monkeypatch.setattr(runner, "execute_case", divergent)
    result = m4.run_numerical_rows(
        model, parent, "A", tmp_path, time.perf_counter_ns() + 180 * 10**9
    )
    assert not result["passed"] and len(result["completed_cases"]) == 6
    report = m4.audit_numerical(tmp_path, parent)
    assert not report["complete"] and not report["passed"]
    assert report["known_required_failure"], report
    assert report["errors"] == [] and report["full_plan_errors"]
    assert len(report["validated_prefix"]["completed_cases"]) == 6
    assert len(report["case_completed_ns"]) == 5
    path = next((tmp_path / "numerical/comparisons").glob("*.jsonl"))
    raw = path.read_bytes()
    path.write_bytes(raw.split(b"\n", 1)[1])
    corrupted = m4.audit_numerical(tmp_path, parent)
    assert not corrupted["known_required_failure"] and corrupted["prefix_errors"]
    assert corrupted["errors"] == corrupted["prefix_errors"]
