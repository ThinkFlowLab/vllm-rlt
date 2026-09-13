"""CPU numerical/projection checks; stand-ins never qualify real graph replay."""

import math
import time
from pathlib import Path

import pytest
import torch

from benchmarks.capture import validation as capture
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation.report import _expected_boundaries
from vllm_lt.validation.schema import read_json

pytestmark = pytest.mark.usefixtures("forbid_cuda")

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (
        read_json(ROOT / "benchmarks/fixtures/ouro-q1.json"),
        read_json(ROOT / "benchmarks/fixtures/ouro-q1-contract.json"),
    )


def test_shared_boundary_projection_leaves_old_coverage_unchanged_and_preserves_all_kv():
    suite, contract = inputs()
    plan = capture.build_model_plan(suite, contract, OuroConfig().to_dict())
    assert plan["contract"]["comparison_policy"] == contract["comparison_policy"]
    fixture = plan["suite"]["fixtures"][0]
    case = plan["execution_order"][1]
    traces = [
        {
            "output_index": i,
            "position": len(fixture["prompt_token_ids"]) - 1 + i,
            "exit_depth": 4,
            "history_sha256": "a" * 64,
        }
        for i in range(9)
    ]
    projected = _expected_boundaries(case, fixture, traces, plan["model_config"])
    legacy_case = {k: v for k, v in case.items() if k != "boundary_projection"}
    legacy = _expected_boundaries(legacy_case, fixture, traces, plan["model_config"])
    assert len(projected) == 93
    assert len(legacy) == 5277
    assert set(meta["operation"] for meta, _ in projected.values()) == set(capture.OPERATIONS)
    assert {k: v for k, v in legacy.items() if v[0]["operation"] in capture.OPERATIONS} == projected
    kv = [
        (meta, shape) for meta, shape in projected.values() if meta["operation"] == "populated_kv"
    ]
    for component in ("keys", "values"):
        assert [
            p for meta, _ in kv if meta["component"] == component for p in meta["positions"]
        ] == list(range(24))
    assert all(shape == [4, 24, 4, 16, 128] for _, shape in kv)
    with pytest.raises(ValueError, match="unknown numerical boundary projection"):
        _expected_boundaries(
            {**case, "boundary_projection": "unknown"}, fixture, traces, plan["model_config"]
        )


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """Exercise real numerical drivers with the shared CPU graph stand-in."""
    import os
    import shutil

    from test_recurrent_graph import FakeRuntime, fake_runtime

    from vllm_lt.core.kv_cache_manager import KVCacheManager
    from vllm_lt.worker import recurrent_graph

    with pytest.MonkeyPatch.context() as patch:
        fake_runtime(patch)
        patch.setattr(recurrent_graph, "_make_runtime", lambda device: FakeRuntime())
        initialize = KVCacheManager.__init__

        def cpu_cache(self, *args, **kwargs):
            backend = kwargs.get("backend", "torch")
            initialize(self, *args, **{**kwargs, "backend": "torch"})
            self.backend = backend

        patch.setattr(KVCacheManager, "__init__", cpu_cache)
        patch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU fixture initialized CUDA"))
        config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
        torch.manual_seed(41)
        model = OuroForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        parent = {
            "plan_sha256": "b" * 64,
            "numerical": capture.build_model_plan(*inputs(), config.to_dict()),
        }
        output = tmp_path_factory.mktemp("m3-capture-cpu")
        prefixes = {}
        for implementation in ("A", "B"):
            result = capture.run_model_rows(
                model, parent, implementation, output, time.perf_counter_ns() + 180 * 10**9
            )
            assert result["complete"] and result["passed"], result["errors"]
            if implementation == "A":
                prefixes[False] = output.parent / (output.name + "-prefix")
                shutil.copytree(output, prefixes[False], copy_function=os.link)
        prefixes[True] = tmp_path_factory.mktemp("m3-capture-failed")
        coda = model.coda
        patch.setattr(model, "coda", lambda hidden: coda(hidden) + 1)
        failed = capture.run_model_rows(
            model, parent, "A", prefixes[True], time.perf_counter_ns() + 180 * 10**9
        )
        assert not failed["passed"] and failed["errors"]
        yield output, parent, prefixes


def test_complete_projected_eager_replay_and_backend_fallback_cpu_pipeline(completed_run):
    output, parent, _ = completed_run
    result = capture.audit_model_rows(output, parent, expected_device="cpu")
    assert result["complete"] and result["passed"], result["errors"]
    assert result["counts"]["verified_cases"] == 15
    assert result["counts"]["verified_comparisons"] == 31
    assert result["counts"]["verified_qualification_comparisons"] == 30
    assert result["counts"]["verified_replay_dispatches"] > 0
    assert result["counts"]["verified_backend_fallbacks"] == 64
    # CPU stand-ins cannot satisfy the real CUDA evidence contract.
    real = capture.audit_model_rows(output, parent)
    assert not real["complete"] and not real["passed"]
    assert "tensor device differs" in real["errors"][-1]["message"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "captured_pointer",
        "pool_alias",
        "counter",
        "request_alias",
        "cleanup",
        "replay_mismatch",
    ],
)
def test_rehashed_or_missing_replay_pointer_lifetime_evidence_cannot_qualify(completed_run, change):
    output, parent, _ = completed_run
    case = next(
        c for c in parent["numerical"]["execution_order"] if c["storage_strategy"] == "graph_replay"
    )
    value = read_json(output / "numerical/cases" / case["case_id"] / "result.json")
    events = value["graph_observations"]
    if change == "missing":
        events.pop()
    elif change == "captured_pointer":
        value["graph_lifecycle"]["setup"]["buckets"]["4"]["captured_inputs"]["hidden"][
            "data_ptr"
        ] += 4
    elif change == "pool_alias":
        value["graph_lifecycle"]["setup"]["buckets"]["8"]["pool_id"] = [0, 999]
    elif change == "counter":
        events[0]["after"]["counters"]["replays"] = 0
    elif change == "request_alias":
        events[0]["request_hidden"][0]["storage_ptr"] = events[0]["after"]["buckets"]["4"][
            "tensors"
        ]["hidden_out"]["storage_ptr"]
    elif change == "replay_mismatch":
        value["graph_lifecycle"]["setup"]["buckets"]["4"]["verification"]["max_abs_diff"] = [0.1, 0]
    elif change == "cleanup":
        value["graph_lifecycle"]["after_close"]["buckets"] = value["graph_lifecycle"]["setup"][
            "buckets"
        ]
    with pytest.raises(ValueError):
        capture._audit_graph_case(
            case, value, parent["numerical"]["model_config"], expected_device="cpu"
        )


def test_post_setup_failure_keeps_primary_and_records_failed_close(monkeypatch):
    from vllm_lt.validation import runner

    class StubRunner:
        def _enable_recurrent_graph(self, **kwargs):
            from types import SimpleNamespace

            self.decode_executor = SimpleNamespace()

        def _graph_snapshot(self):
            return {"enabled": True}

        def _close_recurrent_graph(self):
            raise RuntimeError("close failed")

    class StubEngine:
        def __init__(self, *args, **kwargs):
            self.model_runner = StubRunner()

    monkeypatch.setattr(runner, "ValidationEngine", StubEngine)
    original = runner.observe_native
    lifecycle = {}
    with pytest.raises(KeyboardInterrupt, match="primary interrupt") as caught:
        with capture._execution({"implementation_id": "B"}, [], lifecycle, capture.GRAPH_LIMITS):
            runner.ValidationEngine()
            raise KeyboardInterrupt("primary interrupt")
    assert "close failed" in caught.value.__notes__[0]
    assert lifecycle["close_error"] == {"type": "RuntimeError", "message": "close failed"}
    assert runner.ValidationEngine is StubEngine and runner.observe_native is original


def _settled_prefix(completed_run, destination, *, fail=False):
    import os
    import shutil

    _, parent, prefixes = completed_run
    shutil.copytree(prefixes[fail], destination, copy_function=os.link)
    return destination, parent


@pytest.mark.parametrize("failed", [False, True])
def test_settled_prefix_retains_required_failure_without_claiming_completion(
    completed_run, tmp_path, failed
):
    output, parent = _settled_prefix(completed_run, tmp_path / "prefix", fail=failed)
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["valid_prefix"] and result["errors"] == [], result["errors"]
    assert result["evidence_status"] == "incomplete" and not result["passed"]
    assert result["missing_case_ids"] and result["known_required_failure"] is failed
    assert result["raw_evidence"]["retained_references"]


@pytest.mark.parametrize("change", ["anchor", "summary", "pointer", "lifetime", "worker_claim"])
def test_corrupt_prefix_never_establishes_trusted_failure(completed_run, tmp_path, change):
    output, parent = _settled_prefix(completed_run, tmp_path / change, fail=True)
    folder = output / "numerical"
    if change == "anchor":
        path = next((folder / "spools").glob("*/*.bin"))
        contents = path.read_bytes()
        path.unlink()
        path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    elif change == "summary":
        path = next((folder / "comparisons").glob("*.summary.json"))
        value = read_json(path)
        value["required_failures"] += 1
        capture.write_json(path, value)
    elif change == "pointer":
        ledger = read_json(folder / "ledger.json")
        path = folder / "cases" / ledger["completed_cases"][0] / "result.json"
        value = read_json(path)
        value["graph_observations"].pop()
        capture.write_json(path, value)
    else:
        path = folder / "ledger.json"
        ledger = read_json(path)
        if change == "lifetime":
            ledger["case_lifetimes"][ledger["completed_cases"][0]]["finished_ns"] = 2**62
        else:
            ledger["workers"]["A"]["errors"] = []
        capture.write_json(path, ledger)
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["evidence_status"] == "invalid" and result["errors"]
    assert not result["valid_prefix"] and not result["known_required_failure"]


def test_pending_case_is_unqualified_without_invented_corruption(completed_run, tmp_path):
    output, parent = _settled_prefix(completed_run, tmp_path / "active")
    path = output / "numerical/ledger.json"
    ledger = read_json(path)
    case = parent["numerical"]["execution_order"][len(ledger["completed_cases"])]
    ledger["started_cases"] = [*ledger["completed_cases"], case["case_id"]]
    ledger["active_case"] = {"case_id": case["case_id"], "started_ns": 100, "deadline_ns": 200}
    capture.write_json(path, ledger)
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["evidence_status"] == "incomplete" and result["errors"] == []
    assert not result["valid_prefix"] and not result["known_required_failure"]
