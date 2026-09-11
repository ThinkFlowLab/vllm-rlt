"""M4 plans use fake checkpoint bytes, genuine frozen inputs and no accelerator."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.benchmarks import m4_attention_schema as schema
from vllm_lt.benchmarks.schema import _digest, _file_record, read_json, write_json
from vllm_lt.models import OuroConfig

ROOT = Path(__file__).resolve().parents[1]
AFFINITY = {
    "cpu_ids": [56, 57],
    "numa_status": ["Cpus_allowed_list:\t56-57", "Mems_allowed_list:\t0-1"],
    "numactl_show": {
        "policy": "bind",
        "preferred_node": "1",
        "physcpubind": "56 57",
        "cpubind": "1",
        "nodebind": "1",
        "membind": "1",
    },
}


def rehash(plan):
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


@pytest.fixture(autouse=True)
def no_device_or_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU preparation touched CUDA or loaded checkpoint tensors")

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
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)


@pytest.fixture
def m4_plan(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", OuroConfig().to_dict())
    (model / "tokenizer.json").write_text('{"test":"byte-only-tokenizer"}\n')
    (model / "tokenizer_config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"frozen bytes, deliberately not loadable tensors")
    tokenizer = _file_record(model / "tokenizer.json")
    deps = schema.dependency_manifest()
    deps["official"]["dependencies"].update(transformers="4.55.0", kernels=None)
    deps["official"]["optional_kernels_present"] = False
    baseline = (ROOT / schema.DIFF_PATHS[0]).read_text().replace("BLOCK_T=64,", "BLOCK_T=32,")
    roots = {key: tmp_path / key for key in ("A", "B")}
    for side, root in roots.items():
        for relative in schema.INPUTS.values():
            contents = read_json(ROOT / relative)
            if "provenance" in contents:
                for key in ("size_bytes", "sha256"):
                    contents["provenance"]["tokenizer"][key] = tokenizer[key]
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, contents)
        for relative in (*schema.DIFF_PATHS, "vllm_lt/benchmarks/m4_attention.py"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                (baseline if side == "A" else baseline.replace("BLOCK_T=32,", "BLOCK_T=64,"))
                if relative in schema.DIFF_PATHS
                else "same common harness"
            )

    def probe(root):
        root = Path(root)
        return {
            "source": {
                "root": str(root),
                "commit": ("a" if root.name == "A" else "b") * 40,
                "status": "",
                "files": [
                    _file_record(p, relative_to=root)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                ],
            },
            "imports": {name: name.replace(".", "/") + ".py" for name in schema.IMPORT_MODULES},
            "dependencies": deepcopy(deps),
        }

    monkeypatch.setattr(schema, "probe_checkout", probe)
    monkeypatch.setattr(schema, "affinity_snapshot", lambda: deepcopy(AFFINITY))
    contract = tmp_path / "contract.json"
    write_json(contract, read_json(ROOT / "benchmarks/fixtures/ouro-m4-attention-contract.json"))
    return schema.make_plan(
        baseline_root=roots["A"],
        candidate_root=roots["B"],
        contract_path=contract,
        model_path=model,
        gpu_ids=[7],
        affinity=deepcopy(AFFINITY),
    )


def test_genuine139_row_plan_rebuilds_budget_and_abba_controls_without_cuda(m4_plan):
    schema.verify_plan(m4_plan)
    assert m4_plan["artifact_type"] == "m4_attention_plan"
    rows = m4_plan["execution_order"]
    assert len(rows) == len({r["execution_id"] for r in rows}) == 139
    assert Counter(r["kind"] for r in rows) == {
        "benchmark": 78,
        "numerical": 27,
        "kernel": 30,
        "lifecycle": 4,
    }
    assert [len(w["execution_ids"]) for w in m4_plan["workers"]] == [40, 35, 14, 14, 14, 14, 4, 4]
    assert [r["kind"] for r in rows[:40]] == ["benchmark"] * 7 + ["kernel"] * 15 + [
        "lifecycle"
    ] * 2 + ["numerical"] * 16
    benchmarks = [r for r in rows if r["kind"] == "benchmark"]
    assert Counter(r["phase"] for r in benchmarks) == {
        "feasibility": 14,
        "warmup": 28,
        "measured": 28,
        "diagnostic_warmup": 4,
        "profile": 4,
    }
    measured = [r for r in benchmarks if r["phase"] == "measured"]
    assert [r["worker_id"] for r in measured] == [
        w for w in ("A1", "B1", "B2", "A2") for _ in range(7)
    ]
    assert set(Counter(r["pair_id"] for r in measured).values()) == {2}
    assert all(r["pair_id"].startswith("M4-") for r in measured)
    assert len({r["controls_sha256"] for r in measured}) == 1
    assert all(r["case_lifetime_timeout_s"] == 600 for r in benchmarks)
    assert m4_plan["production_differences"] == list(schema.DIFF_PATHS)
    assert m4_plan["resource_estimates"]["total_artifact_bytes_upper_bound"] < 12 * 1024**3
    assert m4_plan["resource_estimates"]["components"]["dump_index_bytes"] == 498_677_760
    assert schema.execution_view(m4_plan)["contract"] == m4_plan["benchmark_contract"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(artifact_type="m2_ab_plan"),
        lambda p: p.update(unknown=True),
        lambda p: p["contract"]["controls"].update(gpu_ids=None),
        lambda p: p["contract"]["controls"].update(gpu_ids=[True]),
        lambda p: p["contract"]["controls"].update(benchmark_path="persistent"),
        lambda p: p["contract"]["controls"]["affinity"]["numactl_show"].update(
            membind="0", policy="default"
        ),
        lambda p: p["contract"]["acceptance"].update(target_ratio_min=1.0),
        lambda p: p["contract"]["acceptance"].update(peak_increase_bytes_max=128 * 1024**2),
        lambda p: p["contract"]["tile_policy"].update(num_warps=8),
        lambda p: p["contract"]["limits"].update(executions=140),
        lambda p: p["execution_order"].reverse(),
        lambda p: p["execution_order"][0].update(phase="measured"),
        lambda p: p["workers"][0]["execution_ids"].pop(),
        lambda p: p["implementations"]["B"]["source"].update(status=" M file.py"),
        lambda p: p["implementations"]["B"]["imports"].update(
            {schema.IMPORT_MODULES[-1]: "/elsewhere.py"}
        ),
        lambda p: p["dependencies"]["official"].update(optional_kernels_present=True),
        lambda p: p["resource_estimates"].update(total_artifact_bytes_upper_bound=0),
        lambda p: p["held"]["execution_order"].pop(),
        lambda p: p["numerical"]["comparison_order"][0].update(require_exact=True),
    ],
)
def test_rehashed_invalid_policy_order_source_and_nested_plans_fail(m4_plan, mutate):
    changed = deepcopy(m4_plan)
    mutate(changed)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(rehash(changed))


@pytest.mark.parametrize(
    "change", ["warp", "arithmetic", "comment", "expression", "syntax", "second_tile"]
)
def test_even_rehashed_candidate_source_can_only_change_the_exact_literal(m4_plan, change):
    changed = deepcopy(m4_plan)
    source = changed["attention_source"]["B"]
    if change == "warp":
        source = source.replace("num_warps=4", "num_warps=8")
    elif change == "arithmetic":
        source = source.replace("SCALE=q.shape[-1] ** -0.5", "SCALE=q.shape[-1] ** -0.4")
    elif change == "comment":
        source += "\n# unrelated but AST-equivalent change\n"
    elif change == "expression":
        source = source.replace("BLOCK_T=64", "BLOCK_T=32+32")
    elif change == "syntax":
        source += "\nfor invalid syntax\n"
    else:
        source += "\nother(BLOCK_T=64, num_warps=4)\n"
    changed["attention_source"]["B"] = source
    import hashlib

    record = next(
        r
        for r in changed["implementations"]["B"]["source"]["files"]
        if r["path"] == schema.DIFF_PATHS[0]
    )
    record.update(
        size_bytes=len(source.encode()), sha256=hashlib.sha256(source.encode()).hexdigest()
    )
    with pytest.raises(ValueError):
        schema.validate_plan(rehash(changed))


def test_rehashed_extra_metadata_production_difference_is_rejected(m4_plan):
    import hashlib

    changed = deepcopy(m4_plan)
    raw = b"unplanned per-layer metadata change"
    changed["implementations"]["B"]["source"]["files"].append(
        {
            "path": "vllm_lt/core/kv_cache_manager.py",
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    with pytest.raises(ValueError, match="unexpected production differences"):
        schema.validate_plan(rehash(changed))


def test_equal_extra_source_edits_cannot_replace_accepted_baseline(m4_plan):
    import hashlib

    changed = deepcopy(m4_plan)
    for side in ("A", "B"):
        source = changed["attention_source"][side] + "\n# common but unapproved baseline edit\n"
        changed["attention_source"][side] = source
        row = next(
            r
            for r in changed["implementations"][side]["source"]["files"]
            if r["path"] == schema.DIFF_PATHS[0]
        )
        row.update(
            size_bytes=len(source.encode()), sha256=hashlib.sha256(source.encode()).hexdigest()
        )
    with pytest.raises(ValueError, match="accepted compact/padded baseline"):
        schema.validate_plan(rehash(changed))


@pytest.mark.parametrize(
    "change", ["weight", "input", "source", "contract", "affinity", "environment"]
)
def test_verify_rejects_actual_file_or_environment_drift(m4_plan, monkeypatch, change):
    if change == "weight":
        (Path(m4_plan["model_path"]) / "model.safetensors").write_bytes(b"changed bytes")
    elif change == "input":
        Path(m4_plan["inputs"]["benchmark_suite"]["file"]["path"]).write_text("{}\n")
    elif change == "source":
        (Path(m4_plan["implementations"]["B"]["root"]) / schema.DIFF_PATHS[0]).write_text("changed")
    elif change == "contract":
        Path(m4_plan["contract_file"]["path"]).write_text("{}\n")
    elif change == "affinity":
        changed = deepcopy(AFFINITY)
        changed["numactl_show"]["membind"] = "0"
        monkeypatch.setattr(schema, "affinity_snapshot", lambda: changed)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", "changed")
    with pytest.raises(ValueError):
        schema.verify_plan(m4_plan)


def test_rehashed_embedded_fixture_cannot_disagree_with_actual_frozen_input(m4_plan):
    from vllm_lt.validation.m4_attention import build_numerical_plan
    from vllm_lt.validation.schema import FIXTURE_HASH_FIELDS

    changed = deepcopy(m4_plan)
    suite = changed["inputs"]["numerical_suite"]["contents"]
    suite["fixtures"][0]["continuation_input_ids"][0] += 1
    suite["fixtures_sha256"] = _digest({k: suite[k] for k in FIXTURE_HASH_FIELDS})
    changed["numerical"] = build_numerical_plan(
        suite, changed["inputs"]["numerical_contract"]["contents"], changed["model_config"]
    )
    schema.validate_plan(rehash(changed))
    with pytest.raises(ValueError, match="input parsed contents"):
        schema.verify_plan(changed)


def test_all_evidence_components_are_charged_and_over_budget_fails(m4_plan):
    n = deepcopy(m4_plan["numerical"])
    n["resource_estimates"]["comparison_records_upper_bound"] += 1_000_000
    with pytest.raises(ValueError, match="12 GiB"):
        schema.resource_estimates(n, m4_plan["held"])
    assert (
        sum(m4_plan["resource_estimates"]["components"].values())
        == m4_plan["resource_estimates"]["total_artifact_bytes_upper_bound"]
    )
    held = deepcopy(m4_plan["held"])
    held["resource_estimates"]["artifact_bytes_upper_bound"] += 1
    with pytest.raises(ValueError, match="held plans exceed"):
        schema.resource_estimates(m4_plan["numerical"], held)


def test_source_probe_rejects_shadow_import_before_any_device_work(m4_plan, monkeypatch):
    from types import SimpleNamespace

    selected = m4_plan["implementations"]["A"]
    monkeypatch.setattr(schema, "_source_manifest", lambda: deepcopy(selected["source"]))
    monkeypatch.setattr(
        schema.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__="/wrong/" + name.replace(".", "/") + ".py"),
    )
    with pytest.raises(ValueError, match="import outside"):
        schema.source_probe()


def test_new_contract_does_not_mutate_m2_policy():
    from vllm_lt.benchmarks import ab_schema

    assert ab_schema.ACCEPTANCE["target_ratio_min"] == 1.1
    assert ab_schema.LIMITS["executions"] == 105
    assert schema.ACCEPTANCE["target_ratio_min"] == 1.01
    assert schema.LIMITS["executions"] == 139
