"""Frozen quality source, input and protocol checks using synthetic CPU data."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_lt.benchmarks import q2_quality_schema as schema
from vllm_lt.models.config import OuroConfig

ROOT = Path(__file__).resolve().parents[1]
AFFINITY = {
    "cpu_ids": [0, 1],
    "numa_status": ["Cpus_allowed_list:\t0-1", "Mems_allowed_list:\t0-1"],
    "numactl_show": {
        "policy": "bind",
        "preferred_node": "0",
        "physcpubind": "0 1",
        "cpubind": "0",
        "nodebind": "0",
        "membind": "0",
    },
}
UUID = "GPU-cbf66259-f4ab-0ede-1811-82037dde5924"


class FakeTokenizer:
    def get_vocab_size(self, with_added_tokens=True):
        return 49152

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        length = 512 if "Question 0?" in text else 129 if "Question 1?" in text else 16
        return SimpleNamespace(ids=[1] * length)

    def decode(self, ids, skip_special_tokens):
        assert not skip_special_tokens
        return "".join({0: "<|endoftext|>", 1: "reason ", 2: "#### 7", 3: "bad"}[t] for t in ids)


def rehash(plan):
    plan["controls_sha256"] = schema._controls_hash(plan)
    plan["plan_sha256"] = schema.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


@pytest.fixture
def quality_plan(tmp_path, monkeypatch):
    import safetensors.torch
    import torch

    from vllm_lt.benchmarks import ab_schema
    from vllm_lt.benchmarks.schema import _source_manifest

    def forbidden(*a, **k):
        raise AssertionError("no accelerator discovery or tensor checkpoint loading")

    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    data = schema.data_module()
    model = tmp_path / "model"
    model.mkdir()
    schema.write_json(model / "config.json", OuroConfig().to_dict())
    (model / "tokenizer.json").write_text("synthetic local tokenizer")
    (model / "tokenizer_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"not tensors")
    monkeypatch.setattr(
        data, "TOKENIZER_SHA256", schema.file_record(model / "tokenizer.json")["sha256"]
    )
    monkeypatch.setattr(data, "load_tokenizer", lambda p: FakeTokenizer())
    normalized = tmp_path / "test.jsonl"
    normalized.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"{data.DATASET_REVISION}/main/test/{i}",
                    "question": f"Question {i}?",
                    "answer": f"#### {i}",
                }
            )
            + "\n"
            for i in range(1319)
        )
    )
    selection = data.prepare_selection(normalized, model / "tokenizer.json")
    selection_path = tmp_path / "selection.json"
    schema.write_json(selection_path, selection)
    contract = schema.read_json(ROOT / "benchmarks/fixtures/ouro-q2-fp32-quality-contract.json")
    contract_path = tmp_path / "contract.json"
    schema.write_json(contract_path, contract)
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"synthetic raw input")
    readme = tmp_path / "README.md"
    readme.write_text("MIT synthetic source")
    manifest = {
        "schema_version": 1,
        "artifact_type": "q2_quality_dataset_manifest",
        "dataset": contract["dataset"],
        "row_count": 1319,
        "raw": {
            **schema.file_record(raw),
            "sha256": contract["dataset"]["sha256"],
            "size_bytes": contract["dataset"]["size_bytes"],
        },
        "readme": schema.file_record(readme),
        "normalized": schema.file_record(normalized),
        "conversion": {
            "source_string_transformations": "none",
            "device_work": False,
            "producer": schema.file_record(readme),
        },
    }
    manifest_path = tmp_path / "dataset.json"
    schema.write_json(manifest_path, manifest)
    qual = tmp_path / "qualification.json"
    qual.write_text("original reference report stand-in")
    prerequisite = {
        "schema_version": 1,
        "artifact_type": "q2_quality_q1_prerequisite",
        "dtype": "float32",
        "decision": "passed",
        "fp32": {
            "decision": "passed",
            "numerical_required_failures": 0,
            "behavior_required_failures": 0,
        },
        "required_trajectories": 93,
        "families": {"main": 64, "live_gate": 16, "official": 4, "original": 9},
        "q1_evidence_status": "complete",
        "q1_execution_commit": "3fe9e002bd049084df47cda47e82f1e51cb35b2b",
        "q1_plan_sha256": "43d9994bfbf10aa5fc28366f6f106bf70b6ef9222e426401b0ebdc29639f35e9",
        "bf16_adaptive_eligible": False,
        "qualification": {
            "path": str(qual),
            "size_bytes": 669248,
            "sha256": "e1b77c625962aa21b9a68f89dfe29ced9c20def0844fa2f6737043b2e539586f",
        },
    }
    prereq_path = tmp_path / "prerequisite.json"
    schema.write_json(prereq_path, prerequisite)
    source = _source_manifest()
    source["status"] = ""
    # The synthetic probe models the historical PR14 engine named by the frozen
    # quality contract, even after review fixes advance this checkout.
    for record in source["files"]:
        if record["path"].startswith(schema.PRODUCTION_PREFIXES) or record["path"] in (
            schema.PRODUCTION_FILES
        ):
            raw_source = subprocess.check_output(
                ["git", "show", f"630a8fdc0dd47b6da68a3d30db6b851fc08af5c5:{record['path']}"],
                cwd=ROOT,
            )
            record.update(size_bytes=len(raw_source), sha256=hashlib.sha256(raw_source).hexdigest())
    probe = {
        "source": source,
        "imports": {n: n.replace(".", "/") + ".py" for n in schema.IMPORT_MODULES},
        "dependencies": {
            "python": "synthetic CPU",
            "torch": "2.13.0",
            "torch_cuda_build": "13.0",
            "distributions": [{"name": "tokenizers", "version": "0.21.4"}],
            "official": {},
        },
    }
    monkeypatch.setattr(schema, "source_probe", lambda: deepcopy(probe))
    monkeypatch.setattr(ab_schema, "affinity_snapshot", lambda: deepcopy(AFFINITY))
    plan = schema.make_quality_plan(
        contract_path,
        selection_path,
        model,
        dataset_manifest_path=manifest_path,
        prerequisite_path=prereq_path,
        gpu_ids=[2],
        gpu_uuid=UUID,
        affinity=AFFINITY,
    )
    real_record = schema.file_record

    def actual_record(path):
        if Path(path) == raw:
            return {k: manifest["raw"][k] for k in ("path", "size_bytes", "sha256")}
        if Path(path) == qual:
            return prerequisite["qualification"]
        return real_record(path)

    monkeypatch.setattr(schema, "file_record", actual_record)
    return plan


def test_exact_pass_budget_and_natural_eos(quality_plan):
    schema.validate_quality_plan(quality_plan)
    schema.verify_quality_plan(quality_plan)
    rows = quality_plan["execution_order"]
    assert len(rows) == len({r["example_id"] for r in rows}) == 66
    assert [r["phase"] for r in rows] == ["feasibility"] * 2 + ["evaluation"] * 64
    assert [r["run_id"] for r in rows[:3]] == ["F-feas-00", "F-feas-01", "F-eval-00"]
    assert all(r["max_steps"] == (r["prompt_tokens"] + 127) // 128 + 1531 for r in rows)
    assert quality_plan["resource_estimates"]["native_pool_bytes"] == 1207959552
    assert quality_plan["contract"]["engine"]["sampling"]["ignore_eos"] is False
    assert quality_plan["contract"]["acceptance"]["quality_floor"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["contract"]["engine"]["sampling"].update(ignore_eos=True),
        lambda p: p["contract"]["engine"]["cache"].update(num_blocks=160),
        lambda p: p["contract"]["controls"].update(dtype="bfloat16"),
        lambda p: p["contract"]["limits"].update(executions=65),
        lambda p: p["execution_order"][0].update(max_steps=1536),
        lambda p: p["contract"]["controls"].update(gpu_ids=[True]),
        lambda p: p["model_config"].update(eos_token_id=2),
        lambda p: p["model_files"][0].update(sha256="x" * 64),
        lambda p: p["inputs"]["prerequisite"]["contents"]["fp32"].update(
            numerical_required_failures=1
        ),
        lambda p: p["resource_estimates"].update(native_pool_bytes=0),
        lambda p: p["imports"].update({"vllm_lt.models.ouro": "outside.py"}),
        lambda p: p["source"]["files"][0].update(path="../unsafe"),
        lambda p: p.update(unknown=True),
    ],
)
def test_rehashed_contract_or_evidence_drift_rejected(quality_plan, mutation):
    mutation(quality_plan)
    rehash(quality_plan)
    with pytest.raises((ValueError, KeyError)):
        schema.validate_quality_plan(quality_plan)


def test_actual_input_content_drift_even_after_plan_rehash(quality_plan):
    record = quality_plan["inputs"]["selection"]
    record["contents"]["evaluation"][0]["prompt_token_ids"][0] = 2
    record["contents"]["evaluation"][0]["prompt_token_ids_sha256"] = schema.digest(
        record["contents"]["evaluation"][0]["prompt_token_ids"]
    )
    record["contents"]["selection_sha256"] = schema.digest(
        {k: v for k, v in record["contents"].items() if k != "selection_sha256"}
    )
    quality_plan["selection"] = deepcopy(record["contents"])
    rehash(quality_plan)
    with pytest.raises(ValueError, match="embedded input|actual selection"):
        schema.verify_quality_plan(quality_plan)


def test_byte_checkpoint_drift_rejected(quality_plan):
    (Path(quality_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        schema.verify_quality_plan(quality_plan)


@pytest.mark.parametrize("raw", [UUID, UUID.removeprefix("GPU-")])
def test_actual_gpu_uuid_optional_prefix(raw):
    assert schema.canonical_gpu_uuid(raw) == UUID


@pytest.mark.parametrize("raw", [None, "GPU-GPU-" + UUID, " " + UUID, UUID.upper(), "not-a-uuid"])
def test_malformed_gpu_uuid_rejected(raw):
    with pytest.raises(ValueError):
        schema.canonical_gpu_uuid(raw)


def test_duplicate_nonfinite_json_rejected(tmp_path):
    for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e9999}'):
        path = tmp_path / "invalid.json"
        path.write_text(raw)
        with pytest.raises(ValueError):
            schema.read_json(path)
