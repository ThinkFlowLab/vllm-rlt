"""Offline outcome and ACK audit tests; synthetic records are explicitly not model results."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import test_q2_quality_schema as fixtures

from vllm_lt.benchmarks import q2_quality_score as score

quality_plan = fixtures.quality_plan
schema = fixtures.schema


def write(path, value):
    schema.write_json(path, value)


def bind(root, run_id):
    folder = root / "runs" / run_id
    completed = schema.read_json(folder / "completed.json")
    completed["result"] = score._record(folder / "result.json")
    write(folder / "completed.json", completed)
    ack = schema.read_json(folder / "acknowledged.json")
    ack["result"] = completed["result"]
    ack["completion"] = score._record(folder / "completed.json")
    write(folder / "acknowledged.json", ack)


@pytest.fixture
def recorded(quality_plan, tmp_path, monkeypatch):
    plan = quality_plan
    monkeypatch.setattr(score, "schema", schema)
    root = tmp_path / "run"
    root.mkdir()
    (root / "inputs").mkdir()
    write(root / "plan.json", plan)
    for key, name in (
        ("contract", "contract.json"),
        ("selection", "selection.json"),
        ("dataset", "dataset-manifest.json"),
        ("prerequisite", "prerequisite.json"),
    ):
        shutil.copyfile(plan["inputs"][key]["file"]["path"], root / "inputs" / name)
    dataset = plan["inputs"]["dataset"]["contents"]
    for key, name in (
        ("raw", "raw.parquet"),
        ("readme", "README.md"),
        ("normalized", "normalized.jsonl"),
    ):
        shutil.copyfile(dataset[key]["path"], root / "inputs" / name)
    qual = plan["inputs"]["prerequisite"]["contents"]["qualification"]
    shutil.copyfile(qual["path"], root / "inputs/q1-qualification.json")
    shutil.copyfile(plan["selection"]["tokenizer"]["file"]["path"], root / "inputs/tokenizer.json")
    real_record = score._record

    def record(path):
        if Path(path).name == "raw.parquet":
            return {k: dataset["raw"][k] for k in ("size_bytes", "sha256")}
        if Path(path).name == "q1-qualification.json":
            return {k: qual[k] for k in ("size_bytes", "sha256")}
        return real_record(path)

    monkeypatch.setattr(score, "_record", record)
    controls = plan["contract"]["controls"]
    deadline = 7200 * 10**9 + 100
    worker = {
        "schema_version": 1,
        "artifact_type": "q2_quality_worker",
        "plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "started_ns": 300,
        "ended_ns": 1000000,
        "deadline_ns": deadline,
        "completed_runs": [],
        "failures": [],
        "source_probe": {k: plan[k] for k in ("source", "imports", "dependencies")},
        "environment": {
            "cuda_visible_devices": "2",
            "gpu_uuid": fixtures.UUID.removeprefix("GPU-"),
            "actual_torch_threads": {"intraop": 1, "interop": 1},
            "cpu_affinity": controls["affinity"]["cpu_ids"],
            "active_affinity": controls["affinity"],
            "arithmetic": plan["contract"]["arithmetic"],
            "cudnn_allow_tf32": False,
            "runtime_environment": plan["runtime_environment"],
            "scheduler": [{"gpu_id": 2, "type": "RUN", "user": "test"}],
            "logical_device": "cuda:0",
            "account": "test",
        },
        "preparation": {
            "started_ns": 400,
            "ended_ns": 900,
            "native_pool": {
                "size_bytes": 1207959552,
                "num_blocks": 192,
                "block_size": 16,
                "bytes_per_block": 6291456,
                "key_data_ptr": 100,
                "value_data_ptr": 200,
                "dtype": "torch.float32",
                "device": "cuda:0",
                "backend": "triton",
            },
        },
        "request_cleanup": dict.fromkeys(
            ("requests", "queued", "allocated_pages", "used_pages"), 0
        ),
        "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "q2_quality_manifest",
        "plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "started_ns": 100,
        "ended_ns": 1100000,
        "deadline_ns": deadline,
        "completed_runs": [],
        "failures": [],
        "launch": {"started_ns": 200, "ended_ns": 1050000, "returncode": 0},
    }
    for index, row in enumerate(plan["execution_order"]):
        folder = root / "runs" / row["run_id"]
        folder.mkdir(parents=True)
        start = 300 if index == 0 else 1000 * (index + 1)
        marker = {
            "schema_version": 1,
            "run": row,
            "plan_sha256": plan["plan_sha256"],
            "started_ns": start,
            "deadline_ns": start + 600 * 10**9,
        }
        write(folder / "started.json", marker)
        example = plan["selection"][row["phase"]][row["example_index"]]
        output = [2, 0]
        data = schema.data_module()
        texts = data.decode_output(fixtures.FakeTokenizer(), output, "stop")
        parsed = data.parse_answer(texts["scoring_text"])
        reference = data.parse_answer(example["reference_answer"])
        prefill = (row["prompt_tokens"] + 127) // 128
        stages = {"prefill": prefill, "prelude": 1, "recurrent": 4, "coda": 2}
        result = {
            **marker,
            "artifact_type": "q2_quality_result",
            "status": "complete",
            "example_id": row["example_id"],
            "prompt_token_ids": example["prompt_token_ids"],
            "prompt_sha256": example["prompt_sha256"],
            "output_token_ids": output,
            "exit_depths": [4, 4],
            "finish_reason": "stop",
            "truncated": False,
            **texts,
            "parsed_answer": parsed,
            "parsed_reference": reference,
            "correct": parsed["value"] == reference["value"],
            "counts": {"steps": sum(stages.values()), "stage_counts": stages},
            "finite_checks": {
                "lm_head": 2,
                "recurrent": 4 * (prefill + 1) if row["phase"] == "feasibility" else 0,
                "coda": 2 if row["phase"] == "feasibility" else 0,
            },
            "before_request": worker["request_cleanup"],
            "cleanup": worker["request_cleanup"],
            "failures": [],
            "ended_ns": max(start + 100, 1000),
        }
        write(folder / "result.json", result)
        completion = {
            **marker,
            "status": "complete",
            "result": record(folder / "result.json"),
            "completed_ns": result["ended_ns"] + 10,
        }
        write(folder / "completed.json", completion)
        ack = {
            **completion,
            "completion": record(folder / "completed.json"),
            "acknowledged_ns": result["ended_ns"] + 20,
        }
        write(folder / "acknowledged.json", ack)
        worker["completed_runs"].append(row["run_id"])
    manifest["completed_runs"] = list(worker["completed_runs"])
    write(root / "worker.json", worker)
    write(root / "manifest.json", manifest)
    return root, plan


def test_complete_baseline_has_no_accuracy_floor_and_is_portable(recorded):
    root, plan = recorded
    report = score.score_records(plan, root)
    assert report["errors"] == []
    assert report["complete"] and report["evidence_status"] == "complete"
    assert report["counts"]["eligible_evaluation"] == 64
    assert report["accuracy"]["total"] == 64 and report["accuracy"]["fraction"] < 0.1
    assert report["comparisons"] == [] and report["speed_claim"] is False
    assert report["issue_closure"] is False
    moved = root.with_name("relocated")
    shutil.copytree(root, moved)
    assert score.build_report(moved)["accuracy"] == report["accuracy"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(raw_text="forged"),
        lambda r: r.update(correct=not r["correct"]),
        lambda r: r["finite_checks"].update(lm_head=1),
        lambda r: r.update(exit_depths=[4, 3]),
        lambda r: r["counts"]["stage_counts"].update(recurrent=3),
        lambda r: r.update(output_token_ids=[0, 2]),
        lambda r: r.update(truncated=True),
        lambda r: r["cleanup"].update(used_pages=1),
        lambda r: r.update(prompt_sha256="0" * 64),
    ],
)
def test_rehashed_invalid_generation_never_passes(recorded, change):
    root, plan = recorded
    run_id = "F-eval-00"
    path = root / "runs" / run_id / "result.json"
    result = schema.read_json(path)
    change(result)
    write(path, result)
    bind(root, run_id)
    report = score.score_records(plan, root)
    assert not report["complete"] and report["evidence_status"] == "invalid" and report["errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", "bad"),
        ("gpu_uuid", "GPU-00000000-0000-0000-0000-000000000000"),
        ("cudnn_allow_tf32", True),
        ("logical_device", "cuda:1"),
    ],
)
def test_actual_controls_not_metadata_assumptions(recorded, field, value):
    root, plan = recorded
    worker = schema.read_json(root / "worker.json")
    worker["environment"][field] = value
    write(root / "worker.json", worker)
    report = score.score_records(plan, root)
    assert report["errors"] and not report["complete"]


def test_completion_hash_link_is_required(recorded):
    root, plan = recorded
    p = root / "runs/F-eval-00/acknowledged.json"
    ack = schema.read_json(p)
    ack["completion"]["sha256"] = "0" * 64
    write(p, ack)
    assert "ACK completion hash" in score.score_records(plan, root)["errors"][0]


def truncate(root, plan, count, *, failure=True):
    ids = [r["run_id"] for r in plan["execution_order"]]
    for run_id in ids[count:]:
        shutil.rmtree(root / "runs" / run_id)
    for filename in ("worker.json", "manifest.json"):
        value = schema.read_json(root / filename)
        value["completed_runs"] = ids[:count]
        value["status"] = "failed" if failure else "incomplete"
        value["failures"] = [{"type": "RuntimeError", "message": "stopped"}] if failure else []
        write(root / filename, value)


def test_incomplete_slice_never_reduces_denominator(recorded):
    root, plan = recorded
    truncate(root, plan, 10)
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["decision"] == "failed"
    assert report["evidence_status"] == "incomplete" and report["accuracy"] is None
    assert report["denominator"] == 64 and report["counts"]["eligible_evaluation"] == 8


def test_post_ack_failed_tail_preserves_valid_generation_but_excludes_case(recorded):
    root, plan = recorded
    row = plan["execution_order"][-1]
    folder = root / "runs" / row["run_id"]
    marker = schema.read_json(folder / "started.json")
    write(
        folder / "failure.json",
        {**marker, "occurred_ns": 999999, "failure": {"type": "OSError", "message": "late export"}},
    )
    for filename in ("worker.json", "manifest.json"):
        value = schema.read_json(root / filename)
        value.update(status="failed", failures=[{"type": "OSError", "message": "late export"}])
        write(root / filename, value)
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["decision"] == "failed"
    assert report["counts"]["validated_generations"] == 66
    assert report["counts"]["eligible_evaluation"] == 63 and report["accuracy"] is None
    assert report["retained_failed_records"][0]["generation_valid"] is True
    assert report["retained_failed_records"][0]["case_eligible"] is False


def test_partial_failed_feasibility_is_execution_failure_not_accuracy(recorded):
    root, plan = recorded
    truncate(root, plan, 1)
    folder = root / "runs/F-feas-00"
    for name in ("completed.json", "acknowledged.json"):
        (folder / name).unlink()
    result = schema.read_json(folder / "result.json")
    result.update(status="failed", failures=[{"type": "ValueError", "message": "nonfinite logits"}])
    write(folder / "result.json", result)
    for name in ("worker.json", "manifest.json"):
        value = schema.read_json(root / name)
        value["completed_runs"] = []
        write(root / name, value)
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["decision"] == "failed"
    assert report["counts"]["validated_generations"] == 0 and report["accuracy"] is None


def test_missing_cleanup_is_missing_evidence_not_fake_zero(recorded):
    root, plan = recorded
    value = schema.read_json(root / "worker.json")
    del value["teardown_after_workspace_release"]
    write(root / "worker.json", value)
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["evidence_status"] == "incomplete"
    assert "worker/teardown_after_workspace_release" in report["missing"]


def test_unknown_or_symlink_artifact_invalid(recorded):
    root, plan = recorded
    (root / "unexpected").symlink_to(root / "plan.json")
    assert score.score_records(plan, root)["errors"]


def test_standalone_score_import_does_not_import_torch_or_transformers():
    source = Path(score.__file__)
    code = """import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','transformers','safetensors'):
   raise AssertionError('forbidden offline import: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_failed_before_model_preparation_has_no_invented_coverage(recorded):
    root, plan = recorded
    truncate(root, plan, 0)
    worker = schema.read_json(root / "worker.json")
    del worker["environment"], worker["preparation"], worker["teardown_after_workspace_release"]
    write(root / "worker.json", worker)
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["evidence_status"] == "incomplete"
    assert report["decision"] == "failed" and report["counts"]["validated_generations"] == 0
    assert report["accuracy"] is None and "worker/environment" in report["missing"]


def test_available_corrupt_source_not_hidden_by_missing_preparation(recorded):
    root, plan = recorded
    truncate(root, plan, 0)
    worker = schema.read_json(root / "worker.json")
    del worker["preparation"]
    worker["source_probe"]["source"]["commit"] = "0" * 40
    write(root / "worker.json", worker)
    report = score.score_records(plan, root)
    assert report["evidence_status"] == "invalid" and report["errors"]


@pytest.mark.parametrize(
    "ids,reason", [([0], "stop"), ([1] * 255 + [0], "stop"), ([1] * 255 + [2], "length")]
)
def test_natural_completion_boundaries_are_scored_under_same_rule(recorded, ids, reason):
    root, plan = recorded
    run_id = "F-eval-00"
    row = plan["execution_order"][2]
    path = root / "runs" / run_id / "result.json"
    result = schema.read_json(path)
    result.update(
        output_token_ids=ids,
        exit_depths=[4] * len(ids),
        finish_reason=reason,
        truncated=reason == "length",
    )
    data = schema.data_module()
    result.update(data.decode_output(fixtures.FakeTokenizer(), ids, reason))
    result["parsed_answer"] = data.parse_answer(result["scoring_text"])
    result["correct"] = (
        result["parsed_answer"]["value"] is not None
        and result["parsed_answer"]["value"] == result["parsed_reference"]["value"]
    )
    stages = {
        "prefill": (row["prompt_tokens"] + 127) // 128,
        "prelude": len(ids) - 1,
        "recurrent": 4 * (len(ids) - 1),
        "coda": len(ids),
    }
    result["counts"] = {"steps": sum(stages.values()), "stage_counts": stages}
    result["finite_checks"] = {"lm_head": len(ids), "recurrent": 0, "coda": 0}
    write(path, result)
    bind(root, run_id)
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["complete"]
    outcome = next(r for r in report["records"] if r["run_id"] == run_id)
    assert outcome["truncated"] == (reason == "length")
    assert outcome["decode_mean_depth"] == (4 if len(ids) > 1 else None)


def test_unplanned_duplicate_result_is_not_hidden(recorded):
    root, plan = recorded
    shutil.copyfile(root / "runs/F-eval-00/result.json", root / "runs/F-eval-00/extra-result.json")
    report = score.score_records(plan, root)
    assert (
        report["evidence_status"] == "invalid" and "unexpected case record" in report["errors"][0]
    )


def test_controller_failure_before_worker_or_input_copy_is_incomplete(recorded):
    root, plan = recorded
    truncate(root, plan, 0)
    (root / "worker.json").unlink()
    shutil.rmtree(root / "inputs")
    report = score.score_records(plan, root)
    assert report["errors"] == [] and report["evidence_status"] == "incomplete"
    assert report["decision"] == "failed" and report["counts"]["validated_generations"] == 0
    assert report["accuracy"] is None
