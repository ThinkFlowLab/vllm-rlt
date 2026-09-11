"""Portable, no-Torch Q2 scoring: run this file directly for CPU-only readback."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

# Loading siblings by filename avoids the unchanged package initializer's Torch imports.
_SPEC = importlib.util.spec_from_file_location(
    "_q2_quality_schema_offline", Path(__file__).with_name("q2_quality_schema.py")
)
schema = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = schema
_SPEC.loader.exec_module(schema)
require, equal, integer = schema.require, schema.equal, schema.integer


def _clock(value, name, lower=0, upper=None):
    return integer(value, name, lower, upper)


def _identity(value, plan, row):
    equal(value["schema_version"], 1, "record version")
    equal(value["run"], row, "record execution row")
    equal(value["plan_sha256"], plan["plan_sha256"], "record plan")


def _record(path):
    data = path.read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _empty(value):
    equal(
        value,
        {"requests": 0, "queued": 0, "allocated_pages": 0, "used_pages": 0},
        "request cleanup",
    )


def _inventory(root, deadline):
    total, cases, files = 0, {}, {}
    for path in root.rglob("*"):
        require(time.monotonic() < deadline, "offline scoring time cap")
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "unsafe artifact")
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        size = path.stat().st_size
        total += size
        files[str(rel)] = size
        require(len(files) <= 4096, "artifact file-count cap")
        require(total <= schema.LIMITS["artifact_bytes_max"], "artifact total byte cap")
        require(
            path.suffix not in (".bin", ".safetensors", ".pt") and "trace" not in path.name,
            "unplanned tensor or profile",
        )
        if rel.parts[0] == "runs" and len(rel.parts) > 2:
            cases[rel.parts[1]] = cases.get(rel.parts[1], 0) + size
            require(cases[rel.parts[1]] <= schema.LIMITS["case_bytes_max"], "case byte cap")
        if path.suffix == ".json":
            require(size <= 8 * 1024**2, "JSON file byte cap")
    return {"total_bytes": total, "case_bytes": cases, "file_count": len(files)}


def _inputs(plan, root, tokenizer_path):
    mapping = {
        "contract": "contract.json",
        "selection": "selection.json",
        "dataset": "dataset-manifest.json",
        "prerequisite": "prerequisite.json",
    }
    for key, name in mapping.items():
        path = root / "inputs" / name
        expected = plan["inputs"][key]
        equal(
            _record(path),
            {k: expected["file"][k] for k in ("size_bytes", "sha256")},
            "retained input " + key,
        )
        equal(schema.read_json(path), expected["contents"], "retained parsed input " + key)
    manifest = plan["inputs"]["dataset"]["contents"]
    for key, name in (
        ("raw", "raw.parquet"),
        ("normalized", "normalized.jsonl"),
        ("readme", "README.md"),
    ):
        equal(
            _record(root / "inputs" / name),
            {k: manifest[key][k] for k in ("size_bytes", "sha256")},
            "retained dataset " + key,
        )
    prerequisite = plan["inputs"]["prerequisite"]["contents"]
    equal(
        _record(root / "inputs/q1-qualification.json"),
        {k: prerequisite["qualification"][k] for k in ("size_bytes", "sha256")},
        "retained Q1 report",
    )
    path = Path(tokenizer_path) if tokenizer_path else root / "inputs/tokenizer.json"
    expected = plan["selection"]["tokenizer"]["file"]
    equal(_record(path), {k: expected[k] for k in ("size_bytes", "sha256")}, "retained tokenizer")
    data = schema.data_module()
    rebuilt = data.prepare_selection(root / "inputs/normalized.jsonl", path)
    # Relocation changes only the two source paths and their dependent selection digest.
    rebuilt["dataset"]["file"]["path"] = plan["selection"]["dataset"]["file"]["path"]
    rebuilt["tokenizer"]["file"]["path"] = expected["path"]
    rebuilt["selection_sha256"] = schema.digest(
        {k: v for k, v in rebuilt.items() if k != "selection_sha256"}
    )
    equal(rebuilt, plan["selection"], "recomputed population/tokenization/selection")
    return data.load_tokenizer(path)


class _MissingStartup(Exception):
    def __init__(self, missing, failures):
        self.missing, self.failures = missing, failures


def _failures(value):
    require(isinstance(value, list), "failure list")
    for item in value:
        schema.keys(item, ("type", "message"), "failure record")
        require(all(isinstance(v, str) and v for v in item.values()), "failure text")
    return value


def _controls(plan, manifest, worker):
    start = _clock(manifest["started_ns"], "controller start")
    deadline = _clock(manifest["deadline_ns"], "controller deadline", start)
    equal(deadline, start + schema.LIMITS["total_timeout_s"] * 10**9, "global budget")
    end = _clock(manifest["ended_ns"], "controller end", start)
    launch = manifest["launch"]
    launch_start = _clock(launch["started_ns"], "launch start", start, end)
    launch_end = _clock(launch["ended_ns"], "launch end", launch_start, end)
    worker_start = _clock(worker["started_ns"], "worker start", launch_start, launch_end)
    worker_end = _clock(worker.get("ended_ns", launch_end), "worker end", worker_start, launch_end)
    equal(worker["deadline_ns"], deadline, "worker deadline")
    for record, kind in ((manifest, "q2_quality_manifest"), (worker, "q2_quality_worker")):
        equal(record["schema_version"], 1, "manifest version")
        equal(record["artifact_type"], kind, "manifest type")
        equal(record["plan_sha256"], plan["plan_sha256"], "manifest plan")
        require(
            record["status"] in ("running", "complete", "failed", "incomplete"), "manifest status"
        )
        _failures(record["failures"])

    def missing_startup(key):
        require(
            not worker["completed_runs"] and worker["status"] != "complete",
            "completed worker lacks execution controls",
        )
        raise _MissingStartup(["worker/" + key], manifest["failures"] + worker["failures"])

    if "source_probe" not in worker:
        missing_startup("source_probe")
    probe = worker["source_probe"]
    equal(probe, {k: plan[k] for k in ("source", "imports", "dependencies")}, "worker source probe")
    if "environment" not in worker:
        missing_startup("environment")
    env, controls = worker["environment"], plan["contract"]["controls"]
    equal(env["cuda_visible_devices"], str(controls["gpu_ids"][0]), "actual reserved GPU")
    equal(schema.canonical_gpu_uuid(env["gpu_uuid"]), controls["gpu_uuid"], "actual GPU UUID")
    equal(env["actual_torch_threads"], {"intraop": 1, "interop": 1}, "actual Torch threads")
    equal(env["cpu_affinity"], controls["affinity"]["cpu_ids"], "actual CPU affinity")
    equal(env["active_affinity"], controls["affinity"], "actual NUMA policy")
    equal(env["arithmetic"], plan["contract"]["arithmetic"], "actual arithmetic")
    equal(env["cudnn_allow_tf32"], False, "actual cuDNN TF32")
    equal(env["runtime_environment"], plan["runtime_environment"], "actual environment")
    require(isinstance(env.get("scheduler"), list) and len(env["scheduler"]) == 1, "scheduler row")
    reservation = env["scheduler"][0]
    equal(str(reservation["gpu_id"]), str(controls["gpu_ids"][0]), "scheduler GPU")
    equal(reservation["type"], "RUN", "scheduler RUN reservation")
    equal(reservation["user"], env["account"], "scheduler owner")
    equal(env["logical_device"], "cuda:0", "logical device")
    if (
        "preparation" not in worker
        or not {"ended_ns", "native_pool"} <= worker["preparation"].keys()
    ):
        missing_startup("preparation_completion")
    prep = worker["preparation"]
    pstart = _clock(prep["started_ns"], "preparation start", worker_start, worker_end)
    pend = _clock(prep["ended_ns"], "preparation end", pstart, worker_end)
    pool = prep["native_pool"]
    equal(
        pool["size_bytes"], plan["resource_estimates"]["native_pool_bytes"], "resident native pool"
    )
    equal(pool["dtype"], "torch.float32", "native pool dtype")
    equal(pool["device"], "cuda:0", "native pool device")
    equal(
        {k: pool[k] for k in ("num_blocks", "block_size", "bytes_per_block", "backend")},
        {"num_blocks": 192, "block_size": 16, "bytes_per_block": 6291456, "backend": "triton"},
        "native pool geometry",
    )
    integer(pool["key_data_ptr"], "key pointer", 1)
    integer(pool["value_data_ptr"], "value pointer", 1)
    require(pool["key_data_ptr"] != pool["value_data_ptr"], "distinct K/V pool")
    return {
        "worker_start": worker_start,
        "worker_end": worker_end,
        "preparation_end": pend,
        "deadline": deadline,
        "controller_end": end,
        "launch": launch,
    }


def _generation(plan, row, example, result, tokenizer, lower, upper, deadline):
    _identity(result, plan, row)
    equal(result["artifact_type"], "q2_quality_result", "result type")
    equal(result["status"], "complete", "generation status")
    equal(result["failures"], [], "generation failures")
    started = _clock(result["started_ns"], "case start", lower, upper)
    case_deadline = min(deadline, started + schema.LIMITS["case_timeout_s"] * 10**9)
    equal(result["deadline_ns"], case_deadline, "case deadline")
    ended = _clock(result["ended_ns"], "generation end", started, min(upper, case_deadline))
    equal(result["example_id"], example["source_id"], "actual example")
    equal(result["prompt_token_ids"], example["prompt_token_ids"], "actual prompt IDs")
    equal(result["prompt_sha256"], example["prompt_sha256"], "actual prompt hash")
    ids = result["output_token_ids"]
    data = schema.data_module()
    texts = data.decode_output(tokenizer, ids, result["finish_reason"])
    for key, value in texts.items():
        equal(result[key], value, "redecoded " + key)
    output_count = len(ids)
    equal(result["exit_depths"], [4] * output_count, "actual four-loop outputs")
    equal(result["truncated"], result["finish_reason"] == "length", "truncation")
    parsed = data.parse_answer(texts["scoring_text"])
    reference = data.parse_answer(example["reference_answer"])
    equal(reference, example["parsed_reference"], "prepared reference")
    equal(result["parsed_answer"], parsed, "reparsed answer")
    equal(result["parsed_reference"], reference, "reparsed reference")
    correct = parsed["value"] is not None and Decimal(parsed["value"]) == Decimal(
        reference["value"]
    )
    equal(result["correct"], correct, "rescored correctness")
    prefill = (row["prompt_tokens"] + 127) // 128
    stages = {
        "prefill": prefill,
        "prelude": output_count - 1,
        "recurrent": 4 * (output_count - 1),
        "coda": output_count,
    }
    equal(
        result["counts"],
        {"steps": sum(stages.values()), "stage_counts": stages},
        "actual stage work",
    )
    require(sum(stages.values()) <= row["max_steps"], "step budget")
    full = row["phase"] == "feasibility"
    equal(
        result["finite_checks"],
        {
            "lm_head": output_count,
            "recurrent": 4 * (prefill + output_count - 1) if full else 0,
            "coda": output_count if full else 0,
        },
        "finite observation coverage",
    )
    _empty(result["before_request"])
    _empty(result["cleanup"])
    return {
        "run_id": row["run_id"],
        "example_id": example["source_id"],
        "phase": row["phase"],
        "output_count": output_count,
        "prompt_count": row["prompt_tokens"],
        "correct": correct,
        "parser_status": parsed["diagnostics"]["status"],
        "parsed_answer": parsed,
        "finish_reason": result["finish_reason"],
        "truncated": result["truncated"],
        "started_ns": started,
        "ended_ns": ended,
        "deadline_ns": case_deadline,
        "decode_outputs": output_count - 1,
        "decode_mean_depth": 4 if output_count > 1 else None,
        "decode_mean_depth_unavailable_reason": None if output_count > 1 else "no_decode_outputs",
    }


def score_records(plan, records_dir, *, tokenizer_path=None):
    """Audit actual completed prefixes; incomplete coverage never changes denominator64."""
    root = Path(records_dir)
    deadline = time.monotonic() + 120
    report = {
        "schema_version": 1,
        "artifact_type": "q2_quality_score",
        "plan_sha256": plan.get("plan_sha256"),
        "evidence_status": "invalid",
        "decision": "inconclusive",
        "complete": False,
        "errors": [],
        "missing": [],
        "failures": [],
        "records": [],
        "retained_failed_records": [],
        "denominator": 64,
        "accuracy": None,
        "comparisons": [],
        "paired_qualification": "not_applicable:F32_baseline_only",
        "speed_claim": False,
        "issue_closure": False,
    }
    try:
        schema.validate_quality_plan(plan)
        report["inventory"] = _inventory(root, deadline)
        equal(schema.read_json(root / "plan.json"), plan, "retained plan")
        manifest = schema.read_json(root / "manifest.json")
        if not (root / "worker.json").exists():
            equal(manifest["schema_version"], 1, "controller version")
            equal(manifest["artifact_type"], "q2_quality_manifest", "controller type")
            equal(manifest["plan_sha256"], plan["plan_sha256"], "controller plan")
            equal(manifest["completed_runs"], [], "missing worker has no acknowledged cases")
            require(manifest["status"] != "complete", "successful controller has no worker")
            start = _clock(manifest["started_ns"], "controller start")
            _clock(manifest["ended_ns"], "controller end", start)
            equal(
                manifest["deadline_ns"],
                start + schema.LIMITS["total_timeout_s"] * 10**9,
                "controller deadline",
            )
            require(
                not (root / "runs").exists() or not list((root / "runs").iterdir()),
                "case evidence exists without worker provenance",
            )
            raise _MissingStartup(
                ["worker.json", "execution_controls"], _failures(manifest["failures"])
            )
        tokenizer = _inputs(plan, root, tokenizer_path)
        worker = schema.read_json(root / "worker.json")
        bounds = _controls(plan, manifest, worker)
        rows = plan["execution_order"]
        ids = [r["run_id"] for r in rows]
        entries = list((root / "runs").iterdir())
        require(all(p.is_dir() for p in entries), "unexpected execution-root file")
        actual_dirs = sorted(p.name for p in entries)
        allowed = {
            "started.json",
            "result.json",
            "completed.json",
            "acknowledged.json",
            "failure.json",
        }
        for folder in entries:
            require(
                all(
                    p.is_file() and (p.name in allowed or p.name.removesuffix(".tmp") in allowed)
                    for p in folder.iterdir()
                ),
                "unexpected case record",
            )
        require(set(actual_dirs) <= set(ids), "unknown example directory")
        for label, ack in (
            ("parent", manifest["completed_runs"]),
            ("worker", worker["completed_runs"]),
        ):
            require(isinstance(ack, list), label + " acknowledgment list")
            equal(ack, ids[: len(ack)], label + " acknowledgment order")
        require(
            len(manifest["completed_runs"]) <= len(worker["completed_runs"]),
            "parent ACK exceeds worker",
        )
        acknowledged, failed_tail = [], False
        previous = bounds["worker_start"]
        for row in rows:
            require(time.monotonic() < deadline, "offline scoring time cap")
            run_id = row["run_id"]
            folder = root / "runs" / run_id
            if not folder.exists():
                report["missing"].append(run_id)
                continue
            require(
                not report["missing"] and not failed_tail, "execution after missing or failed tail"
            )
            marker = schema.read_json(folder / "started.json")
            _identity(marker, plan, row)
            start = _clock(marker["started_ns"], "started marker", previous, bounds["worker_end"])
            cdeadline = min(bounds["deadline"], start + 600 * 10**9)
            equal(marker["deadline_ns"], cdeadline, "marker deadline")
            failure = None
            if (folder / "failure.json").exists():
                failure = schema.read_json(folder / "failure.json")
                _identity(failure, plan, row)
                _clock(failure["occurred_ns"], "case failure", start, bounds["worker_end"])
                schema.keys(failure["failure"], ("type", "message"), "failure reason")
                require(
                    all(isinstance(v, str) and v for v in failure["failure"].values()),
                    "failure detail",
                )
                report["failures"].append({"run_id": run_id, **failure["failure"]})
                failed_tail = True
            if not (folder / "result.json").exists() or not (folder / "completed.json").exists():
                if (folder / "result.json").exists():
                    partial = schema.read_json(folder / "result.json")
                    _identity(partial, plan, row)
                    equal(partial["started_ns"], start, "partial start")
                    equal(partial["deadline_ns"], cdeadline, "partial deadline")
                    _clock(partial["ended_ns"], "partial end", start, bounds["worker_end"])
                    require(
                        partial["status"] == "failed" and partial.get("failures"),
                        "uncommitted result",
                    )
                    report["failures"].extend({"run_id": run_id, **f} for f in partial["failures"])
                    failed_tail = True
                report["missing"].append(run_id)
                continue
            result = schema.read_json(folder / "result.json")
            example = plan["selection"][row["phase"]][row["example_index"]]
            outcome = _generation(
                plan,
                row,
                example,
                result,
                tokenizer,
                previous,
                bounds["worker_end"],
                bounds["deadline"],
            )
            require(
                result["ended_ns"] >= bounds["preparation_end"], "generation before preparation"
            )
            equal(result["started_ns"], start, "marker/result start")
            completion = schema.read_json(folder / "completed.json")
            _identity(completion, plan, row)
            equal(completion["result"], _record(folder / "result.json"), "completed result hash")
            completed = _clock(
                completion["completed_ns"],
                "completion",
                result["ended_ns"],
                bounds["worker_end"] if failure else min(cdeadline, bounds["worker_end"]),
            )
            equal(completion["started_ns"], start, "completion start")
            equal(completion["deadline_ns"], cdeadline, "completion deadline")
            ack_file = folder / "acknowledged.json"
            if ack_file.exists():
                ack = schema.read_json(ack_file)
                _identity(ack, plan, row)
                equal(ack["result"], _record(folder / "result.json"), "ACK result hash")
                equal(ack["completion"], _record(folder / "completed.json"), "ACK completion hash")
                previous = _clock(
                    ack["acknowledged_ns"],
                    "immutable ACK",
                    completed,
                    bounds["worker_end"] if failure else min(cdeadline, bounds["worker_end"]),
                )
                equal(ack["started_ns"], start, "ACK start")
                equal(ack["deadline_ns"], cdeadline, "ACK deadline")
                acknowledged.append(run_id)
            else:
                report["missing"].append(run_id + "/acknowledged.json")
                previous = completed
            outcome["generation_valid"] = True
            outcome["case_eligible"] = (
                failure is None and ack_file.exists() and run_id in worker["completed_runs"]
            )
            if ack_file.exists() and run_id not in worker["completed_runs"]:
                report["missing"].append(run_id + "/worker_ack")
            outcome["result"] = _record(folder / "result.json")
            report["retained_failed_records" if failure else "records"].append(outcome)
        equal(
            worker["completed_runs"],
            acknowledged[: len(worker["completed_runs"])],
            "immutable and worker ACKs",
        )
        require(
            len(acknowledged) - len(worker["completed_runs"]) <= 1,
            "multiple uncommitted immutable ACKs",
        )
        if worker["status"] == "complete":
            equal(manifest["completed_runs"], acknowledged, "terminal parent ACKs")
            equal(worker["completed_runs"], acknowledged, "terminal worker ACKs")
        for name, value in (("controller", manifest), ("worker", worker)):
            report["failures"].extend({"scope": name, **f} for f in value["failures"])
            if value["status"] != "complete" or "ended_ns" not in value:
                report["missing"].append(name + "/terminal_success")
        if "request_cleanup" in worker:
            _empty(worker["request_cleanup"])
        else:
            report["missing"].append("worker/request_cleanup")
        for key in ("teardown_after_workspace_release",):
            if key not in worker:
                report["missing"].append("worker/" + key)
            else:
                equal(
                    worker[key], {"allocated_bytes": 0, "reserved_bytes": 0}, "final memory cleanup"
                )
        if bounds["launch"]["returncode"] != 0:
            report["failures"].append(
                {
                    "scope": "worker",
                    "type": "WorkerExit",
                    "message": str(bounds["launch"]["returncode"]),
                }
            )
        if bounds["controller_end"] > bounds["deadline"]:
            report["failures"].append(
                {"scope": "controller", "type": "TimeoutError", "message": "global deadline"}
            )
        eligible = [r for r in report["records"] if r["case_eligible"]]
        evaluation = [r for r in eligible if r["phase"] == "evaluation"]
        feasibility = [r for r in eligible if r["phase"] == "feasibility"]
        report["counts"] = {
            "validated_generations": len(report["records"])
            + len(report["retained_failed_records"]),
            "eligible_evaluation": len(evaluation),
            "eligible_feasibility": len(feasibility),
            "correct_known": sum(r["correct"] for r in evaluation),
            "unparseable_known": sum(r["parser_status"] != "parsed" for r in evaluation),
            "truncated_known": sum(r["truncated"] for r in evaluation),
        }
        complete = (
            len(evaluation) == 64
            and len(feasibility) == 2
            and not report["missing"]
            and not report["failures"]
        )
        report["complete"] = complete
        report["evidence_status"] = "complete" if complete else "incomplete"
        report["decision"] = (
            "failed" if report["failures"] else "complete" if complete else "inconclusive"
        )
        if complete:
            report["accuracy"] = {
                "correct": sum(r["correct"] for r in evaluation),
                "total": 64,
                "fraction": sum(r["correct"] for r in evaluation) / 64,
            }
            report["token_summary"] = {
                "prompt_tokens": sum(r["prompt_count"] for r in evaluation),
                "output_tokens": sum(r["output_count"] for r in evaluation),
                "decode_outputs": sum(r["decode_outputs"] for r in evaluation),
                "decode_mean_depth": 4 if any(r["decode_outputs"] for r in evaluation) else None,
            }
    except _MissingStartup as stopped:
        report["missing"].extend(stopped.missing)
        report["missing"].extend(r["run_id"] for r in plan["execution_order"])
        report["failures"].extend(stopped.failures)
        report["evidence_status"] = "incomplete"
        report["decision"] = "failed" if stopped.failures else "inconclusive"
        report["counts"] = {
            "validated_generations": 0,
            "eligible_evaluation": 0,
            "eligible_feasibility": 0,
            "correct_known": 0,
            "unparseable_known": 0,
            "truncated_known": 0,
        }
    except (ValueError, TypeError, KeyError, OSError, IndexError, AttributeError) as error:
        report["errors"].append(type(error).__name__ + ": " + str(error))
    return report


def build_report(output_dir, *, tokenizer_path=None):
    root = Path(output_dir)
    return score_records(schema.read_json(root / "plan.json"), root, tokenizer_path=tokenizer_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path")
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), "score output must be new; preserve prior reports")
    require(
        not output.resolve().is_relative_to(Path(args.run_dir).resolve()),
        "write score outside raw evidence",
    )
    report = build_report(args.run_dir, tokenizer_path=args.tokenizer_path)
    require(
        len(json.dumps(report, allow_nan=False).encode()) <= 16 * 1024**2,
        "offline score output exceeds 16MiB cap",
    )
    schema.write_json(output, report)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "evidence_status": report["evidence_status"],
                "decision": report["decision"],
                "counts": report.get("counts"),
                "errors": report["errors"],
            }
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
