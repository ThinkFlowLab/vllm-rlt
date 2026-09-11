"""One bounded natural-EOS FP32 quality pass; no delivery-performance claim."""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

SAMPLING = {
    "max_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "seed": 0,
    "min_loops": 4,
    "max_loops": 4,
    "exit_threshold": 1.0,
    "ignore_eos": False,
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_record(path):
    data = Path(path).read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def failure(error):
    return {"type": type(error).__name__, "message": str(error)[:2000]}


def check_deadline(deadline_ns):
    if time.perf_counter_ns() >= deadline_ns:
        raise TimeoutError("quality case/global deadline exhausted")


def artifact_usage(root, limits):
    total, cases = 0, {}
    for path in Path(root).rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("quality artifacts must be ordinary files/directories")
        if path.is_file():
            size = path.stat().st_size
            total += size
            parts = path.relative_to(root).parts
            if len(parts) >= 3 and parts[0] == "runs":
                cases[parts[1]] = cases.get(parts[1], 0) + size
    if total > limits["artifact_bytes_max"] or any(
        size > limits["case_bytes_max"] for size in cases.values()
    ):
        raise RuntimeError("quality artifact byte budget exceeded")
    return {"total_bytes": total, "case_bytes": cases}


def request_state(engine):
    return {
        "requests": len(engine.scheduler.requests),
        "queued": sum(len(queue) for queue in engine.scheduler.queues.values()),
        "allocated_pages": len(engine.cache_manager._allocations),
        "used_pages": engine.cache_manager.num_used_blocks,
    }


def require_empty(engine):
    engine.cache_manager._require_usable()
    observed = request_state(engine)
    if any(observed.values()):
        raise ValueError(f"quality request cleanup is not empty: {observed}")
    return observed


def synchronize(engine):
    # CPU tiny-model tests exercise the real engine without any CUDA discovery.
    if engine.cache_manager.device.type == "cuda":
        import torch

        torch.cuda.synchronize()


@contextmanager
def _logit_checks(model):
    counts = {"lm_head": 0}

    def check(module, inputs, output):
        counts["lm_head"] += 1
        if not bool(output.isfinite().all().item()):
            raise ValueError("nonfinite actual lm_head logits before sampling")

    handle = model.lm_head.register_forward_hook(check)
    try:
        yield counts
    finally:
        handle.remove()


def generate_example(engine, row, *, deadline_ns, max_steps):
    """Drain one actual request, checking every returned cumulative prefix."""
    from vllm_lt.benchmarks.profile import finite_checks
    from vllm_lt.sampling_params import SamplingParams

    prompt = list(row["prompt_token_ids"])
    result = {
        "output_token_ids": [],
        "exit_depths": [],
        "finish_reason": None,
        "truncated": False,
        "counts": {
            "steps": 0,
            "stage_counts": dict.fromkeys(("prefill", "prelude", "recurrent", "coda"), 0),
        },
        "finite_checks": {"lm_head": 0, "recurrent": 0, "coda": 0},
        "before_request": require_empty(engine),
        "cleanup": None,
        "failures": [],
    }
    original_error = None
    try:
        if engine.model_runner._persistent is not None:
            raise ValueError("quality requires the compact synchronous execution path")
        if engine.model.config.eos_token_id != 0 or engine.model.config.total_ut_steps != 4:
            raise ValueError("quality requires EOS0 and exactly four model loops")
        if engine.scheduler.config.max_num_batched_tokens != 128:
            raise ValueError("quality requires 128-token prefill chunks")
        if max_steps != math.ceil(len(prompt) / 128) + 1531:
            raise ValueError("quality max_steps differs from frozen 256-output bound")
        with (
            finite_checks(engine.model, row["phase"] == "feasibility") as full,
            _logit_checks(engine.model) as logits,
        ):
            try:
                check_deadline(deadline_ns)
                engine.add_request(row["run_id"], prompt, SamplingParams(**SAMPLING))
                finished = False
                while engine.has_unfinished_requests():
                    check_deadline(deadline_ns)
                    if result["counts"]["steps"] >= max_steps:
                        raise TimeoutError("quality scheduler step budget exhausted")
                    outputs = engine.step()
                    result["counts"]["steps"] += 1
                    batch = engine.last_schedule
                    if batch is None or len(batch.items) != 1:
                        raise ValueError("quality scheduler must execute exactly one request")
                    if batch.items[0].request.request_id != row["run_id"]:
                        raise ValueError("quality scheduler request identity changed")
                    stage = batch.stage.value
                    result["counts"]["stage_counts"][stage] += 1
                    if len(outputs) != (1 if stage == "coda" else 0):
                        raise ValueError("quality stage/output count mismatch")
                    for output in outputs:
                        prior = result["output_token_ids"]
                        depths = result["exit_depths"]
                        if (
                            finished
                            or output.request_id != row["run_id"]
                            or output.prompt_token_ids != prompt
                            or len(output.token_ids) != len(prior) + 1
                            or output.token_ids[:-1] != prior
                            or output.exit_depths[:-1] != depths
                            or output.exit_depths != [4] * len(output.token_ids)
                            or any(
                                type(t) is not int or not 0 <= t < engine.model.config.vocab_size
                                for t in output.token_ids
                            )
                        ):
                            raise ValueError("quality output prefix, depth, or identity changed")
                        ids = list(output.token_ids)
                        expected_reason = (
                            "stop" if ids[-1] == 0 else ("length" if len(ids) == 256 else None)
                        )
                        if (
                            0 in ids[:-1]
                            or len(ids) > 256
                            or output.finish_reason != expected_reason
                            or output.finished != (expected_reason is not None)
                        ):
                            raise ValueError(
                                "quality natural EOS/length completion invariant failed"
                            )
                        result.update(
                            output_token_ids=ids,
                            exit_depths=list(output.exit_depths),
                            finish_reason=expected_reason,
                            truncated=expected_reason == "length",
                        )
                        finished = output.finished
                synchronize(engine)
                check_deadline(deadline_ns)
                if not finished:
                    raise ValueError("quality engine ended without a terminal output")
                count = len(result["output_token_ids"])
                expected = {
                    "prefill": math.ceil(len(prompt) / 128),
                    "prelude": count - 1,
                    "recurrent": 4 * (count - 1),
                    "coda": count,
                }
                if result["counts"] != {"steps": sum(expected.values()), "stage_counts": expected}:
                    raise ValueError("quality scheduler work differs from actual output history")
                if logits["lm_head"] != count:
                    raise ValueError(
                        "quality logits must be checked exactly once per actual output"
                    )
                if row["phase"] == "feasibility" and full != {
                    "recurrent": 4 * (expected["prefill"] + count - 1),
                    "coda": count,
                }:
                    raise ValueError("quality full feasibility check coverage differs")
            finally:
                result["finite_checks"] = {**full, **logits}
    except BaseException as error:
        original_error = error
        result["failures"].append(failure(error))
    finally:
        try:
            synchronize(engine)
            for request_id in list(engine.scheduler.requests):
                engine.abort_request(request_id)
            engine.last_schedule = None
            result["cleanup"] = require_empty(engine)
        except BaseException as error:
            result["failures"].append(failure(error))
            if original_error is None:
                original_error = error
    if original_error is not None:
        original_error.q2_partial_result = result
        raise original_error
    try:
        check_deadline(deadline_ns)
    except BaseException as error:
        result["failures"].append(failure(error))
        error.q2_partial_result = result
        raise
    return result


def execute_example(plan, row, engine, tokenizer, *, started_ns, deadline_ns):
    from vllm_lt.benchmarks.q2_quality_data import decode_output, parse_answer

    example = plan["selection"][row["phase"]][row["example_index"]]
    if example["source_id"] != row["example_id"]:
        raise ValueError("quality row and selected example differ")
    base = {
        "schema_version": 1,
        "artifact_type": "q2_quality_result",
        "run": row,
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "started_ns": started_ns,
        "deadline_ns": deadline_ns,
        "example_id": row["example_id"],
        "prompt_token_ids": example["prompt_token_ids"],
        "prompt_sha256": example["prompt_sha256"],
        "parsed_reference": example["parsed_reference"],
    }
    try:
        base.update(
            generate_example(
                engine, {**example, **row}, deadline_ns=deadline_ns, max_steps=row["max_steps"]
            )
        )
        base.update(decode_output(tokenizer, base["output_token_ids"], base["finish_reason"]))
        base["parsed_answer"] = parse_answer(base["scoring_text"])
        base["correct"] = (
            base["parsed_answer"]["value"] is not None
            and base["parsed_answer"]["value"] == base["parsed_reference"]["value"]
        )
        check_deadline(deadline_ns)
        base.update(status="complete", ended_ns=time.perf_counter_ns())
        return base
    except BaseException as error:
        base.update(getattr(error, "q2_partial_result", {}))
        base.update(status="failed", ended_ns=time.perf_counter_ns())
        base.setdefault("failures", []).append(failure(error))
        error.q2_partial_result = base
        raise


def _schema():
    from vllm_lt.benchmarks import q2_quality_schema

    return q2_quality_schema


def _environment(plan):
    import torch

    from vllm_lt.benchmarks.ab_schema import RUNTIME_VARIABLES, affinity_snapshot
    from vllm_lt.benchmarks.runner import environment

    observed = environment()
    controls = plan["contract"]["controls"]
    observed["runtime_environment"] = {name: os.environ.get(name) for name in RUNTIME_VARIABLES}
    observed["active_affinity"] = affinity_snapshot()
    observed["arithmetic"] = {
        name: getattr(torch.backends.cuda.matmul, name) for name in plan["contract"]["arithmetic"]
    }
    observed["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    if (
        observed["cuda_visible_devices"] != str(controls["gpu_ids"][0])
        or _schema().canonical_gpu_uuid(observed["gpu_uuid"])
        != _schema().canonical_gpu_uuid(controls["gpu_uuid"])
        or observed["cpu_affinity"] != controls["affinity"]["cpu_ids"]
        or observed["active_affinity"] != controls["affinity"]
        or observed["actual_torch_threads"]
        != {"intraop": controls["cpu_threads"], "interop": controls["interop_threads"]}
        or observed["arithmetic"] != plan["contract"]["arithmetic"]
        or observed["cudnn_allow_tf32"]
    ):
        raise ValueError("quality device/affinity/thread/arithmetic controls differ")
    return observed


def _engine(model, plan):
    from vllm_lt.config import CacheConfig, SchedulerConfig
    from vllm_lt.engine.llm_engine import LLMEngine

    settings = plan["contract"]["engine"]
    return LLMEngine(
        model,
        cache_config=CacheConfig(**settings["cache"]),
        scheduler_config=SchedulerConfig(**settings["scheduler"]),
        attention_backend=settings["attention_backend"],
    )


def _pool(engine):
    cache = engine.cache_manager
    return {
        "num_blocks": cache.num_blocks,
        "block_size": cache.block_size,
        "bytes_per_block": cache.bytes_per_block,
        "size_bytes": cache.num_blocks * cache.bytes_per_block,
        "key_data_ptr": cache.key_cache.data_ptr(),
        "value_data_ptr": cache.value_cache.data_ptr(),
        "dtype": str(cache.dtype),
        "device": str(cache.device),
        "backend": cache.backend,
    }


def _complete_case(root, marker, result, worker, limits):
    """Each immutable link is exported and bounded before the next acknowledgment."""
    run_dir = root / "runs" / marker["run"]["run_id"]
    write_json(run_dir / "result.json", result)
    artifact_usage(root, limits)
    check_deadline(marker["deadline_ns"])
    completion = {
        **marker,
        "status": "complete",
        "result": file_record(run_dir / "result.json"),
        "completed_ns": time.perf_counter_ns(),
    }
    write_json(run_dir / "completed.json", completion)
    artifact_usage(root, limits)
    check_deadline(marker["deadline_ns"])
    ack = {
        **completion,
        "completion": file_record(run_dir / "completed.json"),
        "acknowledged_ns": time.perf_counter_ns(),
    }
    write_json(run_dir / "acknowledged.json", ack)
    artifact_usage(root, limits)
    check_deadline(marker["deadline_ns"])
    worker["completed_runs"].append(marker["run"]["run_id"])
    write_json(root / "worker.json", worker)
    artifact_usage(root, limits)
    check_deadline(marker["deadline_ns"])
    write_json(root / "active-case.json", ack)
    check_deadline(marker["deadline_ns"])


def run_worker(plan, output_dir, deadline_ns):
    """One process, one FP32 load/pool, exactly the declared examples, no retries."""
    from tokenizers import Tokenizer

    from vllm_lt.benchmarks.runner import configure_process, load_model, release_device

    root, limits = Path(output_dir), plan["contract"]["limits"]
    started = time.perf_counter_ns()
    worker = {
        "schema_version": 1,
        "artifact_type": "q2_quality_worker",
        "plan_sha256": plan["plan_sha256"],
        "started_ns": started,
        "deadline_ns": deadline_ns,
        "status": "running",
        "completed_runs": [],
        "failures": [],
    }
    write_json(root / "worker.json", worker)
    engine = model = result = None
    device_started = environment_ready = False
    marker = None
    try:
        # The first case's 600 s includes verification, model loading and engine setup.
        first = plan["execution_order"][0]
        marker = {
            "schema_version": 1,
            "run": first,
            "plan_sha256": plan["plan_sha256"],
            "started_ns": started,
            "deadline_ns": min(deadline_ns, started + limits["case_timeout_s"] * 10**9),
        }
        (root / "runs" / first["run_id"]).mkdir(parents=True, exist_ok=False)
        write_json(root / "active-case.json", marker)
        write_json(root / "runs" / first["run_id"] / "started.json", marker)
        _schema().verify_quality_plan(plan)
        worker["source_probe"] = _schema().source_probe()
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(
            plan["contract"]["controls"]["gpu_ids"][0]
        ):
            raise ValueError("quality requires exact scheduler-assigned device before CUDA access")
        configure_process(plan["contract"])
        device_started = True
        worker["environment"] = _environment(plan)
        environment_ready = True
        check_deadline(marker["deadline_ns"])
        preparation = {"started_ns": time.perf_counter_ns()}
        worker["preparation"] = preparation
        write_json(root / "worker.json", worker)
        model = load_model(
            {
                **plan,
                "workload_stats": {
                    "quality": {"pool_bytes": plan["resource_estimates"]["native_pool_bytes"]}
                },
            },
            preparation,
        )
        engine = _engine(model, plan)
        preparation["native_pool"] = _pool(engine)
        if preparation["native_pool"]["size_bytes"] != 1207959552:
            raise ValueError("quality native pool differs from frozen 1152 MiB geometry")
        tokenizer = Tokenizer.from_file(str(Path(plan["model_path"]) / "tokenizer.json"))
        synchronize(engine)
        require_empty(engine)
        preparation["ended_ns"] = time.perf_counter_ns()
        check_deadline(marker["deadline_ns"])
        write_json(root / "worker.json", worker)
        for index, row in enumerate(plan["execution_order"]):
            result = None
            if index:
                started = time.perf_counter_ns()
                marker = {
                    "schema_version": 1,
                    "run": row,
                    "plan_sha256": plan["plan_sha256"],
                    "started_ns": started,
                    "deadline_ns": min(deadline_ns, started + limits["case_timeout_s"] * 10**9),
                }
                (root / "runs" / row["run_id"]).mkdir(parents=True, exist_ok=False)
                write_json(root / "active-case.json", marker)
                write_json(root / "runs" / row["run_id"] / "started.json", marker)
            check_deadline(marker["deadline_ns"])
            if _pool(engine) != preparation["native_pool"]:
                raise ValueError("quality pool identity changed between examples")
            result = execute_example(
                plan,
                row,
                engine,
                tokenizer,
                started_ns=marker["started_ns"],
                deadline_ns=marker["deadline_ns"],
            )
            if result["status"] != "complete":
                raise RuntimeError("quality example did not complete")
            _complete_case(root, marker, result, worker, limits)
        worker["status"] = "complete"
    except BaseException as error:
        worker["status"] = "failed"
        worker["failures"].append(failure(error))
        traceback.print_exc()
        try:
            if marker is not None:
                run_dir = root / "runs" / marker["run"]["run_id"]
                if (run_dir / "completed.json").exists():
                    write_json(
                        run_dir / "failure.json",
                        {
                            **marker,
                            "failure": failure(error),
                            "occurred_ns": time.perf_counter_ns(),
                        },
                    )
                else:
                    partial = getattr(error, "q2_partial_result", result)
                    if partial is not None:
                        partial.update(status="failed")
                        write_json(run_dir / "result.json", partial)
        except BaseException as export_error:
            worker["failures"].append(failure(export_error))
    finally:
        try:
            if engine is not None:
                synchronize(engine)
                for request_id in list(engine.scheduler.requests):
                    engine.abort_request(request_id)
                engine.last_schedule = None
                worker["request_cleanup"] = require_empty(engine)
        except BaseException as error:
            worker["status"] = "failed"
            worker["failures"].append(failure(error))
        result = engine = model = None
        try:
            import torch

            if device_started and (environment_ready or torch.cuda.is_initialized()):
                release_device(worker)
                if worker.get("cleanup_errors") or worker.get(
                    "teardown_after_workspace_release"
                ) != {"allocated_bytes": 0, "reserved_bytes": 0}:
                    raise RuntimeError("quality worker CUDA cleanup not confirmed zero")
            else:
                worker["device_initialization"] = (
                    "not_initialized" if device_started else "not_started"
                )
            check_deadline(deadline_ns)
        except BaseException as error:
            worker["status"] = "failed"
            worker["failures"].append(failure(error))
        worker["ended_ns"] = time.perf_counter_ns()
        write_json(root / "worker.json", worker)
        try:
            artifact_usage(root, limits)
            check_deadline(deadline_ns)
        except BaseException as error:
            worker["status"] = "failed"
            worker["failures"].append(failure(error))
            write_json(root / "worker.json", worker)
    return worker


def _copy_inputs(plan, root):
    """Keep scoring portable without copying checkpoint tensor files."""
    destination = root / "inputs"
    destination.mkdir()
    sources = {name + ".json": item["file"] for name, item in plan["inputs"].items()}
    sources["dataset-manifest.json"] = sources.pop("dataset.json")
    dataset = plan["inputs"]["dataset"]["contents"]
    for source_name, local_name in (
        ("normalized", "normalized.jsonl"),
        ("raw", "raw.parquet"),
        ("readme", "README.md"),
    ):
        sources[local_name] = dataset[source_name]
    sources["tokenizer.json"] = plan["selection"]["tokenizer"]["file"]
    sources["q1-qualification.json"] = plan["inputs"]["prerequisite"]["contents"]["qualification"]
    for name, record in sources.items():
        source = Path(record["path"])
        expected = {key: record[key] for key in ("sha256", "size_bytes")}
        if source.is_symlink() or not source.is_file() or file_record(source) != expected:
            raise ValueError("quality input source changed: " + name)
        shutil.copyfile(source, destination / name)
        if file_record(destination / name) != expected:
            raise ValueError("quality input copy differs: " + name)


def _stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def run_quality(plan, output_dir):
    """Own only this worker's process group; watchdog includes export and cleanup."""
    started = time.perf_counter_ns()
    _schema().validate_quality_plan(plan)
    limits, root = plan["contract"]["limits"], Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    deadline = started + limits["total_timeout_s"] * 10**9
    manifest = {
        "schema_version": 1,
        "artifact_type": "q2_quality_manifest",
        "plan_sha256": plan["plan_sha256"],
        "started_ns": started,
        "deadline_ns": deadline,
        "status": "running",
        "completed_runs": [],
        "failures": [],
    }
    write_json(root / "manifest.json", manifest)
    child, handlers = None, {}

    def interrupt(signum, frame):
        raise RuntimeError(f"quality controller interrupted by signal {signum}")

    def retain_prefix():
        if not (root / "worker.json").exists():
            return None
        worker = read_json(root / "worker.json")
        prefix = worker["completed_runs"]
        expected = [row["run_id"] for row in plan["execution_order"]]
        if (
            prefix != expected[: len(prefix)]
            or prefix[: len(manifest["completed_runs"])] != manifest["completed_runs"]
        ):
            raise ValueError("quality worker ACK ledger is not a monotonic planned prefix")
        manifest["completed_runs"] = list(prefix)
        write_json(root / "manifest.json", manifest)
        return worker

    try:
        _schema().verify_quality_plan(plan)
        _copy_inputs(plan, root)
        artifact_usage(root, limits)
        check_deadline(deadline)
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupt)
        command = [
            plan["interpreter"],
            "-m",
            "vllm_lt.benchmarks.q2_quality",
            "worker",
            "--plan",
            str(root / "plan.json"),
            "--output",
            str(root),
            "--deadline-ns",
            str(deadline),
        ]
        manifest["launch"] = {
            "command": command,
            "started_ns": time.perf_counter_ns(),
            "ended_ns": None,
            "returncode": None,
        }
        write_json(root / "manifest.json", manifest)
        with (root / "worker.log").open("w") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            manifest["launch"]["pid"] = child.pid
            write_json(root / "manifest.json", manifest)
            while child.poll() is None:
                check_deadline(deadline)
                if (root / "active-case.json").exists():
                    active = read_json(root / "active-case.json")
                    if "acknowledged_ns" not in active:
                        check_deadline(active["deadline_ns"])
                artifact_usage(root, limits)
                retain_prefix()
                time.sleep(0.2)
        worker = retain_prefix()
        if (
            child.returncode != 0
            or worker is None
            or worker["status"] != "complete"
            or worker.get("failures")
            or len(manifest["completed_runs"]) != limits["executions"]
            or worker.get("teardown_after_workspace_release")
            != {"allocated_bytes": 0, "reserved_bytes": 0}
        ):
            raise RuntimeError("quality worker did not complete the frozen protocol and cleanup")
        check_deadline(deadline)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failures"].append(failure(error))
        traceback.print_exc()
    finally:
        try:
            if child is not None:
                _stop_child(child)
                manifest["launch"].update(
                    ended_ns=time.perf_counter_ns(), returncode=child.returncode
                )
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["failures"].append(failure(error))
        try:
            retain_prefix()
            artifact_usage(root, limits)
            check_deadline(deadline)
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["failures"].append(failure(error))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        manifest["ended_ns"] = time.perf_counter_ns()
        write_json(root / "manifest.json", manifest)
        try:
            artifact_usage(root, limits)
            check_deadline(deadline)
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["failures"].append(failure(error))
            write_json(root / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-quality")
    for name in ("contract", "selection", "dataset-manifest", "prerequisite", "model", "gpu-uuid"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--gpu-ids", type=int, nargs=1, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("run-quality", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "worker":
            command.add_argument("--deadline-ns", type=int, required=True)
    score = sub.add_parser("score-quality")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-quality":
        plan = _schema().make_quality_plan(
            args.contract,
            args.selection,
            args.model,
            dataset_manifest_path=args.dataset_manifest,
            prerequisite_path=args.prerequisite,
            gpu_ids=args.gpu_ids,
            gpu_uuid=args.gpu_uuid,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "device_work": False}))
        return 0
    if args.command == "score-quality":
        from vllm_lt.benchmarks.q2_quality_score import build_report

        if args.output.exists() or args.output.resolve().is_relative_to(args.run_dir.resolve()):
            raise ValueError("quality score output must be fresh and outside raw evidence")
        report = build_report(args.run_dir)
        write_json(args.output, report)
        print(json.dumps({"evidence_status": report.get("evidence_status")}))
        return 0 if report.get("complete") else 1
    plan = read_json(args.plan)
    result = (
        run_worker(plan, args.output, args.deadline_ns)
        if args.command == "worker"
        else run_quality(plan, args.output)
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
