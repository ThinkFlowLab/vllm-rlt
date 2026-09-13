"""Frozen graph A/B controller; correctness precedes the bounded M1 timing cells."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmarks.capture.schema import (
    execution_view,
    make_plan,
    source_probe,
    validate_plan,
    verify_plan,
)
from vllm_lt.benchmarks import ab
from vllm_lt.benchmarks.ab_schema import RUNTIME_VARIABLES, affinity_snapshot, equal, require
from vllm_lt.benchmarks.runner import write_json
from vllm_lt.benchmarks.schema import read_json

artifact_usage = ab.artifact_usage
audit_worker_controls = ab.audit_worker_controls


def run_worker(plan, *, worker_id, output_dir, deadline_ns):
    from benchmarks.capture.runtime import ExecutionAdapter
    from benchmarks.capture.validation import run_model_rows
    from vllm_lt.benchmarks import runner

    start = time.perf_counter_ns()
    validate_plan(plan)
    worker = next(row for row in plan["workers"] if row["worker_id"] == worker_id)
    implementation = worker["implementation_id"]
    verify_plan(plan, implementation_id=implementation)
    require(time.perf_counter_ns() < deadline_ns, "worker deadline exhausted before device use")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(plan["contract"]["controls"]["gpu_ids"][0]),
        "scheduler visibility differs from frozen physical GPU ID",
    )
    output_dir = Path(output_dir).resolve()
    folder = output_dir / "workers" / worker_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "m3_capture_worker_manifest",
        **worker,
        "plan_sha256": plan["plan_sha256"],
        "source": plan["implementations"][implementation]["source"],
        "harness_sha256": plan["harness"]["sha256"],
        "affinity": affinity_snapshot(),
        "runtime_environment": {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
        "status": "running",
        "passed": False,
        "failures": [],
        "completed_executions": [],
        "started_ns": start,
        "deadline_ns": deadline_ns,
        "model_loads": 0,
        "preparation": {
            "verification_ns": time.perf_counter_ns() - start,
            "downloads": "none; same prepared checkpoint",
            "warmup": "explicit excluded rows only",
            "compilation_ns": None,
        },
    }
    write_json(manifest_path, manifest)
    model, ready = None, False
    try:
        view = execution_view(plan)
        runner.configure_process(view["contract"])
        manifest["environment"] = runner.environment()
        ready = True
        arithmetic = runner.torch.backends.cuda.matmul
        manifest["environment"]["arithmetic"] = {
            "allow_tf32": arithmetic.allow_tf32,
            "cudnn_allow_tf32": runner.torch.backends.cudnn.allow_tf32,
            **{
                key: getattr(arithmetic, key)
                for key in (
                    "allow_bf16_reduced_precision_reduction",
                    "allow_fp16_reduced_precision_reduction",
                )
            },
        }
        model = runner.load_model(view, manifest)
        manifest["model_loads"] = 1
        write_json(manifest_path, manifest)
        rows = [row for row in plan["execution_order"] if row["worker_id"] == worker_id]

        def completed(run_id, result):
            manifest["completed_executions"].append(run_id)
            manifest["artifact_usage"] = artifact_usage(output_dir, plan["contract"]["limits"])
            write_json(manifest_path, manifest)

        if worker_id.startswith("N-"):

            def after_case(case, value):
                completed(case["case_id"], value)

            numerical = run_model_rows(
                model, plan, implementation, output_dir, deadline_ns, after_case=after_case
            )
            manifest["numerical"] = numerical
            # The shared validator retains a fully recorded failed case before
            # stopping, without invoking the success-only after_case callback.
            for execution_id in worker["execution_ids"][len(manifest["completed_executions"]) :]:
                if execution_id not in numerical["completed_cases"]:
                    break
                completed(execution_id, None)
            require(
                numerical["complete"] and numerical["passed"],
                f"numerical prerequisite failed: {numerical['errors']}",
            )
        runner.run_loaded_rows(
            model,
            view,
            [row for row in rows if row["kind"] == "benchmark"],
            output_dir=output_dir,
            deadline=deadline_ns,
            record_completed=completed,
            execution_adapter=ExecutionAdapter(
                implementation_id=implementation, graph_limits=plan["graph_limits"]
            ),
        )
        equal(
            manifest["completed_executions"], worker["execution_ids"], "complete worker row order"
        )
        manifest["status"], manifest["passed"] = "complete", True
    except BaseException as exc:
        manifest["status"] = (
            "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        model = None
        if ready or runner.torch.cuda.is_initialized():
            try:
                runner.release_device(manifest)
            except Exception as exc:
                manifest["status"], manifest["passed"] = "failed", False
                manifest["failures"].append({"type": "cleanup", "message": str(exc)})
        manifest["ended_ns"] = time.perf_counter_ns()
        if manifest["ended_ns"] >= deadline_ns:
            manifest["status"], manifest["passed"] = "incomplete", False
            manifest["failures"].append(
                {"type": "deadline", "message": "global deadline exhausted"}
            )
        write_json(manifest_path, manifest)
    return manifest


def audit_correctness(output_dir, plan):
    from benchmarks.capture.validation import audit_model_rows

    numerical = audit_model_rows(output_dir, plan)
    return {key: numerical[key] for key in ("complete", "passed")} | {"numerical": numerical}


active_case_deadline = ab.active_case_deadline


def run_ab(plan, *, output_dir):
    from functools import partial

    return ab._run_ab(
        plan,
        output_dir=output_dir,
        verify=verify_plan,
        audit=audit_correctness,
        artifact_type="m3_capture_manifest",
        launch_worker=partial(
            ab._launch_worker, module="benchmarks.capture", active_deadline=active_case_deadline
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-probe")
    probe = commands.add_parser("probe")
    for name in ("baseline-root", "candidate-root", "contract", "model-path", "output"):
        probe.add_argument("--" + name, required=True, type=Path)
    probe.add_argument("--gpu-id", required=True, type=int)
    for name in ("run", "worker"):
        child = commands.add_parser(name)
        child.add_argument("--plan", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
        if name == "worker":
            child.add_argument("--worker-id", required=True)
            child.add_argument("--deadline-ns", required=True, type=int)
    report = commands.add_parser("report")
    report.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "source-probe":
            print(json.dumps(source_probe(), allow_nan=False))
            return 0
        if args.command == "probe":
            require(not args.output.exists(), "probe output already exists")
            plan = make_plan(
                baseline_root=args.baseline_root,
                candidate_root=args.candidate_root,
                contract_path=args.contract,
                model_path=args.model_path,
                gpu_ids=[args.gpu_id],
                affinity=affinity_snapshot(),
            )
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "plan.json", plan)
            print(
                json.dumps(
                    {"plan": str(args.output / "plan.json"), "plan_sha256": plan["plan_sha256"]}
                )
            )
            return 0
        if args.command == "report":
            from benchmarks.capture.report import write_report

            result = write_report(args.run_dir)
            print(json.dumps({key: result[key] for key in ("evidence_status", "decision")}))
            return 0 if result["decision"] == "passed" else 1
        plan = read_json(args.plan)
        result = (
            run_worker(
                plan, worker_id=args.worker_id, output_dir=args.output, deadline_ns=args.deadline_ns
            )
            if args.command == "worker"
            else run_ab(plan, output_dir=args.output)
        )
        return 0 if result["status"] == "complete" else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"invalid or incomplete experiment: {exc}", file=sys.stderr)
        return 2
