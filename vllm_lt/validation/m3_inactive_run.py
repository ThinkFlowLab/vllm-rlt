"""Two owned eager workers for the frozen inactive-row correctness protocol."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from vllm_lt.benchmarks import ab, runner
from vllm_lt.benchmarks.ab_schema import RUNTIME_VARIABLES, affinity_snapshot, equal, require
from vllm_lt.benchmarks.schema import read_json

from .m3_inactive import run_model_rows
from .m3_inactive_schema import loading_view, make_plan, validate_plan, verify_plan

write_json = runner.write_json


def controls_view(plan):
    return {**plan, "benchmark_contract": plan["numerical"]["contract"]}


def active_deadline(output_dir, worker):
    return _active_deadline(
        output_dir,
        worker,
        check_directory="kernels",
        check_ids=[value for value in worker["execution_ids"] if value.startswith("K-")],
    )


def _active_deadline(output_dir, worker, *, check_directory, check_ids):
    allowed = set(worker["execution_ids"])
    candidates = []
    ledger_path = output_dir / "numerical/ledger.json"
    if ledger_path.exists():
        active = read_json(ledger_path).get("active_case")
        if active is not None:
            require(active["case_id"] in allowed, "unplanned active numerical case")
            require(
                type(active["started_ns"]) is int
                and type(active["deadline_ns"]) is int
                and 0 < active["started_ns"] < active["deadline_ns"]
                and active["deadline_ns"] - active["started_ns"] <= 600 * 10**9,
                "invalid numerical watchdog interval",
            )
            candidates.append(active["deadline_ns"])
    for evaluation_id in check_ids:
        folder = output_dir / check_directory / "evaluations" / evaluation_id
        marker_path = folder / "started.json"
        if marker_path.exists() and not (folder / "result.json").exists():
            marker = read_json(marker_path)
            require(
                marker["evaluation"]["evaluation_id"] == evaluation_id,
                "active kernel marker identity differs",
            )
            start, end = marker["started_ns"], marker["deadline_ns"]
            require(
                type(start) is int
                and type(end) is int
                and 0 < start < end
                and end - start <= 600 * 10**9,
                "invalid kernel watchdog interval",
            )
            candidates.append(end)
    return min(candidates) if candidates else None


def run_worker(plan, *, worker_id, output_dir, deadline_ns):
    from .m3_inactive_kernels import run_kernel_evaluation

    def checks(plan, implementation, output_dir, deadline_ns):
        for evaluation in plan["kernels"]["execution_order"]:
            if evaluation["implementation_id"] == implementation:
                yield (
                    evaluation["evaluation_id"],
                    run_kernel_evaluation(
                        plan["kernels"],
                        evaluation["evaluation_id"],
                        output_dir / "kernels",
                        device="cuda",
                        deadline_ns=deadline_ns,
                    ),
                )

    return _run_worker(
        plan,
        worker_id=worker_id,
        output_dir=output_dir,
        deadline_ns=deadline_ns,
        validate=validate_plan,
        verify=verify_plan,
        model_rows=run_model_rows,
        checks=checks,
        artifact_type="m3_inactive_worker_manifest",
    )


def _run_worker(
    plan,
    *,
    worker_id,
    output_dir,
    deadline_ns,
    validate,
    verify,
    model_rows,
    checks,
    artifact_type,
):
    """Owned model lifetime shared by the two bounded eager correctness protocols."""
    start = time.perf_counter_ns()
    validate(plan)
    worker = next(row for row in plan["workers"] if row["worker_id"] == worker_id)
    implementation = worker["implementation_id"]
    verify(plan, implementation_id=implementation)
    require(time.perf_counter_ns() < deadline_ns, "global deadline before device use")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(plan["contract"]["controls"]["gpu_ids"][0]),
        "worker requires frozen scheduler visibility",
    )
    output_dir = Path(output_dir).resolve()
    folder = output_dir / "workers" / worker_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        **worker,
        "plan_sha256": plan["plan_sha256"],
        "source": plan["implementations"][implementation]["source"],
        "harness_sha256": plan["harness"]["sha256"],
        "affinity": affinity_snapshot(),
        "runtime_environment": {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
        "started_ns": start,
        "deadline_ns": deadline_ns,
        "status": "running",
        "passed": False,
        "model_loads": 0,
        "completed_executions": [],
        "failures": [],
        "preparation": {
            "verification_ns": time.perf_counter_ns() - start,
            "downloads": "none",
            "warmup": "none; two excluded feasibility cases only",
        },
    }
    write_json(manifest_path, manifest)
    model, ready = None, False
    try:
        view = loading_view(plan)
        runner.configure_process(view["contract"])
        manifest["environment"] = runner.environment()
        ready = True
        arithmetic = runner.torch.backends.cuda.matmul
        manifest["environment"]["arithmetic"] = {
            **{
                name: getattr(arithmetic, name)
                for name in (
                    "allow_tf32",
                    "allow_bf16_reduced_precision_reduction",
                    "allow_fp16_reduced_precision_reduction",
                )
            },
            "cudnn_allow_tf32": runner.torch.backends.cudnn.allow_tf32,
        }
        model = runner.load_model(view, manifest)
        manifest["model_loads"] = 1
        write_json(manifest_path, manifest)

        def after_case(case, value):
            manifest["completed_executions"].append(case["case_id"])
            write_json(manifest_path, manifest)
            if case["phase"] == "feasibility":
                for evaluation_id, result in checks(plan, implementation, output_dir, deadline_ns):
                    require(
                        result["status"] == "complete" and result["passed"],
                        "held-input correctness prerequisite failed",
                    )
                    manifest["completed_executions"].append(evaluation_id)
                    manifest["artifact_usage"] = ab.artifact_usage(
                        output_dir, plan["contract"]["limits"]
                    )
                    write_json(manifest_path, manifest)
            require(
                time.perf_counter_ns() < deadline_ns, "global deadline after case/kernel exports"
            )

        numerical = model_rows(
            model, plan, implementation, output_dir, deadline_ns, after_case=after_case
        )
        manifest["numerical"] = numerical
        # Retain a numerically failed but fully recorded last case in the completed prefix.
        for execution_id in worker["execution_ids"][len(manifest["completed_executions"]) :]:
            if execution_id not in numerical["completed_cases"]:
                break
            manifest["completed_executions"].append(execution_id)
        require(
            numerical["complete"] and numerical["passed"],
            f"model prerequisite failed: {numerical['errors']}",
        )
        equal(
            manifest["completed_executions"], worker["execution_ids"], "worker execution coverage"
        )
        manifest["status"], manifest["passed"] = "complete", True
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = (
            "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
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
            manifest["failures"].append({"type": "deadline", "message": "global budget exhausted"})
        write_json(manifest_path, manifest)
    return manifest


def run(plan, *, output_dir):
    return _run_workers(
        plan,
        output_dir=output_dir,
        verify=verify_plan,
        module="vllm_lt.validation.m3_inactive_run",
        watchdog=active_deadline,
        artifact_type="m3_inactive_manifest",
    )


def _run_workers(plan, *, output_dir, verify, module, watchdog, artifact_type):
    """Run the declared two-worker prefix with the shared owned-process watchdog."""
    start = time.perf_counter_ns()
    verify(plan)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(plan["contract"]["controls"]["gpu_ids"][0]),
        "controller requires frozen scheduler visibility",
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "workers").mkdir()
    write_json(output_dir / "plan.json", plan)
    deadline = start + 3600 * 10**9
    manifest = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "started_ns": start,
        "deadline_ns": deadline,
        "workers": [],
        "completed_workers": [],
        "completed_executions": [],
        "failures": [],
    }
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    previous = None
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise KeyboardInterrupt("inactive-row controller received SIGTERM")

    signal.signal(signal.SIGTERM, stop)
    try:
        for worker in plan["workers"]:
            require(time.perf_counter_ns() < deadline, "global deadline exhausted")
            launch = {
                "worker_id": worker["worker_id"],
                "launched_ns": time.perf_counter_ns(),
                "returned_ns": None,
                "exit_code": None,
            }
            manifest["workers"].append(launch)
            write_json(path, manifest)
            try:
                launch["exit_code"] = ab._launch_worker(
                    plan,
                    worker,
                    output_dir,
                    deadline,
                    module=module,
                    active_deadline=watchdog,
                )
            finally:
                launch["returned_ns"] = time.perf_counter_ns()
            child = read_json(output_dir / "workers" / worker["worker_id"] / "manifest.json")
            equal(
                child["completed_executions"],
                worker["execution_ids"][: len(child["completed_executions"])],
                "worker completed prefix",
            )
            manifest["completed_executions"].extend(child["completed_executions"])
            require(
                launch["exit_code"] == 0
                and child["status"] == "complete"
                and child["passed"] is True,
                "unsuccessful worker; no subsequent worker started",
            )
            equal(child["completed_executions"], worker["execution_ids"], "worker full coverage")
            require(
                launch["launched_ns"]
                <= child["started_ns"]
                <= child["ended_ns"]
                <= launch["returned_ns"]
                < deadline,
                "worker lifetime/deadline invalid",
            )
            equal(child["deadline_ns"], deadline, "worker deadline")
            equal(child["model_loads"], 1, "one loaded model per worker")
            ab.audit_worker_controls(controls_view(plan), child, previous)
            previous = child
            manifest["completed_workers"].append(worker["worker_id"])
            manifest["artifact_usage"] = ab.artifact_usage(output_dir, plan["contract"]["limits"])
            write_json(path, manifest)
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = (
            "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        manifest["ended_ns"] = time.perf_counter_ns()
        if manifest["ended_ns"] >= deadline:
            manifest["status"] = "incomplete"
            manifest["failures"].append({"type": "deadline", "message": "global budget exhausted"})
        write_json(path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    for field in ("baseline-root", "candidate-root", "contract", "model-path", "output"):
        probe.add_argument("--" + field, required=True, type=Path)
    probe.add_argument("--gpu-id", type=int, required=True)
    for name in ("run", "worker"):
        child = commands.add_parser(name)
        child.add_argument("--plan", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
        if name == "worker":
            child.add_argument("--worker-id", choices=("N-A", "N-B"), required=True)
            child.add_argument("--deadline-ns", required=True, type=int)
    report = commands.add_parser("report")
    report.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "probe":
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
        print(json.dumps({"plan_sha256": plan["plan_sha256"]}))
        return 0
    if args.command == "report":
        from .m3_inactive_report import write_report

        value = write_report(args.run_dir)
        print(json.dumps({key: value[key] for key in ("evidence_status", "decision")}))
        return 0 if value["decision"] == "passed" else 1
    plan = read_json(args.plan)
    value = (
        run_worker(
            plan, worker_id=args.worker_id, output_dir=args.output, deadline_ns=args.deadline_ns
        )
        if args.command == "worker"
        else run(plan, output_dir=args.output)
    )
    return 0 if value["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
