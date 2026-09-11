"""Frozen M2 controller and single-implementation workers; no alternative inference loop."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .ab_schema import (
    RUNTIME_VARIABLES,
    affinity_snapshot,
    equal,
    execution_view,
    make_ab_plan,
    require,
    source_probe,
    validate_ab_plan,
    verify_ab_plan,
)
from .runner import write_json
from .schema import read_json


def artifact_usage(output_dir, limits):
    total, traces = 0, 0
    for path in Path(output_dir).rglob("*"):
        require(not path.is_symlink(), "artifact directories cannot contain symbolic links")
        if path.is_file():
            size = path.stat().st_size
            total += size
            if path.name == "trace.json":
                require(size <= limits["profile_trace_bytes_max"], "individual profile byte cap")
                traces += size
    require(traces <= limits["profile_total_bytes_max"], "total profile byte cap")
    require(total <= limits["artifact_bytes_max"], "total artifact byte cap")
    return {"total_bytes": total, "profile_trace_bytes": traces}


def audit_worker_controls(plan, manifest, previous=None, *, require_cleanup=True):
    """Device discovery happened in the worker; this check reads its recorded facts only."""
    controls = plan["contract"]["controls"]
    env = manifest["environment"]
    equal(env["cuda_visible_devices"], str(controls["gpu_ids"][0]), "physical GPU visibility")
    equal(env["logical_device"], "cuda:0", "logical GPU")
    equal(env["cpu_affinity"], controls["affinity"]["cpu_ids"], "CPU affinity")
    equal(env["numa_status"], controls["affinity"]["numa_status"], "NUMA affinity")
    equal(manifest["affinity"], controls["affinity"], "active NUMA memory policy")
    equal(env["actual_torch_threads"], {"intraop": 1, "interop": 1}, "actual thread counts")
    equal(manifest["runtime_environment"], plan["runtime_environment"], "worker runtime variables")
    equal(manifest["plan_sha256"], plan["plan_sha256"], "worker plan identity")
    equal(manifest["harness_sha256"], plan["harness"]["sha256"], "worker harness")
    equal(
        manifest["source"],
        plan["implementations"][manifest["implementation_id"]]["source"],
        "worker source",
    )
    equal(env["python"], plan["dependencies"]["python"], "runtime Python")
    equal(env["torch_cuda_version"], plan["dependencies"]["torch_cuda_build"], "CUDA build")
    packages = {
        row["name"].lower().replace("_", "-"): row["version"]
        for row in plan["dependencies"]["distributions"]
    }
    for name, value in env["software"].items():
        equal(value, packages[name], f"runtime {name}")
    equal(
        env["arithmetic"],
        {**plan["benchmark_contract"]["arithmetic"], "cudnn_allow_tf32": False},
        "actual arithmetic",
    )
    rows = env["scheduler"]
    require(len(rows) == 1 and rows[0]["type"] == "RUN", "one scheduler RUN record required")
    equal(str(rows[0]["gpu_id"]), str(controls["gpu_ids"][0]), "scheduler device")
    equal(rows[0]["user"], env["account"], "reservation account")
    if previous is not None:
        previous = previous["environment"]
        for name in (
            "host",
            "account",
            "gpu_uuid",
            "gpu_name",
            "total_device_bytes",
            "compute_capability",
            "reservation_environment",
            "cuda_visible_devices",
            "cpu_affinity",
            "numa_status",
            "software",
            "actual_torch_threads",
        ):
            equal(env[name], previous[name], f"cross-worker {name}")
    if require_cleanup:
        equal(
            manifest["teardown_after_workspace_release"],
            {"allocated_bytes": 0, "reserved_bytes": 0},
            "worker final CUDA cleanup",
        )


def run_worker(plan, *, worker_id, output_dir, deadline_ns):
    from vllm_lt.validation.m2 import run_numerical_rows

    from . import runner

    start = time.perf_counter_ns()
    validate_ab_plan(plan)
    worker = next(row for row in plan["workers"] if row["worker_id"] == worker_id)
    implementation = worker["implementation_id"]
    verify_ab_plan(plan, implementation_id=implementation)
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
        "artifact_type": "m2_worker_manifest",
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

        runner.run_loaded_rows(
            model,
            view,
            [row for row in rows if row["kind"] == "benchmark"],
            output_dir=output_dir,
            deadline=deadline_ns,
            record_completed=completed,
        )
        if worker_id.startswith("N-"):
            numerical = run_numerical_rows(model, plan, implementation, output_dir, deadline_ns)
            manifest["numerical"] = numerical
            manifest["completed_executions"].extend(numerical["completed_cases"])
            require(
                numerical["complete"] and numerical["passed"],
                f"numerical prerequisite failed: {numerical['errors']}",
            )
        equal(
            manifest["completed_executions"], worker["execution_ids"], "complete worker row order"
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
            manifest["failures"].append(
                {"type": "deadline", "message": "global deadline exhausted"}
            )
        write_json(manifest_path, manifest)
    return manifest


def active_case_deadline(output_dir, worker):
    """Read only task-owned active markers; no GPU polling or invented progress."""
    candidates = []
    for execution_id in worker["execution_ids"]:
        folder = output_dir / "runs" / execution_id
        marker = folder / "started.json"
        if marker.exists():
            result = folder / "result.json"
            if not result.exists() or "case_completed_ns" not in read_json(result):
                candidates.append(read_json(marker)["deadline_ns"])
    ledger = output_dir / "numerical" / "ledger.json"
    if worker["worker_id"].startswith("N-") and ledger.exists():
        active = read_json(ledger).get("active_case")
        if active:
            candidates.append(active["deadline_ns"])
    return min(candidates) if candidates else None


def _launch_worker(
    plan, worker, output_dir, deadline_ns, *, module="vllm_lt.benchmarks.ab", active_deadline=None
):
    implementation = plan["implementations"][worker["implementation_id"]]
    command = [
        plan["interpreter"],
        "-m",
        module,
        "worker",
        "--plan",
        str(output_dir / "plan.json"),
        "--worker-id",
        worker["worker_id"],
        "--output",
        str(output_dir),
        "--deadline-ns",
        str(deadline_ns),
    ]
    env = dict(os.environ, PYTHONPATH=implementation["root"])
    console = output_dir / "workers" / (worker["worker_id"] + ".console.log")
    with console.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=implementation["root"],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                now = time.perf_counter_ns()
                case_limit = (active_deadline or active_case_deadline)(output_dir, worker)
                if case_limit is not None and now >= case_limit:
                    raise TimeoutError("active case lifetime exceeded its 600-second cap")
                remaining = (deadline_ns - now) / 1e9
                if remaining <= 5:
                    raise TimeoutError("global deadline reached process-cleanup reserve")
                try:
                    return process.wait(timeout=min(1.0, remaining - 5))
                except subprocess.TimeoutExpired:
                    artifact_usage(output_dir, plan["contract"]["limits"])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def run_ab(plan, *, output_dir):
    from vllm_lt.validation.m2 import audit_numerical

    start = time.perf_counter_ns()
    verify_ab_plan(plan)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(plan["contract"]["controls"]["gpu_ids"][0]),
        "controller must run inside the frozen scheduler assignment",
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "workers").mkdir()
    write_json(output_dir / "plan.json", plan)
    deadline = start + plan["contract"]["limits"]["total_timeout_s"] * 1_000_000_000
    manifest = {
        "schema_version": 1,
        "artifact_type": "m2_ab_manifest",
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "started_ns": start,
        "deadline_ns": deadline,
        "completed_workers": [],
        "completed_executions": [],
        "failures": [],
        "workers": [],
        "numerical_gate": None,
    }
    write_json(output_dir / "manifest.json", manifest)
    previous = None
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def stopped(signum, frame):
        raise KeyboardInterrupt("controller received SIGTERM")

    signal.signal(signal.SIGTERM, stopped)
    try:
        for worker in plan["workers"]:
            require(time.perf_counter_ns() < deadline, "global deadline exhausted")
            if worker["worker_id"] == "A1":
                gate = audit_numerical(output_dir, plan)
                manifest["numerical_gate"] = gate
                manifest["numerical_gate_ns"] = time.perf_counter_ns()
                write_json(output_dir / "numerical-gate.json", gate)
                require(
                    gate["complete"] and gate["passed"], "numerical evidence cannot qualify timing"
                )
            launched = time.perf_counter_ns()
            launch = {
                "worker_id": worker["worker_id"],
                "exit_code": None,
                "launched_ns": launched,
                "returned_ns": None,
            }
            manifest["workers"].append(launch)
            write_json(output_dir / "manifest.json", manifest)
            try:
                exit_code = _launch_worker(plan, worker, output_dir, deadline)
                launch["exit_code"] = exit_code
            finally:
                launch["returned_ns"] = time.perf_counter_ns()
            path = output_dir / "workers" / worker["worker_id"] / "manifest.json"
            result = read_json(path)
            child_completed = result["completed_executions"]
            equal(
                child_completed,
                worker["execution_ids"][: len(child_completed)],
                "worker completed execution prefix",
            )
            manifest["completed_executions"].extend(child_completed)
            equal(child_completed, worker["execution_ids"], "complete worker execution order")
            require(
                exit_code == 0 and result["status"] == "complete" and result["passed"],
                f"worker {worker['worker_id']} failed; no subsequent worker started",
            )
            require(
                result["started_ns"] >= launched and result["ended_ns"] < deadline,
                "worker clock/deadline is invalid",
            )
            require(result["model_loads"] == 1, "worker must load exactly one model")
            audit_worker_controls(plan, result, previous)
            previous = result
            manifest["completed_workers"].append(worker["worker_id"])
            manifest["artifact_usage"] = artifact_usage(output_dir, plan["contract"]["limits"])
            write_json(output_dir / "manifest.json", manifest)
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = (
            "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        manifest["ended_ns"] = time.perf_counter_ns()
        if manifest["ended_ns"] >= deadline:
            manifest["status"] = "incomplete"
            manifest["failures"].append(
                {"type": "deadline", "message": "overall deadline exhausted"}
            )
        write_json(output_dir / "manifest.json", manifest)
    return manifest


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
            plan = make_ab_plan(
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
            from .ab_report import write_report

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


if __name__ == "__main__":
    raise SystemExit(main())
