"""Finite M4 workers around the existing inference and numerical executors."""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from . import ab
from .ab_schema import RUNTIME_VARIABLES, affinity_snapshot, equal, require
from .runner import write_json
from .schema import read_json


def failure(error):
    """Bound diagnostics without losing the failing I/O operation or traceback."""
    value = {"type": type(error).__name__, "message": str(error)[:2048]}
    if isinstance(error, OSError):
        value.update(errno=error.errno, filename=str(error.filename)[:1024])
    value["traceback"] = "".join(traceback.format_exception(error))[-8192:]
    return value


def _stop(signum, frame):
    raise KeyboardInterrupt("M4 process received SIGTERM")


def artifact_usage(output_dir, plan):
    """Enforce the frozen compositional bound as well as the overall hard cap."""
    limits = plan["contract"]["limits"]
    usage = ab.artifact_usage(output_dir, limits)
    groups = {"numerical": 0, "held": 0, "auxiliary": 0}
    root = Path(output_dir)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if parts[0] == "profiles" and path.name == "trace.json":
            continue
        if parts[0] == "held":
            group = "held"
        elif (
            len(parts) > 1
            and parts[0] == "numerical"
            and parts[1] in ("spools", "dumps", "comparisons")
        ):
            group = "numerical"
        else:
            group = "auxiliary"
        groups[group] += path.stat().st_size
    estimates = plan["resource_estimates"]["components"]
    numerical_cap = sum(
        estimates[key]
        for key in (
            "retained_tensor_bytes",
            "dump_tensor_bytes",
            "retained_index_bytes",
            "dump_index_bytes",
            "comparison_json_bytes",
        )
    )
    require(groups["numerical"] <= numerical_cap, "numerical evidence byte cap")
    require(groups["held"] <= limits["held_evidence_bytes_max"], "held evidence byte cap")
    require(groups["auxiliary"] <= limits["auxiliary_artifact_bytes_max"], "auxiliary byte cap")
    return {**usage, **{key + "_bytes": value for key, value in groups.items()}}


def active_case_deadline(output_dir, worker):
    from vllm_lt.validation import m4_attention as numerical
    from vllm_lt.validation import m4_attention_held as held

    root = Path(output_dir)
    candidates = [ab.active_case_deadline(root, worker)]
    candidates.append(held.active_case_deadline(root / "held", worker["implementation_id"]))
    candidates.append(numerical.active_case_deadline(root, worker["implementation_id"]))
    marker = root / "workers" / worker["worker_id"] / "active-case.json"
    if marker.exists():
        value = read_json(marker)
        if value.get("case_completed_ns") is None:
            candidates.append(value["deadline_ns"])
    return min((value for value in candidates if value is not None), default=None)


def run_worker(plan, *, worker_id, output_dir, deadline_ns):
    from vllm_lt.validation import m4_attention as numerical
    from vllm_lt.validation import m4_attention_held as held

    from . import m4_attention_schema as schema
    from . import runner

    started = time.perf_counter_ns()
    schema.validate_plan(plan)
    worker = next(item for item in plan["workers"] if item["worker_id"] == worker_id)
    side = worker["implementation_id"]
    schema.verify_plan(plan, implementation_id=side)
    require(time.perf_counter_ns() < deadline_ns, "worker deadline exhausted before device use")
    equal(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        str(plan["contract"]["controls"]["gpu_ids"][0]),
        "scheduler visibility",
    )
    root = Path(output_dir).resolve()
    folder = root / "workers" / worker_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "artifact_type": "m4_attention_worker_manifest",
        **worker,
        "plan_sha256": plan["plan_sha256"],
        "source": plan["implementations"][side]["source"],
        "harness_sha256": plan["harness"]["sha256"],
        "affinity": affinity_snapshot(),
        "runtime_environment": {key: os.environ.get(key) for key in RUNTIME_VARIABLES},
        "status": "running",
        "passed": False,
        "failures": [],
        "completed_executions": [],
        "started_ns": started,
        "deadline_ns": deadline_ns,
        "model_loads": 0,
        "preparation": {
            "verification_ns": time.perf_counter_ns() - started,
            "downloads": "none; prepared checkpoint",
            "compilation_ns": None,
            "warmup": "declared excluded rows only",
        },
    }

    def save():
        write_json(folder / "manifest.json", manifest)

    def completed(execution_id, result):
        expected = worker["execution_ids"][len(manifest["completed_executions"])]
        equal(execution_id, expected, "worker acknowledgment order")
        manifest["artifact_usage"] = artifact_usage(root, plan)
        manifest["completed_executions"].append(execution_id)
        save()

    save()
    model, ready = None, False
    old_signal = signal.signal(signal.SIGTERM, _stop)
    try:
        view = schema.execution_view(plan)
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
        require(time.perf_counter_ns() < deadline_ns, "worker deadline exhausted before model load")
        model = runner.load_model(view, manifest)
        manifest["model_loads"] = 1
        save()
        rows = [row for row in plan["execution_order"] if row["worker_id"] == worker_id]
        for row in rows:
            if row["kind"] == "numerical":
                break  # The shared numerical executor owns its complete ordered subset.
            require(time.perf_counter_ns() < deadline_ns, "worker deadline exhausted")
            start = time.perf_counter_ns()
            marker = {
                "execution_id": row["execution_id"],
                "started_ns": start,
                "deadline_ns": min(
                    deadline_ns, start + int(plan["contract"]["limits"]["case_timeout_s"] * 1e9)
                ),
                "case_completed_ns": None,
            }
            write_json(folder / "active-case.json", marker)
            if row["kind"] == "benchmark":
                runner.run_loaded_rows(
                    model,
                    view,
                    [row],
                    output_dir=root,
                    deadline=marker["deadline_ns"],
                    record_completed=completed,
                )
            else:
                held_row = next(
                    item
                    for item in plan["held"]["execution_order"]
                    if item["execution_id"] == row["execution_id"]
                )
                for key in ("kind", "implementation_id"):
                    equal(held_row[key], row[key], "held execution binding")
                value = held.run_held_row(
                    model, plan, held_row, root / "held", marker["deadline_ns"]
                )
                require(
                    value["status"] == "complete" and value["passed"], "held prerequisite failed"
                )
                completed(row["execution_id"], value)
            marker["case_completed_ns"] = time.perf_counter_ns()
            require(
                marker["case_completed_ns"] < marker["deadline_ns"],
                "complete case lifetime exceeded",
            )
            write_json(folder / (row["execution_id"] + ".lifetime.json"), marker)
            write_json(folder / "active-case.json", marker)
            require(
                time.perf_counter_ns() < marker["deadline_ns"],
                "case marker export exceeded deadline",
            )
        if worker_id.startswith("N-"):
            result = numerical.run_numerical_rows(
                model,
                plan,
                side,
                root,
                deadline_ns,
                after_case=lambda case, value: completed(case["case_id"], value),
            )
            manifest["numerical"] = result
            require(result["complete"] and result["passed"], "model fidelity prerequisite failed")
        equal(manifest["completed_executions"], worker["execution_ids"], "complete worker order")
        manifest["status"], manifest["passed"] = "complete", True
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = (
            "incomplete" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append(failure(error))
    finally:
        model = None
        if ready or runner.torch.cuda.is_initialized():
            try:
                runner.release_device(manifest)
            except Exception as error:
                manifest["status"], manifest["passed"] = "failed", False
                manifest["failures"].append({"stage": "cleanup", **failure(error)})
        manifest["ended_ns"] = time.perf_counter_ns()
        if manifest["ended_ns"] >= deadline_ns:
            manifest["status"], manifest["passed"] = "incomplete", False
            manifest["failures"].append(
                {"type": "deadline", "message": "worker deadline exhausted"}
            )
        signal.signal(signal.SIGTERM, old_signal)
        save()
    return manifest


def run_ab(plan, *, output_dir):
    from vllm_lt.validation import m4_attention as numerical
    from vllm_lt.validation import m4_attention_held as held

    from . import m4_attention_schema as schema

    start = time.perf_counter_ns()
    schema.verify_plan(plan)
    equal(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        str(plan["contract"]["controls"]["gpu_ids"][0]),
        "controller reservation",
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "workers").mkdir()
    (root / "held").mkdir()
    write_json(root / "plan.json", plan)
    deadline = start + int(plan["contract"]["limits"]["total_timeout_s"] * 1e9)
    manifest = {
        "schema_version": 1,
        "artifact_type": "m4_attention_manifest",
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

    def save():
        write_json(root / "manifest.json", manifest)

    save()
    previous = None
    old_signal = signal.signal(signal.SIGTERM, _stop)
    try:
        for worker in plan["workers"]:
            require(time.perf_counter_ns() < deadline, "controller deadline exhausted")
            if worker["worker_id"] == "A1":
                checks = {
                    "numerical": numerical.audit_numerical(root, plan),
                    "held": held.audit_held(root / "held", plan),
                }
                gate = {
                    "checks": checks,
                    "complete": all(v["complete"] for v in checks.values()),
                    "passed": all(v["passed"] for v in checks.values()),
                }
                manifest["numerical_gate"], manifest["numerical_gate_ns"] = (
                    gate,
                    time.perf_counter_ns(),
                )
                write_json(root / "numerical-gate.json", gate)
                save()
                require(
                    gate["complete"] and gate["passed"], "raw prerequisites cannot qualify timing"
                )
            require(time.perf_counter_ns() < deadline, "controller deadline exhausted after audit")
            launch = {
                "worker_id": worker["worker_id"],
                "exit_code": None,
                "launched_ns": time.perf_counter_ns(),
                "returned_ns": None,
            }
            manifest["workers"].append(launch)
            save()
            try:
                launch["exit_code"] = ab._launch_worker(
                    plan,
                    worker,
                    root,
                    deadline,
                    module="vllm_lt.benchmarks.m4_attention",
                    active_deadline=active_case_deadline,
                )
            finally:
                launch["returned_ns"] = time.perf_counter_ns()
            result = read_json(root / "workers" / worker["worker_id"] / "manifest.json")
            done = result["completed_executions"]
            equal(done, worker["execution_ids"][: len(done)], "worker acknowledgment prefix")
            manifest["completed_executions"].extend(done)
            equal(done, worker["execution_ids"], "worker completion")
            require(
                launch["exit_code"] == 0 and result["status"] == "complete" and result["passed"],
                "worker failed; no subsequent worker may start",
            )
            require(
                launch["launched_ns"]
                <= result["started_ns"]
                <= result["ended_ns"]
                <= launch["returned_ns"]
                < deadline,
                "worker chronology",
            )
            equal(result["model_loads"], 1, "one model load per worker")
            ab.audit_worker_controls(plan, result, previous)
            previous = result
            manifest["completed_workers"].append(worker["worker_id"])
            manifest["artifact_usage"] = artifact_usage(root, plan)
            save()
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = (
            "incomplete" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["failures"].append(failure(error))
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        manifest["ended_ns"] = time.perf_counter_ns()
        if manifest["ended_ns"] >= deadline:
            manifest["status"] = "incomplete"
            manifest["failures"].append(
                {"type": "deadline", "message": "controller deadline exhausted"}
            )
        save()
    return manifest


def main(argv=None):
    from . import m4_attention_schema as schema

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-probe")
    probe = commands.add_parser("probe")
    for name in ("baseline-root", "candidate-root", "contract", "model-path", "output"):
        probe.add_argument("--" + name, required=True, type=Path)
    probe.add_argument("--gpu-id", type=int, required=True)
    for name in ("run", "worker"):
        child = commands.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "worker":
            child.add_argument("--worker-id", required=True)
            child.add_argument("--deadline-ns", type=int, required=True)
    commands.add_parser("report").add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "source-probe":
            print(json.dumps(schema.source_probe(), allow_nan=False))
            return 0
        if args.command == "probe":
            require(not args.output.exists(), "probe destination must be fresh")
            plan = schema.make_plan(
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
            from .m4_attention_report import write_report

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
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"invalid or incomplete M4 experiment: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
