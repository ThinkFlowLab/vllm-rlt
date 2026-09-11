"""Probe and execute two owned workers for persistent eager storage correctness."""

import argparse
import json
from pathlib import Path

from vllm_lt.benchmarks.ab_schema import affinity_snapshot
from vllm_lt.benchmarks.schema import read_json, write_json

from .m3_inactive_run import _active_deadline, _run_worker, _run_workers, controls_view
from .m3_persistent import run_model_rows
from .m3_persistent_schema import make_plan, validate_plan, verify_plan


def active_deadline(output_dir, worker):
    return _active_deadline(
        output_dir,
        worker,
        check_directory="lifecycle",
        check_ids=[
            value for value in worker["execution_ids"] if not value.startswith("m3-persistent-")
        ],
    )


def run_worker(plan, *, worker_id, output_dir, deadline_ns):
    from .m3_persistent_lifecycle import run_lifecycle_evaluation

    def checks(plan, implementation, output_dir, deadline_ns):
        for evaluation in plan["lifecycle"]["execution_order"]:
            if evaluation["implementation_id"] == implementation:
                yield (
                    evaluation["evaluation_id"],
                    run_lifecycle_evaluation(
                        plan["lifecycle"],
                        evaluation["evaluation_id"],
                        output_dir / "lifecycle",
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
        artifact_type="m3_persistent_worker_manifest",
    )


def run(plan, *, output_dir):
    return _run_workers(
        plan,
        output_dir=output_dir,
        verify=verify_plan,
        module="vllm_lt.validation.m3_persistent_run",
        watchdog=active_deadline,
        artifact_type="m3_persistent_manifest",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    for field in ("baseline-root", "candidate-root", "contract", "model-path", "output"):
        probe.add_argument("--" + field, required=True, type=Path)
    probe.add_argument("--gpu-id", required=True, type=int)
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
        from .m3_persistent_report import write_report

        result = write_report(args.run_dir)
        print(json.dumps({key: result[key] for key in ("evidence_status", "decision")}))
        return 0 if result["decision"] == "passed" else 1
    plan = read_json(args.plan)
    result = (
        run_worker(
            plan, worker_id=args.worker_id, output_dir=args.output, deadline_ns=args.deadline_ns
        )
        if args.command == "worker"
        else run(plan, output_dir=args.output)
    )
    return 0 if result["status"] == "complete" else 1


__all__ = ["active_deadline", "controls_view", "run", "run_worker", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
