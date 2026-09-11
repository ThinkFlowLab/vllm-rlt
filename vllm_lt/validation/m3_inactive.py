"""The bounded inactive-row correctness subset; no throughput or graph qualification."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

from vllm_lt.models.config import OuroConfig

from .m2 import (
    _audit_numerical_view,
    _run_numerical_view,
    build_numerical_plan,
)
from .schema import _digest, _fixture_stats, read_json, write_json

PADDING = {"mode": "odd_slots", "row_count": 8, "table_width": 32, "decode_only": True}


def build_model_plan(suite, contract, model_config):
    """Reuse the original M2 fixture/policy derivation, then resolve exactly13 model cases."""
    base = build_numerical_plan(suite, contract, model_config)
    cases = [
        deepcopy(row)
        for row in base["execution_order"]
        if row["implementation"] == "oracle"
        or row["schedule"] in ("refill", "no_refill")
        or row["family"] == "live_gate"
    ]
    selected = {row["case_id"] for row in cases}
    comparisons = [
        deepcopy(row) for row in base["comparison_order"] if row["candidate_case_id"] in selected
    ]
    fixtures = deepcopy(base["suite"]["fixtures"])
    feasibility = deepcopy(suite["feasibility_fixtures"][0])
    fixtures.append(feasibility)
    feasibility_cases = {}
    for implementation in ("A", "B"):
        row = deepcopy(
            next(
                case
                for case in cases
                if case["implementation_id"] == implementation
                and case["implementation"] == "native"
            )
        )
        row.update(
            case_id=f"m2-{implementation}-feasibility",
            family="feasibility",
            phase="feasibility",
            schedule="serial",
            fixture_ids=[feasibility["fixture_id"]],
            group_id=feasibility["fixture_id"],
            history_mode="teacher_forced",
            exit_policy={"mode": "forced", "threshold": 1.0, "min_loops": 2, "max_loops": 4},
            expected_capacity={feasibility["fixture_id"]: len(feasibility["prompt_token_ids"]) + 8},
            max_steps=len(feasibility["prompt_token_ids"]) + 49,
            fixture_sha256=_digest([feasibility]),
        )
        row["spool_group"] = row["case_id"]
        feasibility_cases[implementation] = row
    cases = [
        feasibility_cases["A"],
        *[c for c in cases if c["implementation_id"] == "A"],
        feasibility_cases["B"],
        *[c for c in cases if c["implementation_id"] == "B"],
    ]
    config = OuroConfig.from_dict(model_config)
    stats = _fixture_stats(feasibility, "float32", config)
    comparisons.append(
        {
            "comparison_id": "m2-B-feasibility--implementation_fidelity",
            "family": "feasibility",
            "dtype": "float32",
            "reference_case_id": feasibility_cases["A"]["case_id"],
            "candidate_case_id": feasibility_cases["B"]["case_id"],
            "fixture_id": feasibility["fixture_id"],
            "comparison_kind": "implementation_fidelity",
            "required": True,
            "require_exact": False,
            "expected_prediction_points": 9,
            "expected_boundary_records": stats["selected_boundary_records"]
            + stats["kv_comparison_records"],
            "counts_are_upper_bounds": False,
            "policy_sha256": cases[0]["policy_sha256"],
        }
    )
    for row in cases:
        row["case_id"] = row["case_id"].replace("m2-", "m3-inactive-", 1)
        row["spool_group"] = row["case_id"]
        row["padding"] = (
            deepcopy(PADDING) if row["implementation_id"] == "B" else {"mode": "compact"}
        )
    order = {case["case_id"]: index for index, case in enumerate(cases)}
    for row in comparisons:
        for field in ("comparison_id", "reference_case_id", "candidate_case_id"):
            row[field] = row[field].replace("m2-", "m3-inactive-", 1)
        row["comparison_id"] = row["comparison_id"].replace(
            "implementation_exact", "implementation_fidelity"
        )
        if row["comparison_kind"] == "implementation_exact":
            row["comparison_kind"] = "implementation_fidelity"
        row["require_exact"] = False
    comparisons.sort(key=lambda row: (order[row["candidate_case_id"]], row["comparison_id"]))
    adapted = deepcopy(base["contract"])
    adapted["artifact_type"] = "m3_inactive_numerical_contract"
    adapted["limits"].update(
        implementation_executions=13, feasibility_fixtures_per_dtype=1, total_timeout_s=3600
    )
    adapted["diagnostics"]["preselected_dump_comparison_kind"] = "implementation_fidelity"
    by_id = {row["fixture_id"]: row for row in fixtures}
    payload = indexes = largest = 0
    for case in cases:
        if not case["retain_evidence"]:
            continue
        rows = [
            _fixture_stats(by_id[key], "float32", config, live=case["family"] == "live_gate")
            for key in case["fixture_ids"]
        ]
        case_bytes = sum(row["reference_spool_payload_bytes"] for row in rows)
        payload += case_bytes
        largest = max(largest, case_bytes)
        indexes += sum(
            row["selected_boundary_records"] + row["kv_comparison_records"] for row in rows
        )
    if (
        len(cases) != 13
        or len(comparisons) != 28
        or largest > adapted["limits"]["group_spool_bytes"]
    ):
        raise ValueError("inactive-row numerical case/count/byte contract is invalid")
    if (
        payload + adapted["limits"]["persisted_dump_bytes"]
        > adapted["limits"]["cumulative_spool_written_bytes"]
    ):
        raise ValueError("inactive-row reference evidence exceeds cumulative cap")
    result = {
        "schema_version": 1,
        "artifact_type": "m3_inactive_numerical_plan",
        "source_inputs": {"suite": deepcopy(suite), "contract": deepcopy(contract)},
        "suite": {
            "schema_version": 1,
            "artifact_type": "m3_inactive_numerical_suite",
            "fixtures": fixtures,
            "source_suite_sha256": _digest(suite),
        },
        "contract": adapted,
        "model_config": config.to_dict(),
        "execution_order": cases,
        "comparison_order": comparisons,
        "resource_estimates": {
            "cases": 13,
            "comparisons": 28,
            "validation_cases": 11,
            "feasibility_cases": 2,
            "validation_comparisons": 27,
            "retained_tensor_bytes_upper_bound": payload,
            "retained_index_records_upper_bound": indexes,
            "comparison_records_upper_bound": sum(
                r["expected_boundary_records"] for r in comparisons
            ),
            "max_retained_case_bytes": largest,
        },
    }
    result["numerical_plan_sha256"] = _digest(result)
    return result


def validate_model_plan(plan):
    expected = build_model_plan(
        plan["source_inputs"]["suite"], plan["source_inputs"]["contract"], plan["model_config"]
    )
    if _digest(plan) != _digest(expected):
        raise ValueError("inactive-row model plan differs from the exact frozen subset")


def model_view(parent):
    validate_model_plan(parent["numerical"])
    return {**parent["numerical"], "plan_sha256": parent["plan_sha256"]}


@contextmanager
def _padding_execution(case, observations, *, runner_context=None):
    """Instance-only runner selection; engine/observer loops remain the shared Q1 driver."""
    import torch

    from . import runner
    from .native import observe_native

    original_engine, original_observer = runner.ValidationEngine, runner.observe_native
    engine_holder = []
    contexts = ExitStack()

    class PaddedEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            engine_holder.append(self)
            padded = self.model_runner._recurrent_padded

            def recurrent(hidden, request_ids, depths, positions):
                if not 1 <= len(request_ids) <= 4:
                    raise ValueError("padded validation only permits one to four live rows")
                return padded(
                    hidden,
                    request_ids,
                    depths,
                    positions,
                    row_indices=tuple(2 * i + 1 for i in range(len(request_ids))),
                    row_count=8,
                    table_width=32,
                )

            self.model_runner._recurrent = recurrent
            if runner_context is not None:
                contexts.enter_context(runner_context(self))

    def physical(batch, outputs):
        if batch.active is None:
            return
        if batch.row_count != 8 or batch.live_rows != tuple(
            2 * i + 1 for i in range(len(batch.rows))
        ):
            raise ValueError("physical recurrent execution differs from frozen padding map")
        expected = torch.zeros(8, device=batch.active.device, dtype=torch.bool)
        expected[list(batch.live_rows)] = True
        hidden, gates = outputs
        if (
            not torch.equal(batch.active, expected)
            or batch.block_tables.shape[1] != 32
            or not bool(hidden.isfinite().all())
            or not bool(gates.isfinite().all())
            or bool(torch.count_nonzero(hidden[~expected]))
            or bool(torch.count_nonzero(gates[~expected]))
        ):
            raise ValueError("inactive hidden/gate isolation or finite output check failed")
        owners = {id(allocation): key for key, allocation in batch.allocations}
        observations.append(
            {
                "row_count": 8,
                "live_rows": list(batch.live_rows),
                "table_width": 32,
                "request_ids": [owners[id(a)] for a, _, _ in batch.rows],
                "depths": [d + 1 for _, d, _ in batch.rows],
                "positions": [p for _, _, p in batch.rows],
                "inactive_hidden_zero": True,
                "inactive_gate_zero": True,
                "all_finite": True,
                "active_mask_matches": True,
            }
        )

    try:
        if case["padding"]["mode"] == "odd_slots":
            runner.ValidationEngine = PaddedEngine
            runner.observe_native = lambda *args, **kwargs: observe_native(
                *args, **kwargs, on_physical=physical
            )
        yield
    finally:
        runner.ValidationEngine, runner.observe_native = original_engine, original_observer
        try:
            contexts.close()
        finally:
            for engine in engine_holder:
                if "_recurrent" in vars(engine.model_runner):
                    del engine.model_runner._recurrent
            engine_holder.clear()


def execute_model_case(model, view, case, output, budget, dumps, deadline):
    from .runner import execute_case

    observations = []
    with _padding_execution(case, observations):
        try:
            return execute_case(model, view, case, output, budget, dumps, deadline)
        finally:
            path = Path(output) / "cases" / case["case_id"] / "result.json"
            if path.exists():
                result = read_json(path)
                result["padding_observations"] = observations
                write_json(path, result)


def run_model_rows(model, parent, implementation, output_dir, deadline_ns, *, after_case=None):
    return _run_numerical_view(
        model,
        model_view(parent),
        implementation,
        output_dir,
        deadline_ns,
        execute=execute_model_case,
        after_case=after_case,
    )


def _add_model_counts(view, result):
    completed = set(result["completed_cases"])
    verified = {row["comparison_id"] for row in result["comparisons"]}
    result["counts"].update(
        planned_qualification_cases=11,
        planned_excluded_feasibility_cases=2,
        verified_qualification_cases=sum(
            case["case_id"] in completed and case["phase"] == "validation"
            for case in view["execution_order"]
        ),
        verified_excluded_feasibility_cases=sum(
            case["case_id"] in completed and case["phase"] == "feasibility"
            for case in view["execution_order"]
        ),
        planned_qualification_comparisons=27,
        planned_excluded_feasibility_comparisons=1,
        verified_qualification_comparisons=sum(
            row["comparison_id"] in verified and row["family"] != "feasibility"
            for row in view["comparison_order"]
        ),
        verified_excluded_feasibility_comparisons=sum(
            row["comparison_id"] in verified and row["family"] == "feasibility"
            for row in view["comparison_order"]
        ),
    )


def _audit_padding_observations(case, evidence):
    observations = evidence["padding_observations"]
    expected = [row for row in evidence["schedule"] if row["stage"] == "recurrent"]
    if case["padding"]["mode"] == "compact":
        if observations:
            raise ValueError("compact execution contains padded observations")
        return
    if len(observations) != len(expected) or not expected:
        raise ValueError("padded dispatch coverage differs from executed recurrent schedule")
    for actual, scheduled in zip(observations, expected):
        ids = scheduled["request_ids"]
        frozen = {
            "row_count": 8,
            "live_rows": [2 * i + 1 for i in range(len(ids))],
            "table_width": 32,
            "request_ids": ids,
            "depths": scheduled["depths_after_step"],
            "positions": scheduled["positions_after_step"],
            "inactive_hidden_zero": True,
            "inactive_gate_zero": True,
            "all_finite": True,
            "active_mask_matches": True,
        }
        if _digest(actual) != _digest(frozen):
            raise ValueError("padding observations differ from logical execution history")


def audit_model_rows(output_dir, parent):
    view = model_view(parent)
    result = _audit_numerical_view(output_dir, view)
    _add_model_counts(view, result)
    if not result["complete"]:
        return result
    try:
        for case in view["execution_order"]:
            path = Path(output_dir) / "numerical" / "cases" / case["case_id"] / "result.json"
            evidence = read_json(path)
            _audit_padding_observations(case, evidence)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["complete"] = result["passed"] = False
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    return result
