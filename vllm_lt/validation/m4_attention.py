"""The M4 FP32 attention subset; independent histories, not M2 exact arithmetic."""

import time
from copy import deepcopy
from pathlib import Path

from . import m2
from .schema import _digest, read_json, write_json


def build_numerical_plan(suite, contract, model_config):
    """Derive 27 cases/51 streams with unchanged Q1 numerical bounds."""
    plan = deepcopy(m2.build_numerical_plan(suite, contract, model_config))
    plan["artifact_type"] = "m4_attention_numerical_plan"
    plan["suite"]["artifact_type"] = "m4_attention_numerical_suite"
    plan["contract"]["artifact_type"] = "m4_attention_numerical_contract"
    plan["contract"]["diagnostics"]["preselected_dump_comparison_kind"] = "implementation_fidelity"
    for case in plan["execution_order"]:
        case["case_id"] = case["case_id"].replace("m2-", "m4-attention-", 1)
        case["spool_group"] = case["case_id"]
    cases = {case["case_id"]: case for case in plan["execution_order"]}
    for row in plan["comparison_order"]:
        for key in ("comparison_id", "reference_case_id", "candidate_case_id"):
            row[key] = row[key].replace("m2-", "m4-attention-", 1)
        if row["require_exact"] and cases[row["candidate_case_id"]]["backend"] == "triton":
            row["require_exact"] = False
            row["comparison_kind"] = "implementation_fidelity"
            row["comparison_id"] = row["comparison_id"].replace(
                "implementation_exact", "implementation_fidelity"
            )
    del plan["numerical_plan_sha256"]
    plan["numerical_plan_sha256"] = _digest(plan)
    return plan


def validate_numerical_plan(plan):
    expected = build_numerical_plan(
        plan["source_inputs"]["suite"], plan["source_inputs"]["contract"], plan["model_config"]
    )
    if _digest(plan) != _digest(expected):
        raise ValueError("M4 numerical plan differs from the exact frozen attention subset")


def _view(parent_plan):
    plan = parent_plan["numerical"]
    validate_numerical_plan(plan)
    return {**plan, "plan_sha256": parent_plan["plan_sha256"]}


def run_numerical_rows(
    model, parent_plan, implementation_id, output_dir, deadline_ns, *, after_case=None
):
    from .runner import execute_case

    folder = Path(output_dir) / "numerical"
    active_path = folder / f"active-{implementation_id}.json"

    def execute(*args):
        marker = read_json(folder / "ledger.json")["active_case"]
        write_json(active_path, marker)
        return execute_case(*args)

    def completed(case, value):
        if after_case is not None:
            after_case(case, value)
        marker = read_json(active_path)
        end = time.perf_counter_ns()
        if end >= marker["deadline_ns"]:
            raise TimeoutError("M4 numerical export/acknowledgement exceeded case deadline")
        marker["case_completed_ns"] = end
        write_json(folder / "cases" / case["case_id"] / "m4-completed.json", marker)
        write_json(active_path, marker)
        if time.perf_counter_ns() >= marker["deadline_ns"]:
            raise TimeoutError("M4 numerical marker export exceeded case deadline")

    return m2._run_numerical_view(
        model,
        _view(parent_plan),
        implementation_id,
        output_dir,
        deadline_ns,
        execute=execute,
        after_case=completed,
    )


def audit_numerical(output_dir, parent_plan):
    view = _view(parent_plan)
    result = m2._audit_numerical_view(output_dir, view)
    folder = Path(output_dir) / "numerical"
    result["missing"] = [
        case["case_id"]
        for case in view["execution_order"]
        if case["case_id"] not in result["completed_cases"]
    ]
    result["known_required_failure"] = False
    if not result["complete"]:
        if len(result["completed_cases"]) < len(view["execution_order"]):
            _audit_prefix(folder, view, result)
        return result
    result["known_required_failure"] = not result["passed"]
    try:
        ledger = read_json(folder / "ledger.json")
        result["case_completed_ns"] = _completion_markers(
            folder, view, result["comparisons"], ledger["case_lifetimes"], require_success=True
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["complete"] = result["passed"] = False
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    return result


def _completion_markers(folder, view, comparisons, lifetimes, *, require_success):
    by_id = {row["comparison_id"]: row for row in view["comparison_order"]}
    failed = {
        by_id[row["comparison_id"]]["candidate_case_id"]
        for row in comparisons
        if row["required_failures"] or row["behavior_failures"]
    }
    completed = {}
    for case_id, lifetime in lifetimes.items():
        path = folder / "cases" / case_id / "m4-completed.json"
        # A required failure stops before the success-only parent ACK callback.
        if not path.exists() and (not require_success or case_id in failed):
            continue
        marker = read_json(path)
        if not (
            marker
            == {
                "case_id": case_id,
                "started_ns": lifetime["started_ns"],
                "deadline_ns": lifetime["deadline_ns"],
                "case_completed_ns": marker["case_completed_ns"],
            }
            and type(marker["case_completed_ns"]) is int
            and lifetime["finished_ns"] <= marker["case_completed_ns"] < lifetime["deadline_ns"]
        ):
            raise ValueError("M4 numerical full-case completion marker differs")
        completed[case_id] = marker["case_completed_ns"]
    return completed


def _audit_prefix(folder, view, result):
    """Audit an actual completed subset, never label it a complete execution plan."""
    from .report import _audit_case, _audit_comparison, _audit_raw_evidence

    result["full_plan_errors"] = result["errors"]
    try:
        ledger = read_json(folder / "ledger.json")
        completed = ledger["completed_cases"]
        rows = view["execution_order"][: len(completed)]
        if (
            not completed
            or completed != [row["case_id"] for row in rows]
            or ledger["plan_sha256"] != view["plan_sha256"]
            or ledger["numerical_plan_sha256"] != view["numerical_plan_sha256"]
        ):
            raise ValueError("numerical stopped prefix identity differs")
        fixtures = {row["fixture_id"]: row for row in view["suite"]["fixtures"]}
        cases, previous_end = {}, 0
        for row in rows:
            case_id = row["case_id"]
            lifetime = ledger["case_lifetimes"][case_id]
            start, end, deadline = (
                lifetime[name] for name in ("started_ns", "finished_ns", "deadline_ns")
            )
            if (
                any(type(n) is not int or n <= 0 for n in (start, end, deadline))
                or not previous_end <= start <= end <= deadline <= start + 600 * 10**9
            ):
                raise ValueError("numerical prefix lifetime differs")
            previous_end = end
            value = read_json(folder / "cases" / case_id / "result.json")
            _audit_case(value, view, row, fixtures)
            cases[case_id] = value
        comparisons = [row for row in view["comparison_order"] if row["candidate_case_id"] in cases]
        checked, observations, reverse = [], {}, {}
        for comparison in comparisons:
            case_id = comparison["candidate_case_id"]
            seen, back = observations.setdefault(case_id, {}), reverse.setdefault(case_id, {})

            def observe(index, fixture_id, key):
                identity = fixture_id, key
                if (index in seen and seen[index] != identity) or (
                    identity in back and back[identity] != index
                ):
                    raise ValueError("numerical prefix observation identity differs")
                seen[index], back[identity] = identity, index

            checked.append(_audit_comparison(folder, view, comparison, cases, fixtures, observe))
        for case_id, seen in observations.items():
            if sorted(seen) != list(range(1, cases[case_id]["observed_boundaries"] + 1)):
                raise ValueError("numerical prefix observation coverage differs")
        # The helper explicitly accepts caller-provided rows and checks every raw
        # payload against those rows. The original frozen plan remains unchanged.
        raw = _audit_raw_evidence(
            folder,
            {**view, "execution_order": rows, "comparison_order": comparisons},
            cases,
            fixtures,
            checked,
            ledger,
        )
        result["validated_prefix"] = {
            "completed_cases": completed,
            "comparisons": checked,
            "raw_evidence": raw,
            "case_lifetimes": ledger["case_lifetimes"],
            "case_completed_ns": _completion_markers(
                folder, view, checked, ledger["case_lifetimes"], require_success=False
            ),
        }
        result["case_completed_ns"] = result["validated_prefix"]["case_completed_ns"]
        result["known_required_failure"] = any(
            row["required_failures"] or row["behavior_failures"] for row in checked
        )
        result["errors"] = []
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["prefix_errors"] = [{"type": type(exc).__name__, "message": str(exc)}]
        result["errors"] = result["prefix_errors"]


def active_case_deadline(output_dir, implementation_id):
    path = Path(output_dir) / "numerical" / f"active-{implementation_id}.json"
    if path.exists():
        marker = read_json(path)
        if "case_completed_ns" not in marker:
            return marker["deadline_ns"]
    return None
