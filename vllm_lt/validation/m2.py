"""The fixed M2 FP32 subset: independent oracle gates plus exact implementation A/B.

This is a 27-case numerical plan, not a shortened Q1 plan. The parent A/B
controller owns source/environment verification, model loading and GPU lifetime.
"""

import math
import time
from copy import deepcopy
from pathlib import Path

from vllm_lt.models.config import OuroConfig

from .schema import (
    _digest,
    _fixture_stats,
    _validate_contract,
    _validate_suite,
    read_json,
    write_json,
)

FIXTURE_IDS = ("Q1-L16-F0", "Q1-L256-F0", "Q1-L64-F2", "Q1-L128-F3")
PRESELECTED = ("Q1-L64-F2", "Q1-L256-F0")


def build_numerical_plan(suite, contract, model_config):
    """Resolve original Q1 inputs into the exact M2 subset without device work."""
    _validate_suite(suite)
    _validate_contract(contract)
    config = OuroConfig.from_dict(model_config)
    if config.total_ut_steps != 4:
        raise ValueError("M2 numerical validation requires four Ouro loops")
    fixtures = {row["fixture_id"]: deepcopy(row) for row in suite["fixtures"]}
    fixtures = {key: fixtures[key] for key in FIXTURE_IDS}
    for fixture in fixtures.values():
        if any(
            token >= config.vocab_size
            for token in fixture["prompt_token_ids"] + fixture["continuation_input_ids"]
        ):
            raise ValueError("fixture token exceeds model vocabulary")
        if len(fixture["prompt_token_ids"]) + 8 > config.max_position_embeddings:
            raise ValueError("fixture exceeds model position capacity")
    adapted = {
        "schema_version": 1,
        "artifact_type": "m2_numerical_contract",
        **{
            key: deepcopy(contract[key])
            for key in ("engine", "arithmetic", "limits", "diagnostics", "comparison_policy")
        },
    }
    adapted["limits"].update(implementation_executions=27, feasibility_fixtures_per_dtype=0)
    adapted["diagnostics"].update(
        preselected_dump_fixture_ids=list(PRESELECTED),
        preselected_dump_dtype="float32",
        preselected_dump_comparison_kind="implementation_exact",
        reference_retention="retain-oracle-and-A-per-execution-through-B-audit",
    )
    order = []
    policy_hash = _digest(adapted["comparison_policy"])

    def add(implementation_id, implementation, backend, schedule, ids, *, live=False):
        family = "live_gate" if live else "main"
        suffix = ids[0] if len(ids) == 1 else "mixed"
        case_id = (
            f"m2-{implementation_id}-{family}-{implementation}-"
            f"{backend or 'dense'}-{schedule}-{suffix}"
        )
        histories = [fixtures[key] for key in ids]
        capacities = {row["fixture_id"]: len(row["prompt_token_ids"]) + 8 for row in histories}
        cache = adapted["engine"]["cache"]
        pages = sum(4 * math.ceil(size / cache["block_size"]) for size in capacities.values())
        if implementation == "native" and pages > cache["num_blocks"]:
            raise ValueError("M2 packed group exceeds reserved KV capacity")
        row = {
            "case_id": case_id,
            "implementation_id": implementation_id,
            "family": family,
            "phase": "validation",
            "dtype": "float32",
            "implementation": implementation,
            "backend": backend,
            "schedule": schedule,
            "fixture_ids": list(ids),
            "group_id": suffix,
            "history_mode": "live_gate" if live else "teacher_forced",
            "exit_policy": {
                "mode": "live"
                if live
                else "forced"
                if any(row["history_policy"] == "forced" for row in histories)
                else "fixed",
                "threshold": 0.7 if live else 1.0,
                "min_loops": 2,
                "max_loops": 4,
            },
            "block_size": cache["block_size"],
            "num_blocks": cache["num_blocks"],
            "chunk_size": adapted["engine"]["scheduler"]["max_num_batched_tokens"],
            "max_tokens": 9,
            "expected_capacity": capacities,
            "max_steps": sum(len(row["prompt_token_ids"]) + 49 for row in histories),
            "fixture_sha256": _digest(histories),
            "policy_sha256": policy_hash,
            "retain_evidence": implementation_id == "A",
            "spool_group": case_id,
        }
        order.append(row)
        return row

    oracles = {}
    for fixture_id in FIXTURE_IDS:
        oracles[("main", fixture_id)] = add("A", "oracle", None, "serial", [fixture_id])
    oracles[("live_gate", FIXTURE_IDS[0])] = add(
        "A", "oracle", None, "serial", [FIXTURE_IDS[0]], live=True
    )
    for implementation_id in ("A", "B"):
        for backend in ("torch", "triton"):
            for fixture_id in FIXTURE_IDS:
                add(implementation_id, "native", backend, "serial", [fixture_id])
        for schedule in ("refill", "no_refill"):
            add(implementation_id, "native", "triton", schedule, FIXTURE_IDS)
        add(implementation_id, "native", "triton", "serial", [FIXTURE_IDS[0]], live=True)
    comparisons = []
    baselines = {}
    for case in order:
        if case["implementation"] != "native":
            continue
        identity = (case["family"], case["backend"], case["schedule"], tuple(case["fixture_ids"]))
        if case["implementation_id"] == "A":
            baselines[identity] = case
        for fixture_id in case["fixture_ids"]:
            refs = [(oracles[(case["family"], fixture_id)], False)]
            if case["implementation_id"] == "B":
                refs.append((baselines[identity], True))
            for reference, exact in refs:
                stats = _fixture_stats(
                    fixtures[fixture_id], "float32", config, live=case["family"] == "live_gate"
                )
                kind = "implementation_exact" if exact else "same_dtype_fidelity"
                comparisons.append(
                    {
                        "comparison_id": f"{case['case_id']}--{fixture_id}--{kind}",
                        "family": case["family"],
                        "dtype": "float32",
                        "reference_case_id": reference["case_id"],
                        "candidate_case_id": case["case_id"],
                        "fixture_id": fixture_id,
                        "comparison_kind": kind,
                        "required": True,
                        "require_exact": exact,
                        "expected_prediction_points": 9,
                        "expected_boundary_records": stats["selected_boundary_records"]
                        + stats["kv_comparison_records"],
                        "counts_are_upper_bounds": stats["counts_are_upper_bounds"],
                        "policy_sha256": policy_hash,
                    }
                )
    retained_bytes = retained_records = max_case_bytes = 0
    for case in order:
        if not case["retain_evidence"]:
            continue
        stats = [
            _fixture_stats(fixtures[key], "float32", config, live=case["family"] == "live_gate")
            for key in case["fixture_ids"]
        ]
        case_bytes = sum(row["reference_spool_payload_bytes"] for row in stats)
        retained_bytes += case_bytes
        retained_records += sum(
            row["selected_boundary_records"] + row["kv_comparison_records"] for row in stats
        )
        max_case_bytes = max(max_case_bytes, case_bytes)
    limits = adapted["limits"]
    if (
        max_case_bytes > limits["group_spool_bytes"]
        or retained_bytes + limits["persisted_dump_bytes"]
        > limits["cumulative_spool_written_bytes"]
    ):
        raise ValueError("M2 retained evidence exceeds the frozen byte limits")
    result = {
        "schema_version": 1,
        "artifact_type": "m2_numerical_plan",
        "source_inputs": {"suite": deepcopy(suite), "contract": deepcopy(contract)},
        "suite": {
            "schema_version": 1,
            "artifact_type": "m2_numerical_suite",
            "fixtures": list(fixtures.values()),
            "source_suite_sha256": _digest(suite),
        },
        "contract": adapted,
        "model_config": config.to_dict(),
        "execution_order": order,
        "comparison_order": comparisons,
        "resource_estimates": {
            "retained_tensor_bytes_upper_bound": retained_bytes,
            "retained_index_records_upper_bound": retained_records,
            "comparison_records_upper_bound": sum(
                row["expected_boundary_records"] for row in comparisons
            ),
            "max_retained_case_bytes": max_case_bytes,
            "cases": len(order),
            "comparisons": len(comparisons),
        },
    }
    result["numerical_plan_sha256"] = _digest(result)
    return result


def validate_numerical_plan(plan):
    """Rebuild the strict subset from its frozen inputs; never access device/files."""
    if not isinstance(plan, dict) or plan.get("artifact_type") != "m2_numerical_plan":
        raise ValueError("expected an M2 numerical plan")
    expected = build_numerical_plan(
        plan["source_inputs"]["suite"], plan["source_inputs"]["contract"], plan["model_config"]
    )
    if _digest(plan) != _digest(expected):
        raise ValueError("M2 numerical plan differs from its exact frozen subset")


def _view(parent_plan):
    numerical = parent_plan["numerical"]
    validate_numerical_plan(numerical)
    return {**numerical, "plan_sha256": parent_plan["plan_sha256"]}


def _save_ledger(folder, ledger):
    temporary = folder / "ledger.tmp.json"
    write_json(temporary, ledger)
    temporary.replace(folder / "ledger.json")


def _restore_budget(folder, view, ledger):
    from .diagnostics import SpoolBudget, TensorSpool

    limits = view["contract"]["limits"]
    budget = SpoolBudget(limits["cumulative_spool_written_bytes"], limits["group_spool_bytes"])
    expected_groups = {}
    for case in view["execution_order"]:
        if case["case_id"] not in ledger["completed_cases"] or not case["retain_evidence"]:
            continue
        size = 0
        for fixture_id in case["fixture_ids"]:
            spool = TensorSpool.open(folder / "spools", case["case_id"], fixture_id)
            size += spool.data_path.stat().st_size
            spool.close()
        expected_groups[case["spool_group"]] = size
    if ledger["diagnostic_dumps"] != {
        "selected_fixture_ids": view["contract"]["diagnostics"]["preselected_dump_fixture_ids"],
        "written_bytes": 0,
        "fixture_written_bytes": {},
    } or any((folder / "dumps").rglob("*.bin")):
        raise ValueError("a passing A worker cannot have consumed B/A-only dump quota")
    if ledger["spool_bytes_by_group"] != expected_groups:
        raise ValueError("cross-worker spool ledger differs from retained A files")
    for group, size in expected_groups.items():
        budget.claim(group, size)
    if budget.written_bytes != ledger["tensor_written_bytes"]:
        raise ValueError("cross-worker cumulative tensor ledger differs")
    return budget


def run_numerical_rows(model, parent_plan, implementation_id, output_dir, deadline_ns):
    """Run A once, then B once against its retained oracle/A evidence."""
    return _run_numerical_view(
        model, _view(parent_plan), implementation_id, output_dir, deadline_ns
    )


def _run_numerical_view(
    model, view, implementation_id, output_dir, deadline_ns, *, execute=None, after_case=None
):
    """Shared execution for a caller-validated frozen A/B numerical view."""
    from .diagnostics import DiagnosticDump, SpoolBudget
    from .runner import execute_case

    if implementation_id not in ("A", "B"):
        raise ValueError("implementation_id must be A or B")
    if execute is None:
        execute = execute_case
    if (
        _digest(model.config.to_dict()) != _digest(view["model_config"])
        or str(next(model.parameters()).dtype) != "torch.float32"
    ):
        raise ValueError("loaded model differs from the frozen FP32 numerical controls")
    folder = Path(output_dir) / "numerical"
    limits = view["contract"]["limits"]
    selected = view["contract"]["diagnostics"]["preselected_dump_fixture_ids"]
    planned = [
        case for case in view["execution_order"] if case["implementation_id"] == implementation_id
    ]
    if implementation_id == "A":
        folder.mkdir(exist_ok=False)
        budget = SpoolBudget(limits["cumulative_spool_written_bytes"], limits["group_spool_bytes"])
        ledger = {
            "schema_version": 1,
            "artifact_type": view["artifact_type"].replace("_plan", "_ledger"),
            "plan_sha256": view["plan_sha256"],
            "numerical_plan_sha256": view["numerical_plan_sha256"],
            "completed_cases": [],
            "started_cases": [],
            "active_case": None,
            "case_lifetimes": {},
            "workers": {},
            "tensor_written_bytes": 0,
            "spool_bytes_by_group": {},
            "diagnostic_dumps": {
                "selected_fixture_ids": selected,
                "written_bytes": 0,
                "fixture_written_bytes": {},
            },
        }
    else:
        ledger = read_json(folder / "ledger.json")
        if (
            ledger["plan_sha256"] != view["plan_sha256"]
            or ledger["numerical_plan_sha256"] != view["numerical_plan_sha256"]
            or set(ledger["workers"]) != {"A"}
            or not ledger["workers"]["A"]["passed"]
            or ledger["completed_cases"]
            != [
                case["case_id"]
                for case in view["execution_order"]
                if case["implementation_id"] == "A"
            ]
            or ledger["started_cases"] != ledger["completed_cases"]
            or ledger["active_case"] is not None
            or set(ledger["case_lifetimes"]) != set(ledger["completed_cases"])
        ):
            raise ValueError("B requires one complete, passing A numerical worker")
        budget = _restore_budget(folder, view, ledger)
    dumps = DiagnosticDump(
        folder / "dumps",
        selected,
        budget=budget,
        per_fixture_bytes=limits["dump_bytes_per_fixture"],
        total_bytes=limits["persisted_dump_bytes"],
    )
    result = {
        "implementation_id": implementation_id,
        "complete": False,
        "passed": False,
        "completed_cases": [],
        "errors": [],
    }
    ledger["workers"][implementation_id] = result

    def checkpoint():
        ledger.update(
            tensor_written_bytes=budget.written_bytes,
            spool_bytes_by_group=dict(budget.group_written_bytes),
            diagnostic_dumps={
                "selected_fixture_ids": list(dumps.selected_fixture_ids),
                "written_bytes": dumps.written_bytes,
                "fixture_written_bytes": dict(dumps.fixture_written_bytes),
            },
        )
        _save_ledger(folder, ledger)

    checkpoint()
    try:
        for case in planned:
            started_ns = time.perf_counter_ns()
            case_deadline_ns = min(
                deadline_ns, started_ns + limits["case_timeout_s"] * 1_000_000_000
            )
            remaining = (case_deadline_ns - started_ns) / 1e9
            if remaining <= 0:
                raise TimeoutError("parent numerical deadline exhausted")
            ledger["active_case"] = {
                "case_id": case["case_id"],
                "started_ns": started_ns,
                "deadline_ns": case_deadline_ns,
            }
            ledger["started_cases"].append(case["case_id"])
            checkpoint()
            remaining = (case_deadline_ns - time.perf_counter_ns()) / 1e9
            if remaining <= 0:
                raise TimeoutError("numerical case deadline exhausted before execution")
            value = execute(model, view, case, folder, budget, dumps, time.monotonic() + remaining)
            if value["status"] != "complete":
                raise RuntimeError("numerical case did not complete")
            for comparison_id in value["comparisons"]:
                summary = read_json(folder / "comparisons" / (comparison_id + ".summary.json"))
                if summary["required_failures"] or summary["behavior_failures"]:
                    result["errors"].append(
                        {
                            "case_id": case["case_id"],
                            "comparison_id": comparison_id,
                            "message": "required numerical or actual-decision gate failed",
                        }
                    )
            if case is planned[-1] or result["errors"]:
                # Keep the watchdog marker live through the final dump index export.
                dumps.close()
            finished_ns = time.perf_counter_ns()
            if (
                finished_ns > case_deadline_ns
                or not 0 <= value["elapsed_s"] <= limits["case_timeout_s"]
            ):
                raise TimeoutError("numerical case exceeded its frozen lifetime deadline")
            ledger["case_lifetimes"][case["case_id"]] = {
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "deadline_ns": case_deadline_ns,
            }
            ledger["completed_cases"].append(case["case_id"])
            result["completed_cases"].append(case["case_id"])
            ledger["active_case"] = None
            checkpoint()
            if result["errors"]:
                break
            if after_case is not None:
                after_case(case, value)
        result["complete"] = result["completed_cases"] == [case["case_id"] for case in planned]
        result["passed"] = result["complete"] and not result["errors"]
    except (Exception, KeyboardInterrupt) as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        try:
            dumps.close()
        except (Exception, KeyboardInterrupt) as exc:
            result["complete"] = result["passed"] = False
            result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        checkpoint()
    return {**result, "ledger": deepcopy(ledger)}


def audit_numerical(output_dir, parent_plan):
    """Offline audit using the same exact boundary/trace and typed-payload audits."""
    return _audit_numerical_view(output_dir, _view(parent_plan))


def _audit_numerical_view(output_dir, view, *, audit_case=None):
    """Require a complete, independently audited numerical prefix."""
    result = _audit_numerical_prefix(output_dir, view, audit_case=audit_case)
    result["artifact_type"] = view["artifact_type"].replace("_plan", "_report")
    workers = result.get("ledger", {}).get("workers", {})
    if (
        result["valid_prefix"]
        and not result["missing_case_ids"]
        and all(worker["complete"] for worker in workers.values())
    ):
        result["complete"] = True
        result["evidence_status"] = "complete"
        result["passed"] = not result["known_required_failure"] and all(
            worker["passed"] for worker in workers.values()
        )
    elif not result["errors"]:
        result["errors"].append({"type": "ValueError", "message": "incomplete numerical ledger"})
    return result


def _audit_numerical_prefix(output_dir, view, *, audit_case=None):
    """Audit a settled prefix without turning its missing suffix into corruption.

    Original case/comparison rows and the full frozen plan hash are preserved.
    Error aggregates are reconstructed from recorded statistics; retained typed
    anchors are byte-verified, not a recomputation of all native tensor errors.
    An active unfinished case is explicitly unaudited and cannot establish a
    trusted failure through this completed-prefix interface.
    """
    from .report import _audit_case, _audit_comparison, _audit_raw_evidence

    def _require(value, message):
        if not value:
            raise ValueError(message)

    folder = Path(output_dir) / "numerical"
    planned = [case["case_id"] for case in view["execution_order"]]
    result = {
        "schema_version": 1,
        "artifact_type": view["artifact_type"].replace("_plan", "_prefix_report"),
        "plan_sha256": view["plan_sha256"],
        "complete": False,
        "passed": False,
        "valid_prefix": False,
        "known_required_failure": False,
        "evidence_status": "incomplete",
        "expected_case_ids": planned,
        "completed_cases": [],
        "missing_case_ids": planned,
        "comparisons": [],
        "required_failures": 0,
        "behavior_failures": 0,
        "errors": [],
        "counts": {
            "planned_cases": len(planned),
            "verified_cases": 0,
            "planned_comparisons": len(view["comparison_order"]),
            "verified_comparisons": 0,
        },
    }
    if not (folder / "ledger.json").exists():
        result["missing_evidence"] = "numerical ledger has not been published"
        return result
    try:
        ledger = read_json(folder / "ledger.json")
        result["ledger"] = ledger
        _require(
            ledger["plan_sha256"] == view["plan_sha256"]
            and ledger["numerical_plan_sha256"] == view["numerical_plan_sha256"],
            "prefix ledger identity differs",
        )
        completed = ledger["completed_cases"]
        _require(
            isinstance(completed, list) and completed == planned[: len(completed)],
            "completed numerical cases are not the frozen prefix",
        )
        started = ledger["started_cases"]
        _require(
            started == planned[: len(started)]
            and len(completed) <= len(started) <= len(completed) + 1,
            "started numerical cases are not the bounded prefix",
        )
        result["completed_cases"] = list(completed)
        result["missing_case_ids"] = planned[len(completed) :]
        active = ledger["active_case"]
        if active is not None:
            _require(
                len(started) == len(completed) + 1 and active["case_id"] == started[-1],
                "active case differs from next frozen case",
            )
            _require(
                set(active) == {"case_id", "started_ns", "deadline_ns"}
                and all(type(active[k]) is int for k in ("started_ns", "deadline_ns"))
                and 0 < active["started_ns"] < active["deadline_ns"]
                and active["deadline_ns"] - active["started_ns"]
                <= view["contract"]["limits"]["case_timeout_s"] * 10**9,
                "pending numerical case marker is malformed or over budget",
            )
            result["pending_case"] = active
            result["missing_evidence"] = (
                "active case and its partial tensor/dump writes remain unaudited"
            )
            return result
        _require(started == completed, "unsettled started case has no active marker")
        _require(
            set(ledger["case_lifetimes"]) == set(completed), "prefix lifetime coverage differs"
        )
        actual = (
            sorted(path.name for path in (folder / "cases").iterdir() if path.is_dir())
            if (folder / "cases").exists()
            else []
        )
        _require(actual == sorted(completed), "unexpected or missing completed case directories")
        subset = {
            **view,
            "execution_order": view["execution_order"][: len(completed)],
            "comparison_order": [
                r for r in view["comparison_order"] if r["candidate_case_id"] in completed
            ],
        }
        fixtures = {f["fixture_id"]: f for f in view["suite"]["fixtures"]}
        cases, previous_end = {}, 0
        for case in subset["execution_order"]:
            case_id = case["case_id"]
            life = ledger["case_lifetimes"][case_id]
            start, end, deadline = (
                life[key] for key in ("started_ns", "finished_ns", "deadline_ns")
            )
            _require(
                all(type(v) is int and v > 0 for v in (start, end, deadline))
                and previous_end <= start <= end <= deadline
                and deadline - start <= view["contract"]["limits"]["case_timeout_s"] * 10**9,
                "prefix case lifetime exceeded its frozen deadline",
            )
            previous_end = end
            value = read_json(folder / "cases" / case_id / "result.json")
            _audit_case(value, view, case, fixtures)
            if audit_case is not None:
                audit_case(case, value, life)
            cases[case_id] = value
            result["counts"]["verified_cases"] += 1
        indices, reverse = {}, {}
        for comparison in subset["comparison_order"]:
            _require(
                comparison["reference_case_id"] in cases,
                "completed comparison lacks its earlier reference",
            )
            case_id = comparison["candidate_case_id"]
            seen, identities = indices.setdefault(case_id, {}), reverse.setdefault(case_id, {})

            def observe(index, fixture_id, key):
                identity = (fixture_id, key)
                _require(
                    (index not in seen or seen[index] == identity)
                    and (identity not in identities or identities[identity] == index),
                    "inconsistent prefix cross-stream observation order",
                )
                seen[index], identities[identity] = identity, index

            result["comparisons"].append(
                _audit_comparison(folder, view, comparison, cases, fixtures, observe)
            )
            result["counts"]["verified_comparisons"] += 1
        for case_id, seen in indices.items():
            _require(
                sorted(seen) == list(range(1, cases[case_id]["observed_boundaries"] + 1)),
                "prefix global observation coverage differs",
            )
        comparison_ids = [c["comparison_id"] for c in subset["comparison_order"]]
        for suffix in (".jsonl", ".summary.json"):
            actual = sorted(
                p.name.removesuffix(suffix) for p in (folder / "comparisons").glob("*" + suffix)
            )
            _require(
                actual == sorted(comparison_ids),
                "unexpected or missing prefix comparison artifacts",
            )
        raw = _audit_raw_evidence(folder, subset, cases, fixtures, result["comparisons"], ledger)
        groups = {}
        for spool in raw["retained_references"]:
            group = cases[spool["namespace"]]["case"]["spool_group"]
            groups[group] = groups.get(group, 0) + spool["size_bytes"]
        groups.update(ledger["diagnostic_dumps"]["fixture_written_bytes"])
        _require(
            groups == ledger["spool_bytes_by_group"]
            and sum(groups.values()) == ledger["tensor_written_bytes"],
            "prefix spool ledger differs from verified retained bytes",
        )
        caps = view["contract"]["limits"]
        _require(
            ledger["tensor_written_bytes"] <= caps["cumulative_spool_written_bytes"]
            and all(0 <= size <= caps["group_spool_bytes"] for size in groups.values())
            and ledger["diagnostic_dumps"]["written_bytes"] <= caps["persisted_dump_bytes"],
            "prefix tensor budget exceeded",
        )
        workers = ledger["workers"]
        sides = [side for side in ("A", "B") if side in workers]
        _require(
            sides and list(workers) == sides and sides in (["A"], ["A", "B"]),
            "prefix worker order differs",
        )
        failures = {
            r["comparison_id"]
            for r in result["comparisons"]
            if r["required_failures"] or r["behavior_failures"]
        }
        for side in sides:
            worker = workers[side]
            expected = [
                c["case_id"] for c in view["execution_order"] if c["implementation_id"] == side
            ]
            observed = [c for c in completed if c in expected]
            failed_ids = {
                c["comparison_id"]
                for c in subset["comparison_order"]
                if c["candidate_case_id"] in observed and c["comparison_id"] in failures
            }
            _require(
                worker["implementation_id"] == side
                and worker["completed_cases"] == observed
                and type(worker["complete"]) is bool
                and type(worker["passed"]) is bool
                and isinstance(worker["errors"], list),
                "prefix worker summary identity differs",
            )
            _require(
                not worker["complete"] or observed == expected,
                "incomplete side falsely marked complete",
            )
            _require(
                worker["passed"] is (worker["complete"] and not worker["errors"]),
                "prefix worker summary pass flag differs from completion/errors",
            )
            claimed = {
                error["comparison_id"] for error in worker["errors"] if "comparison_id" in error
            }
            _require(
                claimed == failed_ids, "worker required-failure claims differ from audited streams"
            )
            if side == "B":
                _require(
                    workers["A"]["passed"], "B started after an unqualified A numerical worker"
                )
        _require(
            all(c["implementation_id"] in sides for c in subset["execution_order"]),
            "completed cases have no owning worker ledger",
        )
        result.update(
            valid_prefix=True,
            raw_evidence=raw,
            required_failures=sum(r["required_failures"] for r in result["comparisons"]),
            behavior_failures=sum(r["behavior_failures"] for r in result["comparisons"]),
        )
        result["known_required_failure"] = bool(
            result["required_failures"] or result["behavior_failures"]
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["evidence_status"] = "invalid"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    return result
