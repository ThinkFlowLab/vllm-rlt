"""Synthetic observations test frozen pairing and cross-evidence gates, never GPU speed."""

from copy import deepcopy

import pytest
import test_benchmark_ab_schema as schema_fixtures

from vllm_lt.benchmarks import ab_report

ab_plan = schema_fixtures.ab_plan


def records_for(plan):
    records = []
    for row in plan["execution_order"]:
        if row["kind"] != "benchmark":
            continue
        pool = plan["workload_stats"][row["workload_id"]]["pool_bytes"]
        result = {
            **row,
            "setup_ns": 100,
            "comparison_eligible": row["phase"] == "measured",
            "requests": [
                {
                    "request_id": "synthetic",
                    "token_ids": [1, 2],
                    "exit_depths": [4, 4],
                    "finished": True,
                    "finish_reason": "length",
                }
            ],
            "counts": {
                key: []
                for key in (
                    "gate_probabilities",
                    "logical_copy_bytes",
                    "nonfinite_gate_probabilities",
                    "recurrent_depth_counts",
                    "recurrent_occupancy",
                    "request_work",
                    "stage_counts",
                    "stage_tokens",
                )
            },
            "memory": {
                "pool_bytes": pool,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "after_engine_release": {"allocated_bytes": 0, "reserved_bytes": 0},
            },
        }
        records.append(
            {
                "planned": deepcopy(row),
                "run_id": row["run_id"],
                "status": "complete",
                "result": result,
                "validation_errors": [],
                "comparison_eligible": row["phase"] == "measured",
                "recomputed_metrics": {
                    "generated_tokens_per_second": 115.0
                    if row["implementation_id"] == "B"
                    else 100.0
                },
            }
        )
    return records


def target(records, implementation="B", repetition=1, cell="W1-refill"):
    return next(
        r
        for r in records
        if r["planned"]["phase"] == "measured"
        and r["planned"]["cell_id"] == cell
        and r["planned"]["implementation_id"] == implementation
        and r["planned"]["repetition"] == repetition
    )


def test_two_pairs_each_cell_exclude_all_other_phases(ab_plan):
    records = records_for(ab_plan)
    # Poison every excluded observation; it must never affect a timing decision.
    for row in records:
        if row["planned"]["phase"] != "measured":
            row["recomputed_metrics"] = {"generated_tokens_per_second": float("nan")}
    result = ab_report.pair_results(ab_plan, records)
    assert len(result) == 7 and all(cell["status"] == "passed" for cell in result)
    assert all(len(cell["pairs"]) == 2 for cell in result)
    assert result[0]["candidate_range"] == [115.0, 115.0]


@pytest.mark.parametrize(
    "metric,value",
    [
        ("throughput", 109),
        ("setup_ns", 100_000_101),
        ("peak_allocated_bytes", 67_108_965),
        ("peak_reserved_bytes", 67_109_065),
    ],
)
def test_one_bad_pair_cannot_be_averaged_away(ab_plan, metric, value):
    rows = records_for(ab_plan)
    row = target(rows)
    if metric == "throughput":
        row["recomputed_metrics"]["generated_tokens_per_second"] = value
    elif metric == "setup_ns":
        row["result"][metric] = value
    else:
        row["result"]["memory"][metric] = value
        row["result"]["memory"]["peak_reserved_bytes"] = max(
            value, row["result"]["memory"]["peak_reserved_bytes"]
        )
    result = ab_report.pair_results(ab_plan, rows)[0]
    assert result["status"] == "failed"
    assert result["pairs"][0]["status"] == "failed"
    assert result["pairs"][1]["status"] == "passed"


def test_passing_ratios_with_overlapping_ranges_are_inconclusive(ab_plan):
    rows = records_for(ab_plan)
    target(rows, "A", 2)["recomputed_metrics"]["generated_tokens_per_second"] = 110
    target(rows, "B", 1)["recomputed_metrics"]["generated_tokens_per_second"] = 110
    target(rows, "B", 2)["recomputed_metrics"]["generated_tokens_per_second"] = 122
    result = ab_report.pair_results(ab_plan, rows)[0]
    assert all(pair["status"] == "passed" for pair in result["pairs"])
    assert result["status"] == "inconclusive" and not result["strict_range_separation"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "tokens",
        "depths",
        "gates",
        "occupancy",
        "work",
        "pool",
        "eligibility",
    ],
)
def test_incomparable_work_or_missing_observation_cannot_qualify(ab_plan, change):
    rows = records_for(ab_plan)
    row = target(rows)
    if change == "missing":
        rows.remove(row)
    elif change == "duplicate":
        rows.append(deepcopy(row))
    elif change in ("tokens", "depths"):
        key = "token_ids" if change == "tokens" else "exit_depths"
        row["result"]["requests"][0][key][0] += 1
    elif change in ("gates", "occupancy", "work"):
        key = {
            "gates": "gate_probabilities",
            "occupancy": "recurrent_occupancy",
            "work": "request_work",
        }[change]
        row["result"]["counts"][key].append(1)
    elif change == "pool":
        # Matching incorrect pool sizes must also fail.
        row["result"]["memory"]["pool_bytes"] = 0
        target(rows, "A")["result"]["memory"]["pool_bytes"] = 0
    else:
        row["comparison_eligible"] = False
    cell = ab_report.pair_results(ab_plan, rows)[0]
    assert cell["status"] == "inconclusive"
    assert cell["pairs"][0]["status"] == "invalid"


def test_control_cell_has_its_own_five_percent_gate(ab_plan):
    rows = records_for(ab_plan)
    target(rows, cell="W5-no_refill")["recomputed_metrics"]["generated_tokens_per_second"] = 94.9
    result = ab_report.pair_results(ab_plan, rows)
    assert result[0]["status"] == "passed"
    assert result[-1]["status"] == "failed"
