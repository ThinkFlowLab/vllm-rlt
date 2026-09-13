"""CPU-only frozen cases and byte budgets for the bounded Q1 numerical suite."""

import math
import subprocess
from pathlib import Path

from vllm_lt.benchmarks.schema import (
    _constants,
    _digest,
    _file_record,
    _integer,
    _keys,
    _model_files,
    _source_manifest,
    _text,
    _tokens,
    _version,
    read_json,
    write_json,
)
from vllm_lt.models.config import OURO_MODEL_ID, OURO_REVISION, OuroConfig

__all__ = [
    "load_suite",
    "load_contract",
    "make_plan",
    "validate_plan_integrity",
    "verify_plan",
    "read_json",
    "write_json",
    "dependency_manifest",
]

PROJECTION = "loop_gate_logits_full_kv_v1"
DTYPES = ("float32", "bfloat16")
LENGTHS = (16, 64, 128, 256)
OPERATIONS = (
    "attention_input",
    "query",
    "key",
    "value",
    "attention_output",
    "layer_output",
    "loop_hidden",
    "gate_logits",
    "logits",
)
ORIGINAL_REVISION = "fa7a2ca5b833f2dd0f3e8db53ce111a7406496f7"
ORIGINAL_PROMPTS = [
    [504, 3575, 282, 4649, 314],
    [34, 1232, 216, 34, 446],
    [504, 7011, 282, 3380, 314],
]
OFFICIAL_HASHES = {
    "modeling_ouro.py": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
    "configuration_ouro.py": "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb",
}
FORCED_DECODE_PATTERNS = {
    2: [2, 4, 3, 2, 4, 2, 3, 4],
    3: [4, 2, 3, 4, 2, 4, 3, 2],
}
FIXTURE_HASH_FIELDS = (
    "fixtures",
    "groups",
    "live_gate_fixture_ids",
    "official_fixture_ids",
    "feasibility_fixtures",
    "original_reproduction",
)


def _sequence(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")


def _fixture(value, length, *, feasibility=False):
    required = (
        "fixture_id",
        "prompt_token_ids",
        "continuation_input_ids",
        "history_policy",
        "forced_exit_depths",
    )
    _keys(value, required, optional=("dtype",) if feasibility else (), name="fixture")
    _text(value["fixture_id"], "fixture_id")
    _tokens(value["prompt_token_ids"], length, "prompt_token_ids")
    _tokens(value["continuation_input_ids"], 8, "continuation_input_ids")
    depths = value["forced_exit_depths"]
    _sequence(depths, 9, "forced_exit_depths")
    if any(type(depth) is not int or depth not in (2, 3, 4) for depth in depths):
        raise ValueError("forced_exit_depths require integer depths 2 through 4")
    if depths[0] != 4:
        raise ValueError("prediction zero must execute depth four")
    if value["history_policy"] not in ("fixed", "forced"):
        raise ValueError("unsupported history_policy")
    if value["history_policy"] == "fixed" and depths != [4] * 9:
        raise ValueError("fixed history must execute depth four at every prediction")
    if value["history_policy"] == "forced" and not any(
        left < right for left, right in zip(depths[1:], depths[2:])
    ):
        raise ValueError("forced history must include shallow-to-deeper transitions")
    if feasibility and value.get("dtype") not in DTYPES:
        raise ValueError("feasibility fixture requires an explicit supported dtype")


def _validate_suite(suite):
    _keys(
        suite,
        (
            "schema_version",
            "artifact_type",
            "suite_id",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "provenance",
            *FIXTURE_HASH_FIELDS,
            "fixtures_sha256",
        ),
        name="validation suite",
    )
    _version(suite, "validation_suite")
    _text(suite["suite_id"], "suite_id")
    if (suite["model_id"], suite["model_revision"], suite["tokenizer_revision"]) != (
        OURO_MODEL_ID,
        OURO_REVISION,
        OURO_REVISION,
    ):
        raise ValueError("Q1 requires the pinned Ouro model/tokenizer revision")
    provenance = suite["provenance"]
    _keys(
        provenance,
        (
            "kind",
            "text_sources",
            "continuation_text_sources",
            "prompt_construction",
            "continuation_construction",
            "feasibility_text_sources",
            "feasibility_continuation_text",
            "feasibility_construction",
            "tokenizer",
        ),
        name="provenance",
    )
    if provenance["kind"] != "synthetic_cpu_tokenized":
        raise ValueError("unsupported fixture provenance")
    for key in ("text_sources", "continuation_text_sources", "feasibility_text_sources"):
        _sequence(provenance[key], 2 if key == "feasibility_text_sources" else 4, key)
        for text in provenance[key]:
            _text(text, key)
    for key in (
        "prompt_construction",
        "continuation_construction",
        "feasibility_continuation_text",
        "feasibility_construction",
    ):
        _text(provenance[key], key)
    tokenizer = provenance["tokenizer"]
    _keys(
        tokenizer,
        ("filename", "sha256", "size_bytes", "library", "library_version", "add_special_tokens"),
        name="tokenizer provenance",
    )
    for key in ("sha256", "library", "library_version"):
        _text(tokenizer[key], key)
    _integer(tokenizer["size_bytes"], "tokenizer size_bytes", 1)
    if tokenizer["filename"] != "tokenizer.json" or tokenizer["add_special_tokens"] is not False:
        raise ValueError("fixtures require tokenizer.json without added special tokens")
    fixtures = suite["fixtures"]
    _sequence(fixtures, 16, "fixtures")
    expected_ids = [f"Q1-L{length}-F{group}" for length in LENGTHS for group in range(4)]
    for fixture, expected_id in zip(fixtures, expected_ids):
        length = int(expected_id.split("-")[1][1:])
        _fixture(fixture, length)
        if fixture["fixture_id"] != expected_id:
            raise ValueError("main fixture IDs/order must be unique and match declared lengths")
        expected_policy = "fixed" if expected_id[-1] in "01" else "forced"
        if fixture["history_policy"] != expected_policy:
            raise ValueError("Q1 requires eight fixed and eight forced fixtures")
        if expected_policy == "forced":
            pattern = FORCED_DECODE_PATTERNS[int(expected_id[-1])]
            rotation = LENGTHS.index(length)
            if fixture["forced_exit_depths"] != [4] + pattern[rotation:] + pattern[:rotation]:
                raise ValueError("forced groups require the frozen heterogeneous depth rotations")
    groups = suite["groups"]
    _sequence(groups, 4, "packed groups")
    for index, group in enumerate(groups):
        _keys(group, ("group_id", "fixture_ids"), name="packed group")
        if group != {
            "group_id": f"Q1-G{index}",
            "fixture_ids": [f"Q1-L{length}-F{index}" for length in LENGTHS],
        }:
            raise ValueError("packed groups must explicitly contain one of each prompt length")
    for key in ("live_gate_fixture_ids", "official_fixture_ids"):
        if suite[key] != [f"Q1-L{length}-F0" for length in LENGTHS]:
            raise ValueError(f"{key} must equal the frozen four-fixture fixed subset")
    _sequence(suite["feasibility_fixtures"], 2, "feasibility_fixtures")
    main_histories = {tuple(row["prompt_token_ids"]) for row in fixtures}
    for dtype, fixture in zip(DTYPES, suite["feasibility_fixtures"]):
        _fixture(fixture, 24, feasibility=True)
        if fixture["dtype"] != dtype or fixture["fixture_id"] != f"Q1-feasibility-{dtype}":
            raise ValueError("exactly one disjoint feasibility fixture is required per dtype")
        history = tuple(fixture["prompt_token_ids"])
        if history in main_histories:
            raise ValueError("feasibility prompt histories must be disjoint")
        main_histories.add(history)
    original = suite["original_reproduction"]
    _keys(
        original,
        (
            "sources",
            "prompt_texts",
            "prompt_token_ids",
            "block_size",
            "num_blocks",
            "depths",
            "tf32",
            "bf16_reduced_precision_reduction",
            "scope",
        ),
        name="original reproduction",
    )
    if original["prompt_token_ids"] != ORIGINAL_PROMPTS:
        raise ValueError("original reproduction must preserve the recorded three input histories")
    for prompt in original["prompt_token_ids"]:
        _tokens(prompt, 5, "original prompt_token_ids")
    _sequence(original["prompt_texts"], 3, "original prompt texts")
    for text in original["prompt_texts"]:
        _text(text, "original prompt text")
    _constants(
        {
            key: original[key]
            for key in (
                "block_size",
                "num_blocks",
                "tf32",
                "bf16_reduced_precision_reduction",
                "scope",
            )
        },
        {
            "block_size": 8,
            "num_blocks": 32,
            "tf32": False,
            "bf16_reduced_precision_reduction": False,
            "scope": "original_all_prompt_positions_prefill_logits_only",
        },
        "original controls",
    )
    if original["depths"] != [1, 2, 3, 4] or any(type(x) is not int for x in original["depths"]):
        raise ValueError("original reproduction requires all four loop depths")
    _sequence(original["sources"], 2, "original source records")
    for dtype, record in zip(DTYPES, original["sources"]):
        _keys(
            record,
            ("dtype", "commit", "path", "git_blob_sha1", "sha256", "size_bytes", "atol", "rtol"),
            name="original source record",
        )
        suffix = "fp32" if dtype == "float32" else "bf16"
        atol, rtol = (0.001, 0.0001) if dtype == "float32" else (0.25, 0.02)
        if (record["dtype"], record["commit"], record["path"]) != (
            dtype,
            ORIGINAL_REVISION,
            f"docs/validation/checkpoint-{suffix}.json",
        ):
            raise ValueError("original record provenance differs from the retained source")
        _constants(
            {"atol": record["atol"], "rtol": record["rtol"]},
            {"atol": atol, "rtol": rtol},
            "original logit tolerance",
        )
        _integer(record["size_bytes"], "original record size", 1)
        for key in ("git_blob_sha1", "sha256"):
            _text(record[key], key)
    if suite["fixtures_sha256"] != _digest({key: suite[key] for key in FIXTURE_HASH_FIELDS}):
        raise ValueError("fixtures_sha256 mismatch")


def load_suite(path):
    suite = read_json(path)
    _validate_suite(suite)
    return suite


def _validate_contract(contract, *, resolved=False):
    _keys(
        contract,
        (
            "schema_version",
            "artifact_type",
            "contract_id",
            "hypothesis",
            "isolated_variables",
            "acceptance_criterion",
            "stop_conditions",
            "dtypes",
            "engine",
            "arithmetic",
            "controls",
            "limits",
            "diagnostics",
            "comparison_policy",
            "official",
        ),
        name="validation contract",
    )
    _version(contract, "validation_contract")
    for key in ("contract_id", "hypothesis", "acceptance_criterion"):
        _text(contract[key], key)
    for key in ("isolated_variables", "stop_conditions"):
        if not isinstance(contract[key], list) or not contract[key]:
            raise ValueError(f"{key} must be a nonempty list")
        for value in contract[key]:
            _text(value, key)
    if contract["dtypes"] != list(DTYPES):
        raise ValueError("Q1 requires ordered FP32 and BF16 passes")
    engine = contract["engine"]
    _keys(engine, ("cache", "scheduler", "sampling"), name="engine")
    _constants(engine["cache"], {"num_blocks": 160, "block_size": 16}, "cache")
    _constants(
        engine["scheduler"],
        {"max_num_seqs": 4, "max_num_batched_tokens": 64, "min_coda_batch_size": 1},
        "scheduler",
    )
    _constants(
        engine["sampling"],
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 0,
            "min_loops": 2,
            "max_loops": 4,
            "max_tokens": 9,
            "ignore_eos": True,
        },
        "sampling",
    )
    _constants(
        contract["arithmetic"],
        {
            "allow_tf32": False,
            "cudnn_allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False,
        },
        "arithmetic",
    )
    controls = contract["controls"]
    _keys(
        controls,
        (
            "cpu_threads",
            "interop_threads",
            "seed",
            "gpu_ids",
            "numa_policy",
            "cache_policy",
            "residency_policy",
            "spool_dtype",
        ),
        name="controls",
    )
    _constants(
        {key: value for key, value in controls.items() if key != "gpu_ids"},
        {
            "cpu_threads": 1,
            "interop_threads": 1,
            "seed": 0,
            "numa_policy": "inherit-and-record",
            "cache_policy": "fresh-state-preserve-allocator",
            "residency_policy": "reuse-immutable-weights-release-per-case-state",
            "spool_dtype": "original-model-dtype",
        },
        "controls",
    )
    ids = controls["gpu_ids"]
    if ids is not None:
        _sequence(ids, 1, "gpu_ids")
        _integer(ids[0], "physical GPU ID")
    elif resolved:
        raise ValueError("plan requires one explicit physical GPU ID before device execution")
    _constants(
        contract["limits"],
        {
            "case_timeout_s": 600,
            "total_timeout_s": 7200,
            "implementation_executions": 168,
            "passes_per_case": 1,
            "feasibility_fixtures_per_dtype": 1,
            "transient_matching_bytes": 64 * 1024**2,
            "group_spool_bytes": 2 * 1024**3,
            "cumulative_spool_written_bytes": 8 * 1024**3,
            "total_disk_bytes": 12 * 1024**3,
            "spool_index_record_bytes": 1024,
            "comparison_record_bytes": 2048,
            "persisted_dump_bytes": 256 * 1024**2,
            "dump_bytes_per_fixture": 64 * 1024**2,
            "dump_fixture_count": 4,
        },
        "limits",
    )
    diagnostic = contract["diagnostics"]
    _keys(
        diagnostic,
        (
            "positions",
            "operations",
            "kv_selection",
            "kv_chunk_positions",
            "preselected_dump_fixture_ids",
            "preselected_dump_dtype",
            "preselected_dump_comparison_kind",
            "additional_failure_fixtures",
            "failure_selection_order",
            "dump_selection",
            "reference_retention",
        ),
        name="diagnostics",
    )
    if diagnostic["operations"] != list(OPERATIONS):
        raise ValueError("diagnostic operations differ from the frozen observer interface")
    if diagnostic["preselected_dump_fixture_ids"] != ["Q1-L16-F2", "Q1-L256-F0"]:
        raise ValueError("exactly two diagnostic fixtures must be preselected before execution")
    _constants(
        {
            key: value
            for key, value in diagnostic.items()
            if key not in ("operations", "preselected_dump_fixture_ids")
        },
        {
            "positions": "last-prompt-token-and-eight-continuation-inputs",
            "kv_selection": "all-populated-final-positions-all-depths-and-layers",
            "kv_chunk_positions": 4,
            "preselected_dump_dtype": "bfloat16",
            "preselected_dump_comparison_kind": "same_dtype_fidelity",
            "additional_failure_fixtures": 2,
            "failure_selection_order": "execution-order-then-observed-boundary-order",
            "dump_selection": "candidate-observations-from-selection-point-forward-until-cap",
            "reference_retention": "retain-fp32-oracle-until-same-fixture-bf16-sensitivity",
        },
        "diagnostics",
    )
    policy = contract["comparison_policy"]
    _keys(
        policy,
        (
            "policy_id",
            "logits",
            "required",
            "diagnostic_only",
            "near_ties",
            "live_gate",
            "live_decisions",
            "original_top1_rule",
            "rationale",
        ),
        name="comparison policy",
    )
    _text(policy["policy_id"], "policy_id")
    _text(policy["rationale"], "policy rationale")
    _keys(policy["logits"], DTYPES, name="logit policies")
    for dtype, atol, rtol in (("float32", 0.001, 0.0001), ("bfloat16", 0.25, 0.02)):
        _constants(policy["logits"][dtype], {"atol": atol, "rtol": rtol}, f"{dtype} logits")
    if policy["required"] != [
        "finite",
        "final_logits_allclose",
        "actual_top1_equal",
        "forced_depth_work",
        "history_and_cleanup_invariants",
    ]:
        raise ValueError("required qualification gates must remain explicit")
    if policy["diagnostic_only"] != [*OPERATIONS[:-1], "populated_kv", "bf16_fp32_sensitivity"]:
        raise ValueError("hidden/KV/gate diagnostics cannot acquire invented acceptance bounds")
    _constants(
        {
            key: policy[key]
            for key in ("near_ties", "live_gate", "live_decisions", "original_top1_rule")
        },
        {
            "near_ties": "report-top-two-margins-no-retroactive-top1-exceptions",
            "live_gate": "record-first-divergence-later-histories-incomparable",
            "live_decisions": "require-exact-token-and-exit-on-matched-prefix",
            "original_top1_rule": "depth-four-only",
        },
        "behavior policy",
    )
    _constants(
        contract["official"],
        {
            "revision": OURO_REVISION,
            "attention_backend": "eager",
            "use_cache": False,
            "exit_at_step": 3,
            "logits_to_keep": 0,
            "local_files_only": True,
            "optional_hub_kernels": False,
            "transformers_version": "4.55.0",
        },
        "official",
    )


def load_contract(path):
    contract = read_json(path)
    _validate_contract(contract)
    return contract


def _fixture_stats(fixture, dtype, config, *, live=False):
    size = 4 if dtype == "float32" else 2
    capacity = len(fixture["prompt_token_ids"]) + 8
    loop_sum = 36 if live else sum(fixture["forced_exit_depths"])
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    # attention_input/output/layer_output, query, key, value; then loop/gate.
    layer_elements = 3 * hidden + query + 2 * kv
    boundary_elements = loop_sum * (config.num_hidden_layers * layer_elements + hidden + 1)
    boundary_elements += 9 * config.vocab_size
    cache_elements = 2 * 4 * config.num_hidden_layers * capacity * kv
    return {
        "capacity": capacity,
        "prediction_points": 9,
        "selected_token_loops": loop_sum,
        "counts_are_upper_bounds": live,
        "selected_boundary_records": (6 * config.num_hidden_layers + 2) * loop_sum + 9,
        "kv_comparison_records": 2 * math.ceil(capacity / 4),
        "selected_boundary_bytes": boundary_elements * size,
        "final_kv_bytes": cache_elements * size,
        "reference_spool_payload_bytes": (boundary_elements + cache_elements) * size,
        "reserved_pages": 4 * math.ceil(capacity / 16),
    }


def _resolve_cases(suite, contract, config):
    fixtures = {row["fixture_id"]: row for row in suite["fixtures"]}
    fixtures.update({row["fixture_id"]: row for row in suite["feasibility_fixtures"]})
    for index, prompt in enumerate(suite["original_reproduction"]["prompt_token_ids"]):
        fixtures[f"Q1-original-{index}"] = {
            "fixture_id": f"Q1-original-{index}",
            "prompt_token_ids": prompt,
        }
    order = []
    policy_hash = _digest(contract["comparison_policy"])

    def add(family, dtype, implementation, backend, schedule, ids, group_id):
        legacy = family == "original"
        live = family == "live_gate"
        phase = "feasibility" if family == "feasibility" else "validation"
        histories = [fixtures[key] for key in ids]
        policy = "live" if live else "fixed" if legacy else histories[0]["history_policy"]
        case_id = f"{family}-{dtype}-{implementation}-{backend or 'dense'}-{schedule}-{group_id}"
        capacities = {
            row["fixture_id"]: len(row["prompt_token_ids"]) + (0 if legacy else 8)
            for row in histories
        }
        if max(capacities.values()) > config.max_position_embeddings:
            raise ValueError("fixture exceeds configured context capacity")
        block_size = 8 if legacy else contract["engine"]["cache"]["block_size"]
        num_blocks = 32 if legacy else contract["engine"]["cache"]["num_blocks"]
        needed = sum(4 * math.ceil(capacity / block_size) for capacity in capacities.values())
        if implementation in ("native", "legacy_packed") and needed > num_blocks:
            raise ValueError("declared packed group exceeds native page capacity")
        order.append(
            {
                "case_id": case_id,
                "family": family,
                "phase": phase,
                "dtype": dtype,
                "implementation": implementation,
                "backend": backend,
                "schedule": schedule,
                "fixture_ids": ids,
                "group_id": group_id,
                "history_mode": "original_prefill"
                if legacy
                else "live_gate"
                if live
                else "teacher_forced",
                "exit_policy": {
                    "mode": policy,
                    "threshold": 0.7 if live else 1.0,
                    "min_loops": 2,
                    "max_loops": 4,
                },
                "block_size": block_size,
                "num_blocks": num_blocks,
                "chunk_size": 15
                if legacy
                else contract["engine"]["scheduler"]["max_num_batched_tokens"],
                "max_tokens": 0 if legacy else 9,
                "expected_capacity": capacities,
                "max_steps": sum(
                    len(row["prompt_token_ids"]) + (4 if legacy else 49) for row in histories
                ),
                "fixture_sha256": _digest(histories),
                "policy_sha256": policy_hash,
            }
        )

    # Both feasibility sets precede every qualifying execution; no repeats/warmups.
    for fixture in suite["feasibility_fixtures"]:
        dtype, ids = fixture["dtype"], [fixture["fixture_id"]]
        add("feasibility", dtype, "oracle", None, "serial", ids, ids[0])
        for backend, schedule in (
            ("torch", "serial"),
            ("triton", "serial"),
            ("triton", "refill"),
            ("triton", "no_refill"),
        ):
            add("feasibility", dtype, "native", backend, schedule, ids, ids[0])
    for dtype in DTYPES:
        for group in suite["groups"]:
            ids, group_id = group["fixture_ids"], group["group_id"]
            for fixture_id in ids:
                add("main", dtype, "oracle", None, "serial", [fixture_id], group_id)
            for backend in ("torch", "triton"):
                for fixture_id in ids:
                    add("main", dtype, "native", backend, "serial", [fixture_id], group_id)
            for schedule in ("refill", "no_refill"):
                add("main", dtype, "native", "triton", schedule, ids, group_id)
        ids = suite["live_gate_fixture_ids"]
        for fixture_id in ids:
            add("live_gate", dtype, "oracle", None, "serial", [fixture_id], "Q1-live")
        for backend in ("torch", "triton"):
            for fixture_id in ids:
                add("live_gate", dtype, "native", backend, "serial", [fixture_id], "Q1-live")
        for schedule in ("refill", "no_refill"):
            add("live_gate", dtype, "native", "triton", schedule, ids, "Q1-live")
        for fixture_id in suite["official_fixture_ids"]:
            add("official", dtype, "official", None, "serial", [fixture_id], "Q1-official")
        original_ids = [f"Q1-original-{index}" for index in range(3)]
        for fixture_id in original_ids:
            add("original", dtype, "legacy_dense", None, "serial", [fixture_id], "Q1-original")
        for backend in ("torch", "triton"):
            add("original", dtype, "legacy_packed", backend, "packed", original_ids, "Q1-original")
    # Serial executions in a group need their fixture suffix to be unique.
    for row in order:
        if len(row["fixture_ids"]) == 1:
            row["case_id"] += "-" + row["fixture_ids"][0]
    if len(order) != 168 or len({row["case_id"] for row in order}) != 168:
        raise ValueError("Q1 must resolve to exactly 168 unique implementation executions")
    return order, fixtures


def _comparisons(order, fixtures, contract, config):
    comparisons = []
    oracles = {
        (row["family"], row["dtype"], row["fixture_ids"][0]): row
        for row in order
        if row["implementation"] in ("oracle", "legacy_dense")
    }

    def add(candidate, reference, fixture_id, kind, *, required=True, logits_only=False):
        legacy = candidate["family"] == "original"
        fixture = fixtures[fixture_id]
        if legacy:
            records, predictions = 4, 4 * len(fixture["prompt_token_ids"])
            upper = False
        else:
            stats = _fixture_stats(
                fixture, candidate["dtype"], config, live=candidate["family"] == "live_gate"
            )
            records = (
                9
                if logits_only
                else (stats["selected_boundary_records"] + stats["kv_comparison_records"])
            )
            predictions, upper = 9, stats["counts_are_upper_bounds"]
        comparisons.append(
            {
                "comparison_id": f"{candidate['case_id']}--{fixture_id}--{kind}",
                "family": candidate["family"],
                "dtype": candidate["dtype"],
                "reference_case_id": reference["case_id"],
                "candidate_case_id": candidate["case_id"],
                "fixture_id": fixture_id,
                "comparison_kind": kind,
                "required": required,
                "expected_prediction_points": predictions,
                "expected_boundary_records": records,
                "counts_are_upper_bounds": upper,
                "policy_sha256": _digest(contract["comparison_policy"]),
            }
        )

    for row in order:
        implementation = row["implementation"]
        if implementation in ("native", "legacy_packed", "official"):
            family = "main" if implementation == "official" else row["family"]
            for fixture_id in row["fixture_ids"]:
                reference = oracles[(family, row["dtype"], fixture_id)]
                add(
                    row,
                    reference,
                    fixture_id,
                    "official_fixed_fidelity"
                    if implementation == "official"
                    else "same_dtype_fidelity",
                    required=row["phase"] != "feasibility",
                    logits_only=implementation == "official",
                )
            if implementation == "legacy_packed" and row["backend"] == "triton":
                reference = next(
                    other
                    for other in order
                    if other["family"] == "original"
                    and other["dtype"] == row["dtype"]
                    and other["implementation"] == "legacy_packed"
                    and other["backend"] == "torch"
                )
                for fixture_id in row["fixture_ids"]:
                    add(row, reference, fixture_id, "legacy_triton_torch_fidelity")
        elif implementation == "oracle" and row["family"] == "main" and row["dtype"] == "bfloat16":
            fixture_id = row["fixture_ids"][0]
            add(
                row,
                oracles[("main", "float32", fixture_id)],
                fixture_id,
                "bf16_fp32_sensitivity",
                required=False,
            )
    return comparisons


def _resource_estimates(suite, contract, config, order, comparisons, fixtures):
    stats = {
        dtype: {
            key: _fixture_stats(fixture, dtype, config)
            for key, fixture in fixtures.items()
            if "continuation_input_ids" in fixture
        }
        for dtype in DTYPES
    }
    oracle_bytes = oracle_records = legacy_bytes = legacy_records = 0
    by_dtype = {dtype: 0 for dtype in DTYPES}
    group_bytes = {}
    for row in order:
        size = 4 if row["dtype"] == "float32" else 2
        if row["implementation"] == "oracle":
            fixture_id = row["fixture_ids"][0]
            item = _fixture_stats(
                fixtures[fixture_id], row["dtype"], config, live=row["family"] == "live_gate"
            )
            payload = item["reference_spool_payload_bytes"]
            oracle_bytes += payload
            by_dtype[row["dtype"]] += payload
            oracle_records += item["selected_boundary_records"] + item["kv_comparison_records"]
            group = f"{row['family']}-{row['dtype']}-{row['group_id']}"
            group_bytes[group] = group_bytes.get(group, 0) + payload
        elif row["implementation"] == "legacy_dense" or (
            row["implementation"] == "legacy_packed" and row["backend"] == "torch"
        ):
            count = sum(len(fixtures[key]["prompt_token_ids"]) for key in row["fixture_ids"])
            payload = count * 4 * config.vocab_size * size
            legacy_bytes += payload
            legacy_records += 4 * len(row["fixture_ids"])
            by_dtype[row["dtype"]] += payload
            group = f"{row['family']}-{row['dtype']}-{row['group_id']}"
            group_bytes[group] = group_bytes.get(group, 0) + payload
    total_payload = oracle_bytes + legacy_bytes
    native_pool = {}
    for dtype in DTYPES:
        size = 4 if dtype == "float32" else 2
        native_pool[dtype] = (
            2
            * config.num_hidden_layers
            * 160
            * 16
            * config.num_key_value_heads
            * config.head_dim
            * size
        )
    h, q, kv = (
        config.hidden_size,
        config.num_attention_heads * config.head_dim,
        config.num_key_value_heads * config.head_dim,
    )
    parameters = (
        2 * config.vocab_size * h
        + config.num_hidden_layers
        * (2 * h * q + 2 * h * kv + 3 * h * config.intermediate_size + 4 * h)
        + 2 * h
        + 1
    )
    result = {
        "implementation_executions": len(order),
        "execution_counts_by_family": {
            family: sum(row["family"] == family for row in order)
            for family in ("feasibility", "main", "live_gate", "official", "original")
        },
        "comparison_trajectories": len(comparisons),
        "comparison_boundary_records_upper_bound": sum(
            row["expected_boundary_records"] for row in comparisons
        ),
        "main_native_prediction_comparisons": sum(
            row["expected_prediction_points"]
            for row in comparisons
            if row["family"] == "main" and row["comparison_kind"] == "same_dtype_fidelity"
        ),
        "oracle_boundary_records_upper_bound": oracle_records,
        "reference_spool_records_upper_bound": oracle_records + legacy_records,
        "reference_spool_payload_bytes_upper_bound": total_payload,
        "reference_spool_payload_bytes_by_dtype": by_dtype,
        "reference_group_payload_bytes": group_bytes,
        "largest_reference_group_payload_bytes": max(group_bytes.values()),
        "legacy_reference_logits_bytes": legacy_bytes,
        "parameter_count": parameters,
        "parameter_bytes": {
            dtype: parameters * (4 if dtype == "float32" else 2) for dtype in DTYPES
        },
        "native_pool_bytes": native_pool,
        "largest_oracle_cache_bytes": {
            dtype: max(row["final_kv_bytes"] for row in stats[dtype].values()) for dtype in DTYPES
        },
        "largest_native_group_pages": max(
            sum(stats[dtype][key]["reserved_pages"] for key in group["fixture_ids"])
            for dtype in DTYPES
            for group in suite["groups"]
        ),
        "kv_snapshot_pair_bytes_fp32": 2 * 4 * config.num_hidden_layers * 4 * kv * 4,
        "working_memory_policy": (
            "four-position KV snapshots; stream matching values; enforce total transient cap"
        ),
        "activation_memory_policy": (
            "runtime capacity preflight and bounded feasibility; no invented activation estimate"
        ),
        "byte_estimate_scope": (
            "exact typed tensor payload upper bounds; live gates assume full depth; "
            "JSON/filesystem overhead separately capped by total disk budget"
        ),
        "fixture_stats": stats,
    }
    limits = contract["limits"]
    result["reference_index_bytes_upper_bound"] = (
        result["reference_spool_records_upper_bound"] * limits["spool_index_record_bytes"]
    )
    result["comparison_jsonl_bytes_upper_bound"] = (
        result["comparison_boundary_records_upper_bound"] * limits["comparison_record_bytes"]
    )
    result["auxiliary_artifact_allowance_bytes"] = 1024**3
    result["planned_artifact_bytes_with_auxiliary_allowance"] = (
        total_payload
        + result["reference_index_bytes_upper_bound"]
        + result["comparison_jsonl_bytes_upper_bound"]
        + limits["persisted_dump_bytes"]
        + result["auxiliary_artifact_allowance_bytes"]
    )
    result["artifact_budget_policy"] = (
        "12 GiB runtime cap includes tensor files, indexes, comparisons, dumps, "
        "summaries, manifests and logs; the 1 GiB auxiliary allowance is a planning reserve"
    )
    if total_payload + limits["persisted_dump_bytes"] > limits["cumulative_spool_written_bytes"]:
        raise ValueError(
            "reference payload and diagnostic dumps exceed the declared spool/disk budget"
        )
    if result["largest_reference_group_payload_bytes"] > limits["group_spool_bytes"]:
        raise ValueError("reference group exceeds the declared live spool byte budget")
    if result["planned_artifact_bytes_with_auxiliary_allowance"] > limits["total_disk_bytes"]:
        raise ValueError("planned evidence exceeds the declared total artifact byte budget")
    return result


def _official_files(path):
    records = []
    for name, expected in sorted(OFFICIAL_HASHES.items()):
        record = _file_record(path / name, relative_to=path)
        if record["sha256"] != expected:
            raise ValueError(f"reviewed official source hash mismatch: {name}")
        records.append(record)
    return records


def dependency_manifest():
    """Freeze installed dependency metadata without discovery of CUDA devices."""
    import importlib.metadata
    import sys

    import torch

    from .official import official_provenance

    distributions = [
        {"name": item.metadata["Name"], "version": item.version}
        for item in importlib.metadata.distributions()
    ]
    distributions.sort(key=lambda item: (item["name"].lower(), item["version"], item["name"]))
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "distributions": distributions,
        "official": official_provenance(),
    }


def _validate_dependencies(dependencies):
    _keys(
        dependencies,
        ("python", "torch", "torch_cuda_build", "distributions", "official"),
        name="dependencies",
    )
    for key in ("python", "torch"):
        _text(dependencies[key], key)
    if dependencies["torch_cuda_build"] is not None:
        _text(dependencies["torch_cuda_build"], "torch CUDA build")
    distributions = dependencies["distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("dependencies require installed distribution metadata")
    for record in distributions:
        _keys(record, ("name", "version"), name="distribution")
        _text(record["name"], "distribution name")
        _text(record["version"], "distribution version")
    if distributions != sorted(
        distributions, key=lambda item: (item["name"].lower(), item["version"], item["name"])
    ):
        raise ValueError("dependency distributions must be sorted")
    official = dependencies["official"]
    _keys(
        official,
        (
            "model_id",
            "revision",
            "source_sha256",
            "dependencies",
            "required_transformers",
            "optional_kernels_present",
            "attention_implementation",
            "total_ut_steps",
            "exit_at_step",
            "use_cache",
            "weight_storage",
        ),
        name="official dependencies",
    )
    _constants(
        {
            key: official[key]
            for key in (
                "model_id",
                "revision",
                "required_transformers",
                "attention_implementation",
                "total_ut_steps",
                "exit_at_step",
                "use_cache",
                "weight_storage",
            )
        },
        {
            "model_id": OURO_MODEL_ID,
            "revision": OURO_REVISION,
            "required_transformers": "4.55.0",
            "attention_implementation": "eager",
            "total_ut_steps": 4,
            "exit_at_step": 3,
            "use_cache": False,
            "weight_storage": "shared immutable parameter mapping",
        },
        "official dependency contract",
    )
    if official["source_sha256"] != OFFICIAL_HASHES:
        raise ValueError("official dependency provenance differs from reviewed source hashes")
    if type(official["optional_kernels_present"]) is not bool:
        raise ValueError("optional kernels presence must be a boolean")
    _keys(
        official["dependencies"],
        ("transformers", "torch", "huggingface-hub", "tokenizers", "safetensors", "kernels"),
        name="official package versions",
    )
    for version in official["dependencies"].values():
        if version is not None:
            _text(version, "official package version")


def _verify_original_sources(suite):
    import hashlib
    import json

    root = Path(__file__).resolve().parents[2]
    original = suite["original_reproduction"]
    for record in original["sources"]:
        name = f"{record['commit']}:{record['path']}"
        raw = subprocess.check_output(["git", "-C", str(root), "show", name])
        blob = subprocess.check_output(["git", "-C", str(root), "rev-parse", name]).decode().strip()
        if (hashlib.sha256(raw).hexdigest(), len(raw), blob) != (
            record["sha256"],
            record["size_bytes"],
            record["git_blob_sha1"],
        ):
            raise ValueError("original source content provenance mismatch")
        document = json.loads(raw)
        if any(document[key] != original[key] for key in ("prompt_texts", "prompt_token_ids")):
            raise ValueError("copied original inputs differ from their immutable source record")


def _config(values):
    config, expected = OuroConfig.from_dict(values), OuroConfig()
    for key in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "total_ut_steps",
    ):
        if getattr(config, key) != getattr(expected, key):
            raise ValueError(f"Q1 model configuration differs from pinned Ouro: {key}")
    return config


def make_plan(suite_path, contract_path, model_path, official_code_path=None, *, gpu_ids=None):
    """Hash prepared local bytes and resolve 168 cases without device/weight loading.

    The checked-in contract is a template with null GPU IDs. A probe must supply
    exactly one physical ID selected from fresh scheduler status; no ID is inferred.
    """
    from copy import deepcopy

    suite_path, contract_path, model_path = (
        Path(path).resolve() for path in (suite_path, contract_path, model_path)
    )
    official_code_path = Path(
        official_code_path or Path(__file__).with_name("reference_code")
    ).resolve()
    suite, contract = load_suite(suite_path), deepcopy(load_contract(contract_path))
    existing_ids = contract["controls"]["gpu_ids"]
    if gpu_ids is not None:
        if existing_ids is not None and existing_ids != gpu_ids:
            raise ValueError("explicit GPU IDs conflict with the contract")
        contract["controls"]["gpu_ids"] = gpu_ids
    _validate_contract(contract, resolved=True)
    config = _config(read_json(model_path / "config.json"))
    files = _model_files(model_path)
    tokenizer = next(record for record in files if record["path"] == "tokenizer.json")
    if any(
        tokenizer[key] != suite["provenance"]["tokenizer"][key] for key in ("sha256", "size_bytes")
    ):
        raise ValueError("prepared tokenizer does not match frozen fixture provenance")
    _verify_original_sources(suite)
    order, fixtures = _resolve_cases(suite, contract, config)
    comparisons = _comparisons(order, fixtures, contract, config)
    plan = {
        "schema_version": 1,
        "artifact_type": "validation_plan",
        "suite": suite,
        "contract": contract,
        "model_path": str(model_path),
        "model_config": config.to_dict(),
        "official_code_path": str(official_code_path),
        "official_files": _official_files(official_code_path),
        "source": _source_manifest(),
        "dependencies": dependency_manifest(),
        "model_files": files,
        "inputs": {"suite": _file_record(suite_path), "contract": _file_record(contract_path)},
        "execution_order": order,
        "comparison_order": comparisons,
        "resource_estimates": _resource_estimates(
            suite, contract, config, order, comparisons, fixtures
        ),
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def _file_records(records, name):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} must be a nonempty file record list")
    paths = []
    for record in records:
        _keys(record, ("path", "size_bytes", "sha256"), name=name)
        _text(record["path"], "file path")
        _integer(record["size_bytes"], "file size")
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("file sha256 must be 64 lowercase hexadecimal characters")
        paths.append(record["path"])
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name} contains duplicate file paths")


def validate_plan_integrity(plan):
    """Validate every resolved case/count offline, without filesystem/device reads."""
    _keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "suite",
            "contract",
            "model_path",
            "model_config",
            "official_code_path",
            "official_files",
            "source",
            "dependencies",
            "model_files",
            "inputs",
            "execution_order",
            "comparison_order",
            "resource_estimates",
            "plan_sha256",
        ),
        name="validation plan",
    )
    _version(plan, "validation_plan")
    if plan["plan_sha256"] != _digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("validation plan content hash mismatch")
    _validate_suite(plan["suite"])
    _validate_contract(plan["contract"], resolved=True)
    _validate_dependencies(plan["dependencies"])
    config = _config(plan["model_config"])
    for key in ("model_path", "official_code_path"):
        _text(plan[key], key)
        if not Path(plan[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    _file_records(plan["model_files"], "model files")
    _file_records(plan["official_files"], "official files")
    if {row["path"]: row["sha256"] for row in plan["official_files"]} != OFFICIAL_HASHES:
        raise ValueError("plan official code differs from reviewed source hashes")
    _keys(plan["inputs"], ("suite", "contract"), name="inputs")
    _file_records(list(plan["inputs"].values()), "input files")
    source = plan["source"]
    _keys(source, ("root", "commit", "status", "files"), name="source")
    _text(source["root"], "source root")
    _text(source["commit"], "source commit")
    if not isinstance(source["status"], str):
        raise ValueError("source status must be a string")
    _file_records(source["files"], "source files")
    order, fixtures = _resolve_cases(plan["suite"], plan["contract"], config)
    comparisons = _comparisons(order, fixtures, plan["contract"], config)
    expected = {
        "execution_order": order,
        "comparison_order": comparisons,
        "resource_estimates": _resource_estimates(
            plan["suite"], plan["contract"], config, order, comparisons, fixtures
        ),
    }
    for key, value in expected.items():
        if plan[key] != value:
            raise ValueError(f"resolved {key} differs from the frozen executable contract")


def verify_plan(plan):
    """Rehash controls, source, weights, tokenizer and official code before device use."""
    validate_plan_integrity(plan)
    current = make_plan(
        plan["inputs"]["suite"]["path"],
        plan["inputs"]["contract"]["path"],
        plan["model_path"],
        plan["official_code_path"],
        gpu_ids=plan["contract"]["controls"]["gpu_ids"],
    )
    if current != plan:
        changed = sorted(key for key in current if current[key] != plan.get(key))
        raise ValueError(f"frozen Q1 source/inputs/model/official controls changed: {changed}")
