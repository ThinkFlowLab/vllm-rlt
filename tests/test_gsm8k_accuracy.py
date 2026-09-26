"""CPU checks for the accuracy protocol and comparison failure modes."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks import gsm8k
from benchmarks.gsm8k import (
    DEFAULT_CASE,
    FIXED_EXIT,
    baseline_fingerprint,
    baseline_result,
    build_records,
    compare,
    depth_summary,
    digest,
    exit_settings,
    file_digest,
    make_task,
    score,
    select_doc_ids,
)


@pytest.fixture
def task(monkeypatch):
    pytest.importorskip("lm_eval")
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    def download(self, *args, **kwargs):
        self.dataset = DatasetDict(
            {
                split: Dataset.from_dict(
                    {
                        "question": [f"What is 6 times 3? Question #{i}" for i in range(1319)],
                        "answer": ["#### 18"] * 1319,
                    }
                )
                for split in ("train", "test")
            }
        )

    monkeypatch.setattr(ConfigurableTask, "download", download)
    return make_task()


def test_three_shot_prompt_and_strict_scoring(task):
    task.build_all_requests(limit=1, rank=0, world_size=1)
    instance = task.instances[0]
    assert instance.args[0].count("Q:") == 4  # three demonstrations plus the question
    row = {"doc": instance.doc}
    assert score(task, row, "6 * 3 = 18. The answer is 18.")["correct"]
    # A bare number must not silently pass the harness's strict extraction rule.
    assert score(task, row, "18")["unparseable"]
    assert not score(task, row, "The answer is 19.")["correct"]


def test_sample_is_fixed_and_does_not_use_the_first_hundred():
    selected = select_doc_ids(1319, limit=100, seed=0, split="test")
    assert len(selected) == len(set(selected)) == 100
    assert selected == sorted(selected)
    assert selected == select_doc_ids(1319, limit=100, seed=0, split="test")
    assert selected != list(range(100))
    assert selected != select_doc_ids(1319, limit=100, seed=1, split="test")
    assert select_doc_ids(1319, limit=None, seed=0, split="test") == list(range(1319))


@pytest.mark.parametrize("population,limit", [(10, 100), (10, 0), (0, None)])
def test_selection_rejects_missing_coverage(population, limit):
    with pytest.raises(ValueError):
        select_doc_ids(population, limit=limit, seed=0, split="test")


def test_sampled_requests_keep_original_dataset_ids(task):
    tokenizer = SimpleNamespace(encode=lambda prompt, **kwargs: [1, 2, 3])
    records, selection = build_records(
        task, tokenizer, split="test", limit=100, seed=0, max_new_tokens=10, max_length=20
    )
    assert len(records) == 100
    assert [row["id"] for row in records] == selection["ids"]
    for row in records:
        assert row["doc"]["question"] == f"What is 6 times 3? Question #{row['id']}"
        assert row["prompt"].count("Q:") == 4
    with pytest.raises(ValueError, match="truncated"):
        build_records(
            task, tokenizer, split="test", limit=100, seed=0, max_new_tokens=20, max_length=20
        )


def test_default_case_preserves_the_87_observed_questions(task):
    case = json.loads(DEFAULT_CASE.read_text())
    ids = case["source_ids"]
    assert len(ids) == case["baseline"]["examples"] == 87
    assert case["baseline"]["correct"] == 59
    assert ids == select_doc_ids(1319, limit=100, seed=0, split="test")[:87]
    assert ids[-1] == 1136
    assert ids != select_doc_ids(1319, limit=87, seed=0, split="test")
    records, selection = build_records(
        task,
        SimpleNamespace(encode=lambda *a, **kw: [1, 2, 3]),
        split="test",
        limit=None,
        seed=0,
        max_new_tokens=10,
        max_length=20,
        doc_ids=ids,
    )
    assert selection["method"] == "fixed-source-ids-v1"
    assert [row["id"] for row in records] == ids
    for row in records:
        assert row["doc"]["question"] == f"What is 6 times 3? Question #{row['id']}"


@pytest.mark.parametrize("ids", [[], [5, 5], [11, 5], [-1], [1319]])
def test_invalid_fixed_questions_rejected(task, ids):
    with pytest.raises(ValueError, match="Question IDs"):
        build_records(
            task,
            None,
            split="test",
            limit=None,
            seed=0,
            max_new_tokens=10,
            max_length=20,
            doc_ids=ids,
        )


def test_baseline_fingerprint_tracks_recipe_not_checkout():
    protocol = {
        "model_revision": "pinned",
        "model_files": {"weights": "hash"},
        "task_config": {"metric": "strict"},
        "records": [{"id": 5, "prompt_ids": [1, 2, 3]}],
        "max_new_tokens": 1024,
        "max_length": 2048,
        "dtype": "bfloat16",
        "loops": 4,
        "batch_size": 1,
        "add_special_tokens": False,
        "apply_chat_template": False,
        "source": {"sha": "old", "packages": {"transformers": "4.55.0"}},
    }
    expected = baseline_fingerprint(protocol)
    protocol["source"]["sha"] = "new"
    protocol["model"] = "/another/checkpoint/path"
    assert baseline_fingerprint(protocol) == expected
    changed_prompt = deepcopy(protocol)
    changed_prompt["records"][0]["prompt_ids"].append(4)
    changed_length = {**protocol, "max_new_tokens": 512}
    changed_packages = deepcopy(protocol)
    changed_packages["source"]["packages"]["transformers"] = "different"
    for changed in (changed_prompt, changed_length, changed_packages):
        assert baseline_fingerprint(changed) != expected


def baseline_fixture(correct=59):
    case = json.loads(DEFAULT_CASE.read_text())
    records = [{"id": i, "prompt_ids": [i]} for i in case["source_ids"]]
    protocol = {
        "baseline": case["baseline"],
        "records": records,
        "max_regression_pp": 1.0,
        "min_reference_accuracy_pct": None,
    }
    rows = [
        {"id": row["id"], "prompt_sha256": digest(row["prompt_ids"]), "correct": n < correct}
        for n, row in enumerate(records)
    ]
    return protocol, rows


@pytest.mark.parametrize("correct,passes", [(59, True), (58, False), (60, True)])
def test_stored_hf_baseline_gate(correct, passes):
    protocol, rows = baseline_fixture(correct)
    result = baseline_result(protocol, rows)
    assert result["reference_accuracy_pct"] == pytest.approx(100 * 59 / 87)
    assert result["delta_pp"] == pytest.approx(100 * (correct - 59) / 87)
    assert result["passes_observed_accuracy_gate"] is passes


@pytest.mark.parametrize("change", ["partial", "wrong_id", "wrong_prompt", "floor"])
def test_stored_baseline_rejects_invalid_run(change):
    protocol, rows = baseline_fixture()
    if change == "floor":
        protocol["min_reference_accuracy_pct"] = 70
        assert not baseline_result(protocol, rows)["passes_observed_accuracy_gate"]
        return
    if change == "partial":
        rows.pop()
    elif change == "wrong_id":
        rows[0]["id"] = 0
    else:
        rows[0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError):
        baseline_result(protocol, rows)


@pytest.mark.parametrize("correct,exit_code", [(59, 0), (58, 1)])
def test_native_cli_enforces_stored_baseline(monkeypatch, correct, exit_code):
    protocol, rows = baseline_fixture(correct)
    monkeypatch.setattr(
        gsm8k, "run", lambda args: {"baseline_comparison": baseline_result(protocol, rows)}
    )
    monkeypatch.setattr(
        sys, "argv", ["gsm8k", "run", "--backend", "native", "--protocol", "p", "--output", "o"]
    )
    if exit_code:
        with pytest.raises(SystemExit) as exc:
            gsm8k.main()
        assert exc.value.code == exit_code
    else:
        gsm8k.main()


def comparison_fixture(tmp_path, native_correct=True, examples=1):
    for backend, correct in (("transformers", True), ("native", native_correct)):
        path = tmp_path / backend
        path.mkdir()
        row = {
            "id": 0,
            "prompt_sha256": "same",
            "correct": correct,
            "answer": "18" if correct else "19",
        }
        (path / "samples.jsonl").write_text(
            "".join(json.dumps(dict(row, id=i)) + "\n" for i in range(examples))
        )
        summary = {
            "backend": backend,
            "protocol_sha256": "same",
            "expected_examples": examples,
            "split": "test",
            "max_regression_pp": 1.0,
            "min_reference_accuracy_pct": 75.92,
            "complete": True,
            "examples": examples,
            "correct": examples * int(correct),
            "accuracy": float(correct),
            "samples_sha256": file_digest(path / "samples.jsonl"),
        }
        (path / "summary.json").write_text(json.dumps(summary))
    return SimpleNamespace(
        transformers=tmp_path / "transformers",
        native=tmp_path / "native",
        output=tmp_path / "comparison.json",
    )


def test_accuracy_regression_fails(tmp_path):
    result = compare(comparison_fixture(tmp_path, native_correct=False))
    assert result["delta_pp"] == -100
    assert result["reference_correct_native_wrong"] == 1
    assert not result["passes_observed_accuracy_gate"]


def test_measured_transformers_score_is_the_default_baseline(tmp_path):
    args = comparison_fixture(tmp_path, native_correct=False)
    for backend in ("transformers", "native"):
        folder = tmp_path / backend
        samples = folder / "samples.jsonl"
        row = json.loads(samples.read_text())
        row.update(correct=False, answer="19")
        samples.write_text(json.dumps(row) + "\n")
        path = folder / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(
            correct=0,
            accuracy=0.0,
            min_reference_accuracy_pct=None,
            samples_sha256=file_digest(samples),
        )
        path.write_text(json.dumps(summary))
    result = compare(args)
    assert result["transformers_correct"] == result["native_correct"] == 0
    assert result["delta_pp"] == 0
    assert result["passes_observed_accuracy_gate"]


@pytest.mark.parametrize("change", ["different_protocol", "partial", "wrong_score", "duplicate"])
def test_invalid_comparisons_rejected(tmp_path, change):
    args = comparison_fixture(tmp_path, examples=2 if change == "duplicate" else 1)
    path = args.native / "summary.json"
    summary = json.loads(path.read_text())
    if change == "different_protocol":
        summary["protocol_sha256"] = "different"
    elif change == "partial":
        summary["complete"] = False
    elif change == "wrong_score":
        summary["accuracy"] = 0.0
    else:
        rows = args.native / "samples.jsonl"
        records = [json.loads(line) for line in rows.read_text().splitlines()]
        records[1]["id"] = records[0]["id"]
        rows.write_text("".join(json.dumps(row) + "\n" for row in records))
        summary.update(samples_sha256=file_digest(rows))
    path.write_text(json.dumps(summary))
    message = "Missing or duplicate examples" if change == "duplicate" else None
    with pytest.raises(ValueError, match=message):
        compare(args)


def test_default_exit_is_the_fixed_recipe():
    assert exit_settings("native") == FIXED_EXIT
    assert exit_settings("transformers") == FIXED_EXIT


@pytest.mark.parametrize(
    "backend,options",
    [
        ("native", dict(mode="ouro", threshold=0.5, min_loops=2)),
        ("native", dict(mode="ouro_delayed", threshold=0.2, min_loops=2, async_scheduling=True)),
        ("transformers", dict(mode="ouro", threshold=0.9, min_loops=1)),
    ],
)
def test_adaptive_exit_settings(backend, options):
    settings = exit_settings(backend, **options)
    assert settings["max_loops"] == 4
    assert settings["threshold"] == options["threshold"]
    assert settings["min_loops"] == options["min_loops"]
    assert settings["async_scheduling"] == options.get("async_scheduling", False)
    assert settings != FIXED_EXIT


@pytest.mark.parametrize(
    "backend,options,message",
    [
        ("native", dict(threshold=1.5), "\\[0, 1\\]"),
        ("native", dict(threshold=0.5), "--min-loops"),
        ("native", dict(threshold=0.5, min_loops=5), "--min-loops"),
        # Loop and scheduling options would be silently ignored at fixed depth.
        ("native", dict(min_loops=2), "below 1"),
        ("native", dict(mode="ouro_delayed"), "below 1"),
        ("native", dict(threshold=0.5, min_loops=2, async_scheduling=True), "ouro_delayed"),
        ("transformers", dict(threshold=0.5, min_loops=2), "min-loops 1"),
        ("transformers", dict(mode="ouro_delayed", threshold=0.5, min_loops=1), "min-loops 1"),
    ],
)
def test_invalid_exit_settings_rejected(backend, options, message):
    with pytest.raises(ValueError, match=message):
        exit_settings(backend, **options)


def test_depth_summary_excludes_the_prefill_token():
    rows = [{"exit_depths": [4, 2, 3]}, {"exit_depths": [4, 4]}, {"exit_depths": [4]}]
    summary = depth_summary(rows)
    assert summary["decode_tokens"] == 3
    assert summary["mean_decode_depth"] == pytest.approx(3.0)
    assert summary["decode_depth_histogram"] == {"2": 1, "3": 1, "4": 1}
    assert summary["decode_loops_per_question_mean"] == pytest.approx(3.0)
    assert depth_summary([{"token_ids": [1]}]) is None


def test_compare_reports_exit_policies_between_native_runs(tmp_path):
    args = comparison_fixture(tmp_path)
    adaptive = exit_settings("native", mode="ouro", threshold=0.5, min_loops=2)
    depth = depth_summary([{"exit_depths": [4, 2]}])
    for backend, extra in (
        ("transformers", {"backend": "native"}),
        ("native", {"exit": adaptive, "depth": depth}),
    ):
        path = tmp_path / backend / "summary.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **extra}))
    result = compare(args)
    # A summary without exit settings predates them and used the fixed recipe.
    assert result["reference"] == {"backend": "native", "exit": FIXED_EXIT, "depth": None}
    assert result["candidate"] == {"backend": "native", "exit": adaptive, "depth": depth}
    assert (result["reference_correct"], result["candidate_correct"]) == (1, 1)
    # Backend-named keys would mislabel a native-vs-native comparison, so they are omitted.
    assert not any(key.startswith(("transformers_", "native_")) for key in result)
    assert "reference_correct_native_wrong" not in result


def test_compare_keeps_backend_named_keys_for_hf_vs_native(tmp_path):
    result = compare(comparison_fixture(tmp_path, native_correct=False))
    assert result["reference_correct_candidate_wrong"] == 1
    assert result["transformers_correct"] == result["reference_correct"] == 1
    assert result["native_correct"] == result["candidate_correct"] == 0
    assert result["native_accuracy_pct"] == result["candidate_accuracy_pct"] == 0
