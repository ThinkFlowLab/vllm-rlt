"""Pure parsing and synthetic source-selection checks, without dataset downloads."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from decimal import localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

# Direct source loading also supports the scorer's genuinely Torch-free entry.
_SOURCE = Path(__file__).parents[1] / "vllm_lt/benchmarks/q2_quality_data.py"
_SPEC = importlib.util.spec_from_file_location("quality_data_test_subject", _SOURCE)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#### -1,234.50", "-1234.5"),
        ("#### +0", "0"),
        ("#### -0.00", "0"),
        ("#### 000123.45000", "123.45"),
        ("reasoning\n#### 12", "12"),
        ("#### 1\n#### 2\n", "2"),
        ("#### 2\n#### bananas", "2"),
        ("\t####\t+1,234.00 \r\n\r\n", "1234"),
        ("#### 17\r", "17"),
    ],
)
def test_complete_final_matching_lines_and_exact_values(text, expected):
    result = data.parse_answer(text)
    assert result["value"] == expected and result["lexeme"] is not None
    assert result["diagnostics"]["status"] == "parsed"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "the answer is 12",
        "#### 1,23",
        "#### 1234,567",
        "#### 1,234,56",
        "#### 1e3",
        "#### 1/2",
        "#### $3",
        "#### 1%",
        "#### NaN",
        "#### Infinity",
        "#### −1",
        "#### １２",
        "#### .5",
        "#### 1.",
        "#### 1 because",
        "#### 1<file_sep>2",
        "####12",
        "x#### 1",
        "#### 1\r\r",
    ],
)
def test_absent_or_malformed_answers_are_not_repaired(text):
    result = data.parse_answer(text)
    assert result["value"] is None and result["lexeme"] is None
    assert result["diagnostics"]["selected_line_index"] is None


def test_exact_decimal_does_not_round_to_context_or_binary_float():
    number = "123456789012345678901234567890123456789.1234500"
    with localcontext() as context:
        context.prec = 2
        assert data.parse_answer("#### " + number)["value"] == number[:-2]


def test_multiple_markers_preserve_bounded_last_marker_diagnostics():
    result = data.parse_answer("work\n#### 1\n#### 2\n#### bad\n")
    assert result == {
        "lexeme": "2",
        "value": "2",
        "diagnostics": {
            "status": "parsed",
            "marker_line_count": 3,
            "valid_line_count": 2,
            "selected_line_index": 2,
            "last_marker_line_index": 3,
            "last_marker_valid": False,
        },
    }


class TextTokenizer:
    pieces = {0: "<|endoftext|>", 1: "#### 1", 2: "<file_sep>", 3: "2", 4: "\n"}

    def get_vocab_size(self, with_added_tokens):
        assert with_added_tokens
        return 5

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.pieces[token] for token in ids)


def test_decode_only_removes_terminal_actual_eos_and_preserves_other_specials():
    tokenizer = TextTokenizer()
    actual = data.decode_output(tokenizer, [1, 0], "stop")
    assert actual == {"raw_text": "#### 1<|endoftext|>", "scoring_text": "#### 1"}
    assert data.parse_answer(actual["scoring_text"])["value"] == "1"
    poisoned = data.decode_output(tokenizer, [1, 2, 3, 0], "stop")
    assert poisoned["scoring_text"] == "#### 1<file_sep>2"
    assert data.parse_answer(poisoned["scoring_text"])["value"] is None
    first_eos = data.decode_output(tokenizer, [0], "stop")
    assert first_eos["scoring_text"] == ""
    assert data.parse_answer(first_eos["scoring_text"])["value"] is None


def test_natural_eos_at_cap_has_stop_precedence_and_length_text_is_scored():
    tokenizer = TextTokenizer()
    ids = [4] * 254 + [1, 0]
    assert (
        data.parse_answer(data.decode_output(tokenizer, ids, "stop")["scoring_text"])["value"]
        == "1"
    )
    with pytest.raises(ValueError, match="non-EOS"):
        data.decode_output(tokenizer, ids, "length")
    cap = [4] * 255 + [1]
    assert (
        data.parse_answer(data.decode_output(tokenizer, cap, "length")["scoring_text"])["value"]
        == "1"
    )


@pytest.mark.parametrize(
    "ids,reason",
    [
        ([], "stop"),
        ([True, 0], "stop"),
        ([5, 0], "stop"),
        ([0, 1, 0], "stop"),
        ([1], "stop"),
        ([1], "length"),
        ([1] * 257, "length"),
        ([0], "abort"),
    ],
)
def test_invalid_or_unfinished_histories_cannot_become_scored_completions(ids, reason):
    with pytest.raises(ValueError):
        data.decode_output(TextTokenizer(), ids, reason)


class SelectionTokenizer:
    def __init__(self):
        self.seen = []

    def get_vocab_size(self, with_added_tokens):
        return 49152

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        self.seen.append(text)
        length = (
            513 if "LONG" in text else 512 if "BOUNDARY" in text else 0 if "EMPTY" in text else 7
        )
        return SimpleNamespace(ids=[1] * length)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rows = [
        {
            "source_id": f"{data.DATASET_REVISION}/main/test/{i}",
            "question": f"Question {i}?",
            "answer": f"Reference reasoning\n#### {i}",
        }
        for i in range(1319)
    ]
    rows[0]["question"] = "LONG source question must not be truncated"
    rows[1]["question"] = "BOUNDARY source question is eligible"
    rows[2]["question"] = "EMPTY encoded prompt is excluded"
    source = tmp_path / "test.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("synthetic tokenizer for isolated test")
    monkeypatch.setattr(
        data, "TOKENIZER_SHA256", hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    )
    fake = SelectionTokenizer()
    monkeypatch.setattr(data, "load_tokenizer", lambda path: fake)
    return source, tokenizer, rows, fake


def test_selection_uses_full_prompt_length_pinned_id_hash_rank_and_disjoint_feasibility(inputs):
    source, tokenizer, rows, fake = inputs
    result = data.prepare_selection(source, tokenizer)
    assert result["dataset"]["row_count"] == 1319
    assert result["eligible_count"] == 1317 and result["excluded_count"] == 2
    assert result["population"][0]["exclusion_reason"] == "prompt_too_long"
    assert result["population"][1]["prompt_token_count"] == 512
    assert result["population"][1]["eligible"] is True
    assert result["population"][2]["exclusion_reason"] == "empty_prompt"
    expected = sorted(
        (r for i, r in enumerate(rows) if i not in (0, 2)),
        key=lambda r: (
            hashlib.sha256(
                json.dumps(
                    ["q2-fp32-quality-v1", 0, r["source_id"]], separators=(",", ":")
                ).encode()
            ).hexdigest(),
            r["source_id"],
        ),
    )
    assert [r["source_id"] for r in result["evaluation"]] == [r["source_id"] for r in expected[:64]]
    assert [r["source_id"] for r in result["feasibility"]] == [
        r["source_id"] for r in expected[64:66]
    ]
    assert fake.seen == [data.PROMPT_TEMPLATE.format(question=row["question"]) for row in rows]
    assert all("Reference reasoning" not in prompt for prompt in fake.seen)
    for selected in result["evaluation"] + result["feasibility"]:
        assert selected["question"] == rows[selected["source_row_index"]]["question"]
        assert selected["parsed_reference"]["value"] == str(selected["source_row_index"])
    data.validate_selection(json.loads(json.dumps(result)))


@pytest.mark.parametrize(
    "corruption", ["duplicate", "short", "extra", "bad_reference", "extra_field", "duplicate_json"]
)
def test_source_corruption_is_preparation_failure_not_eligibility_filter(inputs, corruption):
    source, tokenizer, rows, _ = inputs
    if corruption == "duplicate":
        rows[1]["source_id"] = rows[0]["source_id"]
    elif corruption == "short":
        rows.pop()
    elif corruption == "extra":
        rows.append(rows[-1])
    elif corruption == "bad_reference":
        rows[0]["answer"] = (
            "#### not a number"  # Even an excluded prompt must have a valid reference.
        )
    elif corruption == "extra_field":
        rows[0]["unused"] = True
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if corruption == "duplicate_json":
        source.write_text(source.read_text().replace('"answer":', '"answer":"#### 1","answer":', 1))
    with pytest.raises(ValueError):
        data.prepare_selection(source, tokenizer)


@pytest.mark.parametrize(
    "mutation", ["selected_rank", "parser", "reference", "token_count", "population_key"]
)
def test_self_rehashed_selection_cannot_change_declared_relationships(inputs, mutation):
    source, tokenizer, _, _ = inputs
    selection = deepcopy(data.prepare_selection(source, tokenizer))
    if mutation == "selected_rank":
        selection["evaluation"].reverse()
    elif mutation == "parser":
        selection["parser"]["version"] = "post-hoc-parser"
    elif mutation == "reference":
        selection["evaluation"][0]["parsed_reference"]["value"] = "999"
    elif mutation == "token_count":
        selection["evaluation"][0]["prompt_token_count"] += 1
    elif mutation == "population_key":
        selection["population"][0]["selection_key"] = "0" * 64
    selection["selection_sha256"] = data._digest(
        {k: v for k, v in selection.items() if k != "selection_sha256"}
    )
    with pytest.raises(ValueError):
        data.validate_selection(selection)


def test_constants_match_the_preselection_committed_contract():
    contract = json.loads(
        (_SOURCE.parents[2] / "benchmarks/fixtures/ouro-q2-fp32-quality-contract.json").read_text()
    )
    assert data.PROMPT_TEMPLATE == contract["prompt_template"]
    assert data.PARSER_PATTERN == contract["parser"]["grammar"]
    assert data.LINE_POLICY == contract["parser"]["line_policy"]
    assert data.SELECTION_ALGORITHM == contract["selection"]["key"]
    assert data.DATASET_REVISION == contract["dataset"]["revision"]


def test_direct_cpu_import_and_parser_need_no_torch_or_network():
    script = (
        "import importlib.util,sys,socket\n"
        "def forbidden(*args,**kwargs): raise AssertionError('network')\n"
        "socket.socket.connect=forbidden\n"
        "spec=importlib.util.spec_from_file_location('quality_data',sys.argv[1])\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "assert module.parse_answer('#### 1,234.50')['value']=='1234.5'\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script, str(_SOURCE)], check=True, timeout=10)
