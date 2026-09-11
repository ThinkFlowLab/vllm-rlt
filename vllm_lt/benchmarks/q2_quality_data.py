"""Frozen Q2 question selection and exact answer parsing; no Torch/model imports."""

import hashlib
import importlib.metadata
import json
import re
from decimal import Decimal
from pathlib import Path

DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_ROWS = 1319
TOKENIZER_VERSION = "0.21.4"
TOKENIZER_SHA256 = "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d"
PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning, then give the final answer "
    "on its own line as #### <number>.\n\nQuestion: {question}\n\nAnswer:"
)
PARSER_VERSION = "q2-final-number-line-v1"
PARSER_PATTERN = r"[ \t]*####[ \t]+([+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?)[ \t]*"
LINE_POLICY = (
    "last fully matching LF line, optional terminal CR; end-of-generation terminates a line"
)
SELECTION_SEED = 0
SELECTION_VERSION = "q2-fp32-quality-v1"
SELECTION_ALGORITHM = "sha256(canonical_compact_json([version,seed,source_id]))"
ENCODE_OPTIONS = {"add_special_tokens": False, "padding": False, "truncation": False}
DECODE_OPTIONS = {"skip_special_tokens": False, "terminal_eos_only": True, "eos_token_id": 0}
_NUMBER_LINE = re.compile(PARSER_PATTERN)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, names, label):
    _require(isinstance(value, dict) and set(value) == set(names), f"invalid {label} fields")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _canonical_decimal(lexeme):
    # Decimal construction and fixed-point formatting preserve every digit;
    # normalize() would apply the caller's precision context and can round.
    number = Decimal(lexeme.replace(",", ""))
    if number == 0:
        return "0"
    value = format(number, "f")
    return value.rstrip("0").rstrip(".") if "." in value else value


def parse_answer(text):
    """Return the last complete matching line and bounded, JSON-safe diagnostics."""
    _require(isinstance(text, str), "answer text must be a string")
    selected, lexeme, valid, markers, last_marker, last_valid = None, None, 0, 0, None, None
    for index, line in enumerate(text.split("\n")):
        # The committed policy permits one terminal CR, including at EOF.
        line = line.removesuffix("\r")
        match = _NUMBER_LINE.fullmatch(line)
        if line.lstrip(" \t").startswith("####"):
            markers += 1
            last_marker, last_valid = index, match is not None
        if match is not None:
            valid += 1
            selected, lexeme = index, match.group(1)
    return {
        "lexeme": lexeme,
        "value": None if lexeme is None else _canonical_decimal(lexeme),
        "diagnostics": {
            "status": "parsed"
            if lexeme is not None
            else "malformed_marker"
            if markers
            else "missing_marker",
            "marker_line_count": markers,
            "valid_line_count": valid,
            "selected_line_index": selected,
            "last_marker_line_index": last_marker,
            "last_marker_valid": last_valid,
        },
    }


def _tokens(ids, maximum):
    _require(
        isinstance(ids, list) and all(type(token) is int and 0 <= token < maximum for token in ids),
        "token IDs must be a list of vocabulary integers",
    )


def decode_output(tokenizer, ids, finish_reason):
    """Preserve special text and remove only terminal actual EOS0 for scoring."""
    _tokens(ids, tokenizer.get_vocab_size(with_added_tokens=True))
    _require(1 <= len(ids) <= 256, "completed output must contain 1..256 tokens")
    if finish_reason == "stop":
        _require(ids[-1] == 0 and 0 not in ids[:-1], "stop requires exactly one terminal EOS0")
        scoring_ids = ids[:-1]
    elif finish_reason == "length":
        _require(len(ids) == 256 and 0 not in ids, "length requires 256 non-EOS outputs")
        scoring_ids = ids
    else:
        raise ValueError("only completed stop/length output can be decoded for scoring")
    return {
        "raw_text": tokenizer.decode(ids, skip_special_tokens=False),
        "scoring_text": tokenizer.decode(scoring_ids, skip_special_tokens=False),
    }


def load_tokenizer(path):
    """Open only the pinned local Rust tokenizer; no network, Transformers or weights."""
    record = _file_record(path)
    _require(record["sha256"] == TOKENIZER_SHA256, "pinned tokenizer hash mismatch")
    _require(
        importlib.metadata.version("tokenizers") == TOKENIZER_VERSION, "tokenizers version mismatch"
    )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _source_id(index):
    return f"{DATASET_REVISION}/main/test/{index}"


def _selection_key(source_id):
    return _digest([SELECTION_VERSION, SELECTION_SEED, source_id])


def _eligibility(length):
    return None if 1 <= length <= 512 else "empty_prompt" if length == 0 else "prompt_too_long"


def prepare_selection(normalized_dataset_path, tokenizer_path):
    """Validate all 1319 source rows before freezing 64 evaluation plus two feasibility IDs."""
    dataset_file = _file_record(normalized_dataset_path)
    tokenizer_file = _file_record(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    population, eligible = [], []
    with Path(normalized_dataset_path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            _require(index < DATASET_ROWS, "dataset has more than 1319 rows")
            row = json.loads(line, object_pairs_hook=_pairs, parse_constant=_reject_constant)
            _fields(row, ("source_id", "question", "answer"), "dataset row")
            _require(row["source_id"] == _source_id(index), "source ID/order mismatch")
            _require(
                all(
                    isinstance(row[key], str) and row[key].strip() for key in ("question", "answer")
                ),
                "question and answer must be nonempty strings",
            )
            reference = parse_answer(row["answer"])
            _require(
                reference["value"] is not None, f"unparseable dataset reference: {row['source_id']}"
            )
            text = PROMPT_TEMPLATE.format(question=row["question"])
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            _tokens(ids, tokenizer.get_vocab_size(with_added_tokens=True))
            length, key = len(ids), _selection_key(row["source_id"])
            reason = _eligibility(length)
            content = _digest({"question": row["question"], "answer": row["answer"]})
            common = {
                "source_id": row["source_id"],
                "source_row_index": index,
                "content_sha256": content,
                "prompt_token_count": length,
                "selection_key": key,
            }
            population.append({**common, "eligible": reason is None, "exclusion_reason": reason})
            if reason is None:
                eligible.append(
                    {
                        **common,
                        "question": row["question"],
                        "reference_answer": row["answer"],
                        "parsed_reference": reference,
                        "prompt_text": text,
                        "prompt_sha256": _text_hash(text),
                        "prompt_token_ids": ids,
                        "prompt_token_ids_sha256": _digest(ids),
                    }
                )
    _require(len(population) == DATASET_ROWS, "dataset must contain exactly 1319 rows")
    eligible.sort(key=lambda row: (row["selection_key"], row["source_id"]))
    _require(len(eligible) >= 66, "fewer than 66 prompt-length-eligible examples")
    result = {
        "schema_version": 1,
        "artifact_type": "q2_quality_selection",
        "dataset": {
            "revision": DATASET_REVISION,
            "config": "main",
            "split": "test",
            "row_count": DATASET_ROWS,
            "file": dataset_file,
        },
        "tokenizer": {
            "file": tokenizer_file,
            "library": "tokenizers",
            "version": TOKENIZER_VERSION,
            "encode_options": dict(ENCODE_OPTIONS),
            "decode_options": dict(DECODE_OPTIONS),
        },
        "prompt_template": PROMPT_TEMPLATE,
        "parser": {
            "version": PARSER_VERSION,
            "pattern": PARSER_PATTERN,
            "line_policy": LINE_POLICY,
        },
        "selection_seed": SELECTION_SEED,
        "selection_algorithm": SELECTION_ALGORITHM,
        "eligible_count": len(eligible),
        "excluded_count": DATASET_ROWS - len(eligible),
        "population": population,
        "evaluation": eligible[:64],
        "feasibility": eligible[64:66],
    }
    result["selection_sha256"] = _digest(result)
    validate_selection(result)
    return result


def validate_selection(selection):
    """Check frozen ranking/content relationships without accessing files or tokenizing.

    Rebuild prepare_selection from retained local files to prove every population
    length and selected tokenization; this self-contained audit cannot establish
    facts about question text omitted from the unselected population records.
    """
    _fields(
        selection,
        (
            "schema_version",
            "artifact_type",
            "dataset",
            "tokenizer",
            "prompt_template",
            "parser",
            "selection_seed",
            "selection_algorithm",
            "eligible_count",
            "excluded_count",
            "population",
            "evaluation",
            "feasibility",
            "selection_sha256",
        ),
        "selection",
    )
    _require(
        selection["schema_version"] == 1 and selection["artifact_type"] == "q2_quality_selection",
        "selection version mismatch",
    )
    _require(
        selection["selection_sha256"]
        == _digest({k: v for k, v in selection.items() if k != "selection_sha256"}),
        "selection hash mismatch",
    )
    dataset = selection["dataset"]
    _fields(dataset, ("revision", "config", "split", "row_count", "file"), "dataset")
    _require(
        {k: dataset[k] for k in ("revision", "config", "split", "row_count")}
        == {
            "revision": DATASET_REVISION,
            "config": "main",
            "split": "test",
            "row_count": DATASET_ROWS,
        },
        "dataset identity mismatch",
    )
    tokenizer = selection["tokenizer"]
    _fields(
        tokenizer, ("file", "library", "version", "encode_options", "decode_options"), "tokenizer"
    )
    for record in (dataset["file"], tokenizer["file"]):
        _fields(record, ("path", "size_bytes", "sha256"), "file record")
        _require(
            isinstance(record["path"], str) and Path(record["path"]).is_absolute(),
            "absolute file path required",
        )
        _require(
            type(record["size_bytes"]) is int and record["size_bytes"] > 0, "invalid file size"
        )
        _require(
            isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
            "invalid file hash",
        )
    _require(tokenizer["file"]["sha256"] == TOKENIZER_SHA256, "pinned tokenizer hash mismatch")
    _require(
        {k: tokenizer[k] for k in ("library", "version", "encode_options", "decode_options")}
        == {
            "library": "tokenizers",
            "version": TOKENIZER_VERSION,
            "encode_options": ENCODE_OPTIONS,
            "decode_options": DECODE_OPTIONS,
        },
        "tokenizer policy mismatch",
    )
    _require(
        selection["prompt_template"] == PROMPT_TEMPLATE
        and selection["parser"]
        == {
            "version": PARSER_VERSION,
            "pattern": PARSER_PATTERN,
            "line_policy": LINE_POLICY,
        },
        "prompt/parser policy mismatch",
    )
    _require(
        type(selection["selection_seed"]) is int
        and selection["selection_seed"] == 0
        and selection["selection_algorithm"] == SELECTION_ALGORITHM,
        "selection policy mismatch",
    )
    population = selection["population"]
    _require(
        isinstance(population, list) and len(population) == DATASET_ROWS,
        "population coverage mismatch",
    )
    eligible = []
    for index, row in enumerate(population):
        _fields(
            row,
            (
                "source_id",
                "source_row_index",
                "content_sha256",
                "prompt_token_count",
                "selection_key",
                "eligible",
                "exclusion_reason",
            ),
            "population row",
        )
        _require(
            type(row["source_row_index"]) is int
            and row["source_row_index"] == index
            and row["source_id"] == _source_id(index),
            "population source order mismatch",
        )
        length = row["prompt_token_count"]
        _require(type(length) is int and length >= 0, "invalid prompt length")
        reason = _eligibility(length)
        _require(
            row["eligible"] is (reason is None) and row["exclusion_reason"] == reason,
            "eligibility mismatch",
        )
        _require(row["selection_key"] == _selection_key(row["source_id"]), "selection key mismatch")
        _require(
            isinstance(row["content_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", row["content_sha256"]),
            "invalid source content hash",
        )
        if reason is None:
            eligible.append(row)
    eligible.sort(key=lambda row: (row["selection_key"], row["source_id"]))
    _require(
        type(selection["eligible_count"]) is int
        and selection["eligible_count"] == len(eligible)
        and type(selection["excluded_count"]) is int
        and selection["excluded_count"] == DATASET_ROWS - len(eligible),
        "population counts mismatch",
    )
    _require(
        isinstance(selection["evaluation"], list)
        and len(selection["evaluation"]) == 64
        and isinstance(selection["feasibility"], list)
        and len(selection["feasibility"]) == 2,
        "selection coverage mismatch",
    )
    rows = selection["evaluation"] + selection["feasibility"]
    _require(
        [row["source_id"] for row in rows] == [row["source_id"] for row in eligible[:66]],
        "selected rank/order mismatch",
    )
    for row, population_row in zip(rows, eligible[:66]):
        _fields(
            row,
            (
                "source_id",
                "source_row_index",
                "content_sha256",
                "prompt_token_count",
                "selection_key",
                "question",
                "reference_answer",
                "parsed_reference",
                "prompt_text",
                "prompt_sha256",
                "prompt_token_ids",
                "prompt_token_ids_sha256",
            ),
            "selected row",
        )
        _require(
            all(
                row[key] == population_row[key]
                for key in (
                    "source_id",
                    "source_row_index",
                    "content_sha256",
                    "prompt_token_count",
                    "selection_key",
                )
            ),
            "selected population identity mismatch",
        )
        _require(
            all(
                isinstance(row[key], str) and row[key].strip()
                for key in ("question", "reference_answer")
            ),
            "invalid selected text",
        )
        parsed = parse_answer(row["reference_answer"])
        _require(
            parsed["value"] is not None and row["parsed_reference"] == parsed,
            "reference parser mismatch",
        )
        _require(
            row["content_sha256"]
            == _digest({"question": row["question"], "answer": row["reference_answer"]}),
            "selected content hash mismatch",
        )
        _require(
            row["prompt_text"] == PROMPT_TEMPLATE.format(question=row["question"])
            and row["prompt_sha256"] == _text_hash(row["prompt_text"]),
            "selected prompt mismatch",
        )
        _tokens(row["prompt_token_ids"], 49152)
        _require(
            len(row["prompt_token_ids"]) == row["prompt_token_count"]
            and row["prompt_token_ids_sha256"] == _digest(row["prompt_token_ids"]),
            "selected token accounting mismatch",
        )
