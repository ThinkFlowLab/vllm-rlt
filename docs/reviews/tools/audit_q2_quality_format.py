"""Post-hoc format inventory only: no model, tokenizer execution or answer rescoring."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def record(path):
    raw = Path(path).read_bytes()
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def audit(run_dir, report_path, model_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    run_dir, model_dir = Path(run_dir), Path(model_dir)
    original = record(report_path)
    report = json.loads(Path(report_path).read_text())
    assert report["complete"] and report["denominator"] == 64
    rows, statuses, finishes, flags, sources = [], Counter(), Counter(), Counter(), {}
    for row in report["records"]:
        if row["phase"] != "evaluation":
            continue
        path = run_dir / "runs" / row["run_id"] / "result.json"
        source = record(path)
        assert source == row["result"]
        raw = json.loads(path.read_text())
        sources[str(path)] = source
        text = raw["scoring_text"]
        tags = {
            "contains_four_hashes": "####" in text,
            "contains_short_hash_line": bool(re.search(r"(?m)^[ \t]*#{1,3}[ \t]+\S", text)),
            "contains_question_heading": bool(re.search(r"(?m)^[ \t]*Question:", text)),
            "contains_chat_start": "<|im_start|>" in text,
            "contains_chat_end": "<|im_end|>" in text,
        }
        for name, present in tags.items():
            flags[name] += int(present)
        status = raw["parsed_answer"]["diagnostics"]["status"]
        assert status == row["parser_status"]
        assert raw["finish_reason"] == row["finish_reason"]
        statuses[status] += 1
        finishes[raw["finish_reason"]] += 1
        rows.append(
            {
                "run_id": row["run_id"],
                "example_id": row["example_id"],
                "result": source,
                "original_parser_status": status,
                "finish_reason": raw["finish_reason"],
                "format_flags": tags,
            }
        )
    assert len(rows) == len({r["example_id"] for r in rows}) == 64
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "README.md",
    )
    model_files = {name: record(model_dir / name) for name in names}
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    model = json.loads((model_dir / "config.json").read_text())
    result = {
        "schema_version": 1,
        "artifact_type": "q2_quality_posthoc_format_inventory",
        "primary_report": original,
        "primary_accuracy_unchanged": report["accuracy"],
        "evaluation_records": 64,
        "parser_counts": dict(statuses),
        "finish_counts": dict(finishes),
        "overlapping_format_flag_counts": dict(flags),
        "tokenizer_interface": {
            "files": model_files,
            "tokenizer_class": config["tokenizer_class"],
            "bos_token": config.get("bos_token"),
            "eos_token": config.get("eos_token"),
            "model_bos_token_id": model.get("bos_token_id"),
            "model_eos_token_id": model.get("eos_token_id"),
            "post_processor": tok.get("post_processor"),
            "chat_template_present": bool(config.get("chat_template")),
            "chat_template_sha256": hashlib.sha256(
                config.get("chat_template", "").encode()
            ).hexdigest(),
            "special_tokens": [
                {k: t[k] for k in ("id", "content", "special")}
                for t in tok["added_tokens"]
                if t["id"] in (0, 1, 2)
            ],
        },
        "records": rows,
        "answer_rescoring": False,
        "gpu_runs_added": 0,
        "limitations": [
            "Post-hoc lexical flags overlap; they are not semantic or correctness judgments.",
            "The original strict parser, fixed denominator and 256-output cap are unchanged.",
            "This inventory cannot localize an engine defect "
            "or establish checkpoint task accuracy.",
            "Long-history native/official and alternative-template/output-budget "
            "runs remain unexecuted.",
        ],
    }
    assert record(report_path) == original
    assert all(record(path) == value for path, value in sources.items())
    assert all(record(model_dir / name) == value for name, value in model_files.items())
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "primary_accuracy_unchanged",
                    "parser_counts",
                    "finish_counts",
                    "overlapping_format_flag_counts",
                    "answer_rescoring",
                    "gpu_runs_added",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.report, args.model_dir, args.output)
