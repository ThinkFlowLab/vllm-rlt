"""Collect profile_device_loop.py timing JSON files into one CSV (one row per file).

Usage: python -m benchmarks.summarize_device_loop <result dir>... --csv out.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

FIELDS = (
    "source",
    "mode",
    "execution",
    "backend",
    "concurrency",
    "k",
    "measure",
    "eos",
    "prompt_tokens",
    "committed_tokens",
    "n",
    "tok_s_median",
    "tok_s_min",
    "tok_s_max",
    "tok_s_rel_spread",
    "elapsed_s_median",
    "steps",
    "acceptance_rate",
    "accepted_per_round",
    "peak_mem_gib",
    "outputs_identical",
    "git_sha",
    "warmup_tok_s",
)


def rows(paths):
    for path in paths:
        d = json.loads(path.read_text())
        if "summary" not in d or d.get("status") != "ok":
            continue
        c, s, runs = d["config"], d["summary"], d["runs"]
        spec = runs[0].get("spec")
        yield dict(
            source=str(path),
            mode=c["mode"],
            execution=c.get("execution", "eager"),
            backend=c["backend"],
            concurrency=c["concurrency"],
            k=c["k"] if c["mode"] == "spec" else "",
            measure=c["measure"],
            eos="natural" if c["natural_eos"] else "ignore",
            prompt_tokens=sum(runs[0]["prompt_tokens"]),
            committed_tokens=runs[0]["committed_tokens"],
            n=s["n"],
            tok_s_median=round(s["tokens_per_s_median"], 2),
            tok_s_min=round(s["tokens_per_s_min"], 2),
            tok_s_max=round(s["tokens_per_s_max"], 2),
            tok_s_rel_spread=round(
                (s["tokens_per_s_max"] - s["tokens_per_s_min"]) / s["tokens_per_s_median"], 4
            ),
            elapsed_s_median=round(s["elapsed_s_median"], 4),
            steps=statistics.median(r["window_steps"] for r in runs),
            acceptance_rate=round(spec["acceptance_rate"], 4) if spec else "",
            # Committed tokens per request-round include the correction/bonus token.
            accepted_per_round=round(spec["committed_tokens"] / spec["rounds"], 3) if spec else "",
            peak_mem_gib=round(max(r["peak_memory_bytes"] for r in runs) / 2**30, 3),
            outputs_identical=s["outputs_identical"],
            git_sha=d["environment"]["git_sha"][:10],
            warmup_tok_s=";".join(f"{w['tokens_per_s']:.1f}" for w in d["warmup"]),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(
        f for d in args.dirs for f in d.rglob("*.json") if not f.name.endswith(("analysis.json",))
    )
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows(paths):
            w.writerow(row)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
