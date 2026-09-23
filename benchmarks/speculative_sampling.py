"""Empirical target-distribution checks for filtered rejection sampling."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampling import probabilities, rejection_sample

CASES = ((0.8, 3, 1.0), (1.0, -1, 0.8), (0.6, 4, 0.9))
TARGET_LOGITS = torch.tensor([2.0, 1.5, 0.0, -1.0, 2.5])
PROPOSAL_LOGITS = torch.tensor([1.0, 2.0, 0.5, -0.5, 1.5])


def sample_distribution(*, temperature, top_k, top_p, seed, trials):
    """Draw proposals, apply correction, and compare counts with target p."""
    if trials < 1:
        raise ValueError("trials must be positive")
    params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)
    p = probabilities(TARGET_LOGITS, params)
    q = probabilities(PROPOSAL_LOGITS, params)
    generator = torch.Generator().manual_seed(seed)
    counts = torch.zeros(len(p), dtype=torch.int64)
    accepted = 0
    for _ in range(trials):
        candidate = int(torch.multinomial(q, 1, generator=generator))
        token, was_accepted = rejection_sample(candidate, p, q, generator)
        counts[token] += 1
        accepted += int(was_accepted)
    empirical = counts.float() / trials
    errors = (empirical - p).abs()
    standard_errors = torch.sqrt(p * (1 - p) / trials)
    return {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "trials": trials,
        "target_p": p.tolist(),
        "proposal_q": q.tolist(),
        "counts": counts.tolist(),
        "accepted_fraction": accepted / trials,
        "max_abs_deviation": errors.max().item(),
        "max_standardized_deviation": max(
            (errors[i] / standard_errors[i]).item() for i in range(len(p)) if standard_errors[i] > 0
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[431, 982, 2026])
    args = parser.parse_args(argv)
    if args.trials < 1 or not args.seeds:
        parser.error("trials and seed list must be nonempty and positive")
    results = {
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "method": "one-draft rejection and residual correction on filtered p/q; independent seeds",
        "cases": [
            sample_distribution(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                trials=args.trials,
            )
            for temperature, top_k, top_p in CASES
            for seed in args.seeds
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    worst = max(results["cases"], key=lambda case: case["max_standardized_deviation"])
    print(
        json.dumps(
            {
                "cases": len(results["cases"]),
                "max_abs_deviation": max(case["max_abs_deviation"] for case in results["cases"]),
                "max_standardized_deviation": worst["max_standardized_deviation"],
                "worst_seed": worst["seed"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
