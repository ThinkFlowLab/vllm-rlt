"""Numerical metric checks for the fixed-chain speculative evaluation."""

import subprocess
import sys

import pytest
import torch


def test_error_statistics_use_fp32_and_count_every_element():
    from benchmarks.speculative_accuracy import error_statistics

    actual = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    expected = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    assert error_statistics(actual, expected) == {
        "max_abs": 1.0,
        "rms": 1.0,
        "elements": 2,
    }


def test_error_statistics_reject_shape_mismatch():
    from benchmarks.speculative_accuracy import error_statistics

    with pytest.raises(ValueError, match="shape"):
        error_statistics(torch.ones(2), torch.ones(1, 2))


def test_fixed_chain_reports_hidden_kv_and_logit_errors():
    from benchmarks.speculative_accuracy import compare_fixed_chain
    from vllm_rlt.models import OuroConfig, OuroForCausalLM

    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny())
    report = compare_fixed_chain(model, backend="torch", prefix=[2, 3, 4], current=5, k=2)
    assert report["positions_compared"] == 3
    assert set(report["hidden_by_depth"]) == {"0", "1", "2", "3"}
    assert set(report["kv_by_depth"]) == {"0", "1", "2", "3"}
    assert report["logits"]["max_abs"] < 1e-4
    assert report["argmax_mismatches"] == 0
    assert len(report["logit_rows"]) == 3
    assert all(row["reuse_top1"] == row["oracle_top1"] for row in report["logit_rows"])


def test_accuracy_cli_exposes_fixed_chain_options():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.speculative_accuracy", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--model" in result.stdout
    assert "--ks" in result.stdout


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [(0.8, 3, 1.0), (1.0, -1, 0.8), (0.6, 4, 0.9)],
)
@pytest.mark.parametrize("seed", [431, 982, 2026])
def test_filtered_rejection_sampling_recovers_target_distribution(temperature, top_k, top_p, seed):
    from vllm_rlt.sampling_params import SamplingParams
    from vllm_rlt.worker.sampling import probabilities, rejection_sample

    params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)
    p = probabilities(torch.tensor([2.0, 1.5, 0.0, -1.0, 2.5]), params)
    q = probabilities(torch.tensor([1.0, 2.0, 0.5, -0.5, 1.5]), params)
    generator = torch.Generator().manual_seed(seed)
    trials = 6000
    counts = torch.zeros_like(p)
    for _ in range(trials):
        candidate = int(torch.multinomial(q, 1, generator=generator))
        token, _ = rejection_sample(candidate, p, q, generator)
        counts[token] += 1
    empirical = counts / trials
    tolerance = 5 * torch.sqrt(p * (1 - p) / trials) + 0.008
    assert torch.all((empirical - p).abs() <= tolerance)
