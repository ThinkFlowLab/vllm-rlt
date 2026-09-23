"""First-divergence selection for fixed-prefix diagnostics."""

from benchmarks.speculative_replay import first_divergence


def test_first_divergence_uses_first_shared_prefix_difference():
    result = {
        "cases": [
            {
                "runs": [
                    {
                        "native": {"token_ids": [[1, 2, 3], [4, 5, 6]]},
                        "speculative": {"token_ids": [[1, 2, 3], [4, 7, 6]]},
                    }
                ]
            }
        ]
    }
    assert first_divergence(result) == (0, 0, 1, 1)
    assert first_divergence(result, case_index=0) == (0, 0, 1, 1)
    assert first_divergence(result, case_index=1) is None
