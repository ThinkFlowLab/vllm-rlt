import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager


def make_cache(**kwargs):
    parameters = dict(
        num_layers=2, num_kv_heads=1, head_dim=4, num_blocks=18, block_size=2, max_loops=3
    )
    parameters.update(kwargs)
    return KVCacheManager(**parameters)


def test_admission_is_atomic_and_counts_all_depths():
    cache = make_cache(num_blocks=12)
    assert cache.required_blocks(3) == 6
    assert cache.allocate("a", 3)
    assert cache.num_used_blocks == 6
    assert not cache.allocate("b", 5)
    assert cache.num_used_blocks == 6
    with pytest.raises(ValueError, match="pool has"):
        cache.allocate("impossible", 9)
    with pytest.raises(ValueError, match="already owns"):
        cache.allocate("a", 3)
    assert cache.allocate("b", 4)
    assert cache.num_free_blocks == 0
    cache.free("never-admitted")
    cache.free("a")
    cache.free("a")
    assert cache.num_free_blocks == 6
    assert cache.allocate("c", 4)
    a_pages = {page for depth in range(3) for page in cache.get_block_table("c", depth)}
    b_pages = {page for depth in range(3) for page in cache.get_block_table("b", depth)}
    assert a_pages.isdisjoint(b_pages)


def test_last_exited_copies_every_layer_and_preserves_executed_depths():
    cache = make_cache()
    assert cache.allocate("a", 4)
    for layer in range(2):
        for depth in range(2):
            keys = torch.full((1, 1, 4), float(10 * layer + depth))
            cache.write(layer, ["a"], [depth], [0], keys, keys + 100)
    cache.finalize_token("a", 0, exit_depth=1)
    for layer in range(2):
        first_keys, first_values = cache.read(layer, "a", 0, 1)
        second_keys, second_values = cache.read(layer, "a", 1, 1)
        last_keys, last_values = cache.read(layer, "a", 2, 1)
        torch.testing.assert_close(first_keys, torch.full_like(first_keys, float(10 * layer)))
        torch.testing.assert_close(
            first_values, torch.full_like(first_values, float(10 * layer + 100))
        )
        torch.testing.assert_close(last_keys, second_keys)
        torch.testing.assert_close(last_values, second_values)
    # They are physical copies: a later write cannot mutate another depth.
    cache.write(0, ["a"], [1], [0], torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    torch.testing.assert_close(cache.read(0, "a", 2, 1)[0], torch.ones(1, 1, 4))


def test_early_exit_then_later_token_loops_deeper_across_block_boundary():
    cache = make_cache()
    assert cache.allocate("a", 3)
    for position, exit_depth in [(0, 0), (1, 1), (2, 2)]:
        for depth in range(exit_depth + 1):
            for layer in range(2):
                keys = torch.full((1, 1, 4), float(position * 10 + depth + layer))
                cache.write(layer, ["a"], [depth], [position], keys, keys + 100)
                attended = cache.attend(layer, ["a"], [depth], [position], torch.zeros(1, 2, 4))
                expected_values = torch.tensor(
                    [100 + p * 10 + min(p, depth) + layer for p in range(position + 1)]
                )
                torch.testing.assert_close(
                    attended, torch.full_like(attended, expected_values.float().mean())
                )
        cache.finalize_token("a", position, exit_depth)
    assert cache.read(0, "a", 2)[0].shape == (3, 1, 4)


def test_block_reuse_never_exposes_stale_request_history():
    cache = make_cache(num_blocks=6)
    assert cache.allocate("old", 4)
    keys = torch.full((4, 1, 4), 7.0)
    cache.write(0, ["old"] * 4, [0] * 4, list(range(4)), keys, keys)
    cache.free("old")
    assert cache.allocate("new", 4)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.attend(0, ["new"], [0], [0], torch.ones(1, 1, 4))
    cache.write(0, ["new"], [0], [0], torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
    torch.testing.assert_close(
        cache.attend(0, ["new"], [0], [0], torch.zeros(1, 1, 4)), torch.ones(1, 1, 4)
    )
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(0, "new", 0, 2)


def test_validation_happens_before_mutation():
    cache = make_cache()
    cache.allocate("a", 3)
    k = torch.ones(2, 1, 4)
    with pytest.raises(ValueError, match="position"):
        cache.write(0, ["a", "a"], [0, 0], [0, 3], k, k)
    assert cache.read(0, "a", 0)[0].shape[0] == 0
    with pytest.raises(ValueError, match="duplicate"):
        cache.write(0, ["a", "a"], [0, 0], [0, 0], k, k)
    with pytest.raises(RuntimeError, match="every layer"):
        cache.finalize_token("a", 0, 0)
    with pytest.raises(ValueError, match="one-dimensional integer"):
        cache.write(0, ["a"], [0], torch.tensor([0.5]), k[:1], k[:1])
    with pytest.raises(ValueError, match="depth"):
        cache.get_block_table("a", 3)
    with pytest.raises(ValueError, match="dtype"):
        cache.write(0, ["a"], [0], [0], k[:1].half(), k[:1].half())


def test_out_of_order_writes_only_expose_complete_prefix():
    cache = make_cache()
    cache.allocate("a", 4)
    k = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    cache.write(0, ["a", "a"], [0, 0], [3, 1], k[[3, 1]], k[[3, 1]])
    assert cache.read(0, "a", 0)[0].shape[0] == 0
    cache.write(0, ["a"], [0], [0], k[:1], k[:1])
    torch.testing.assert_close(cache.read(0, "a", 0)[0], k[:2])
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.attend(0, ["a"], [0], [3], torch.zeros(1, 1, 4))
    cache.write(0, ["a"], [0], [2], k[2:3], k[2:3])
    torch.testing.assert_close(cache.read(0, "a", 0)[0], k)


def test_triton_is_explicit_and_rejects_cpu():
    with pytest.raises(ValueError, match="requires a CUDA"):
        make_cache(backend="triton")
    with pytest.raises(ValueError, match="backend"):
        make_cache(backend="automatic")
