"""Inactive addressing, finite output and held-input equivalence; no default CUDA use."""

import pytest
import torch
from test_attention import dense_attention

from vllm_lt.kernels.paged_attention import torch_paged_attention, triton_paged_attention


@pytest.fixture(autouse=True)
def no_unreserved_cuda(request, monkeypatch):
    if request.node.get_closest_marker("gpu"):
        return

    def forbidden(*args, **kwargs):
        pytest.fail("CPU kernel tests must not discover or initialize CUDA")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def inputs(device="cpu"):
    # Layer slices exercise the real noncontiguous cache strides.
    keys = torch.arange(6 * 2 * 16 * 2 * 8, dtype=torch.float32, device=device).reshape(
        6, 2, 16, 2, 8
    )
    keys = (keys.remainder(31) - 15) / 16
    values = keys.flip(-1).clone()
    query = torch.full((4, 4, 8), float("nan"), device=device)
    query[1] = 0.25
    query[3] = -0.5
    tables = torch.tensor(
        [[2**30, -1], [1, -1], [-1, 2**30], [3, 5]], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([0, 3, 0, 17], dtype=torch.int32, device=device)
    # A strided mask must retain its logical row meaning.
    active = torch.tensor([False, False, True, False, False, False, True, False], device=device)[
        ::2
    ]
    return query, keys, values, tables, lengths, active


def test_holey_rows_never_index_poisoned_tables_and_preserve_live_gqa():
    q, keys, values, tables, lengths, active = inputs()
    expected = torch_paged_attention(
        q[[1, 3]], keys[:, 1], values[:, 1], tables[[1, 3]], lengths[[1, 3]]
    )
    actual = torch_paged_attention(q, keys[:, 1], values[:, 1], tables, lengths, active)
    assert torch.equal(actual[[1, 3]], expected)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[[0, 2]], torch.zeros_like(actual[[0, 2]]))
    assert not torch.signbit(actual[[0, 2]]).any()


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_all_inactive_is_zero_without_valid_addresses(rows):
    q = torch.full((rows, 2, 8), float("nan"))
    # No cache block exists at all: an accidental read cannot be hidden by a dummy page.
    cache = torch.empty(0, 16, 2, 8)
    output = torch_paged_attention(
        q,
        cache,
        cache,
        torch.full((rows, 1), 2**30),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.bool),
    )
    assert output.shape == q.shape and torch.equal(output, torch.zeros_like(output))
    assert not torch.signbit(output).any()


@pytest.mark.parametrize("active", [torch.ones(4), torch.ones(4, 1, dtype=torch.bool), [True] * 4])
def test_mask_contract_rejected_before_attention(active):
    q, keys, values, tables, lengths, _ = inputs()
    with pytest.raises(ValueError, match="active must"):
        torch_paged_attention(q, keys[:, 1], values[:, 1], tables, lengths, active)


def test_none_and_explicit_all_active_preserve_compact_arithmetic():
    q, keys, values, tables, lengths, _ = inputs()
    args = q[[1, 3]], keys[:, 1], values[:, 1], tables[[1, 3]], lengths[[1, 3]]
    assert torch.equal(
        torch_paged_attention(*args), torch_paged_attention(*args, torch.ones(2, dtype=torch.bool))
    )
    with pytest.raises(ValueError, match="requires a CUDA"):
        triton_paged_attention(*args)


@pytest.mark.gpu
def test_reserved_masked_scatter_and_attention_preserve_strided_neighbors():
    from vllm_lt.kernels.triton_kv_write import masked_kv_write

    q, keys, values, tables, lengths, active = inputs("cuda")
    expected_k, expected_v = keys.clone(), values.clone()
    k = torch.full((4, 2, 8), float("nan"), device="cuda")
    v = k.clone()
    k[1], k[3], v[1], v[3] = 0.75, -0.25, -0.5, 0.125
    blocks = torch.tensor([-1, 1, 2**30, 5], device="cuda")
    offsets = torch.tensor([2**30, 2, -1, 0], device="cuda")
    expected_k[blocks[[1, 3]], 1, offsets[[1, 3]]] = k[[1, 3]]
    expected_v[blocks[[1, 3]], 1, offsets[[1, 3]]] = v[[1, 3]]
    masked_kv_write(keys[:, 1], values[:, 1], blocks, offsets, k, v, active)
    torch.cuda.synchronize()
    assert torch.equal(keys, expected_k) and torch.equal(values, expected_v)
    actual = triton_paged_attention(q, keys[:, 1], values[:, 1], tables, lengths, active)
    expected = triton_paged_attention(
        q[[1, 3]], keys[:, 1], values[:, 1], tables[[1, 3]], lengths[[1, 3]]
    )
    torch.cuda.synchronize()
    assert torch.equal(actual[[1, 3]], expected)
    assert torch.isfinite(actual).all() and torch.equal(
        actual[[0, 2]], torch.zeros_like(actual[[0, 2]])
    )
    assert not torch.signbit(actual[[0, 2]]).any()


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [7, 32, 64, 128, 256])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_dim_buckets_match_independent_dense_attention(head_dim, groups, dtype):
    torch.manual_seed(271)
    keys = torch.randn(4, 16, 2, head_dim, device="cuda", dtype=dtype)
    values = torch.randn_like(keys)
    before_k, before_v = keys.clone(), values.clone()
    q = torch.randn(4, 2 * groups, head_dim, device="cuda", dtype=dtype)
    q[[0, 2]] = float("nan")
    tables = torch.tensor(
        [[2**30, -1, -1], [1, 3, 0], [-1, 2**30, -1], [2, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([0, 35, 0, 1], dtype=torch.int32, device="cuda")
    active = torch.tensor([False, True, False, True], device="cuda")
    actual = triton_paged_attention(q, keys, values, tables, lengths, active)
    for row, blocks, count in [(1, [1, 3, 0], 35), (3, [2], 1)]:
        k = keys[blocks].reshape(-1, 2, head_dim)[:count]
        v = values[blocks].reshape(-1, 2, head_dim)[:count]
        expected = dense_attention(q[row], k, v)
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-5
        torch.testing.assert_close(actual[row], expected, atol=tolerance, rtol=tolerance)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[[0, 2]], torch.zeros_like(actual[[0, 2]]))
    assert not torch.signbit(actual[[0, 2]]).any()
    assert torch.equal(keys, before_k) and torch.equal(values, before_v)
