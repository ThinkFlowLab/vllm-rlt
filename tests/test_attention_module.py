"""M8 contracts independent of logical KV allocation and device ownership."""

from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.attention import (
    AttentionRows,
    backend_capabilities,
    build_attention_metadata,
    create_backend,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager


def test_backend_factory_preserves_reference_and_explicit_rejections():
    backend = create_backend("torch", torch.device("cpu"), torch.float32, 8, 2)
    assert callable(backend)
    with pytest.raises(ValueError, match="unknown attention backend"):
        create_backend("unknown", torch.device("cpu"), torch.float32, 8, 2)
    with pytest.raises(ValueError, match="requires a CUDA or ROCm device"):
        create_backend("triton", torch.device("cpu"), torch.float32, 8, 2)
    with pytest.raises(ValueError, match="head_dim <= 256"):
        create_backend("triton", torch.device("cuda"), torch.float32, 257, 2)


def test_backend_name_is_checked_before_other_cache_options():
    with pytest.raises(ValueError, match="unknown attention backend"):
        KVCacheManager(1, 1, 8, 4, 2, 1, dtype=torch.int32, backend="missing")


def test_packed_prefill_capability_is_an_explicit_backend_contract():
    reference = create_backend("torch", torch.device("cpu"), torch.float32, 8, 2)
    assert backend_capabilities(reference).packed_prefill is False
    assert backend_capabilities(SimpleNamespace(generation=4)).packed_prefill is True
    cache = KVCacheManager(1, 1, 8, 4, 2, 1)
    assert cache.attention_capabilities.packed_prefill is False
    cache.attention = SimpleNamespace(generation=4)
    assert cache.attention_capabilities.packed_prefill is True


def test_metadata_is_depth_selected_padded_and_borrowed_for_one_traversal():
    rows = AttentionRows(block_tables=[(4, 7, 8), (2, 3)], positions=[4, 1])
    metadata = build_attention_metadata(rows, block_size=2, device=torch.device("cpu"))
    assert metadata.block_tables.tolist() == [[4, 7, 8], [2, 3, -1]]
    assert metadata.context_lengths.tolist() == [5, 2]
    assert metadata.block_tables.dtype == metadata.context_lengths.dtype == torch.int32
    assert metadata.cu_seqlens_q is None
    assert metadata.max_seqlen_q == 1


def test_packed_metadata_groups_contiguous_query_chunks():
    rows = AttentionRows(
        block_tables=[(4, 7, 8), (4, 7, 8), (2, 3)],
        positions=[2, 3, 1],
        sequence_keys=[("a", 0), ("a", 0), ("b", 1)],
    )
    metadata = build_attention_metadata(
        rows, block_size=2, device=torch.device("cpu"), packed_prefill=True
    )
    assert metadata.block_tables.tolist() == [[4, 7], [2, 3]]
    assert metadata.context_lengths.tolist() == [4, 2]
    assert metadata.cu_seqlens_q.tolist() == [0, 2, 3]
    assert metadata.max_seqlen_q == 2
    with pytest.raises(ValueError, match="contiguous"):
        build_attention_metadata(
            AttentionRows(
                block_tables=[(4, 7, 8), (4, 7, 8)],
                positions=[2, 4],
                sequence_keys=[("a", 0), ("a", 0)],
            ),
            block_size=2,
            device=torch.device("cpu"),
            packed_prefill=True,
        )
    with pytest.raises(ValueError, match="one sequence"):
        build_attention_metadata(
            AttentionRows(
                block_tables=[(4, 7, 8), (2, 3), (4, 7, 8)],
                positions=[2, 1, 3],
                sequence_keys=[("a", 0), ("b", 1), ("a", 0)],
            ),
            block_size=2,
            device=torch.device("cpu"),
            packed_prefill=True,
        )


def test_metadata_rejects_misaligned_row_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        build_attention_metadata(
            AttentionRows(block_tables=[(1,)], positions=[0, 1]),
            block_size=2,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="equal lengths"):
        build_attention_metadata(
            AttentionRows(block_tables=[(1,)], positions=[0], sequence_keys=[("a", 0), ("a", 0)]),
            block_size=2,
            device=torch.device("cpu"),
            packed_prefill=True,
        )
