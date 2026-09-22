"""Large KV pools must promote physical-address arithmetic before multiplying."""

import pytest
import torch


@pytest.mark.gpu
def test_paged_attention_int32_block_ids_use_64bit_physical_offsets():
    from vllm_rlt.kernels.triton_attention import paged_attention

    # Small logical rows separated by a large physical layer/block stride.
    # The last block's offset is 2**31 elements, although ID and stride each
    # fit in int32. Touch only the two rows used by this test, not all 8 GiB.
    if torch.cuda.mem_get_info()[0] < 9 * 2**30:
        pytest.skip("large-address regression requires 9 GiB free CUDA memory")
    shape = (2049, 1, 1, 16)
    strides = (2**20, 16, 16, 1)
    key = torch.empty_strided(shape, strides, device="cuda", dtype=torch.float16)
    value = torch.empty_strided(shape, strides, device="cuda", dtype=torch.float16)
    key[0].zero_()
    value[0].fill_(-3)
    key[2048].fill_(1)
    expected = torch.arange(16, device="cuda", dtype=torch.float16).view(1, 1, 16)
    value[2048].copy_(expected)
    query = torch.ones((1, 1, 16), device="cuda", dtype=torch.float16)
    table = torch.tensor([[2048]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1], device="cuda", dtype=torch.int32)
    actual = paged_attention(query, key, value, table, lengths)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
