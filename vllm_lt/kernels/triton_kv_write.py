"""Masked KV scatter for physical padding rows; compact writes retain their path.

Strides are compilation constants: distinct layouts compile separate variants.
Captured callers must hold layouts fixed within each declared bucket.
"""

import triton
import triton.language as tl


@triton.jit
def _masked_kv_write_kernel(
    K,
    V,
    KEY_CACHE,
    VALUE_CACHE,
    BLOCKS,
    OFFSETS,
    ACTIVE,
    k_row: tl.constexpr,
    k_head: tl.constexpr,
    k_dim: tl.constexpr,
    v_row: tl.constexpr,
    v_head: tl.constexpr,
    v_dim: tl.constexpr,
    kc_block: tl.constexpr,
    kc_token: tl.constexpr,
    kc_head: tl.constexpr,
    kc_dim: tl.constexpr,
    vc_block: tl.constexpr,
    vc_token: tl.constexpr,
    vc_head: tl.constexpr,
    vc_dim: tl.constexpr,
    blocks_stride: tl.constexpr,
    offsets_stride: tl.constexpr,
    active_stride: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    enabled = tl.load(ACTIVE + row * active_stride)
    block = tl.load(BLOCKS + row * blocks_stride, mask=enabled, other=0)
    offset = tl.load(OFFSETS + row * offsets_stride, mask=enabled, other=0)
    elements = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    heads, dims = elements // HEAD_DIM, elements % HEAD_DIM
    valid = enabled & (elements < HEADS * HEAD_DIM)
    keys = tl.load(K + row * k_row + heads * k_head + dims * k_dim, mask=valid, other=0)
    values = tl.load(V + row * v_row + heads * v_head + dims * v_dim, mask=valid, other=0)
    tl.store(
        KEY_CACHE + block * kc_block + offset * kc_token + heads * kc_head + dims * kc_dim,
        keys,
        mask=valid,
    )
    tl.store(
        VALUE_CACHE + block * vc_block + offset * vc_token + heads * vc_head + dims * vc_dim,
        values,
        mask=valid,
    )


def masked_kv_write(key_cache, value_cache, write_blocks, write_offsets, k, v, active):
    """Write validated active rows to strided [block, token, KV head, dim] views.

    The manager validates ownership, shapes, metadata, and duplicate live
    destinations. All inactive address and source loads/stores are masked.
    """
    if k.device.type != "cuda":
        raise ValueError("the Triton KV write requires a CUDA or ROCm device")
    if k.shape[0] == 0:
        return
    _masked_kv_write_kernel[(k.shape[0], triton.cdiv(k.shape[1] * k.shape[2], 256))](
        k,
        v,
        key_cache,
        value_cache,
        write_blocks,
        write_offsets,
        active,
        *k.stride(),
        *v.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        write_blocks.stride(0),
        write_offsets.stride(0),
        active.stride(0),
        HEADS=k.shape[1],
        HEAD_DIM=k.shape[2],
        BLOCK=256,
    )
