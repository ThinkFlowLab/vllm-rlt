"""Measure eight-row synchronous metadata preparation on a reserved GPU."""

import statistics
import time

import torch
import triton

from vllm_rlt.core.kv_cache_manager import KVCacheManager

cache = KVCacheManager(1, 2, 64, 256, 16, 2, device="cuda", dtype=torch.bfloat16, backend="triton")
ids = [f"r{i}" for i in range(8)]
for rid in ids:
    assert cache.allocate(rid, 128)
depths = [i % 2 for i in range(8)]
positions = [63, 47, 31, 15, 55, 39, 23, 7]
for _ in range(50):
    cache._prepare_batch(ids, depths, positions)
torch.cuda.synchronize()
samples = []
for _ in range(7):
    start = time.perf_counter()
    for _ in range(300):
        cache._prepare_batch(ids, depths, positions)
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) / 300 * 1e6)
print(f"metadata_us_median={statistics.median(samples):.1f}")
print("metadata_us_samples=" + ",".join(f"{x:.1f}" for x in samples))
print(f"torch={torch.__version__} triton={triton.__version__}")
print(f"gpu={torch.cuda.get_device_name()}")
