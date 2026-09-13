import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu", action="store_true", help="Run CUDA tests inside a GPU reservation"
    )


def pytest_collection_modifyitems(config, items):
    enabled = config.getoption("--run-gpu")
    if enabled and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise pytest.UsageError("--run-gpu requires scheduler-provided CUDA_VISIBLE_DEVICES")
    for item in items:
        if "gpu" in item.keywords and not enabled:
            item.add_marker(pytest.mark.skip(reason="use --run-gpu inside a GPU reservation"))


@pytest.fixture(autouse=True, scope="session")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def forbid_cuda(monkeypatch):
    """Opt-in guard for CPU graph tests; individual tests install their own fakes."""

    def forbidden(*args, **kwargs):
        pytest.fail("CPU test attempted CUDA discovery or device work")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
        "current_stream",
        "Stream",
        "CUDAGraph",
        "MemPool",
        "get_allocator_backend",
        "memory_allocated",
        "memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    return forbidden
