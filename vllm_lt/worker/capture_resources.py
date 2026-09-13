"""CUDA resource ownership; idle capture streams are reused only after safe close."""

import threading

import torch

_IDLE_STREAMS = {}
_STREAM_LOCK = threading.Lock()


class _CudaRuntime:
    """The small CUDA setup surface; CPU tests replace it without device discovery."""

    def __init__(self, device):
        self.device = device
        self._leased_stream = None

    def current_stream(self):
        return torch.cuda.current_stream(self.device)

    def new_stream(self):
        if self._leased_stream is not None:
            raise RuntimeError("capture stream is already leased")
        key = torch.device(self.device)
        with _STREAM_LOCK:
            idle = _IDLE_STREAMS.setdefault(key, [])
            stream = idle.pop() if idle else torch.cuda.Stream(device=self.device)
        self._leased_stream = stream
        return stream

    def release_stream(self, stream):
        if stream is not self._leased_stream:
            raise RuntimeError("capture stream does not belong to this runtime")
        # Caller has synchronized, reset every graph and dropped pool owners.
        # Never return a stream while another executor can still use its graphs.
        with _STREAM_LOCK:
            _IDLE_STREAMS.setdefault(torch.device(self.device), []).append(stream)
        self._leased_stream = None

    @staticmethod
    def stream_context(stream):
        return torch.cuda.stream(stream)

    @staticmethod
    def new_graph():
        return torch.cuda.CUDAGraph(keep_graph=False)

    def new_pool(self):
        # Public MemPool owns the user-created pool until its targeted teardown.
        # A graph reset alone only makes an implicit graph pool freeable.
        if torch.cuda.get_allocator_backend() != "native":
            raise RuntimeError("owned recurrent graph pools require the native CUDA allocator")
        with torch.cuda.device(self.device):
            return torch.cuda.MemPool()

    def memory(self):
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }

    def reset_peaks(self):
        torch.cuda.reset_peak_memory_stats(self.device)


def _make_runtime(device):
    return _CudaRuntime(device) if device.type == "cuda" else None
