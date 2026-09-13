"""Graph experiment adapter for the existing, timed M1 inference loop."""

from contextlib import contextmanager

import torch

from vllm_lt.benchmarks.ab_schema import equal, require
from vllm_lt.benchmarks.profile import Capture
from vllm_lt.benchmarks.runner import write_json


class GraphCapture(Capture):
    """Label actual replay calls after setup; never instrument a captured body."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatches = []

    def __enter__(self):
        super().__enter__()
        try:
            return self._install_graph_wrappers()
        except BaseException as primary:
            try:
                self.__exit__(None, None, None)
            except BaseException as secondary:
                if hasattr(primary, "add_note"):
                    primary.add_note(f"profile wrapper cleanup failed: {secondary}")
            raise

    def _install_graph_wrappers(self):
        executor = self.engine.model_runner.decode_executor
        original = executor.recurrent
        self.saved.append(
            (executor, "recurrent", "recurrent" in vars(executor), vars(executor).get("recurrent"))
        )

        def dispatch(hidden, request_ids, depths, positions, **kwargs):
            if not self.active:
                return original(hidden, request_ids, depths, positions, **kwargs)
            bucket, kind = executor.preview_dispatch(request_ids, positions)
            dispatch_id = executor.counters["calls"] + 1
            label = f"vllm_lt::graph_dispatch::{dispatch_id}::bucket::{bucket or 0}::{kind}"
            with torch.profiler.record_function(label):
                result = original(hidden, request_ids, depths, positions, **kwargs)
            actual = executor.last_dispatch
            equal(
                (actual["dispatch_id"], actual["bucket_id"], actual["kind"]),
                (dispatch_id, bucket, kind),
                "profile actual dispatch",
            )
            self.dispatches.append(
                {
                    "dispatch_id": dispatch_id,
                    "bucket_id": bucket,
                    "kind": kind,
                    "generation": actual["generation"],
                    "graph_exec_id": executor.buckets[bucket]["graph_exec_id"] if bucket else None,
                    "scope": label,
                }
            )
            return result

        executor.recurrent = dispatch
        for rows, bucket in executor.buckets.items():
            graph = bucket["graph"]
            if graph is None:
                continue
            replay = graph.replay
            self.saved.append((graph, "replay", "replay" in vars(graph), vars(graph).get("replay")))

            def recorded_replay(replay=replay, rows=rows, executable=bucket["graph_exec_id"]):
                if self.active:
                    with torch.profiler.record_function(
                        f"vllm_lt::graph_replay::{rows}::exec::{executable}"
                    ):
                        return replay()
                return replay()

            graph.replay = recorded_replay
        return self


@contextmanager
def finite_checks(engine, enabled):
    """Check real returned decode outputs, including replay, only in feasibility."""
    counts = {"recurrent": 0, "coda": 0}
    if not enabled:
        yield counts
        return
    saved = []

    def install(owner, method, category):
        original = getattr(owner, method)
        saved.append((owner, method, method in vars(owner), vars(owner).get(method)))

        def checked(*args, **kwargs):
            result = original(*args, **kwargs)
            values = result if isinstance(result, tuple) else (result,)
            if not all(bool(value.isfinite().all().item()) for value in values):
                raise ValueError(f"nonfinite {category} result during graph feasibility")
            counts[category] += 1
            return result

        setattr(owner, method, checked)

    try:
        install(engine.model, "recurrent", "recurrent")
        install(engine.model_runner, "_recurrent", "recurrent")
        install(engine.model, "coda", "coda")
        yield counts
    finally:
        for owner, method, existed, value in reversed(saved):
            if existed:
                setattr(owner, method, value)
            else:
                delattr(owner, method)


class ExecutionAdapter:
    """One experiment configuration, with an explicit fresh executor per engine."""

    capture_type = GraphCapture
    finite_checks = staticmethod(finite_checks)

    @staticmethod
    def _failure_record(path, evidence, primary):
        try:
            write_json(path, evidence)
        except BaseException as secondary:
            if hasattr(primary, "add_note"):
                primary.add_note(f"failure evidence export failed at {path.name}: {secondary}")

    def __init__(self, *, implementation_id, graph_limits):
        require(implementation_id in ("A", "B"), "unknown graph implementation")
        self.implementation_id = implementation_id
        self.use_graphs = implementation_id == "B"
        self.graph_limits = dict(graph_limits)
        self.initial = None
        self.engine = None

    def prepare(self, engine, run, run_dir):
        equal(run["implementation_id"], self.implementation_id, "adapter implementation")
        equal(run["use_graphs"], self.use_graphs, "adapter replay configuration")
        self.initial = None
        self.engine = engine
        try:
            engine.model_runner._enable_recurrent_graph(
                use_graphs=self.use_graphs, limits=self.graph_limits
            )
            executor = engine.model_runner.decode_executor
            require(
                executor is not None,
                "graph setup declined its budget; frozen A/B qualification requires the executor",
            )
            executor.record_dispatch_tensors = run["phase"] == "profile"
            self.initial = executor.snapshot()
            write_json(run_dir / "graph-setup.json", self.initial)
        except BaseException as primary:
            executor = engine.model_runner.decode_executor
            evidence = {"error": {"type": type(primary).__name__, "message": str(primary)}}
            evidence["runner_setup"] = engine.model_runner._graph_snapshot()
            if executor is not None:
                evidence["failed_setup"] = executor.snapshot()
                try:
                    executor.close()
                    evidence["after_close"] = executor.snapshot()
                except BaseException as secondary:
                    evidence["cleanup_error"] = {
                        "type": type(secondary).__name__,
                        "message": str(secondary),
                    }
                    if hasattr(primary, "add_note"):
                        primary.add_note(f"graph setup cleanup failed: {secondary}")
            self._failure_record(run_dir / "graph-setup-failure.json", evidence, primary)
            raise

    def finish(self, engine, capture):
        executor = engine.model_runner.decode_executor
        final = executor.snapshot()
        executor.close()
        closed = executor.snapshot()
        require(closed["status"] == "closed" and not closed["buckets"], "graph ownership remains")
        result = {
            "schema_version": 1,
            "implementation_id": self.implementation_id,
            "use_graphs": self.use_graphs,
            "initial": self.initial,
            "final": final,
            "closed": closed,
            "cleanup": {"closed": True},
            "profile_dispatches": capture.dispatches if capture is not None else [],
        }
        self.initial = None
        self.engine = None
        return result

    def abort(self, primary, run_dir):
        """Settle failures occurring outside the ordinary inference recovery block."""
        if self.engine is None:
            return
        engine = self.engine
        evidence = {"primary": {"type": type(primary).__name__, "message": str(primary)}}
        executor = engine.model_runner.decode_executor
        try:
            if executor is not None and executor.status != "closed":
                executor.settle_failure(primary)
            for request_id in list(engine.scheduler.requests):
                engine.abort_request(request_id)
            if executor is not None:
                evidence["before_close"] = executor.snapshot()
                executor.close()
                evidence["after_close"] = executor.snapshot()
            else:
                engine.model_runner._close_recurrent_graph()
                evidence["after_close"] = engine.model_runner._graph_snapshot()
            self.engine = None
        except BaseException as secondary:
            evidence["cleanup_error"] = {
                "type": type(secondary).__name__,
                "message": str(secondary),
            }
            if hasattr(primary, "add_note"):
                primary.add_note(f"graph execution cleanup failed: {secondary}")
        finally:
            self._failure_record(run_dir / "graph-failure-cleanup.json", evidence, primary)
