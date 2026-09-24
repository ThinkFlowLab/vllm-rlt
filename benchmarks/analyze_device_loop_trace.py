"""Summarize a profile_device_loop.py trace: nsys sqlite export or torch.profiler JSON.

Input is either ``nsys export --type=sqlite`` output (NVTX ranges) or the Kineto/CUPTI
``.json[.gz]`` written by ``--profiler torch`` (record_function ranges); both are
normalized to the same host-range / API / device-activity tuples.

Per ``engine.step`` NVTX range (host interval) it reports CUDA API counts, time spent in
blocking API calls, kernels/memcpys launched from the range, GPU busy time as the union
of device intervals (overlaps are not double-counted), and GPU idle time inside the
span of that step's device work. Each GPU idle gap in the profiled window is attributed
to the innermost NVTX range that launched the next device operation.

Usage: python -m benchmarks.analyze_device_loop_trace trace.{sqlite,json.gz} [--json out.json]
"""

import argparse
import bisect
import gzip
import json
import sqlite3
import statistics
from collections import Counter, defaultdict

BLOCKING = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cuStreamSynchronize",
    "cuCtxSynchronize",
)
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def _table(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None


def load(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    nvtx = [
        (s, e, t or names.get(tid, ""), g)
        for s, e, t, tid, g in db.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
            "WHERE eventType = 59 AND end IS NOT NULL"
        )
    ]
    api = [
        (s, e, names[n].split("_v")[0], c, g)
        for s, e, n, c, g in db.execute(
            "SELECT start, end, nameId, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    ]
    device = []  # (start, end, kind, correlationId, detail)
    if _table(db, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for s, e, c, n in db.execute(
            "SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            device.append((s, e, "kernel", c, names.get(n, "?")))
    if _table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for s, e, c, k, b in db.execute(
            "SELECT start, end, correlationId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            device.append((s, e, f"memcpy.{COPY_KIND.get(k, k)}", c, b))
    if _table(db, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for s, e, c in db.execute(
            "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            device.append((s, e, "memset", c, None))
    device.sort()
    return nvtx, api, device


def load_kineto(path):
    """Kineto chrome trace: timestamps are microseconds; convert to integer ns."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    ns = lambda us: int(round(us * 1000))  # noqa: E731
    nvtx, api, device = [], [], []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, s = ev.get("cat"), ns(ev["ts"])
        e = s + ns(ev.get("dur", 0))
        corr = (ev.get("args") or {}).get("correlation")
        if cat == "user_annotation":
            nvtx.append((s, e, ev["name"], ev.get("tid")))
        elif cat in ("cuda_runtime", "cuda_driver"):
            api.append((s, e, ev["name"], corr, ev.get("tid")))
        elif cat == "kernel":
            device.append((s, e, "kernel", corr, ev["name"]))
        elif cat == "gpu_memcpy":
            kind = ev["name"].split()[1] if " " in ev["name"] else ev["name"]
            device.append((s, e, f"memcpy.{kind}", corr, (ev.get("args") or {}).get("bytes")))
        elif cat == "gpu_memset":
            device.append((s, e, "memset", corr, None))
    device.sort()
    return nvtx, api, device


def union(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def innermost(ranges, t):
    """Innermost NVTX push/pop range containing host time t (ranges sorted by start)."""
    best = None
    for s, e, text, _ in ranges:
        if s > t:
            break
        if e >= t and (best is None or s >= best[0]):
            best = (s, e, text)
    return best[2] if best else "<no range>"


def analyze(path, min_gap_ns=0):
    nvtx, api, device = load(path) if str(path).endswith(".sqlite") else load_kineto(path)
    steps = sorted((s, e) for s, e, t, _ in nvtx if t == "engine.step")
    ranges = sorted(nvtx)
    api_by_corr = {c: (s, e, n, g) for s, e, n, c, g in api}
    api_starts = sorted((s, e, n, c) for s, e, n, c, _ in api)
    starts = [a[0] for a in api_starts]

    step_rows = []
    for index, (s, e) in enumerate(steps):
        children = sorted(
            {t for rs, re_, t, _ in nvtx if s < rs and re_ <= e and t != "engine.step"}
        )
        kind = (
            "spec"
            if "spec.round" in children
            else "native"
            if "native.execute" in children
            else "?"
        )
        lo, hi = bisect.bisect_left(starts, s), bisect.bisect_right(starts, e)
        calls = api_starts[lo:hi]
        corr = {c for _, _, _, c in calls}
        ops = [d for d in device if d[3] in corr]
        busy = union([(a, b) for a, b, *_ in ops])
        span = (max(b for _, b, *_ in ops) - min(a for a, *_ in ops)) if ops else 0
        count = Counter(n for _, _, n, _ in calls)
        blocking = sum(b - a for a, b, n, _ in calls if n.startswith(BLOCKING))
        step_rows.append(
            dict(
                index=index,
                kind=kind,
                host_ns=e - s,
                api_calls=len(calls),
                api_counts=dict(count.most_common()),
                blocking_api_ns=blocking,
                stream_syncs=sum(v for k, v in count.items() if "Synchronize" in k),
                kernels=sum(1 for d in ops if d[2] == "kernel"),
                memcpy=dict(Counter(d[2] for d in ops if d[2].startswith("memcpy"))),
                memcpy_dtoh_bytes=[d[4] for d in ops if d[2] == "memcpy.DtoH"][:4],
                gpu_busy_ns=busy,
                gpu_span_ns=span,
                gpu_idle_in_span_ns=span - busy,
            )
        )

    # Idle gaps across the whole window: attribute each to the launcher of the next op.
    gaps = Counter()
    gap_ns = Counter()
    # Gap size histogram separates intra-graph kernel spacing (~1-2 us, inflated by
    # CUPTI) from host-induced idle; only gaps >= min_gap_ns are attributed.
    buckets = ((0, 2_000), (2_000, 10_000), (10_000, 100_000), (100_000, None))
    hist = {f"{lo // 1000}-{hi // 1000 if hi else 'inf'}us": [0, 0] for lo, hi in buckets}
    if steps:
        window = [d for d in device if steps[0][0] <= d[0] <= steps[-1][1] + 10**9]
    else:  # CUDA-only trace (no CPU ranges): use the whole device timeline
        window = list(device)
    frontier = None
    for s, e, kind, c, _ in window:
        if frontier is not None and s > frontier:
            size = s - frontier
            for (lo, hi), key in zip(buckets, hist):
                if size >= lo and (hi is None or size < hi):
                    hist[key][0] += 1
                    hist[key][1] += size
            if size >= min_gap_ns:
                launch = api_by_corr.get(c)
                label = innermost(ranges, launch[0]) if launch else "<unknown>"
                gaps[label] += 1
                gap_ns[label] += size
        frontier = e if frontier is None else max(frontier, e)
    busy_all = union([(s, e) for s, e, *_ in window])
    if steps:
        wall = steps[-1][1] - steps[0][0]
    else:
        wall = (max(e for _, e, *_ in window) - window[0][0]) if window else 0
    by_kind = defaultdict(list)
    for row in step_rows:
        by_kind[row["kind"]].append(row)

    def med(rows, key):
        return statistics.median(r[key] for r in rows) if rows else None

    return dict(
        trace=str(path),
        steps=len(step_rows),
        window_wall_ns=wall,
        window_gpu_busy_ns=busy_all,
        window_gpu_busy_fraction=busy_all / wall if wall else None,
        per_kind_median={
            k: {
                key: med(rows, key)
                for key in (
                    "host_ns",
                    "api_calls",
                    "stream_syncs",
                    "blocking_api_ns",
                    "kernels",
                    "gpu_busy_ns",
                    "gpu_span_ns",
                    "gpu_idle_in_span_ns",
                )
            }
            | {"n": len(rows)}
            for k, rows in by_kind.items()
        },
        min_gap_ns=min_gap_ns,
        idle_gap_histogram={k: dict(count=v[0], total_ns=v[1]) for k, v in hist.items()},
        idle_gaps_by_launching_range={
            k: dict(count=gaps[k], total_ns=gap_ns[k]) for k, _ in gap_ns.most_common()
        },
        step_rows=step_rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("--json")
    p.add_argument("--min-gap-us", type=float, default=0.0, help="attribute only larger gaps")
    args = p.parse_args()
    result = analyze(args.trace, int(args.min_gap_us * 1000))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    print(
        f"{result['trace']}: {result['steps']} engine steps, wall "
        f"{result['window_wall_ns'] / 1e6:.2f} ms, GPU busy (union) "
        f"{result['window_gpu_busy_ns'] / 1e6:.2f} ms "
        f"({(result['window_gpu_busy_fraction'] or 0):.1%})"
    )
    for kind, row in result["per_kind_median"].items():
        print(
            f"  [{kind}] n={row['n']} median host {row['host_ns'] / 1e3:.0f} us, "
            f"GPU busy {row['gpu_busy_ns'] / 1e3:.0f} us, idle-in-span "
            f"{row['gpu_idle_in_span_ns'] / 1e3:.0f} us, api {row['api_calls']}, "
            f"syncs {row['stream_syncs']}, blocking-api {row['blocking_api_ns'] / 1e3:.0f} us, "
            f"kernels {row['kernels']}"
        )
    print("  GPU idle gap histogram (count, total ms):")
    for key, v in result["idle_gap_histogram"].items():
        print(f"    {key:14s} {v['count']:7d} {v['total_ns'] / 1e6:9.3f}")
    print(
        f"  GPU idle gaps >= {result['min_gap_ns'] / 1e3:.0f} us by launching range "
        "(count, total ms):"
    )
    for label, v in list(result["idle_gaps_by_launching_range"].items())[:12]:
        print(f"    {label:32s} {v['count']:6d} {v['total_ns'] / 1e6:9.3f}")


if __name__ == "__main__":
    main()
