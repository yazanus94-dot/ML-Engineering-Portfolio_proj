"""Timing and correctness harness."""

from __future__ import annotations
import platform
import statistics
import time
import tracemalloc
import numpy as np
from . import model as M

# Maximum allowed error relative to the dynamic range of the float64 reference output.
TOLERANCE = {"loops": 1e-9, "per_sample": 1e-4, "batched32": 1e-4, "int8": 5e-2}


def _time(fn, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def _peak_alloc(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def build_impls(m: M.MLP):
    p32, pq = M.prepare_batched32(m), M.prepare_int8(m)
    return {
        "reference": (lambda x: M.forward_reference(m, x), m),
        "loops": (lambda x: M.forward_loops(m, x), m),
        "per_sample": (lambda x: M.forward_per_sample(p32, x), p32),
        "batched32": (lambda x: M.forward_batched32(p32, x), p32),
        "int8": (lambda x: M.forward_int8(pq, x), pq),
    }


def relative_error(out, ref):
    return float(
        np.max(np.abs(np.asarray(out, dtype=np.float64) - ref))
        / max(np.ptp(ref), 1e-12)
    )


def run(sizes=(256, 512, 512, 10), batch=256, repeats=7, seed=0, loops_batch=8) -> dict:
    m = M.MLP.random(list(sizes), seed)
    x = np.random.default_rng(seed + 1).standard_normal((batch, sizes[0]))
    impls = build_impls(m)
    ref = M.forward_reference(m, x)
    rows = []
    for name, (fn, prep) in impls.items():
        xb = (
            x[:loops_batch] if name == "loops" else x
        )  # pure-Python loops get a smaller batch
        refb = ref[: len(xb)]
        out = fn(xb)
        ts = _time(lambda: fn(xb), repeats if name != "loops" else max(2, repeats // 3))
        med = statistics.median(ts)
        err = 0.0 if name == "reference" else relative_error(out, refb)
        agree = float((np.asarray(out).argmax(1) == refb.argmax(1)).mean())
        rows.append(
            {
                "impl": name,
                "batch": len(xb),
                "median_ms": med * 1e3,
                "samples_per_s": len(xb) / med,
                "rel_error": err,
                "tolerance": TOLERANCE.get(name, 0.0),
                "within_tolerance": err <= TOLERANCE.get(name, 0.0),
                "argmax_agreement": agree,
                "weight_bytes": M.weight_bytes(prep, name),
                "peak_alloc_bytes": _peak_alloc(lambda: fn(xb)),
            }
        )
    return {
        "config": {
            "sizes": list(sizes),
            "batch": batch,
            "repeats": repeats,
            "seed": seed,
            "loops_batch": loops_batch,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "system": platform.platform(),
        },
        "results": rows,
    }


def to_markdown(res: dict) -> str:
    c, e = res["config"], res["environment"]
    L = [
        "# Benchmark results",
        "",
        f"MLP {c['sizes']}, batch {c['batch']} (the `loops` row uses batch {c['loops_batch']}), {c['repeats']} repeats, seed {c['seed']}.",
        f"Environment: Python {e['python']}, NumPy {e['numpy']}, {e['system']}.",
        "",
        "Throughput is only comparable between rows with the same batch size. Timings depend on the machine; re-run on your own hardware.",
        "",
        "| impl | batch | median ms | samples/s | rel. error | tolerance | ok | argmax agree | weights (KiB) | peak alloc (KiB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["results"]:
        L.append(
            f"| {r['impl']} | {r['batch']} | {r['median_ms']:.3f} | {r['samples_per_s']:.0f} | {r['rel_error']:.2e} | {r['tolerance']:.0e} | "
            f"{'yes' if r['within_tolerance'] else 'NO'} | {r['argmax_agreement']:.3f} | {r['weight_bytes']/1024:.1f} | {r['peak_alloc_bytes']/1024:.1f} |"
        )
    return "\n".join(L) + "\n"
