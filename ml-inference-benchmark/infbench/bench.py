"""Timing and correctness harness."""

from __future__ import annotations

import json
import os
import pathlib
import platform
import statistics
import time
import tracemalloc

import numpy as np

from . import model as M

# Maximum allowed error relative to the dynamic range of the float64 reference output.
TOLERANCE = {"loops": 1e-9, "per_sample": 1e-4, "batched32": 1e-4, "int8": 5e-2}

METHODOLOGY = {
    "latency": "wall-clock time per call, median of repeated measurements after one warm-up",
    "throughput": "batch / median wall-clock seconds",
    "per_sample_latency": "median wall-clock milliseconds / batch size",
    "variability": "sample standard deviation across repeats, plus minimum and maximum",
    "memory": "tracemalloc peak during one call; an already-active tracing session is preserved",
}


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
    """Return peak traced bytes without disturbing an existing tracing session."""
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    try:
        fn()
        return float(tracemalloc.get_traced_memory()[1])
    finally:
        if started_here:
            tracemalloc.stop()


def _summarise(ts):
    if not ts:
        raise ValueError("at least one timing measurement is required")
    std = statistics.stdev(ts) if len(ts) > 1 else 0.0
    return {
        "median": statistics.median(ts),
        "std": std,
        "min": min(ts),
        "max": max(ts),
    }


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


def run(
    sizes=(256, 512, 512, 10),
    batch=256,
    repeats=7,
    seed=0,
    loops_batch=None,
) -> dict:
    if batch < 1:
        raise ValueError("batch must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if loops_batch is not None and not 1 <= loops_batch <= batch:
        raise ValueError("loops_batch must be between 1 and batch")

    m = M.MLP.random(list(sizes), seed)
    x = np.random.default_rng(seed + 1).standard_normal((batch, sizes[0]))
    impls = build_impls(m)
    ref = M.forward_reference(m, x)
    rows = []
    for name, (fn, prep) in impls.items():
        xb = x if loops_batch is None else x[:loops_batch]
        refb = ref[: len(xb)]
        out = fn(xb)
        ts = _time(lambda: fn(xb), repeats if name != "loops" else max(2, repeats // 3))
        stats = _summarise(ts)
        med = stats["median"]
        err = 0.0 if name == "reference" else relative_error(out, refb)
        agree = float((np.asarray(out).argmax(1) == refb.argmax(1)).mean())
        rows.append(
            {
                "impl": name,
                "batch": len(xb),
                "median_ms": med * 1e3,
                "std_ms": stats["std"] * 1e3,
                "min_ms": stats["min"] * 1e3,
                "max_ms": stats["max"] * 1e3,
                "median_ms_per_sample": (med / len(xb)) * 1e3,
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
            "comparison_batch": batch,
            "repeats": repeats,
            "seed": seed,
            "loops_batch": loops_batch if loops_batch is not None else batch,
        },
        "methodology": METHODOLOGY,
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
    loops_note = (
        ""
        if c["loops_batch"] == c["batch"]
        else f" The `loops` row uses batch {c['loops_batch']} for practicality."
    )
    L = [
        "# Benchmark results",
        "",
        f"MLP {c['sizes']}, comparison batch {c['batch']}, {c['repeats']} repeats, seed {c['seed']}.{loops_note}",
        f"Environment: Python {e['python']}, NumPy {e['numpy']}, {e['system']}.",
        "",
        "Throughput = batch / median wall-clock seconds. `ms/sample` = median wall-clock milliseconds / batch size.",
        "Variability is the sample standard deviation across repeats; min and max are the observed extrema.",
        "Timings depend on the machine; re-run on your own hardware. Compare throughput only at equal batch size.",
        "",
        "| impl | batch | median ms | std ms | min ms | max ms | ms/sample | samples/s | rel. error | tolerance | ok | argmax agree | weights (KiB) | peak alloc (KiB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["results"]:
        L.append(
            f"| {r['impl']} | {r['batch']} | {r['median_ms']:.3f} | {r['std_ms']:.3f} | "
            f"{r['min_ms']:.3f} | {r['max_ms']:.3f} | {r['median_ms_per_sample']:.4f} | "
            f"{r['samples_per_s']:.0f} | {r['rel_error']:.2e} | {r['tolerance']:.0e} | "
            f"{'yes' if r['within_tolerance'] else 'NO'} | {r['argmax_agreement']:.3f} | "
            f"{r['weight_bytes'] / 1024:.1f} | {r['peak_alloc_bytes'] / 1024:.1f} |"
        )
    return "\n".join(L) + "\n"


def _write_text_atomic(path: pathlib.Path, text: str) -> None:
    """Write text through a temporary file so a failure cannot truncate the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_report(res: dict, out_dir) -> pathlib.Path:
    """Write JSON and Markdown reports atomically.

    Both payloads are rendered before any target is touched. If rendering or
    writing fails, the previously committed report is left unchanged.
    """
    out = pathlib.Path(out_dir)
    json_text = json.dumps(res, indent=2) + "\n"
    md_text = to_markdown(res)
    _write_text_atomic(out / "benchmark.json", json_text)
    _write_text_atomic(out / "benchmark.md", md_text)
    return out
