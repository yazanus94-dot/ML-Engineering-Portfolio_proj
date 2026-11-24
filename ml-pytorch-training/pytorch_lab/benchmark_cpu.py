"""CPU inference benchmark for a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from . import __version__
from .model import load_checkpoint


def benchmark(
    checkpoint_path: str | Path,
    batch_size: int = 64,
    repeats: int = 50,
    warmup: int = 5,
) -> dict:
    """Measure batched CPU forward latency with warmup and repeats."""
    model, _ = load_checkpoint(checkpoint_path)
    example = torch.randn(batch_size, 2)
    with torch.no_grad():
        for _ in range(warmup):
            model(example)
        timings_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(example)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings_ms)
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup": warmup,
        "median_ms": round(statistics.median(timings_ms), 4),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        "samples_per_second": round(
            batch_size / (statistics.median(timings_ms) / 1000.0), 1
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "project_version": __version__,
        },
        "caveat": "latency is machine-specific; compare implementations on the same host",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="reports/checkpoint.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", default="reports/cpu_benchmark.json")
    args = parser.parse_args(argv)
    result = benchmark(args.checkpoint, args.batch_size, args.repeats)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
