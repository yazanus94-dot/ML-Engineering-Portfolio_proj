"""Run golden solutions through their tests and write an honest automated report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .tasks import TASKS, Task


def run_task(task: Task, python: str = sys.executable, timeout: int = 120) -> dict:
    """Execute one task's golden solution against its tests in a temp directory."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"{task.id}_") as directory:
        root = Path(directory)
        (root / "solution.py").write_text(task.golden + "\n", encoding="utf-8")
        (root / "test_solution.py").write_text(task.tests + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [python, "-m", "pytest", "-q"], cwd=root,
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            returncode, output = completed.returncode, completed.stdout + completed.stderr
        except subprocess.TimeoutExpired:
            returncode, output = -1, f"TIMEOUT after {timeout}s"
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "id": task.id,
        "title": task.title,
        "category": task.category,
        "dimensions": list(task.dimensions),
        "passed": returncode == 0,
        "returncode": returncode,
        "duration_ms": duration_ms,
        "tail": output.strip().splitlines()[-1] if output.strip() else "",
    }


def build_report(results: list[dict]) -> dict:
    by_dimension: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for result in results:
        for key, bucket in ((dim, by_dimension[dim]) for dim in result["dimensions"]):
            bucket["total"] += 1
            bucket["passed"] += int(result["passed"])
        bucket = by_category[result["category"]]
        bucket["total"] += 1
        bucket["passed"] += int(result["passed"])
    return {
        "benchmark": "coding-agent-benchmark",
        "version": "0.1.0",
        "execution_mode": "golden solutions executed against their own tests (self-check)",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": results,
        "summary": {
            "tasks": len(results),
            "passed": sum(int(item["passed"]) for item in results),
            "failed": sum(int(not item["passed"]) for item in results),
            "by_dimension": dict(sorted(by_dimension.items())),
            "by_category": dict(sorted(by_category.items())),
            "total_duration_ms": round(sum(item["duration_ms"] for item in results), 1),
        },
        "human_review": {
            "status": "pending",
            "note": "No human-reviewed agent solutions exist in this repository. "
                    "These numbers validate the benchmark harness against golden "
                    "solutions only; they are not agent performance scores.",
        },
        "limitations": [
            "Golden-solution self-check is not an evaluation of any coding agent.",
            "Task set is small and synthetic; it does not measure real-world ML engineering work.",
            "Timings are machine-specific and include pytest process startup.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/evaluation_results.json")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional subset of task ids")
    args = parser.parse_args(argv)
    tasks = [task for task in TASKS if not args.ids or task.id in args.ids]
    results = [run_task(task) for task in tasks]
    report = build_report(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['summary']['passed']}/{report['summary']['tasks']} golden tasks passed; "
          f"report: {output}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
