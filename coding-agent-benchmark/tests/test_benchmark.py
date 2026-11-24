"""Self-checks for the benchmark harness (golden solutions only)."""

from __future__ import annotations

from bench.run import build_report, run_task
from bench.tasks import TASKS


def test_task_inventory():
    assert len(TASKS) == 24
    assert len({task.id for task in TASKS}) == 24
    for task in TASKS:
        assert task.prompt and task.golden and task.tests
        assert task.dimensions


def test_golden_solutions_pass_subset():
    for task_id in ("T01", "T13", "T24"):
        result = run_task(next(task for task in TASKS if task.id == task_id))
        assert result["passed"] is True, result


def test_report_builder_counts_dimensions():
    results = [run_task(TASKS[0]), run_task(TASKS[1])]
    report = build_report(results)
    assert report["summary"]["tasks"] == 2
    assert report["summary"]["passed"] == 2
    assert report["human_review"]["status"] == "pending"
    assert "not agent performance scores" in report["human_review"]["note"]
