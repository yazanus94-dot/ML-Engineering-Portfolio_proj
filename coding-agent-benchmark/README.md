# Coding-agent benchmark (24 tasks)

A reproducible, self-contained suite of small Python engineering tasks designed
to exercise the failure modes that matter when reviewing agent-generated code:
correctness, edge cases, numerical stability, performance, regression risk and
test coverage.

## What it actually measures

**This suite currently validates the benchmark harness against golden
solutions only.** It does **not** report any coding agent's performance.
`reports/evaluation_results.json` records `human_review.status = "pending"`
because no human-reviewed agent solutions exist here. Do not read 24/24 as an
agent score.

## Dimensions

| Dimension | Tasks |
|---|---|
| correctness | 23 |
| edge_cases | 8 |
| numerical_stability | 5 |
| performance | 4 |
| regression_risk | 5 |
| test_coverage | 2 |

Categories: algorithms (7), numerical (8), strings (3), data (4), robustness (1),
ml-features (1).

## Layout

```
bench/tasks.py          # 24 Task objects: prompt, golden source, pytest source
bench/run.py            # isolated temp-dir runner + report builder
reports/evaluation_results.json   # generated, automated, human review pending
tests/test_benchmark.py # metadata + runner self-checks
```

## Commands

```bash
python -m pytest -q                       # benchmark self-tests
python -m bench.run                       # execute all 24 golden tasks
python -m bench.run --ids T01 T13 T24     # quick subset
```

Each task writes `solution.py` and `test_solution.py` into a fresh temporary
directory and runs `python -m pytest -q` there. Timings include process startup
and are machine-specific.

## Adding a task

Add a `Task` to `bench/tasks.py` with a unique id, a prompt, a self-contained
golden implementation and pytest tests that import `from solution import ...`.
Add at least one dimension tag. Re-run `python -m bench.run` and commit the
regenerated report only if the result is genuinely produced by the code.

## Limitations

- Synthetic toy tasks, not real ML engineering work.
- Golden solutions are not agent submissions.
- No human-reviewed agent evaluation has been performed.
- The suite cannot prove an agent avoided leakage or wrote good architecture;
  it is a narrow automated signal.
