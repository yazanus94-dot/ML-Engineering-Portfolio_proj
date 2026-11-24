# Technical audit — ML Engineering Portfolio

## Scope

Audited the six subprojects, the experiment registry, root CI and documentation
without modifying unrelated repositories.

## Test baseline

At audit start: 45 tests across five subprojects. Final verified state:

| Suite | Tests | Result |
|---|---:|---|
| ml-numpy-from-scratch | 33 | pass |
| ml-numerical-debugging-lab | 11 | pass |
| ml-inference-benchmark | 11 | pass |
| ml-pytorch-training | 8 | pass |
| coding-agent-benchmark | 3 | pass |
| applied-biomedical-ml | 18 | pass |
| experiments | 12 | pass |
| **Total** | **96** | pass |

Environment: Python 3.14.7, NumPy 2.5.4, Windows 11. CI targets Python 3.11,
3.12 and 3.13; no feature-branch run exists, so cross-version execution is not
verified.

## Findings and actions

| Component | Problem | Severity | Action taken | Verification |
|---|---|---|---|---|
| `lab/cases.py` C04 | scaler fitted on train + test; test split not transformed | high | fit on train only, transform both splits, guard zero variance | 2 dedicated tests |
| `lab/cases.py` C03 | only the mean was asserted; constant features produced NaN | medium | assert mean and std; unit scale for zero-variance features | 3 dedicated tests |
| `infbench/model.py` | weight conversion ran inside the timed call | medium | hoist preparation via `prepare_per_sample` | unchanged correctness tests |
| `infbench/bench.py` | tracing could stay active after an exception | medium | `try/finally` with ownership check | tracing-preservation test |
| Benchmark report | median only; README claimed error bars | medium | record std/min/max; update README to the real method | schema test |
| Benchmark Markdown | committed file was a traceback | high | atomic report writes; regenerate from a successful run | report test asserts no traceback |
| `nnscratch/layers.py` | dimensions validated implicitly by NumPy | medium | explicit dtype/shape/parameter validation | constructor and backward-shape tests |
| `nnscratch/optim.py` | no parameter validation; no convergence test | low | validate parameters and hyperparameters; add exact-step and convergence tests | 6 optimizer tests |
| Root CI | covered three of five projects; old action majors | medium | seven jobs, Python 3.11/3.12/3.13, `checkout@v7`/`setup-python@v7` | workflow inspection |
| Applied research project | absent; no verified dataset | high | add WDBC project with CC BY 4.0 licence, checksum and 18 tests | dataset, leakage and end-to-end tests |
| Experiment management | absent | medium | add validated config registry and deterministic runner | 12 tests |
| Root `LICENSE` | missing while README linked to it | low | add MIT licence; note CC BY 4.0 for the dataset | file exists, README link resolves |
| Root/sub READMEs | stale test counts and claims | low | align with actual counts and implemented methods | manual cross-check |
| `coding-agent-benchmark` | golden self-check is not an agent score | low | keep human review pending | README states this |

## Security review

- No secrets, credentials or private patient data are tracked. The supplied
  GitHub token was never used, printed or committed; it should be rotated.
- The committed WDBC file is public, de-identified research data (CC BY 4.0).
- `archive/*.zip` files remain immutable and are not installed by CI.
- Report writers use atomic replaces, so a failed run cannot commit a partial
  report.
