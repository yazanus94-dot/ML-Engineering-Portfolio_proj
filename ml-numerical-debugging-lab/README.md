# Numerical Debugging Lab for ML Code

Eight realistic machine-learning bugs (softmax overflow, log of zero, wrong axis, train/test leakage, float16 accumulation, variance cancellation, softplus overflow, silent broadcasting). Each case has the symptom, root cause, fix, and a test that reproduces the bug and proves the fix.

> **Status:** self-directed portfolio project by Yazan Aljabari. It demonstrates machine-learning engineering practice (implementation, numerical correctness, reproducibility, testing). It is not production software, and any results reported are from the exact commands shown below.

## Contents
| Path | Purpose |
|---|---|
| `lab/cases.py` | The eight cases: buggy and fixed implementations, demo input, pass/fail verdict |
| `lab/report.py` | Prints a table and generates `docs/cases.md` |
| `docs/cases.md` | Generated write-up: symptom, cause, fix, property, buggy vs fixed output |
| `docs/debugging-playbook.md` | General checklist for numerical and system-level ML failures |
| `tests/` | 11 tests: every bug is reproduced and every fix verified, including the C03 mean/std property and the C04 train-only scaling contract |

## Run
```bash
pip install -r requirements.txt
python -m pytest -q                      # or: python tests/run_tests.py
python -m lab.report                     # table + regenerates docs/cases.md
```

## Output
`python -m lab.report` shows, for example, softmax of `[1000, 1001, 1002]` returning NaN in the buggy version and `[0.0900, 0.2447, 0.6652]` in the fixed one; a float16 sum of 10,000 x 0.1 stalling at 256; and the variance of data near 1e9 returned as 256 instead of 0.02. All values are deterministic and listed in `docs/cases.md`.

## Limitations
* The cases are small, deterministic reproductions of bug classes, not bugs taken from a specific production system.
* Float behaviour is NumPy/IEEE-754 on CPU; GPU kernels and mixed-precision frameworks add further failure modes that are not covered.

## How an AI coding agent was used
See [`docs/agent-workflow.md`](docs/agent-workflow.md).
