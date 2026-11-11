# ML Engineering Portfolio

[![Tests](https://github.com/yazanus94-dot/ML-Engineering-Portfolio_proj/actions/workflows/ci.yml/badge.svg)](https://github.com/yazanus94-dot/ML-Engineering-Portfolio_proj/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Three projects around the part of ML engineering that decides whether a model
ever reaches production: **numerical correctness, reproducibility, and
measurement**. Every claim in these projects is checked by tests, and every
benchmark run is seeded and deterministic.

| # | Project | What it shows | Validation |
|---|---------|---------------|------------|
| 1 | [ml-numpy-from-scratch](ml-numpy-from-scratch/) | A neural network (Linear, ReLU, Tanh, LayerNorm, causal self-attention) with **hand-written backpropagation**, stable softmax cross-entropy, SGD/Adam — and **central-difference gradient checking of every layer** | 11 tests incl. gradient checks; deterministic training run (spiral dataset, 97.3% test accuracy, committed `reports/spiral_run.json`) |
| 2 | [ml-numerical-debugging-lab](ml-numerical-debugging-lab/) | **Eight realistic ML bugs** — softmax overflow, log-of-zero, wrong axis reduction, train/test leakage, float16 accumulation, variance cancellation, softplus overflow, silent broadcasting — each with symptom, root cause, fix, and a test that reproduces the bug | 6 tests: every bug reproduced, every fix verified; generated `docs/cases.md` write-up |
| 3 | [ml-inference-benchmark](ml-inference-benchmark/) | One seeded MLP **implemented five ways** (float64 reference, pure-Python, per-sample NumPy, batched float32, int8-quantised with int32 accumulation) with a harness measuring median latency, throughput, weight memory and peak allocation | 6 tests: correctness within explicit tolerance, determinism, quantisation error bound, memory ratio, report completeness |

## The engineering habits

- **Gradient-checked backprop.** Hand-derived gradients are verified against
  finite differences — not assumed correct because training loss goes down.
- **Bugs as test cases.** The debugging lab reproduces each numerical failure
  first, then proves the fix, so the knowledge is executable.
- **Measurement with error bars.** The benchmark reports median-of-repeats
  latency, memory via `tracemalloc`, and validates every implementation
  against the float64 reference within a stated tolerance — no silent drift.
- **Determinism.** Seeded generators everywhere; the training-run report is
  committed and regenerated identically on each run.

## Quick start

```bash
# 1. NN from scratch: gradient checks + deterministic training
cd ml-numpy-from-scratch
python -m pytest -q
python scripts/train_spiral.py

# 2. Numerical debugging lab: reproduce 8 bugs and their fixes
cd ../ml-numerical-debugging-lab
python -m pytest -q
python -m lab.report

# 3. Inference benchmark: five implementations, latency + memory + error
cd ../ml-inference-benchmark
python -m pytest -q
python -m infbench.run --batch 64 --repeats 5
```

CI runs all three suites on Python 3.11 / 3.12 / 3.13.

## Scope

Self-directed portfolio projects by Yazan Aljabari demonstrating ML
engineering practice (implementation, numerical correctness, reproducibility,
testing). They are not production software; reported results come from the
exact commands shown.

## License

MIT, see [LICENSE](LICENSE).
