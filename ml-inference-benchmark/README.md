# Inference Performance and Correctness Benchmark

One seeded MLP implemented five ways (float64 reference, pure-Python loops, per-sample NumPy, batched float32, and int8 quantised with int32 accumulation), with a harness that measures median latency, its spread across repeats, throughput, per-sample cost, weight memory and peak allocation, and checks every implementation against the reference within an explicit tolerance.

> **Status:** self-directed portfolio project by Yazan Aljabari. It demonstrates machine-learning engineering practice (implementation, numerical correctness, reproducibility, testing). It is not production software, and any results reported are from the exact commands shown below.

## Contents
| Path | Purpose |
|---|---|
| `infbench/model.py` | Seeded MLP and the five implementations, including symmetric per-channel int8 quantisation |
| `infbench/bench.py` | Timing (warm-up, median/std/min/max of repeats), tracemalloc peak, error vs reference, tolerances, atomic report writes |
| `infbench/run.py` | CLI that writes `reports/benchmark.json` and `reports/benchmark.md` |
| `tests/` | 11 tests: correctness within tolerance, determinism, quantisation error bound, memory ratio, report completeness, variability fields, tracing preservation, failure-safe writes |

## Run
```bash
pip install -r requirements.txt
python -m pytest -q
python -m infbench.run --batch 256 --repeats 7 --seed 0
```
The command prints a table and writes `reports/benchmark.md` and `reports/benchmark.json`.

## Results

`reports/benchmark.md` and `reports/benchmark.json` are committed from a real
run on the author's machine (Python 3.14.7, NumPy 2.5.4, Windows 11); the
environment is recorded inside both files. Absolute timings are machine-specific
— re-run the command above on your hardware.

Each row reports the median, sample standard deviation, minimum and maximum
across repeats. Read the spread before comparing close rows: a difference smaller
than the run-to-run spread is not evidence of a speed-up.

What to look for: batched float32 should beat per-sample inference; the int8 path should use about a quarter of the weight memory of float32 but stay within its 5% tolerance. In a quick sandbox check the NumPy int8 path was far slower than float32, because NumPy integer matrix multiplication does not use BLAS. Quantisation saves memory here, not time, unless an optimised int8 kernel is used. Confirm this on your own machine.

## Limitations
* NumPy on CPU only. Real speed-ups from quantisation need int8 kernels (for example in PyTorch, ONNX Runtime or hardware libraries), which this project does not use.
* Timing is wall-clock on a possibly busy machine; use several repeats and compare only rows with equal batch size. Spread is reported, not hidden.
* Error is measured against the float64 reference relative to the output's dynamic range, on random inputs, not on a real task metric.

## How an AI coding agent was used
See [`docs/agent-workflow.md`](docs/agent-workflow.md).
