# Inference Performance and Correctness Benchmark

One seeded MLP implemented five ways (float64 reference, pure-Python loops, per-sample NumPy, batched float32, and int8 quantised with int32 accumulation), with a harness that measures median latency, throughput, weight memory and peak allocation, and checks every implementation against the reference within an explicit tolerance.

> **Status:** self-directed portfolio project by Yazan Aljabari. It demonstrates machine-learning engineering practice (implementation, numerical correctness, reproducibility, testing). It is not production software, and any results reported are from the exact commands shown below.

## Contents
| Path | Purpose |
|---|---|
| `infbench/model.py` | Seeded MLP and the five implementations, including symmetric per-channel int8 quantisation |
| `infbench/bench.py` | Timing (warm-up, median of repeats), tracemalloc peak, error vs reference, tolerances |
| `infbench/run.py` | CLI that writes `reports/benchmark.json` and `reports/benchmark.md` |
| `tests/` | Correctness within tolerance, determinism, quantisation error bound, memory ratio, report completeness |

## Run
```bash
pip install -r requirements.txt
python -m pytest -q                                   # or: python tests/run_tests.py
python -m infbench.run --batch 256 --repeats 7 --seed 0
```
The command prints a table and writes `reports/benchmark.md`. **No timings are committed to this repository on purpose**: they depend on the machine. Run it on your own hardware and paste the table here.

## Results
_Run the command above and paste `reports/benchmark.md` here with your machine details._

What to look for: batched float32 should beat per-sample inference; the int8 path should use about a quarter of the weight memory of float32 but stay within its 5% tolerance. In a quick sandbox check the NumPy int8 path was far slower than float32, because NumPy integer matrix multiplication does not use BLAS. Quantisation saves memory here, not time, unless an optimised int8 kernel is used. Confirm this on your own machine.

## Limitations
* NumPy on CPU only. Real speed-ups from quantisation need int8 kernels (for example in PyTorch, ONNX Runtime or hardware libraries), which this project does not use.
* Timing is wall-clock on a possibly busy machine; use several repeats and compare only rows with equal batch size. The pure-Python row uses a smaller batch.
* Error is measured against the float64 reference relative to the output's dynamic range, on random inputs, not on a real task metric.

## How an AI coding agent was used
See [`docs/agent-workflow.md`](docs/agent-workflow.md).
