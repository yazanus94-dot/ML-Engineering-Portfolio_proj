# Architecture

The repository is a monorepo of independent ML engineering projects, a small
experiment registry, and root CI.

- `ml-numpy-from-scratch/` implements a neural network using NumPy, with explicit
  forward/backward caches, hand-derived gradients and finite-difference verification.
- `ml-numerical-debugging-lab/` reproduces eight numerical bugs and verifies their fixes.
- `ml-inference-benchmark/` implements one MLP five ways and measures latency (median and
  spread), throughput, memory and numerical error.
- `ml-pytorch-training/` demonstrates training, checkpointing, ONNX export and numerical validation.
- `coding-agent-benchmark/` provides a deterministic task harness with golden solutions.
- `applied-biomedical-ml/` classifies WDBC malignancy with a checksum-pinned public
  dataset, train-only preprocessing and a from-scratch logistic-regression baseline.
- `experiments/` holds validated JSON configs and a deterministic runner that records
  metrics, environment and command for a run.
- `archive/` stores immutable zip copies of three projects.

## Data flow

```text
raw public data (WDBC, checksum) ──► strict loader ──► stratified split
                                                       │
                                       train-only standardiser
                                                       │
                                             logistic regression
                                                       │
                                     metrics + Wilson intervals
                                                       │
                                       reports/metrics.json + .md

experiments/configs/*.json ──► experiment_runner ──► experiments/results/*.json
```

## Boundaries

- Each project is independently installable and testable; the root CI runs all six
  suites plus the experiment runner on Python 3.11, 3.12 and 3.13.
- No project imports another project's package at runtime. The experiment runner
  deliberately reuses the NumPy training entry point and says so in its docstring.
- The applied project commits its raw dataset because it is small (124 KB), public
  and CC BY 4.0; the checksum makes silent substitution visible.
