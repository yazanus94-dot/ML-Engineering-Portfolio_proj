# Architecture

The repository is a monorepo of independent ML engineering projects plus root CI.

- `ml-numpy-from-scratch/` implements a neural network using NumPy, with explicit forward/backward
  caches, hand-derived gradients and finite-difference verification.
- `ml-numerical-debugging-lab/` reproduces eight numerical bugs and verifies their fixes.
- `ml-inference-benchmark/` implements one MLP five ways and measures latency, throughput, memory
  and numerical error.
- `ml-pytorch-training/` demonstrates training, checkpointing, ONNX export and numerical validation.
- `coding-agent-benchmark/` provides a deterministic task harness with golden solutions.
- `archive/` stores immutable zip copies of three projects.

Each project is independently installable and testable; the root CI runs all five suites.
