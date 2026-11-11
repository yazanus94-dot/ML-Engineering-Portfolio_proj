# Neural Network From Scratch (NumPy) with Gradient Checking

Linear, ReLU, Tanh, LayerNorm and single-head self-attention (optionally causal) with hand-written backpropagation, a numerically stable softmax cross-entropy, SGD and Adam. Every layer's gradient is verified against finite differences, and training is deterministic.

> **Status:** self-directed portfolio project by Yazan Aljabari. It demonstrates machine-learning engineering practice (implementation, numerical correctness, reproducibility, testing). It is not production software, and any results reported are from the exact commands shown below.

## Contents
| Path | Purpose |
|---|---|
| `nnscratch/layers.py` | Layers with explicit forward and backward passes |
| `nnscratch/losses.py` | Stable softmax cross-entropy (log-sum-exp) |
| `nnscratch/optim.py` | SGD with momentum, Adam |
| `nnscratch/gradcheck.py` | Central-difference gradient checker |
| `scripts/train_spiral.py` | Reproducible training run on a synthetic dataset |
| `tests/` | 11 tests: gradient checks per layer, causal-mask test, loss stability, determinism, learning |

## Run
```bash
pip install -r requirements.txt
python -m pytest -q                      # or: python tests/run_tests.py
python scripts/train_spiral.py --seed 0  # writes reports/spiral_run.json
```

## Result of the included run (synthetic data)
`python scripts/train_spiral.py --seed 0` (MLP 2-64-64-3, Adam, 300 epochs): loss 1.128 to 0.014, train accuracy 99.3%, test accuracy 97.3% on a synthetic noisy three-arm spiral set. This only shows that the training loop and backpropagation work; it says nothing about real-world data or large models.

## Limitations
* NumPy only, float64, CPU. No GPU, no autograd, no convolutions. It is for understanding and verification, not production training.
* Gradient checks sample up to 40 entries per tensor rather than every entry.
* The attention layer is single-head and has no dropout or positional encoding.

## How an AI coding agent was used
See [`docs/agent-workflow.md`](docs/agent-workflow.md).
