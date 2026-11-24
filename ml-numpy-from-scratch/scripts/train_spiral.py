"""Train an MLP on the synthetic spiral dataset. Deterministic for a given seed.

    python scripts/train_spiral.py --seed 0 --epochs 300
Writes reports/spiral_run.json. The data is SYNTHETIC; this demonstrates the training loop, not real-world performance.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nnscratch import Linear, ReLU, Sequential, Adam, softmax_cross_entropy
from nnscratch.data import make_spirals, train_test_split


def train(seed=0, epochs=300, hidden=64, lr=0.01):
    X, y = make_spirals(seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    rng = np.random.default_rng(seed)
    net = Sequential(
        Linear(2, hidden, rng),
        ReLU(),
        Linear(hidden, hidden, rng),
        ReLU(),
        Linear(hidden, 3, rng),
    )
    opt = Adam(net.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        loss, d = softmax_cross_entropy(net.forward(Xtr), ytr)
        net.backward(d)
        opt.step()
        losses.append(loss)

    def acc(A, b):
        return float((net.forward(A).argmax(1) == b).mean())

    return {
        "seed": seed,
        "epochs": epochs,
        "hidden": hidden,
        "lr": lr,
        "dataset": "synthetic spirals (not real data)",
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "train_accuracy": acc(Xtr, ytr),
        "test_accuracy": acc(Xte, yte),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    res = train(a.seed, a.epochs)
    out = pathlib.Path(__file__).resolve().parents[1] / "reports"
    out.mkdir(exist_ok=True)
    (out / "spiral_run.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
