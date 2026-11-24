"""A seeded MLP and four inference implementations of the same function.

reference   float64 NumPy, batched. Ground truth for correctness checks.
loops       pure-Python loops (no NumPy in the hot path). Deliberately naive.
per_sample  NumPy, one sample at a time (a common serving mistake).
batched32   NumPy, whole batch, float32.
int8        weights and activations quantised to int8 (symmetric, per-output-channel weight scales,
            per-tensor dynamic activation scale), int32 accumulation, float32 rescale.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MLP:
    weights: list  # float64 (in, out)
    biases: list  # float64 (out,)

    @classmethod
    def random(cls, sizes, seed=0):
        rng = np.random.default_rng(seed)
        W = [
            rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        B = [rng.standard_normal(b) * 0.1 for b in sizes[1:]]
        return cls(W, B)

    @property
    def sizes(self):
        return [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]


def forward_reference(m: MLP, x):
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(zip(m.weights, m.biases)):
        h = h @ W + b
        if i < len(m.weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def forward_loops(m: MLP, x):
    """Pure-Python implementation (lists and loops)."""
    Ws = [w.tolist() for w in m.weights]
    Bs = [b.tolist() for b in m.biases]
    out = []
    for row in np.asarray(x, dtype=np.float64).tolist():
        h = row
        for li, (W, b) in enumerate(zip(Ws, Bs)):
            n_out = len(b)
            nxt = list(b)
            for i, hi in enumerate(h):
                if hi != 0.0:
                    Wi = W[i]
                    for j in range(n_out):
                        nxt[j] += hi * Wi[j]
            h = [v if v > 0.0 else 0.0 for v in nxt] if li < len(Ws) - 1 else nxt
        out.append(h)
    return np.array(out)


def prepare_per_sample(m: MLP):
    """Prepare float32 weights and biases once for per-sample inference."""
    return [w.astype(np.float32) for w in m.weights], [
        b.astype(np.float32) for b in m.biases
    ]


def forward_per_sample(prep, x):
    """Per-sample inference using a prepared float32 representation."""
    W32, B32 = prep
    out = []
    for row in np.asarray(x, dtype=np.float32):
        h = row
        for i, (W, b) in enumerate(zip(W32, B32)):
            h = h @ W + b
            if i < len(W32) - 1:
                h = np.maximum(h, 0.0)
        out.append(h)
    return np.array(out)


def prepare_batched32(m: MLP):
    return [w.astype(np.float32) for w in m.weights], [
        b.astype(np.float32) for b in m.biases
    ]


def forward_batched32(prep, x):
    W, B = prep
    h = np.asarray(x, dtype=np.float32)
    for i, (w, b) in enumerate(zip(W, B)):
        h = h @ w + b
        if i < len(W) - 1:
            h = np.maximum(h, 0.0)
    return h


def quantise_weights(W):
    """Symmetric int8, one scale per output channel (column). Returns (Wq int8, scale float32 (out,))."""
    scale = np.abs(W).max(axis=0) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(W / scale), -127, 127).astype(np.int8), scale.astype(
        np.float32
    )


def prepare_int8(m: MLP):
    return [quantise_weights(w) for w in m.weights], [
        b.astype(np.float32) for b in m.biases
    ]


def forward_int8(prep, x):
    qW, B = prep
    h = np.asarray(x, dtype=np.float32)
    for i, ((wq, ws), b) in enumerate(zip(qW, B)):
        sx = max(
            float(np.abs(h).max()) / 127.0, 1e-12
        )  # per-tensor dynamic activation scale
        hq = np.clip(np.round(h / sx), -127, 127).astype(np.int8)
        acc = hq.astype(np.int32) @ wq.astype(np.int32)  # int32 accumulation
        h = acc.astype(np.float32) * (sx * ws) + b
        if i < len(qW) - 1:
            h = np.maximum(h, 0.0)
    return h


def weight_bytes(prep_or_model, kind: str) -> int:
    if kind in ("reference", "loops"):
        return sum(w.nbytes for w in prep_or_model.weights) + sum(
            b.nbytes for b in prep_or_model.biases
        )
    if kind in ("per_sample", "batched32"):
        W, B = prep_or_model
        return sum(w.nbytes for w in W) + sum(b.nbytes for b in B)
    qW, B = prep_or_model
    return sum(w.nbytes + s.nbytes for w, s in qW) + sum(b.nbytes for b in B)
