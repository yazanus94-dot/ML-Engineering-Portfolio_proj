"""Seeded synthetic two-spirals dataset — no external data download required."""

from __future__ import annotations

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Make torch/NumPy RNGs deterministic for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_spirals(n_per_class: int = 250, seed: int = 0, noise: float = 0.18):
    """Return ``(X, y)`` tensors of shape ``(2*n, 2)`` and ``(2*n,)``."""
    rng = np.random.default_rng(seed)
    xs = np.zeros((2 * n_per_class, 2), dtype=np.float32)
    ys = np.zeros(2 * n_per_class, dtype=np.int64)
    for class_index in range(2):
        ix = slice(class_index * n_per_class, (class_index + 1) * n_per_class)
        t = np.linspace(0, 1, n_per_class)
        r = 0.25 + 2.5 * t
        theta = 1.5 * np.pi * t + class_index * np.pi
        xs[ix] = np.column_stack(
            [
                r * np.sin(theta) + rng.normal(0, noise, n_per_class),
                r * np.cos(theta) + rng.normal(0, noise, n_per_class),
            ]
        )
        ys[ix] = class_index
    return torch.from_numpy(xs), torch.from_numpy(ys)
