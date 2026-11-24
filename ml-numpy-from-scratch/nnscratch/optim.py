"""SGD with momentum and Adam optimizers."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np


Params = Iterable[Tuple[str, np.ndarray, np.ndarray]]


class SGD:
    def __init__(self, params: Params, lr: float = 0.1, momentum: float = 0.9):
        self.params: List[Tuple[str, np.ndarray, np.ndarray]] = list(params)
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got lr={lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got momentum={momentum}")
        self.lr, self.m = lr, momentum
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self) -> None:
        for v, (_, p, g) in zip(self.v, self.params):
            v *= self.m
            v -= self.lr * g
            p += v


class Adam:
    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got lr={lr}")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"b1 must be in [0, 1), got b1={b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"b2 must be in [0, 1), got b2={b2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got eps={eps}")
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for _, p, _ in self.params]
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        for m, v, (_, p, g) in zip(self.m, self.v, self.params):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
