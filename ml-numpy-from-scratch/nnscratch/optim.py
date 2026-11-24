"""SGD with momentum and Adam optimizers."""

from __future__ import annotations

import numbers
from typing import Iterable, List, Tuple

import numpy as np

Params = Iterable[Tuple[str, np.ndarray, np.ndarray]]


def _finite_positive(value, name: str) -> float:
    if not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def _validate_params(params: Params) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    try:
        items = list(params)
    except TypeError as exc:
        raise TypeError(
            "params must be an iterable of (name, parameter, gradient)"
        ) from exc
    if not items:
        raise ValueError("params must contain at least one parameter")
    checked: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for index, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(
                f"parameter {index} must be a (name, parameter, gradient) triple"
            )
        name, p, g = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"parameter {index} must have a non-empty string name")
        if not isinstance(p, np.ndarray) or not isinstance(g, np.ndarray):
            raise TypeError(f"parameter {name!r} and its gradient must be NumPy arrays")
        if p.shape != g.shape:
            raise ValueError(
                f"parameter {name!r} shape {p.shape} does not match gradient shape {g.shape}"
            )
        if not np.issubdtype(p.dtype, np.number) or np.issubdtype(
            p.dtype, np.complexfloating
        ):
            raise TypeError(f"parameter {name!r} must have a real numeric dtype")
        if not np.issubdtype(g.dtype, np.number) or np.issubdtype(
            g.dtype, np.complexfloating
        ):
            raise TypeError(f"gradient {name!r} must have a real numeric dtype")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
            raise ValueError(f"parameter {name!r} and its gradient must be finite")
        checked.append((name, p, g))
    return checked


class SGD:
    def __init__(self, params: Params, lr: float = 0.1, momentum: float = 0.9):
        self.params = _validate_params(params)
        if not isinstance(momentum, numbers.Real) or not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got momentum={momentum!r}")
        self.lr = _finite_positive(lr, "learning rate")
        self.m = float(momentum)
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self) -> None:
        for v, (_, p, g) in zip(self.v, self.params):
            if p.shape != g.shape:
                raise ValueError("parameter and gradient shapes diverged")
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
        self.params = _validate_params(params)
        self.lr = _finite_positive(lr, "learning rate")
        if not isinstance(b1, numbers.Real) or not 0.0 <= b1 < 1.0:
            raise ValueError(f"b1 must be in [0, 1), got b1={b1!r}")
        if not isinstance(b2, numbers.Real) or not 0.0 <= b2 < 1.0:
            raise ValueError(f"b2 must be in [0, 1), got b2={b2!r}")
        self.eps = _finite_positive(eps, "eps")
        self.b1, self.b2, self.t = float(b1), float(b2), 0
        self.m = [np.zeros_like(p) for _, p, _ in self.params]
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self) -> None:
        self.t += 1
        for m, v, (_, p, g) in zip(self.m, self.v, self.params):
            if p.shape != g.shape:
                raise ValueError("parameter and gradient shapes diverged")
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
