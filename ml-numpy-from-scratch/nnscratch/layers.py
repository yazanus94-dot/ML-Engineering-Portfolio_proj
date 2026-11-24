"""Layers with explicit forward/backward. All arithmetic is NumPy; shapes are documented per layer.

Every layer exposes:
  forward(x) -> y        (caches what backward needs)
  backward(dy) -> dx     (fills self.g[name] with parameter gradients)
  p, g                   dicts of parameters and gradients (same keys)
"""

from __future__ import annotations

import numbers

import numpy as np


def _positive_int(value, name: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _numeric_dtype(dtype):
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise TypeError(f"dtype must be a NumPy numeric dtype, got {dtype!r}") from exc
    if not np.issubdtype(resolved, np.number) or np.issubdtype(
        resolved, np.complexfloating
    ):
        raise TypeError(f"dtype must be real numeric, got {resolved}")
    return resolved


def _numeric_array(value, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy raises the useful message
        raise TypeError(f"{name} must be array-like") from exc
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(
        arr.dtype, np.complexfloating
    ):
        raise TypeError(
            f"{name} must contain real numeric values, got dtype {arr.dtype}"
        )
    return arr


class Module:
    def __init__(self):
        self.p: dict[str, np.ndarray] = {}
        self.g: dict[str, np.ndarray] = {}

    def parameters(self):
        return [(n, self.p[n], self.g[n]) for n in self.p]


class Linear(Module):
    """y = x @ W + b.  x: (..., in_features) -> (..., out_features). He-normal init."""

    def __init__(
        self, n_in: int, n_out: int, rng: np.random.Generator, dtype=np.float64
    ):
        super().__init__()
        self.n_in = _positive_int(n_in, "n_in")
        self.n_out = _positive_int(n_out, "n_out")
        resolved = _numeric_dtype(dtype)
        self.p["W"] = (
            rng.standard_normal((self.n_in, self.n_out)) * np.sqrt(2.0 / self.n_in)
        ).astype(resolved)
        self.p["b"] = np.zeros(self.n_out, dtype=resolved)
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._check_shapes()

    def _check_shapes(self) -> None:
        if self.p["W"].shape != (self.n_in, self.n_out) or self.p["b"].shape != (
            self.n_out,
        ):
            raise ValueError("Linear parameter shapes are invalid")

    def forward(self, x):
        self._check_shapes()
        x = _numeric_array(x, "x")
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.n_in:
            raise ValueError(f"x last dimension must be {self.n_in}, got {x.shape[-1]}")
        self._shape = x.shape
        self._x = x.reshape(-1, x.shape[-1])
        return (self._x @ self.p["W"] + self.p["b"]).reshape(*x.shape[:-1], -1)

    def backward(self, dy):
        if not hasattr(self, "_x"):
            raise RuntimeError("Linear.backward called before forward")
        dy = _numeric_array(dy, "dy")
        expected = (*self._shape[:-1], self.n_out)
        if dy.shape != expected:
            raise ValueError(f"dy must have shape {expected}, got {dy.shape}")
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.g["W"][...] = self._x.T @ dy2
        self.g["b"][...] = dy2.sum(axis=0)
        return (dy2 @ self.p["W"].T).reshape(self._shape)


class ReLU(Module):
    def forward(self, x):
        x = _numeric_array(x, "x")
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy):
        if not hasattr(self, "_mask"):
            raise RuntimeError("ReLU.backward called before forward")
        dy = _numeric_array(dy, "dy")
        if dy.shape != self._mask.shape:
            raise ValueError(f"dy must have shape {self._mask.shape}, got {dy.shape}")
        return dy * self._mask


class Tanh(Module):
    def forward(self, x):
        x = _numeric_array(x, "x")
        self._y = np.tanh(x)
        return self._y

    def backward(self, dy):
        if not hasattr(self, "_y"):
            raise RuntimeError("Tanh.backward called before forward")
        dy = _numeric_array(dy, "dy")
        if dy.shape != self._y.shape:
            raise ValueError(f"dy must have shape {self._y.shape}, got {dy.shape}")
        return dy * (1.0 - self._y**2)


class LayerNorm(Module):
    """Normalise over the last axis, then scale and shift: y = gamma * (x - mu)/sqrt(var + eps) + beta."""

    def __init__(self, n: int, eps: float = 1e-5, dtype=np.float64):
        super().__init__()
        self.n = _positive_int(n, "n")
        if not isinstance(eps, numbers.Real) or eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        self.eps = float(eps)
        resolved = _numeric_dtype(dtype)
        self.p["gamma"] = np.ones(self.n, dtype=resolved)
        self.p["beta"] = np.zeros(self.n, dtype=resolved)
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._check_shapes()

    def _check_shapes(self) -> None:
        if self.p["gamma"].shape != (self.n,) or self.p["beta"].shape != (self.n,):
            raise ValueError("LayerNorm parameter shapes are invalid")

    def forward(self, x):
        self._check_shapes()
        x = _numeric_array(x, "x")
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.n:
            raise ValueError(f"x last dimension must be {self.n}, got {x.shape[-1]}")
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xhat = (x - mu) * self._inv
        return self.p["gamma"] * self._xhat + self.p["beta"]

    def backward(self, dy):
        if not hasattr(self, "_xhat"):
            raise RuntimeError("LayerNorm.backward called before forward")
        dy = _numeric_array(dy, "dy")
        if dy.shape != self._xhat.shape:
            raise ValueError(f"dy must have shape {self._xhat.shape}, got {dy.shape}")
        n = dy.shape[-1]
        axes = tuple(range(dy.ndim - 1))
        self.g["gamma"][...] = (dy * self._xhat).sum(axis=axes)
        self.g["beta"][...] = dy.sum(axis=axes)
        dxhat = dy * self.p["gamma"]
        return (self._inv / n) * (
            n * dxhat
            - dxhat.sum(-1, keepdims=True)
            - self._xhat * (dxhat * self._xhat).sum(-1, keepdims=True)
        )


class SelfAttention(Module):
    """Single-head scaled dot-product self-attention. x: (B, T, D) -> (B, T, D).

    S = Q K^T / sqrt(D);  A = softmax(S);  out = (A V) Wo.   `causal=True` masks future positions.
    """

    def __init__(
        self, d: int, rng: np.random.Generator, causal: bool = False, dtype=np.float64
    ):
        super().__init__()
        self.d = _positive_int(d, "d")
        self.causal = bool(causal)
        resolved = _numeric_dtype(dtype)
        for n in ("Wq", "Wk", "Wv", "Wo"):
            self.p[n] = (
                rng.standard_normal((self.d, self.d)) / np.sqrt(self.d)
            ).astype(resolved)
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._check_shapes()

    def _check_shapes(self) -> None:
        for name in ("Wq", "Wk", "Wv", "Wo"):
            if self.p[name].shape != (self.d, self.d):
                raise ValueError(f"SelfAttention parameter {name} has an invalid shape")

    def forward(self, x):
        self._check_shapes()
        x = _numeric_array(x, "x")
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, D), got {x.shape}")
        if x.shape[-1] != self.d:
            raise ValueError(f"x last dimension must be {self.d}, got {x.shape[-1]}")
        self._x = x
        self._q, self._k, self._v = x @ self.p["Wq"], x @ self.p["Wk"], x @ self.p["Wv"]
        s = self._q @ self._k.transpose(0, 2, 1) / np.sqrt(self.d)
        if self.causal:
            t = x.shape[1]
            s = np.where(np.tril(np.ones((t, t), dtype=bool)), s, -np.inf)
        s = s - s.max(axis=-1, keepdims=True)  # stable softmax
        e = np.exp(s)
        self._a = e / e.sum(axis=-1, keepdims=True)
        self._o = self._a @ self._v
        return self._o @ self.p["Wo"]

    def backward(self, dy):
        if not hasattr(self, "_x"):
            raise RuntimeError("SelfAttention.backward called before forward")
        dy = _numeric_array(dy, "dy")
        if dy.shape != self._x.shape:
            raise ValueError(f"dy must have shape {self._x.shape}, got {dy.shape}")
        x, q, k, v, a, o = self._x, self._q, self._k, self._v, self._a, self._o
        self.g["Wo"][...] = np.einsum("btd,bte->de", o, dy)
        do = dy @ self.p["Wo"].T
        da = do @ v.transpose(0, 2, 1)
        dv = a.transpose(0, 2, 1) @ do
        ds = (
            a * (da - (da * a).sum(axis=-1, keepdims=True)) / np.sqrt(self.d)
        )  # masked entries have a == 0
        dq = ds @ k
        dk = ds.transpose(0, 2, 1) @ q
        self.g["Wq"][...] = np.einsum("btd,bte->de", x, dq)
        self.g["Wk"][...] = np.einsum("btd,bte->de", x, dk)
        self.g["Wv"][...] = np.einsum("btd,bte->de", x, dv)
        return dq @ self.p["Wq"].T + dk @ self.p["Wk"].T + dv @ self.p["Wv"].T


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def parameters(self):
        out = []
        for i, layer in enumerate(self.layers):
            out += [(f"{i}.{n}", p, g) for n, p, g in layer.parameters()]
        return out
