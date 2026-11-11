"""Layers with explicit forward/backward. All arithmetic is NumPy; shapes are documented per layer.

Every layer exposes:
  forward(x) -> y        (caches what backward needs)
  backward(dy) -> dx     (fills self.g[name] with parameter gradients)
  p, g                   dicts of parameters and gradients (same keys)
"""
from __future__ import annotations
import numpy as np


class Module:
    def __init__(self):
        self.p: dict[str, np.ndarray] = {}
        self.g: dict[str, np.ndarray] = {}

    def parameters(self):
        return [(n, self.p[n], self.g[n]) for n in self.p]


class Linear(Module):
    """y = x @ W + b.  x: (..., in_features) -> (..., out_features). He-normal init."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, dtype=np.float64):
        super().__init__()
        self.p["W"] = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype)
        self.p["b"] = np.zeros(n_out, dtype=dtype)
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}

    def forward(self, x):
        self._shape = x.shape
        self._x = x.reshape(-1, x.shape[-1])
        return (self._x @ self.p["W"] + self.p["b"]).reshape(*x.shape[:-1], -1)

    def backward(self, dy):
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.g["W"][...] = self._x.T @ dy2
        self.g["b"][...] = dy2.sum(axis=0)
        return (dy2 @ self.p["W"].T).reshape(self._shape)


class ReLU(Module):
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy):
        return dy * self._mask


class Tanh(Module):
    def forward(self, x):
        self._y = np.tanh(x)
        return self._y

    def backward(self, dy):
        return dy * (1.0 - self._y ** 2)


class LayerNorm(Module):
    """Normalise over the last axis, then scale and shift: y = gamma * (x - mu)/sqrt(var + eps) + beta."""

    def __init__(self, n: int, eps: float = 1e-5, dtype=np.float64):
        super().__init__()
        self.eps = eps
        self.p["gamma"] = np.ones(n, dtype=dtype)
        self.p["beta"] = np.zeros(n, dtype=dtype)
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xhat = (x - mu) * self._inv
        return self.p["gamma"] * self._xhat + self.p["beta"]

    def backward(self, dy):
        n = dy.shape[-1]
        axes = tuple(range(dy.ndim - 1))
        self.g["gamma"][...] = (dy * self._xhat).sum(axis=axes)
        self.g["beta"][...] = dy.sum(axis=axes)
        dxhat = dy * self.p["gamma"]
        return (self._inv / n) * (n * dxhat - dxhat.sum(-1, keepdims=True)
                                  - self._xhat * (dxhat * self._xhat).sum(-1, keepdims=True))


class SelfAttention(Module):
    """Single-head scaled dot-product self-attention. x: (B, T, D) -> (B, T, D).

    S = Q K^T / sqrt(D);  A = softmax(S);  out = (A V) Wo.   `causal=True` masks future positions.
    """

    def __init__(self, d: int, rng: np.random.Generator, causal: bool = False, dtype=np.float64):
        super().__init__()
        self.d, self.causal = d, causal
        for n in ("Wq", "Wk", "Wv", "Wo"):
            self.p[n] = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(dtype)
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}

    def forward(self, x):
        self._x = x
        self._q, self._k, self._v = x @ self.p["Wq"], x @ self.p["Wk"], x @ self.p["Wv"]
        s = self._q @ self._k.transpose(0, 2, 1) / np.sqrt(self.d)
        if self.causal:
            t = x.shape[1]
            s = np.where(np.tril(np.ones((t, t), dtype=bool)), s, -np.inf)
        s = s - s.max(axis=-1, keepdims=True)           # stable softmax
        e = np.exp(s)
        self._a = e / e.sum(axis=-1, keepdims=True)
        self._o = self._a @ self._v
        return self._o @ self.p["Wo"]

    def backward(self, dy):
        x, q, k, v, a, o = self._x, self._q, self._k, self._v, self._a, self._o
        self.g["Wo"][...] = np.einsum("btd,bte->de", o, dy)
        do = dy @ self.p["Wo"].T
        da = do @ v.transpose(0, 2, 1)
        dv = a.transpose(0, 2, 1) @ do
        ds = a * (da - (da * a).sum(axis=-1, keepdims=True)) / np.sqrt(self.d)   # masked entries have a == 0
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
