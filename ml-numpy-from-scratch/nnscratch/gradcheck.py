"""Finite-difference gradient checking (central differences, float64)."""
from __future__ import annotations
import numpy as np


def _rel_err(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check_module(module, x, rng, eps: float = 1e-6, max_checks: int = 40) -> dict:
    """Compare analytic and numerical gradients of L = sum(R * module(x)) for the input and every parameter.

    Returns {name: max relative error}. Float64 and a smooth loss should give errors below ~1e-6.
    """
    y0 = module.forward(x)
    r = rng.standard_normal(y0.shape)

    def loss():
        return float((module.forward(x) * r).sum())

    module.forward(x)
    dx = module.backward(r)
    analytic = {"input": dx, **{n: g.copy() for n, _, g in module.parameters()}}
    targets = {"input": x, **{n: p for n, p, _ in module.parameters()}}
    out = {}
    for name, arr in targets.items():
        flat = arr.reshape(-1)
        idx = rng.choice(flat.size, size=min(max_checks, flat.size), replace=False)
        num, ana = [], []
        for i in idx:
            old = flat[i]
            flat[i] = old + eps; lp = loss()
            flat[i] = old - eps; lm = loss()
            flat[i] = old
            num.append((lp - lm) / (2 * eps))
            ana.append(analytic[name].reshape(-1)[i])
        out[name] = _rel_err(np.array(ana), np.array(num))
    return out
