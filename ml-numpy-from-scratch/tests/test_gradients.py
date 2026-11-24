import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nnscratch import (
    Linear,
    ReLU,
    Tanh,
    LayerNorm,
    SelfAttention,
    Sequential,
    check_module,
)

TOL = 1e-6


def _check(module, x, seed=0):
    errs = check_module(module, x, np.random.default_rng(seed))
    assert max(errs.values()) < TOL, errs


def test_linear_2d_and_3d():
    rng = np.random.default_rng(1)
    _check(Linear(5, 4, rng), rng.standard_normal((6, 5)))
    _check(Linear(5, 4, rng), rng.standard_normal((2, 3, 5)))


def test_tanh():
    rng = np.random.default_rng(2)
    _check(Tanh(), rng.standard_normal((4, 5)))


def test_relu_away_from_kink():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 5))
    x[np.abs(x) < 0.05] = 0.5  # finite differences are invalid at the kink
    _check(ReLU(), x)


def test_layernorm():
    rng = np.random.default_rng(4)
    ln = LayerNorm(6)
    ln.p["gamma"][:] = rng.standard_normal(6)
    ln.p["beta"][:] = rng.standard_normal(6)
    _check(ln, rng.standard_normal((3, 6)))
    _check(ln, rng.standard_normal((2, 4, 6)))


def test_attention_full_and_causal():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 4, 6))
    _check(SelfAttention(6, rng), x)
    _check(SelfAttention(6, rng, causal=True), x)


def test_causal_attention_ignores_future():
    rng = np.random.default_rng(6)
    att = SelfAttention(6, rng, causal=True)
    x = rng.standard_normal((1, 5, 6))
    y1 = att.forward(x).copy()
    x2 = x.copy()
    x2[:, 3:] += 10.0  # change only future positions
    y2 = att.forward(x2)
    assert np.allclose(y1[:, :3], y2[:, :3])


def test_sequential_stack():
    rng = np.random.default_rng(7)
    net = Sequential(Linear(4, 8, rng), Tanh(), LayerNorm(8), Linear(8, 3, rng))
    _check(net, rng.standard_normal((5, 4)))
