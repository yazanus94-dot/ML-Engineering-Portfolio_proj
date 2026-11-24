"""Edge-case and input-validation tests for nnscratch."""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from nnscratch import (
    Linear,
    LayerNorm,
    SelfAttention,
    Adam,
    SGD,
    softmax_cross_entropy,
    check_module,
)


def test_linear_rejects_wrong_input_dim():
    rng = np.random.default_rng(0)
    layer = Linear(5, 4, rng)
    with pytest.raises(ValueError):
        layer.forward(rng.standard_normal((3, 6)))  # last dim 6 != 5


def test_linear_rejects_non_numeric():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    with pytest.raises(TypeError):
        layer.forward(np.array([["a", "b", "c"]]))


def test_softmax_cross_entropy_rejects_bad_shapes():
    z = np.zeros((4, 3))
    y = np.zeros((4, 1))  # should be 1-D
    with pytest.raises(ValueError):
        softmax_cross_entropy(z, y)

    y2 = np.array([0, 1, 2, 5])  # label 5 out of range for 3 classes
    with pytest.raises(ValueError):
        softmax_cross_entropy(z, y2)


def test_adam_rejects_bad_lr():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    with pytest.raises(ValueError):
        Adam(layer.parameters(), lr=-0.01)


def test_sgd_rejects_bad_momentum():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    with pytest.raises(ValueError):
        SGD(layer.parameters(), momentum=1.5)


def test_sgd_step_decreases_simple_quadratic():
    """SGD should make progress on a simple quadratic loss."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 3))
    w_true = rng.standard_normal((3, 1))
    y = x @ w_true + 0.01 * rng.standard_normal((10, 1))

    layer = Linear(3, 1, rng)
    opt = SGD(layer.parameters(), lr=0.1)
    losses = []
    for _ in range(200):
        pred = layer.forward(x)
        loss = float(((pred - y) ** 2).mean())
        losses.append(loss)
        d = 2.0 * (pred - y) / pred.size
        layer.backward(d)
        opt.step()
    assert losses[-1] < losses[0]


def test_layernorm_rejects_wrong_feature_dim():
    rng = np.random.default_rng(0)
    ln = LayerNorm(4)
    with pytest.raises(ValueError):
        ln.forward(rng.standard_normal((3, 5)))  # last dim 5 != 4


def test_self_attention_rejects_wrong_dim():
    rng = np.random.default_rng(0)
    att = SelfAttention(6, rng)
    with pytest.raises(ValueError):
        att.forward(rng.standard_normal((2, 4, 5)))  # last dim 5 != 6


def test_gradient_check_detects_wrong_gradient():
    """A deliberately wrong layer should fail its gradient check."""
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    orig_backward = layer.backward

    def wrong_backward(dy):
        dx = orig_backward(dy)
        layer.g["W"] *= 2.0  # wrong gradient
        return dx

    layer.backward = wrong_backward
    x = rng.standard_normal((4, 3))
    errs = check_module(layer, x, rng)
    assert max(errs.values()) > 1e-4, errs
