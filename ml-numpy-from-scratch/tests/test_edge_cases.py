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


def test_linear_rejects_invalid_constructor_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        Linear(0, 4, rng)
    with pytest.raises(ValueError):
        Linear(3, -1, rng)
    with pytest.raises(TypeError):
        Linear(3, 2, rng, dtype=object)


def test_linear_rejects_corrupt_parameter_shapes():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    layer.p["W"] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        layer.forward(rng.standard_normal((4, 3)))


def test_linear_backward_rejects_wrong_shape():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    layer.forward(rng.standard_normal((4, 3)))
    with pytest.raises(ValueError):
        layer.backward(rng.standard_normal((4, 3)))


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


def test_softmax_cross_entropy_rejects_float_labels():
    with pytest.raises(ValueError):
        softmax_cross_entropy(np.zeros((2, 3)), np.array([0.0, 1.0]))


def test_softmax_cross_entropy_is_finite_for_extreme_logits():
    loss, grad = softmax_cross_entropy(
        np.array([[-1000.0, 0.0, 1000.0]]), np.array([2])
    )
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


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


def test_layernorm_rejects_invalid_eps():
    with pytest.raises(ValueError):
        LayerNorm(4, eps=0.0)
    with pytest.raises(ValueError):
        LayerNorm(4, eps=-1.0)


def test_self_attention_rejects_wrong_dim():
    rng = np.random.default_rng(0)
    att = SelfAttention(6, rng)
    with pytest.raises(ValueError):
        att.forward(rng.standard_normal((2, 4, 5)))  # last dim 5 != 6


def test_self_attention_rejects_wrong_rank():
    rng = np.random.default_rng(0)
    att = SelfAttention(6, rng)
    with pytest.raises(ValueError):
        att.forward(rng.standard_normal((4, 6)))


def test_optimizers_reject_malformed_parameters():
    with pytest.raises(ValueError):
        SGD([("w", np.zeros(2))])
    with pytest.raises(ValueError):
        SGD([("w", np.zeros(2), np.zeros(3))])
    with pytest.raises(TypeError):
        SGD([("w", [1.0], [0.0])])


def test_sgd_exact_parameter_update():
    p = np.array([1.0])
    g = np.array([0.5])
    opt = SGD([("w", p, g)], lr=0.1, momentum=0.0)
    opt.step()
    assert np.allclose(p, 0.95)


def test_adam_exact_first_step_and_state():
    p = np.array([1.0])
    g = np.array([0.5])
    opt = Adam([("w", p, g)], lr=0.1)
    opt.step()
    expected = 0.9  # first Adam step is approximately -lr * sign(g)
    assert np.allclose(p, expected, atol=1e-6)
    assert opt.t == 1
    assert not np.allclose(opt.m[0], 0.0)
    assert not np.allclose(opt.v[0], 0.0)


def test_adam_converges_on_quadratic():
    p = np.array([1.0, -2.0])
    g = np.zeros_like(p)
    opt = Adam([("w", p, g)], lr=0.1)
    for _ in range(500):
        g[...] = p
        opt.step()
    assert np.max(np.abs(p)) < 1e-3


def test_adam_rejects_invalid_parameters():
    p, g = np.zeros(1), np.zeros(1)
    with pytest.raises(ValueError):
        Adam([("w", p, g)], b1=1.0)
    with pytest.raises(ValueError):
        Adam([("w", p, g)], b2=-0.1)
    with pytest.raises(ValueError):
        Adam([("w", p, g)], eps=0.0)


def test_gradient_check_is_sampled_and_validates_max_checks():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    x = rng.standard_normal((4, 3))
    errs = check_module(layer, x, rng, max_checks=1)
    assert set(errs) == {"input", "W", "b"}
    assert max(errs.values()) < 1e-6
    with pytest.raises(ValueError):
        check_module(layer, x, rng, max_checks=0)


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
