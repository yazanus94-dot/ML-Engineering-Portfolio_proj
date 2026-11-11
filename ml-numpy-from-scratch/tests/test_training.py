import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from nnscratch import softmax_cross_entropy
from train_spiral import train


def test_loss_finite_for_huge_logits():
    loss, d = softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([1]))
    assert np.isfinite(loss) and abs(loss - 1000.0) < 1e-9 and np.all(np.isfinite(d))


def test_loss_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    z, y = rng.standard_normal((4, 3)), np.array([0, 2, 1, 2])
    _, d = softmax_cross_entropy(z, y)
    num = np.zeros_like(z)
    for i in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy(); zp[i] += 1e-6; zm[i] -= 1e-6
        num[i] = (softmax_cross_entropy(zp, y)[0] - softmax_cross_entropy(zm, y)[0]) / 2e-6
    assert np.allclose(d, num, atol=1e-7)


def test_training_is_deterministic():
    assert train(seed=0, epochs=30) == train(seed=0, epochs=30)


def test_training_learns():
    r = train(seed=0, epochs=300)
    assert r["final_loss"] < 0.5 * r["first_loss"]
    assert r["test_accuracy"] > 0.8        # chance is 0.33
