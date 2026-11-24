"""Eight realistic ML numerical/engineering bugs.

Each Case has:
  buggy / fixed   two implementations with the same signature
  demo()          returns (buggy_output, fixed_output)
  verdict(b, f)   True if the BUG is demonstrated (buggy output violates the property) AND the FIX satisfies it
All inputs are small and deterministic, so results are exactly reproducible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


def _quiet_ctx():
    return np.errstate(all="ignore")


@dataclass
class Case:
    id: str
    title: str
    symptom: str
    cause: str
    fix: str
    property: str
    buggy: Callable
    fixed: Callable
    demo: Callable
    verdict: Callable


# 1 ---------------------------------------------------------------- softmax overflow
def softmax_buggy(x):
    with _quiet_ctx():
        e = np.exp(x)
        return e / e.sum()


def softmax_fixed(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# 2 ---------------------------------------------------------------- log of zero probability
def xent_buggy(logits, y):
    p = softmax_fixed(logits)  # even a stable softmax underflows to exactly 0.0
    with _quiet_ctx():
        return float(-np.log(p[y]))


def xent_fixed(logits, y):
    z = logits - logits.max()
    return float(-(z[y] - np.log(np.exp(z).sum())))


# 3 ---------------------------------------------------------------- wrong axis in standardisation
def standardise_buggy(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(
        axis=1, keepdims=True
    )  # per-sample, not per-feature


def standardise_fixed(X):
    """Standardise each feature using its own mean and standard deviation.

    Constant features have zero standard deviation; they are mapped to zero
    rather than producing NaN or infinity.
    """
    X = np.asarray(X)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


# 4 ---------------------------------------------------------------- train/test leakage in scaling
def scale_buggy(train, test):
    allx = np.vstack([train, test])  # statistics see the test set
    mu, sd = allx.mean(0), allx.std(0)
    return (train - mu) / sd


def scale_fixed(train, test):
    """Fit mean/std on training data only and transform both splits.

    Returns ``(train_scaled, test_scaled)``. Zero-variance features are left
    unchanged rather than producing NaNs or infinities.
    """
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


# 5 ---------------------------------------------------------------- low-precision accumulation
def accumulate_buggy(values):
    s = np.float16(0)
    for v in values:
        s = np.float16(
            s + np.float16(v)
        )  # float16 accumulator stalls: 256 + 0.1 == 256
    return float(s)


def accumulate_fixed(values):
    s = np.float32(0)
    for v in values:
        s = np.float32(s + np.float32(v))
    return float(s)


# 6 ---------------------------------------------------------------- catastrophic cancellation in variance
def variance_buggy(x):
    return float(np.mean(x**2) - np.mean(x) ** 2)  # E[x^2] - E[x]^2


def variance_fixed(x):
    m = np.mean(x)
    return float(np.mean((x - m) ** 2))  # two-pass


# 7 ---------------------------------------------------------------- softplus / logistic loss overflow
def softplus_buggy(x):
    with _quiet_ctx():
        return float(np.log(1.0 + np.exp(x)))


def softplus_fixed(x):
    return float(np.logaddexp(0.0, x))


# 8 ---------------------------------------------------------------- silent broadcasting
def mse_buggy(pred, target):
    return float(
        np.mean((pred - target) ** 2)
    )  # pred (N,1) - target (N,) broadcasts to (N,N)


def mse_fixed(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    return float(np.mean((pred - target) ** 2))


# ------------------------------------------------------------------ demos & verdicts
_X = np.array([1000.0, 1001.0, 1002.0])
_LOG = np.array([1000.0, 0.0])
_DATA = np.array(
    [[1.0, 10.0, 100.0], [2.0, 20.0, 300.0], [3.0, 30.0, 200.0], [4.0, 50.0, 500.0]]
)
_TRAIN = np.array([[1.0], [2.0], [3.0]])
_TEST_A, _TEST_B = np.array([[4.0]]), np.array([[400.0]])
_VALS = [0.1] * 10000
_BIG = 1e9 + np.array([0.1, 0.2, 0.3, 0.4, 0.5])
_P, _T = np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0])


def _mse_fixed_raises():
    try:
        mse_fixed(_P, _T)
    except ValueError:
        return "ValueError"
    return "no error"


CASES = [
    Case(
        "C01",
        "Softmax overflow",
        "Probabilities are NaN for large logits.",
        "exp(1000) overflows to inf; inf/inf = NaN.",
        "Subtract the maximum logit before exponentiating (mathematically identical).",
        "Output is finite and sums to 1; softmax([1000,1001,1002]) = [0.0900, 0.2447, 0.6652].",
        softmax_buggy,
        softmax_fixed,
        lambda: (softmax_buggy(_X), softmax_fixed(_X)),
        lambda b, f: (not np.all(np.isfinite(b)))
        and np.allclose(f, [0.09003057, 0.24472847, 0.66524096], atol=1e-7),
    ),
    Case(
        "C02",
        "Cross-entropy of a zero probability",
        "Loss is inf (or the training loss jumps to inf) for a confidently wrong prediction.",
        "The probability underflows to 0.0 and log(0) = -inf.",
        "Compute the loss from logits with log-sum-exp; never take log of a probability.",
        "Loss for logits [1000, 0] and true class 1 is exactly 1000.",
        xent_buggy,
        xent_fixed,
        lambda: (xent_buggy(_LOG, 1), xent_fixed(_LOG, 1)),
        lambda b, f: np.isinf(b) and abs(f - 1000.0) < 1e-9,
    ),
    Case(
        "C03",
        "Standardisation on the wrong axis",
        "Features are not zero-mean; model quality silently drops.",
        "axis=1 normalises each sample across its features instead of each feature across samples.",
        "Use axis=0 for (samples, features) data.",
        "Every feature (column) has mean 0 and std 1.",
        standardise_buggy,
        standardise_fixed,
        lambda: (
            (standardise_buggy(_DATA).mean(0), standardise_fixed(_DATA).mean(0)),
            standardise_fixed(_DATA).std(0),
        ),
        lambda means, stds: (not np.allclose(means[0], 0, atol=1e-9))
        and np.allclose(means[1], 0, atol=1e-9)
        and np.allclose(stds, 1, atol=1e-9),
    ),
    Case(
        "C04",
        "Train/test leakage through scaling",
        "Validation looks better than production; changing the test set changes the training features.",
        "Scaler statistics were computed on train + test together.",
        "Fit the scaler on the training split only and reuse it for the test split.",
        "The transformed training data must not depend on the test data.",
        scale_buggy,
        scale_fixed,
        lambda: (
            np.abs(scale_buggy(_TRAIN, _TEST_A) - scale_buggy(_TRAIN, _TEST_B)).max(),
            np.abs(
                scale_fixed(_TRAIN, _TEST_A)[0] - scale_fixed(_TRAIN, _TEST_B)[0]
            ).max(),
        ),
        lambda b, f: b > 0.1 and f == 0.0,
    ),
    Case(
        "C05",
        "Low-precision accumulation",
        "Summing 10,000 values of 0.1 returns 256 instead of ~1000.",
        "float16 has 11 significant bits; once the sum reaches 256, adding 0.1 is below half the spacing and rounds away.",
        "Accumulate in float32 (or higher), cast at the end if needed.",
        "The sum is within 0.5% of 1000.",
        accumulate_buggy,
        accumulate_fixed,
        lambda: (accumulate_buggy(_VALS), accumulate_fixed(_VALS)),
        lambda b, f: abs(b - 1000) > 100 and abs(f - 1000) / 1000 < 0.005,
    ),
    Case(
        "C06",
        "Catastrophic cancellation in variance",
        "Variance of data near 1e9 comes out as 256 (true value 0.02) or negative.",
        "E[x^2] - E[x]^2 subtracts two nearly equal ~1e18 numbers, losing all significant digits.",
        "Use the two-pass formula (or Welford's algorithm).",
        "Variance of 1e9 + [0.1..0.5] equals 0.02 to 6 significant digits.",
        variance_buggy,
        variance_fixed,
        lambda: (variance_buggy(_BIG), variance_fixed(_BIG)),
        lambda b, f: abs(b - 0.02) > 1.0 and abs(f - 0.02) < 1e-6,
    ),
    Case(
        "C07",
        "Softplus / logistic loss overflow",
        "log(1 + exp(x)) returns inf for x = 1000 (true value is 1000).",
        "exp(x) overflows before the log can compress it.",
        "Use logaddexp(0, x), which is stable.",
        "softplus(1000) = 1000.",
        softplus_buggy,
        softplus_fixed,
        lambda: (softplus_buggy(1000.0), softplus_fixed(1000.0)),
        lambda b, f: np.isinf(b) and f == 1000.0,
    ),
    Case(
        "C08",
        "Silent broadcasting in a loss",
        "MSE of perfect predictions is not 0; no error is raised.",
        "pred (N,1) minus target (N,) broadcasts to an (N,N) matrix.",
        "Validate shapes explicitly and fail loudly.",
        "Perfect predictions give 0; a shape mismatch raises an error instead of returning a number.",
        mse_buggy,
        mse_fixed,
        lambda: (mse_buggy(_P, _T), _mse_fixed_raises()),
        lambda b, f: b > 0.1 and f == "ValueError",
    ),
]


def run_all():
    rows = []
    for c in CASES:
        b, f = c.demo()
        rows.append((c, b, f, bool(c.verdict(b, f))))
    return rows
