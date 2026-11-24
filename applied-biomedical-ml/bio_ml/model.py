"""Train-only feature scaling and a from-scratch logistic-regression model."""

from __future__ import annotations

import numbers

import numpy as np


def _numeric_matrix(X, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return X


class Standardizer:
    """Per-feature z-scoring whose statistics are fitted on training data only.

    A feature with zero variance is mapped to zero instead of producing NaN.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X) -> "Standardizer":
        X = _numeric_matrix(X)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, X) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer.transform called before fit")
        X = _numeric_matrix(X)
        if X.shape[1] != self.mean_.size:
            raise ValueError(
                f"X has {X.shape[1]} features, scaler was fitted on {self.mean_.size}"
            )
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function, finite for any finite input."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    """L2-regularised logistic regression trained by full-batch gradient descent.

    No randomness is involved: initial weights are zero and the update order is
    fixed, so two fits on identical data produce identical parameters.
    """

    def __init__(
        self,
        lr: float = 0.5,
        epochs: int = 4000,
        l2: float = 1e-3,
        tol: float = 1e-10,
    ) -> None:
        if not isinstance(lr, numbers.Real) or not lr > 0:
            raise ValueError("lr must be a positive number")
        if (
            not isinstance(epochs, numbers.Integral)
            or isinstance(epochs, bool)
            or epochs < 1
        ):
            raise ValueError("epochs must be a positive integer")
        if not isinstance(l2, numbers.Real) or l2 < 0:
            raise ValueError("l2 must be non-negative")
        if not isinstance(tol, numbers.Real) or tol <= 0:
            raise ValueError("tol must be positive")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    def fit(self, X, y) -> "LogisticRegression":
        X = _numeric_matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.size != X.shape[0]:
            raise ValueError("y must be one-dimensional with one label per row")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must contain only 0 and 1")
        y = y.astype(np.float64)

        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        self.loss_history_ = []

        for epoch in range(1, self.epochs + 1):
            z = X @ w + b
            p = _sigmoid(z)
            loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * self.l2 * w @ w)
            self.loss_history_.append(loss)

            grad_w = X.T @ (p - y) / n + self.l2 * w
            grad_b = float(np.mean(p - y))
            w -= self.lr * grad_w
            b -= self.lr * grad_b

            if max(np.max(np.abs(grad_w)), abs(grad_b)) < self.tol:
                self.n_iter_ = epoch
                break
        else:
            self.n_iter_ = self.epochs

        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def decision_function(self, X) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LogisticRegression.predict called before fit")
        X = _numeric_matrix(X)
        if X.shape[1] != self.coef_.size:
            raise ValueError(
                f"X has {X.shape[1]} features, model expects {self.coef_.size}"
            )
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(np.int64)
