"""Binary classification metrics with explicit edge-case behaviour."""

from __future__ import annotations

import math

import numpy as np

# 97.5th percentile of the standard normal distribution.
_Z95 = 1.959963984540054


def _as_binary(y_true) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if y.size == 0:
        raise ValueError("y_true must not be empty")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    return y.astype(np.int64)


def confusion_matrix(y_true, y_pred) -> dict:
    y = _as_binary(y_true)
    p = _as_binary(y_pred)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def wilson_interval(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval; returns ``(0.0, 1.0)``-clipped bounds."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    if successes == 0:
        return 0.0, min(1.0, z * z / (n + z * z))
    if successes == n:
        return max(0.0, n / (n + z * z)), 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def roc_auc(y_true, y_score) -> float:
    """Rank-based AUROC (equivalent to the Mann-Whitney U statistic).

    Ties in the score receive the average rank, so a constant score yields 0.5.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if not np.isfinite(s).all():
        raise ValueError("y_score must be finite")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one sample of each class")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Accuracy, class-wise rates, F1, AUROC and Wilson intervals."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
        raise ValueError("y_score must be probabilities in [0, 1]")

    pred = (s >= threshold).astype(np.int64)
    cm = confusion_matrix(y, pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = int(y.size)
    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else 0.0

    acc_lo, acc_hi = wilson_interval(tp + tn, n)
    se_lo, se_hi = wilson_interval(tp, tp + fn)
    sp_lo, sp_hi = wilson_interval(tn, tn + fp)

    return {
        "threshold": float(threshold),
        "n": n,
        "prevalence": float(y.mean()),
        "confusion": cm,
        "accuracy": float(accuracy),
        "accuracy_ci95": [acc_lo, acc_hi],
        "sensitivity": float(sensitivity),
        "sensitivity_ci95": [se_lo, se_hi],
        "specificity": float(specificity),
        "specificity_ci95": [sp_lo, sp_hi],
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "auroc": roc_auc(y, s),
    }
