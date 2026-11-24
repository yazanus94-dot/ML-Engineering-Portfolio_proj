import numpy as np


def make_spirals(
    n_per_class: int = 200, classes: int = 3, noise: float = 0.45, seed: int = 0
):
    """Synthetic multi-arm spiral dataset: (X: (N, 2), y: (N,))."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = (
            np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class)
            + rng.standard_normal(n_per_class) * noise
        )
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def train_test_split(X, y, test_frac=0.25, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]
