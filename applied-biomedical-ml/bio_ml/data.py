"""Loading, validation and splitting of the WDBC dataset.

The committed file is the raw UCI ``wdbc.data`` export: 569 rows, no header,
column 0 = sample ID, column 1 = diagnosis (M/B), columns 2-31 = 30 real
features. Nothing is imputed, deduplicated or filtered.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

import numpy as np

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "wdbc.data"

# SHA-256 of the committed raw file, downloaded from
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
EXPECTED_SHA256 = "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a"
EXPECTED_ROWS = 569
EXPECTED_FEATURES = 30

_BASES = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
)
FEATURE_NAMES = tuple(f"{b}_{s}" for s in ("mean", "se", "worst") for b in _BASES)


@dataclass(frozen=True)
class Dataset:
    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray  # 1 = malignant (positive class), 0 = benign

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wdbc(path=DATA_PATH, verify_checksum: bool = True) -> Dataset:
    """Parse and validate the raw WDBC file.

    Raises ``ValueError`` for anything that does not match the documented
    shape, labels, identifier uniqueness or finite-value contract.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WDBC data not found at {path}")

    if verify_checksum:
        actual = sha256_file(path)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"WDBC checksum mismatch: expected {EXPECTED_SHA256}, got {actual}. "
                "Do not train on an unverified copy."
            )

    ids, labels, rows = [], [], []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != EXPECTED_FEATURES + 2:
                raise ValueError(
                    f"line {line_no}: expected {EXPECTED_FEATURES + 2} fields, "
                    f"got {len(parts)}"
                )
            sample_id, diagnosis = parts[0], parts[1]
            if diagnosis not in {"M", "B"}:
                raise ValueError(f"line {line_no}: unknown diagnosis {diagnosis!r}")
            try:
                values = [float(v) for v in parts[2:]]
            except ValueError as exc:
                raise ValueError(f"line {line_no}: non-numeric feature value") from exc
            ids.append(sample_id)
            labels.append(1 if diagnosis == "M" else 0)
            rows.append(values)

    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    ids_arr = np.asarray(ids, dtype=str)

    if X.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(
            f"expected shape ({EXPECTED_ROWS}, {EXPECTED_FEATURES}), got {X.shape}"
        )
    if not np.isfinite(X).all():
        raise ValueError("features contain NaN or infinity")
    if len(set(ids_arr.tolist())) != len(ids_arr):
        raise ValueError("duplicate sample IDs")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("both benign and malignant samples are required")

    return Dataset(ids=ids_arr, X=X, y=y)


def stratified_split(
    y: np.ndarray, test_frac: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` with class proportions preserved.

    The split is deterministic for a given seed. Grouping is impossible here:
    the WDBC file has no patient identifier (see ``docs/DATASET.md``).
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be strictly between 0 and 1")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_test = min(max(n_test, 1), len(idx) - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("train and test indices overlap")
    if set(np.unique(y[train_idx]).tolist()) != {0, 1}:
        raise AssertionError("training split is missing a class")
    if set(np.unique(y[test_idx]).tolist()) != {0, 1}:
        raise AssertionError("test split is missing a class")
    return train_idx, test_idx
