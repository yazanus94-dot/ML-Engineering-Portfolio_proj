import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bio_ml.data import (  # noqa: E402
    DATA_PATH,
    EXPECTED_SHA256,
    FEATURE_NAMES,
    load_wdbc,
    sha256_file,
    stratified_split,
)
from bio_ml.metrics import (  # noqa: E402
    binary_metrics,
    confusion_matrix,
    roc_auc,
    wilson_interval,
)
from bio_ml.model import LogisticRegression, Standardizer, _sigmoid  # noqa: E402
from bio_ml.train import run, write_report  # noqa: E402


def test_dataset_shape_labels_and_ids():
    ds = load_wdbc()
    assert ds.X.shape == (569, 30)
    assert ds.y.shape == (569,)
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert int(ds.y.sum()) == 212  # malignant
    assert np.isfinite(ds.X).all()
    assert len(set(ds.ids.tolist())) == 569
    assert len(FEATURE_NAMES) == 30
    assert FEATURE_NAMES[0] == "radius_mean"
    assert FEATURE_NAMES[-1] == "fractal_dimension_worst"


def test_committed_file_matches_documented_checksum():
    assert sha256_file(DATA_PATH) == EXPECTED_SHA256


def test_loader_rejects_tampered_file(tmp_path):
    text = DATA_PATH.read_text(encoding="utf-8")
    tampered = tmp_path / "wdbc.data"
    tampered.write_text(
        text.replace("842302,M,17.99", "842302,M,18.99", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum"):
        load_wdbc(tampered)


def test_loader_rejects_malformed_row(tmp_path):
    text = DATA_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    lines[0] = ",".join(lines[0].split(",")[:-1])
    broken = tmp_path / "wdbc.data"
    broken.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_wdbc(broken, verify_checksum=False)


def test_split_is_disjoint_stratified_and_deterministic():
    ds = load_wdbc()
    train_a, test_a = stratified_split(ds.y, test_frac=0.2, seed=42)
    train_b, test_b = stratified_split(ds.y, test_frac=0.2, seed=42)
    train_c, _ = stratified_split(ds.y, test_frac=0.2, seed=7)

    assert np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)
    assert not np.array_equal(train_a, train_c)
    assert np.intersect1d(train_a, test_a).size == 0
    assert train_a.size + test_a.size == 569
    assert set(np.unique(ds.y[train_a]).tolist()) == {0, 1}
    assert set(np.unique(ds.y[test_a]).tolist()) == {0, 1}
    assert abs(ds.y[test_a].mean() - ds.y.mean()) < 0.05


def test_split_rejects_invalid_fraction():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        stratified_split(y, test_frac=0.0)
    with pytest.raises(ValueError):
        stratified_split(y, test_frac=1.0)
    with pytest.raises(ValueError):
        stratified_split(y, test_frac=0.5, seed=True)


def test_standardizer_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((50, 4)) * 3 + 10
    test = rng.standard_normal((20, 4)) * 100 - 50

    scaler = Standardizer().fit(train)
    mu, sd = scaler.mean_.copy(), scaler.scale_.copy()

    transformed = scaler.transform(test)
    assert np.isfinite(transformed).all()
    assert np.allclose(mu, train.mean(axis=0))
    assert np.allclose(sd, train.std(axis=0))
    assert np.allclose(scaler.transform(train).mean(axis=0), 0.0, atol=1e-12)

    # Changing the test rows must not move the fitted statistics.
    scaler.transform(test + 1_000.0)
    assert np.array_equal(mu, scaler.mean_)
    assert np.array_equal(sd, scaler.scale_)


def test_standardizer_handles_constant_features():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Z = Standardizer().fit_transform(X)
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, 0], 0.0)
    assert np.allclose(Z[:, 1].mean(), 0.0, atol=1e-12)


def test_standardizer_rejects_wrong_width_and_non_finite():
    scaler = Standardizer().fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.ones((3, 3)))
    with pytest.raises(ValueError, match="NaN"):
        Standardizer().fit(np.array([[1.0, np.nan]]))


def test_sigmoid_is_stable_for_extreme_inputs():
    z = np.array([-1000.0, -40.0, 0.0, 40.0, 1000.0])
    p = _sigmoid(z)
    assert np.isfinite(p).all()
    assert p[2] == 0.5
    assert p[0] == 0.0 and p[-1] == 1.0
    assert np.all(np.diff(p) >= 0)


def test_logistic_regression_learns_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.standard_normal((60, 2)) - 2.0, rng.standard_normal((60, 2)) + 2.0]
    )
    y = np.array([0] * 60 + [1] * 60)
    model = LogisticRegression(epochs=2000).fit(X, y)
    assert (model.predict(X) == y).mean() == 1.0
    assert np.all((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1))


def test_logistic_regression_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    a = LogisticRegression(epochs=300).fit(X, y)
    b = LogisticRegression(epochs=300).fit(X, y)
    assert np.array_equal(a.coef_, b.coef_)
    assert a.intercept_ == b.intercept_


def test_metrics_match_hand_computed_values():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 0, 1, 1, 0])
    m = binary_metrics(y, p, threshold=0.5)

    assert m["confusion"] == {"tp": 2, "tn": 2, "fp": 1, "fn": 1}
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["ppv"] == pytest.approx(2 / 3)
    assert m["npv"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert 0.0 <= m["accuracy_ci95"][0] <= m["accuracy"] <= m["accuracy_ci95"][1] <= 1.0


def test_confusion_matrix_and_metric_validation():
    assert confusion_matrix([1, 0], [0, 1]) == {"tp": 0, "tn": 0, "fp": 1, "fn": 1}
    with pytest.raises(ValueError):
        confusion_matrix([1, 2], [0, 1])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.2, 1.2])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.2, 0.8], threshold=1.5)


def test_auroc_known_cases():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.3])


def test_wilson_interval_is_bounded_and_ordered():
    lo, hi = wilson_interval(40, 50)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_end_to_end_run_generalises_and_is_deterministic():
    first = run(seed=42, test_frac=0.2)
    second = run(seed=42, test_frac=0.2)

    assert first["metrics"]["test"] == second["metrics"]["test"]
    assert first["split"]["n_train"] + first["split"]["n_test"] == 569
    assert first["metrics"]["test"]["accuracy"] >= 0.90
    assert first["metrics"]["test"]["auroc"] >= 0.95
    assert (
        first["metrics"]["train"]["accuracy"]
        >= first["metrics"]["test"]["accuracy"] - 0.05
    )
    assert first["model"]["n_iter"] <= 4000
    assert first["dataset"]["license"] == "CC BY 4.0"


def test_report_is_valid_and_contains_no_traceback(tmp_path):
    result = run(seed=42, test_frac=0.2)
    out = write_report(result, tmp_path)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    assert markdown.startswith("# WDBC classification report")
    assert "Traceback" not in markdown
    assert "| AUROC |" in markdown
    assert (
        payload["metrics"]["test"]["accuracy"] == result["metrics"]["test"]["accuracy"]
    )
    assert payload["dataset"]["sha256"] == EXPECTED_SHA256
    assert "not a clinical device" in markdown
