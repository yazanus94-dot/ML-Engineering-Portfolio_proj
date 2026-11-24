"""Train and evaluate a logistic-regression baseline on WDBC.

Run from ``applied-biomedical-ml``:

    python -m bio_ml.train --output reports

The output is deterministic for a given seed. The test set is touched exactly
once, after fitting, for the reported numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys

import numpy as np

from .data import (
    DATA_PATH,
    EXPECTED_SHA256,
    FEATURE_NAMES,
    load_wdbc,
    stratified_split,
)
from .metrics import binary_metrics
from .model import LogisticRegression, Standardizer

SEED = 42
TEST_FRAC = 0.2
THRESHOLD = 0.5
HYPERPARAMETERS = {"lr": 0.5, "epochs": 4000, "l2": 1e-3, "tol": 1e-10}

DATASET = {
    "name": "Breast Cancer Wisconsin (Diagnostic)",
    "uci_id": 17,
    "doi": "10.24432/C5DW2B",
    "license": "CC BY 4.0",
    "source_url": (
        "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
    ),
    "raw_url": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "breast-cancer-wisconsin/wdbc.data"
    ),
    "sha256": EXPECTED_SHA256,
    "n_samples": 569,
    "n_features": 30,
    "positive_class": "malignant",
    "target": "Diagnosis (M=1, B=0)",
}

LIMITATIONS = [
    "WDBC has no patient identifier, so patient-level (group) leakage cannot be constructed or measured. "
    "UCI documents one fine-needle aspirate per mass; the split is stratified by diagnosis only.",
    "Preprocessing (mean/std) is fitted on the training split only; no resampling, feature selection or "
    "hyperparameter search is performed on the test split.",
    "Hyperparameters were fixed a priori and not tuned against test performance.",
    "This is a portfolio demonstration on a 1993 research dataset with 569 samples, not a clinical device. "
    "It must not be used for diagnosis or treatment decisions.",
]


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(seed: int = SEED, test_frac: float = TEST_FRAC) -> dict:
    ds = load_wdbc(DATA_PATH, verify_checksum=True)
    train_idx, test_idx = stratified_split(ds.y, test_frac=test_frac, seed=seed)

    scaler = Standardizer().fit(ds.X[train_idx])
    X_train = scaler.transform(ds.X[train_idx])
    X_test = scaler.transform(ds.X[test_idx])

    model = LogisticRegression(**HYPERPARAMETERS).fit(X_train, ds.y[train_idx])

    train_metrics = binary_metrics(
        ds.y[train_idx], model.predict_proba(X_train), threshold=THRESHOLD
    )
    test_metrics = binary_metrics(
        ds.y[test_idx], model.predict_proba(X_test), threshold=THRESHOLD
    )

    order = np.argsort(-np.abs(model.coef_))[:8]
    top_coefficients = [
        {
            "feature": FEATURE_NAMES[i],
            "standardised_coefficient": float(model.coef_[i]),
            "odds_ratio_per_sd": float(np.exp(model.coef_[i])),
        }
        for i in order
    ]

    return {
        "dataset": DATASET,
        "split": {
            "seed": int(seed),
            "test_frac": float(test_frac),
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "train_positives": int(ds.y[train_idx].sum()),
            "test_positives": int(ds.y[test_idx].sum()),
            "stratified": True,
            "group_level_split": False,
        },
        "model": {
            "name": "L2-regularised logistic regression (from scratch)",
            "hyperparameters": dict(HYPERPARAMETERS),
            "threshold": THRESHOLD,
            "n_iter": int(model.n_iter_),
            "final_training_loss": float(model.loss_history_[-1]),
            "intercept": float(model.intercept_),
            "n_parameters": int(model.coef_.size + 1),
        },
        "metrics": {"train": train_metrics, "test": test_metrics},
        "top_coefficients": top_coefficients,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "command": "python -m bio_ml.train --output reports",
        "limitations": LIMITATIONS,
    }


def to_markdown(result: dict) -> str:
    ds, sp, md = result["dataset"], result["split"], result["model"]
    tr, te = result["metrics"]["train"], result["metrics"]["test"]
    cm = te["confusion"]
    lines = [
        "# WDBC classification report",
        "",
        f"Dataset: **{ds['name']}** (UCI {ds['uci_id']}, DOI {ds['doi']}), "
        f"license {ds['license']}.",
        f"Raw file SHA-256: `{ds['sha256']}`.",
        "",
        f"Model: {md['name']}, {md['n_parameters']} parameters, "
        f"{md['n_iter']} full-batch iterations, threshold {md['threshold']}.",
        f"Split: stratified {int((1 - sp['test_frac']) * 100)}/{int(sp['test_frac'] * 100)} "
        f"with seed {sp['seed']} — {sp['n_train']} train / {sp['n_test']} test.",
        "",
        "| metric | train | test |",
        "|---|---|---|",
        f"| accuracy | {tr['accuracy']:.4f} | {te['accuracy']:.4f} |",
        f"| 95% CI (accuracy) | "
        f"[{tr['accuracy_ci95'][0]:.4f}, {tr['accuracy_ci95'][1]:.4f}] | "
        f"[{te['accuracy_ci95'][0]:.4f}, {te['accuracy_ci95'][1]:.4f}] |",
        f"| sensitivity (malignant) | {tr['sensitivity']:.4f} | {te['sensitivity']:.4f} |",
        f"| specificity (benign) | {tr['specificity']:.4f} | {te['specificity']:.4f} |",
        f"| precision (PPV) | {tr['ppv']:.4f} | {te['ppv']:.4f} |",
        f"| NPV | {tr['npv']:.4f} | {te['npv']:.4f} |",
        f"| F1 | {tr['f1']:.4f} | {te['f1']:.4f} |",
        f"| AUROC | {tr['auroc']:.4f} | {te['auroc']:.4f} |",
        "",
        "## Test confusion matrix",
        "",
        "| | predicted benign | predicted malignant |",
        "|---|---|---|",
        f"| actual benign | {cm['tn']} | {cm['fp']} |",
        f"| actual malignant | {cm['fn']} | {cm['tp']} |",
        "",
        "## Largest standardised coefficients",
        "",
        "| feature | coefficient (per SD) | odds ratio per SD |",
        "|---|---|---|",
    ]
    for row in result["top_coefficients"]:
        lines.append(
            f"| {row['feature']} | {row['standardised_coefficient']:+.4f} | "
            f"{row['odds_ratio_per_sd']:.4f} |"
        )
    lines += [
        "",
        "Coefficients describe associations in this dataset; they are not causal "
        "or clinical evidence.",
        "",
        "## Leakage control and limitations",
        "",
    ]
    lines += [f"- {item}" for item in result["limitations"]]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd applied-biomedical-ml",
        "python -m pytest -q",
        "python -m bio_ml.train --output reports",
        "```",
        "",
        f"Environment: Python {result['environment']['python']}, "
        f"NumPy {result['environment']['numpy']}.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict, out_dir) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    payload = json.dumps(result, indent=2) + "\n"
    markdown = to_markdown(result)
    _atomic_write_text(out / "metrics.json", payload)
    _atomic_write_text(out / "report.md", markdown)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--output", default="reports")
    args = parser.parse_args(argv)

    try:
        result = run(seed=args.seed, test_frac=args.test_frac)
        out = pathlib.Path(args.output)
        write_report(result, out)
    except (OSError, ValueError, TypeError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1

    te = result["metrics"]["test"]
    print(
        f"test accuracy {te['accuracy']:.4f} "
        f"(95% CI {te['accuracy_ci95'][0]:.4f}-{te['accuracy_ci95'][1]:.4f}), "
        f"sensitivity {te['sensitivity']:.4f}, specificity {te['specificity']:.4f}, "
        f"AUROC {te['auroc']:.4f}"
    )
    print(f"wrote {out / 'metrics.json'} and {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
