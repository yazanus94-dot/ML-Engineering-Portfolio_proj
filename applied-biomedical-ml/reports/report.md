# WDBC classification report

Dataset: **Breast Cancer Wisconsin (Diagnostic)** (UCI 17, DOI 10.24432/C5DW2B), license CC BY 4.0.
Raw file SHA-256: `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`.

Model: L2-regularised logistic regression (from scratch), 31 parameters, 4000 full-batch iterations, threshold 0.5.
Split: stratified 80/20 with seed 42 — 456 train / 113 test.

| metric | train | test |
|---|---|---|
| accuracy | 0.9912 | 0.9558 |
| 95% CI (accuracy) | [0.9777, 0.9966] | [0.9006, 0.9810] |
| sensitivity (malignant) | 0.9765 | 0.9524 |
| specificity (benign) | 1.0000 | 0.9577 |
| precision (PPV) | 1.0000 | 0.9302 |
| NPV | 0.9862 | 0.9714 |
| F1 | 0.9881 | 0.9412 |
| AUROC | 0.9980 | 0.9960 |

## Test confusion matrix

| | predicted benign | predicted malignant |
|---|---|---|
| actual benign | 68 | 3 |
| actual malignant | 2 | 40 |

## Largest standardised coefficients

| feature | coefficient (per SD) | odds ratio per SD |
|---|---|---|
| radius_se | +1.5553 | 4.7363 |
| texture_worst | +1.5370 | 4.6506 |
| concave_points_mean | +1.3806 | 3.9773 |
| area_se | +1.1684 | 3.2170 |
| area_worst | +1.1561 | 3.1776 |
| radius_worst | +1.1383 | 3.1213 |
| perimeter_worst | +1.0832 | 2.9540 |
| smoothness_worst | +1.0753 | 2.9309 |

Coefficients describe associations in this dataset; they are not causal or clinical evidence.

## Leakage control and limitations

- WDBC has no patient identifier, so patient-level (group) leakage cannot be constructed or measured. UCI documents one fine-needle aspirate per mass; the split is stratified by diagnosis only.
- Preprocessing (mean/std) is fitted on the training split only; no resampling, feature selection or hyperparameter search is performed on the test split.
- Hyperparameters were fixed a priori and not tuned against test performance.
- This is a portfolio demonstration on a 1993 research dataset with 569 samples, not a clinical device. It must not be used for diagnosis or treatment decisions.

## Reproduce

```bash
cd applied-biomedical-ml
python -m pytest -q
python -m bio_ml.train --output reports
```

Environment: Python 3.14.7, NumPy 2.5.4.
