# Dataset provenance and license

## Identity

| Field | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| UCI ID | 17 |
| DOI | [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Instances | 569 |
| Features | 30 real-valued |
| Target | Diagnosis: M = malignant (212), B = benign (357) |
| Missing values | none |
| Donated | 1995-10-31 |

Source page:
<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>

Raw file:
<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data>

## Committed copy

`data/wdbc.data` is the raw export with no header, no cleaning or imputation:

- 569 lines, 124,103 bytes
- column 0: sample ID (unique)
- column 1: diagnosis `M`/`B`
- columns 2–31: radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension — each as mean, standard
  error and "worst" value

SHA-256:

```text
d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a
```

The loader verifies this checksum by default and refuses to train on a
mismatch. Verify manually with:

```bash
python -c "from bio_ml.data import DATA_PATH, sha256_file; print(sha256_file(DATA_PATH))"
```

## Re-download

```bash
curl -o data/wdbc.data https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```

Then re-run the checksum command above; it must equal the value in this file.

## Leakage-relevant properties

WDBC has **no patient identifier**. The UCI documentation describes features
computed from a digitised image of a fine-needle aspirate (FNA) of a breast
mass, and the dataset is distributed as one row per sample with no grouping
variable. Consequences, stated plainly:

- A patient-level or hospital-level group split **cannot** be constructed from
  this file. The report says so instead of pretending otherwise.
- The only leakage controls that can actually be implemented here are
  preprocessing fitted on the training split, a fixed prior specification, and
  a test set untouched until evaluation. All three are implemented and tested.
- Any claim of "subject-level leakage prevention" on WDBC would be unverifiable
  and is therefore not made.

## Citation

W. Street, W. Wolberg, O. Mangasarian. *Nuclear feature extraction for breast
tumor diagnosis.* IS&T/SPIE 1993 International Symposium on Electronic Imaging:
Science and Technology, 1905, 861–870 (1993).
DOI: [10.1117/12.148698](https://doi.org/10.1117/12.148698)

Dataset citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
*Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning
Repository. DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). CC BY 4.0.

## Ethical scope

This is a methodological portfolio demonstration on de-identified, publicly
released research data. It is not medical advice, has no regulatory clearance,
and must not be used to support clinical decisions. Reported metrics come from a
single 113-sample test split and should be read with the accompanying confidence
interval, not as a performance guarantee.
