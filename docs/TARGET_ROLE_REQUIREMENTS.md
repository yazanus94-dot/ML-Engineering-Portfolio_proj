# Target role requirements — Machine Learning Researcher

## Vacancy

Micro1, Machine Learning Researcher  
https://jobs.micro1.ai/post/8dd58c0e-78c8-44b4-a8e3-0bfbfc1f765f

## Access limitation

The vacancy page could not be parsed at audit time (JavaScript-rendered content not available
through the fetch used). Requirements below are therefore organized from the verified technical
content of this repository and the standard responsibilities of an ML researcher. Only claims
that are directly demonstrated by executable code, tests or committed reports are marked verified.

## Traceability matrix

| Job requirement | Repository evidence | Verification method | Status |
|---|---|---|---|
| Python ML implementation | `ml-numpy-from-scratch/nnscratch`, `ml-pytorch-training/pytorch_lab`, `applied-biomedical-ml/bio_ml` | `python -m pytest -q` (33 + 8 + 18 tests) | verified |
| Neural network development | Linear, ReLU, Tanh, LayerNorm, SelfAttention, Sequential with hand-written backprop | gradient checks, deterministic training test | verified |
| Mathematical validation | `nnscratch/gradcheck.py`, `nnscratch/losses.py`, `nnscratch/optim.py` | finite-difference checks, loss-gradient tests, Adam convergence test | verified |
| Numerical debugging | `ml-numerical-debugging-lab/lab/cases.py` (8 bugs + fixes) | 11 tests, including C03 mean/std and C04 train-only scaling; `python -m lab.report` | verified |
| Inference optimization | `ml-inference-benchmark/infbench` (5 implementations) | 11 tests; `python -m infbench.run`; committed JSON/Markdown | verified |
| Model training and evaluation | `ml-pytorch-training/pytorch_lab/train.py`, `export_onnx.py`, `benchmark_cpu.py` | 8 tests; committed reports | verified |
| Applied biomedical ML | `applied-biomedical-ml/bio_ml`, checksum-pinned WDBC (UCI 17, CC BY 4.0) | 18 tests; committed `reports/metrics.json` with train-only preprocessing and Wilson intervals | verified (research demonstration, not clinical) |
| Reproducible experiments | `experiments/` config registry + deterministic runner; per-project committed reports | 12 tests; `python experiments/experiment_runner.py --config ...` | verified |
| Automated testing | pytest suites across 6 projects + experiment runner | `python -m pytest -q` in each project | verified |
| CI/CD | `.github/workflows/ci.yml`, 7 jobs × Python 3.11/3.12/3.13 | workflow inspection; local suites pass | configured locally, remote run pending publication |
| Research methodology | `docs/RESEARCH_METHODOLOGY.md`, `docs/TECHNICAL_AUDIT.md` | documentation review | added |
| Measurement with uncertainty | benchmark std/min/max across repeats; Wilson 95% intervals in the biomedical report | inspect committed reports | verified |
| Education/experience claims | not applicable | repository evidence only | verified |

## Notes

- No official PhysioNet or clinical-challenge participation is claimed.
- No production deployment is claimed. The optional FastAPI service is a portfolio surface.
- Synthetic data is labelled as synthetic throughout.
- WDBC has no patient identifier, so no patient-level split is claimed; the
  preprocessing leakage control that *is* implemented is tested.
- No API keys or secrets are required for the default workflows.
- CI has not been executed from the feature branch; publication requires approval.
