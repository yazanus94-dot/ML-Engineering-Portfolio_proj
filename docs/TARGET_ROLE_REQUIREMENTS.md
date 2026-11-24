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

| Job requirement | Existing repository evidence | Implementation needed | Verification method | Status |
|---|---|---|---|---|
| Python ML implementation | `ml-numpy-from-scratch/nnscratch`, `ml-pytorch-training/pytorch_lab` | none | `pytest -q` (20 tests pass) | verified |
| Neural network development | Linear, ReLU, Tanh, LayerNorm, SelfAttention, Sequential with hand-written backprop | none | gradient checks, deterministic training test | verified |
| Mathematical validation | `nnscratch/gradcheck.py`, `nnscratch/losses.py` | none | finite-difference checks, loss gradient test | verified |
| Numerical debugging | `ml-numerical-debugging-lab/lab/cases.py` (8 bugs + fixes) | none | 6 tests pass, `python -m lab.report` | verified |
| Inference optimization | `ml-inference-benchmark/infbench` (5 implementations) | none | 6 tests pass; `python -m infbench.run` | verified |
| Model training and evaluation | `ml-pytorch-training/pytorch_lab/train.py`, `export_onnx.py`, `benchmark_cpu.py` | none | 8 tests pass; committed reports | verified |
| Reproducible experiments | fixed seeds, committed reports, deterministic checks | none | rerun commands and compare reports | verified |
| Automated testing | pytest suites across 5 projects | none | `pytest -q` in each project | verified |
| CI/CD | `.github/workflows/ci.yml` matrix Python 3.11/3.12/3.13 | upgrade action versions | inspect workflow and GitHub runs | local only |
| Research methodology | README and docs | add `RESEARCH_METHODOLOGY.md` | documentation review | added |
| Applied ML research | not present | a verified public dataset is required | dataset verification | blocker |
| Education/experience claims | not applicable | repository evidence only | README | verified |

## Notes
- No official PhysioNet or clinical challenge participation is claimed.
- No production deployment is claimed. The optional FastAPI service is a portfolio surface.
- Synthetic data is labelled as synthetic throughout.
- No API keys or secrets are required for the default workflows.