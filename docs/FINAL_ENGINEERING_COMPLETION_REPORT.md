# Final engineering completion report

## 1. Executive summary

The portfolio was audited, then completed in place without replacing any
existing project. The completion pass fixed the defects recorded in the audit,
added an applied biomedical project on a verified public dataset, added a small
experiment registry, repaired the corrupted benchmark report, and aligned the
documentation with the code. Nothing has been pushed or published; the public
repository is still `main` at `b614b91`.

## 2. Problems found at audit time

- C04 (`scale_fixed`) returned only the training split and fitted statistics on
  train + test together.
- C03 standardisation was checked for mean only, not standard deviation, and
  divided by zero for constant features.
- Benchmark setup work ran inside the timed region for the per-sample path.
- `_peak_alloc` could leave `tracemalloc` running if the measured call raised.
- Benchmark reports contained only a median, despite a README claim of "error
  bars".
- The committed benchmark Markdown had been replaced by a Python traceback.
- Neural-network layers validated dimensions only implicitly, via NumPy errors.
- No Adam convergence test existed.
- No applied biomedical project, no experiment registry, no root `LICENSE`.
- Root and sub-project READMEs contained stale test counts.
- CI covered three of five projects; the two newer projects had no job.
- No feature-branch CI run existed.

## 3. Job requirement mapping

See `docs/TARGET_ROLE_REQUIREMENTS.md`. The vacancy page was not machine-readable,
so the matrix uses requirements demonstrable from the repository.

## 4. Technical work completed

- **C04 fixed and tested**: statistics fitted on the training split only; both
  splits transformed; zero variance mapped to a unit scale.
- **C03 fixed and tested**: the property now asserts mean **and** standard
  deviation; constant features are handled without NaN.
- **Benchmark harness hardened**: preparation hoisted out of the timed call,
  `try/finally` around tracing, standard deviation/min/max recorded, report
  writes made atomic, and a regression test that a rendering failure cannot
  corrupt the existing report.
- **Benchmark report regenerated** from a real run; JSON and Markdown now agree.
- **Neural-network validation**: explicit dimension/dtype checks in layers,
  parameter validation in SGD/Adam, plus tests for constructor errors, corrupt
  parameter shapes, exact first Adam step, Adam convergence, and sampled
  gradient checking.
- **Applied biomedical project added**: WDBC (UCI 17, CC BY 4.0,
  checksum-pinned raw file), strict loader, train-only standardiser, from-scratch
  L2 logistic regression, AUROC and Wilson intervals, 18 tests, committed report.
- **Experiment registry added**: validated JSON config plus a deterministic
  runner recording dataset, model, seed, hyperparameters, metrics, environment
  and command; 12 tests.
- **Root `LICENSE` added** (MIT for code; CC BY 4.0 noted for the dataset).
- **CI expanded** to seven jobs covering all six projects and the experiment
  runner on Python 3.11/3.12/3.13, with `checkout@v7` / `setup-python@v7`.
- **Documentation aligned** with actual counts, commands and limitations.

## 5. New files

- `applied-biomedical-ml/` — package, tests, dataset, reports and dataset
  provenance document
- `experiments/` — config, runner, results and tests
- `LICENSE`
- `docs/LOCAL_WORK_INVENTORY.md` — pre-completion snapshot

## 6. Files modified

- `ml-numerical-debugging-lab/lab/cases.py`, `tests/test_cases.py`,
  `docs/cases.md`, `README.md`
- `ml-inference-benchmark/infbench/bench.py`, `infbench/run.py`,
  `tests/test_infbench.py`, `reports/benchmark.json`, `reports/benchmark.md`,
  `README.md`
- `ml-numpy-from-scratch/nnscratch/layers.py`, `nnscratch/optim.py`,
  `nnscratch/gradcheck.py`, `tests/test_edge_cases.py`, `README.md`
- `.github/workflows/ci.yml`
- `README.md`
- `docs/REPRODUCIBILITY.md`, `docs/TARGET_ROLE_REQUIREMENTS.md`,
  `docs/FINAL_ENGINEERING_COMPLETION_REPORT.md`

## 7. Dataset and research methodology

- **WDBC**: 569 samples, 30 real features, binary diagnosis. UCI ID 17, DOI
  `10.24432/C5DW2B`, CC BY 4.0. The raw file is committed unmodified and
  checksum-verified before training.
- Preprocessing is fitted on the training split only and the test split is
  touched once, after fitting. Hyperparameters were fixed before evaluation.
- WDBC has **no patient identifier**, so a patient-level split cannot be
  constructed. This limitation is stated in the project README, the dataset
  document and the generated report; no subject-level leakage prevention is
  claimed.
- Synthetic data remains labelled synthetic.

## 8. Actual experiment results

- `ml-pytorch-training/reports/experiment.json`: seeded spiral run; committed
  metrics and checksum from a real run.
- `ml-numpy-from-scratch/reports/spiral_run.json`: seed 0, 300 epochs, hidden 64,
  first loss 1.1282, final loss 0.01445, train accuracy 0.9933, test accuracy
  0.9733.
- `ml-inference-benchmark/reports/benchmark.json`: five implementations at batch
  64 with median, standard deviation, minimum and maximum latency, memory and
  error versus the float64 reference (Python 3.14.7, NumPy 2.5.4, Windows 11).
- `applied-biomedical-ml/reports/metrics.json`: seed 42, stratified 80/20,
  456 train / 113 test. Test accuracy 0.9558 (Wilson 95% CI 0.9006–0.9810),
  sensitivity 0.9524, specificity 0.9577, AUROC 0.9960.
- `experiments/results/spiral_numpy_seed0.json`: recorded run of the spiral
  configuration with metrics, environment and command.

## 9. Actual test results

Final local run, Python 3.14.7:

| Suite | Passed | Failed | Skipped | Errors |
|---|---:|---:|---:|---:|
| `ml-numpy-from-scratch` | 33 | 0 | 0 | 0 |
| `ml-numerical-debugging-lab` | 11 | 0 | 0 | 0 |
| `ml-inference-benchmark` | 11 | 0 | 0 | 0 |
| `ml-pytorch-training` | 8 | 0 | 0 | 0 |
| `coding-agent-benchmark` | 3 | 0 | 0 | 0 |
| `applied-biomedical-ml` | 18 | 0 | 0 | 0 |
| `experiments` | 12 | 0 | 0 | 0 |
| **Total** | **96** | **0** | **0** | **0** |

## 10. CI workflow status

Configuration covers seven jobs on Python 3.11, 3.12 and 3.13. It has **not**
been executed remotely from this feature branch because publication requires
explicit approval. Cross-version execution is therefore not verified.

## 11. Security review

- No secrets, credentials or environment files were added.
- The GitHub token supplied in a local attachment was never used, printed or
  committed. It should be rotated regardless.
- The committed WDBC file is public, de-identified research data under CC BY 4.0.
- No unrelated repositories were modified.

## 12. Git state

- Branch: `feat/ml-research-portfolio-upgrade`
- Published `main`: `b614b9167ae7d272e6d39b2af1470828c958671a`
- The completion work is committed locally on top of the six commits recorded in
  `docs/LOCAL_WORK_INVENTORY.md`.
- Run `git log -1 --format=%H` for the exact completion commit; no SHA is
  embedded here because a file cannot contain the hash of the commit that adds
  it.

## 13. Remaining limitations

- CI has not run remotely; the three-version matrix is configured, not observed.
- WDBC provides no patient identifier, so group-level leakage prevention cannot
  be demonstrated. Only train-only preprocessing, a fixed prior specification
  and an untouched test split are demonstrated.
- The biomedical result comes from a single 113-sample test split and is a
  portfolio demonstration, not external validation or a clinical claim.
- Absolute benchmark timings are machine-specific and are reported with spread,
  not as universal numbers.
- Gradient checking remains sampled at up to 40 entries per tensor.
- The coding-agent benchmark golden self-check is not an agent score.

## 14. Approval still required

- Pushing the feature branch and opening a pull request.
- Merging to `main` and publishing the revised README.
- Any statement beyond the committed reports about real-world or clinical
  performance.
