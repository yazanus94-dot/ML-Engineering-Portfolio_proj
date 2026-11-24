# Local work inventory (pre-implementation snapshot)

Recorded before the completion pass on `feat/ml-research-portfolio-upgrade`.

## Git state

- Branch: `feat/ml-research-portfolio-upgrade`
- HEAD: `99c46f91c2eebb07856283e2cc154211fea5fdd9`
- Published `main`: `b614b9167ae7d272e6d39b2af1470828c958671a`
- Working tree: clean, no staged changes, no untracked files.
- Local commits ahead of `origin/main`:
  1. `af01e64` Add PyTorch training-to-serving lab and coding-agent benchmark
  2. `15e7f6f` Improve ML portfolio engineering docs and optimizer validation
  3. `67a0a96` Record final audit commit SHA
  4. `6fdc3e4` Improve ML portfolio: input validation, C04 train/test split, reproducibility docs and CI action versions
  5. `f683513` Record final audit commit SHA
  6. `99c46f9` Complete audit docs and clean lint/format

## Work that must be preserved

- C04 `scale_fixed(train, test)` fits statistics on the training split only and returns both transformed splits.
- C04 zero-variance handling replaces zero standard deviations with `1.0`.
- Benchmark preparation is outside the timed call through `prepare_per_sample`.
- `_peak_alloc` is protected by `try/finally`.
- Neural-network loss and optimizer validation added in `nnscratch/losses.py` and `nnscratch/optim.py`.
- Five CI jobs exist in `.github/workflows/ci.yml`.
- Six engineering documents exist under `docs/`.
- The six local commits above remain intact.

## Known gaps at the snapshot

- `ml-inference-benchmark/reports/benchmark.md` contains a Python traceback.
- C03 has no explicit standard-deviation assertion.
- Benchmark variability is not reported.
- Layer input validation is implicit and relies on NumPy errors.
- No central `experiments/` management exists.
- No `applied-biomedical-ml/` project exists.
- Root `LICENSE` is missing while the README links to it.
- Several documentation statements are stale.
- No feature-branch CI run exists.

This inventory is a historical snapshot. The completion report records the final state.
