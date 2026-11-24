# Research methodology

Only the existing synthetic spiral experiment is run and verified. No real dataset was downloaded
or claimed.

- `ml-numpy-from-scratch/scripts/train_spiral.py` trains a 64-hidden-unit MLP for 300 epochs.
- The committed report is `ml-numpy-from-scratch/reports/spiral_run.json`:
  - seed 0
  - first loss 1.1282, final loss 0.01445
  - train accuracy 0.9933, test accuracy 0.9733
- The dataset is explicitly labelled as synthetic in the report.
- The benchmark uses a seeded MLP, seeded NumPy input, median latency, peak allocation via
  `tracemalloc`, and a float64 reference for numerical comparison.
- The PyTorch lab exports ONNX and compares runtime logits against PyTorch with a maximum
  difference of 9.54e-07.
- No medical diagnostic, clinical or production-readiness claim is made.
