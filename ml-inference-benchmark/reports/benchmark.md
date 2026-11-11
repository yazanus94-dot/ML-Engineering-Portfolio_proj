# Benchmark results

MLP [256, 512, 512, 10], batch 64 (the `loops` row uses batch 8), 2 repeats, seed 0.
Environment: Python 3.14.7, NumPy 2.5.4, Windows-11-10.0.26300-SP0.

Throughput is only comparable between rows with the same batch size. Timings depend on the machine; re-run on your own hardware.

| impl | batch | median ms | samples/s | rel. error | tolerance | ok | argmax agree | weights (KiB) | peak alloc (KiB) |
|---|---|---|---|---|---|---|---|---|---|
| reference | 64 | 24.400 | 2623 | 0.00e+00 | 0e+00 | yes | 1.000 | 3120.1 | 833.5 |
| loops | 8 | 206.006 | 39 | 4.79e-16 | 1e-09 | yes | 1.000 | 3120.1 | 12652.6 |
| per_sample | 64 | 5.336 | 11993 | 2.54e-07 | 1e-04 | yes | 1.000 | 1560.0 | 1641.2 |
| batched32 | 64 | 0.785 | 81539 | 2.27e-07 | 1e-04 | yes | 1.000 | 1560.0 | 417.5 |
| int8 | 64 | 55.705 | 1149 | 1.24e-02 | 5e-02 | yes | 0.938 | 397.1 | 1569.6 |
