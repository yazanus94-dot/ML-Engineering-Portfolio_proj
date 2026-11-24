# Benchmark results

MLP [256, 512, 512, 10], comparison batch 64, 5 repeats, seed 0.
Environment: Python 3.14.7, NumPy 2.5.4, Windows-11-10.0.26300-SP0.

Throughput = batch / median wall-clock seconds. `ms/sample` = median wall-clock milliseconds / batch size.
Variability is the sample standard deviation across repeats; min and max are the observed extrema.
Timings depend on the machine; re-run on your own hardware. Compare throughput only at equal batch size.

| impl | batch | median ms | std ms | min ms | max ms | ms/sample | samples/s | rel. error | tolerance | ok | argmax agree | weights (KiB) | peak alloc (KiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 64 | 1.251 | 1.210 | 1.231 | 3.961 | 0.0195 | 51151 | 0.00e+00 | 0e+00 | yes | 1.000 | 3120.1 | 833.5 |
| loops | 64 | 1798.162 | 94.573 | 1731.289 | 1865.035 | 28.0963 | 36 | 6.53e-16 | 1e-09 | yes | 1.000 | 3120.1 | 13129.0 |
| per_sample | 64 | 2.604 | 0.583 | 2.386 | 3.534 | 0.0407 | 24574 | 2.54e-07 | 1e-04 | yes | 1.000 | 1560.0 | 80.5 |
| batched32 | 64 | 1.093 | 0.666 | 0.966 | 2.550 | 0.0171 | 58554 | 2.27e-07 | 1e-04 | yes | 1.000 | 1560.0 | 417.5 |
| int8 | 64 | 39.402 | 5.023 | 36.994 | 49.871 | 0.6157 | 1624 | 1.24e-02 | 5e-02 | yes | 0.938 | 397.1 | 1569.4 |
