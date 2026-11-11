# Debugging playbook for numerical ML failures

1. **Reproduce on the smallest input**, with fixed seeds. Print dtype, shape and min/max/NaN count at each stage.
2. **Look for the usual suspects:** exp/log of large or zero values, subtracting nearly equal numbers, low-precision accumulation, wrong axis, silent broadcasting, statistics computed before the train/test split, train/eval mode differences.
3. **Compare with a slow, simple reference** (float64, explicit loops) and quantify the difference relative to the output range.
4. **Use gradient checking** to separate a wrong backward pass from a bad optimiser setting.
5. **Fix at the cause** (stable formulation, validation of shapes) rather than clipping the symptom.
6. **Add a regression test** that fails on the old code and passes on the new.
