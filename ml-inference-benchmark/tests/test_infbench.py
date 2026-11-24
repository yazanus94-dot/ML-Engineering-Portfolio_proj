import json
import sys
import pathlib
import tracemalloc

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from infbench import model as M
from infbench.bench import (
    TOLERANCE,
    _peak_alloc,
    build_impls,
    relative_error,
    run,
    to_markdown,
    write_report,
)

SIZES = [16, 32, 32, 5]


def _setup(seed=0, n=12):
    m = M.MLP.random(SIZES, seed)
    return m, np.random.default_rng(seed + 1).standard_normal((n, SIZES[0]))


def test_all_impls_match_reference_within_tolerance():
    m, x = _setup()
    ref = M.forward_reference(m, x)
    for name, (fn, _) in build_impls(m).items():
        if name == "reference":
            continue
        assert relative_error(fn(x), ref) <= TOLERANCE[name], name


def test_models_are_deterministic_per_seed():
    a, b, c = M.MLP.random(SIZES, 0), M.MLP.random(SIZES, 0), M.MLP.random(SIZES, 1)
    assert all(np.array_equal(p, q) for p, q in zip(a.weights, b.weights))
    assert not np.array_equal(a.weights[0], c.weights[0])


def test_quantisation_error_bounded_by_half_step():
    W = np.random.default_rng(0).standard_normal((20, 7))
    wq, s = M.quantise_weights(W)
    assert wq.dtype == np.int8 and np.abs(wq).max() <= 127
    assert np.all(np.abs(wq * s - W) <= s / 2 + 1e-7)


def test_int8_weights_are_about_4x_smaller_than_float32():
    m, _ = _setup()
    b32 = M.weight_bytes(M.prepare_batched32(m), "batched32")
    b8 = M.weight_bytes(M.prepare_int8(m), "int8")
    assert b8 < b32 / 3


def test_relu_zero_rows_do_not_break_loops_impl():
    m, _ = _setup()
    x = np.zeros((2, SIZES[0]))
    assert np.allclose(M.forward_loops(m, x), M.forward_reference(m, x))


def test_run_returns_complete_report():
    res = run(sizes=(16, 32, 5), batch=16, repeats=3)
    names = [r["impl"] for r in res["results"]]
    assert names == ["reference", "loops", "per_sample", "batched32", "int8"]
    assert all(r["within_tolerance"] for r in res["results"])
    assert all(r["batch"] == 16 for r in res["results"])
    assert all(
        {"median_ms", "std_ms", "min_ms", "max_ms", "median_ms_per_sample"} <= r.keys()
        for r in res["results"]
    )
    assert res["config"]["comparison_batch"] == 16
    assert "| impl |" in to_markdown(res)


def test_markdown_contains_variability_and_methodology():
    res = run(sizes=(16, 32, 5), batch=8, repeats=3)
    md = to_markdown(res)
    assert "| impl | batch | median ms | std ms | min ms | max ms |" in md
    assert "Throughput = batch / median wall-clock seconds" in md
    assert "Variability is the sample standard deviation" in md
    for r in res["results"]:
        assert r["std_ms"] >= 0.0
        assert r["min_ms"] <= r["median_ms"] <= r["max_ms"]
        assert r["median_ms_per_sample"] > 0.0
        assert r["samples_per_s"] > 0.0


def test_invalid_measurement_settings_are_rejected():
    with pytest.raises(ValueError):
        run(sizes=(16, 32, 5), batch=8, repeats=0)
    with pytest.raises(ValueError):
        run(sizes=(16, 32, 5), batch=4, repeats=2, loops_batch=5)
    with pytest.raises(ValueError):
        run(sizes=(16, 32, 5), batch=0, repeats=2)


def test_peak_alloc_preserves_an_active_tracing_session():
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        peak = _peak_alloc(lambda: np.ones((100, 100)))
        assert peak > 0
        assert tracemalloc.is_tracing()
    finally:
        if not was_tracing:
            tracemalloc.stop()


def test_write_report_produces_valid_files(tmp_path):
    res = run(sizes=(16, 32, 5), batch=8, repeats=2)
    out = write_report(res, tmp_path)
    md = (out / "benchmark.md").read_text(encoding="utf-8")
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert md.startswith("# Benchmark results")
    assert "| impl |" in md
    assert "Traceback" not in md
    assert payload["config"]["comparison_batch"] == 8
    assert len(payload["results"]) == 5


def test_report_failure_does_not_corrupt_existing_markdown(tmp_path, monkeypatch):
    res = run(sizes=(16, 32, 5), batch=8, repeats=2)
    out = write_report(res, tmp_path)
    before = (out / "benchmark.md").read_text(encoding="utf-8")

    def fail(_):
        raise RuntimeError("render failed")

    monkeypatch.setattr("infbench.bench.to_markdown", fail)
    with pytest.raises(RuntimeError):
        write_report(res, tmp_path)
    assert (out / "benchmark.md").read_text(encoding="utf-8") == before
