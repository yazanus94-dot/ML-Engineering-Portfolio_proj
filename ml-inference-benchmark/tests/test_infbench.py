import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from infbench import model as M
from infbench.bench import run, to_markdown, build_impls, relative_error, TOLERANCE

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
    res = run(sizes=(16, 32, 5), batch=16, repeats=2, loops_batch=2)
    names = [r["impl"] for r in res["results"]]
    assert names == ["reference", "loops", "per_sample", "batched32", "int8"]
    assert all(r["within_tolerance"] for r in res["results"])
    assert "| impl |" in to_markdown(res)
