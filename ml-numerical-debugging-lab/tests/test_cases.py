import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from lab.cases import CASES, run_all, softmax_fixed, variance_fixed, xent_fixed


def test_eight_cases_unique_ids():
    assert len(CASES) == 8 and len({c.id for c in CASES}) == 8


def test_every_bug_reproduced_and_fix_verified():
    for c, b, f, ok in run_all():
        assert ok, (c.id, b, f)


def test_every_case_is_documented():
    for c in CASES:
        assert all([c.symptom, c.cause, c.fix, c.property])


def test_fixed_softmax_matches_reference_and_is_shift_invariant():
    x = np.array([0.5, -1.0, 2.0])
    assert np.allclose(softmax_fixed(x), softmax_fixed(x + 1000.0))
    assert abs(softmax_fixed(x).sum() - 1) < 1e-12


def test_fixed_variance_matches_numpy():
    x = np.random.default_rng(0).standard_normal(1000) + 1e6
    assert abs(variance_fixed(x) - np.var(x)) < 1e-9


def test_fixed_xent_matches_naive_on_safe_inputs():
    z = np.array([0.2, 1.0, -0.5])
    p = np.exp(z) / np.exp(z).sum()
    assert abs(xent_fixed(z, 1) + np.log(p[1])) < 1e-12
