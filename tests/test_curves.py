"""Offline unit tests for the curve math — no network, pure numerics."""
import numpy as np
from scipy import integrate

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import curves

# A representative Svensson parameter set (percent), in the style of GSW output.
P = dict(b0=2.5, b1=-1.0, b2=-2.0, b3=3.0, t1=1.5, t2=10.0)


def test_zero_limit_at_zero():
    y0 = curves.svensson_zero(0.0, **P)
    assert abs(float(y0) - (P["b0"] + P["b1"])) < 1e-9


def test_zero_matches_manual():
    n = 7.0
    x1, x2 = n / P["t1"], n / P["t2"]
    d1 = (1 - np.exp(-x1)) / x1
    manual = (P["b0"] + P["b1"] * d1
              + P["b2"] * (d1 - np.exp(-x1))
              + P["b3"] * ((1 - np.exp(-x2)) / x2 - np.exp(-x2)))
    assert abs(float(curves.svensson_zero(n, **P)) - manual) < 1e-12


def test_inst_forward_integral_equals_n_times_zero():
    # integral_0^n f(u) du must equal n * y(n)
    for n in [1.0, 5.0, 10.0, 30.0]:
        val, _ = integrate.quad(lambda u: float(curves.svensson_inst_forward(u, **P)), 0, n)
        assert abs(val - n * float(curves.svensson_zero(n, **P))) < 1e-6


def test_one_year_forwards_telescope():
    mats = np.arange(1, 31)
    zeros = curves.svensson_zero(mats, **P)
    fwd = curves.one_year_forwards(zeros, short_rate=float(curves.svensson_zero(0, **P)))
    # sum of the first N one-year forwards must equal N * y(N)
    for N in [1, 5, 10, 30]:
        assert abs(fwd[:N].sum() - N * float(zeros[N - 1])) < 1e-9
    # f(0,1) must equal y(1)
    assert abs(fwd[0] - float(zeros[0])) < 1e-12


def test_forward_from_zeros_matches_strip():
    mats = np.arange(0, 31)
    zeros = curves.svensson_zero(mats, **P)
    f_5_10 = curves.forward_from_zeros(mats, zeros, 5, 10)
    manual = (10 * float(zeros[10]) - 5 * float(zeros[5])) / 5
    assert abs(f_5_10 - manual) < 1e-12


def test_discount_factor_monotone_and_bounded():
    mats = np.arange(1, 31)
    zeros = curves.svensson_zero(mats, **P)
    dfs = curves.discount_factor(mats, zeros)
    assert np.all(dfs > 0) and np.all(dfs <= 1.0000001)
    assert np.all(np.diff(dfs) < 0)  # strictly decreasing for positive rates


def test_par_yield_reasonable():
    mats = np.arange(1, 31, dtype=float)
    zeros = curves.svensson_zero(mats, **P)
    par = curves.par_yield(mats, zeros, freq=2)
    # par yields should sit near the zero curve (within a point or so here)
    assert np.all(np.isfinite(par))
    assert abs(par[-1] - float(zeros[-1])) < 1.5


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
