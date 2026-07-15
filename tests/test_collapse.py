"""Tests for collapsing a rate term structure to a single equivalent rate."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import collapse

GRID = np.arange(1, 101, dtype=float)


def test_flat_curve_collapses_to_itself():
    r = collapse.collapse_rate(GRID, np.full_like(GRID, 6.0), growth=2.0)
    assert abs(r - 6.0) < 1e-4


def test_collapsed_rate_lies_within_the_curve_range():
    # a rising curve collapses to somewhere between its min and max
    curve = np.linspace(5.0, 12.0, len(GRID))
    r = collapse.collapse_rate(GRID, curve, growth=2.0)
    assert 5.0 < r < 12.0


def test_reprices_the_cashflows_exactly():
    curve = 6.0 + 3.0 * (GRID > 40)                     # a tail bump
    cf = (1.02) ** GRID
    r = collapse.collapse_rate(GRID, curve, cf)
    pv_ts = np.sum(cf / np.cumprod(1 + curve / 100))
    pv_flat = np.sum(cf / (1 + r / 100) ** GRID)
    assert abs(pv_ts - pv_flat) / pv_ts < 1e-6


def test_tail_bump_barely_moves_the_single_rate():
    # a curve that spikes only past year 40 collapses close to its front level,
    # because those cash flows are already heavily discounted (the critique, quantified)
    flat = collapse.collapse_rate(GRID, np.full_like(GRID, 6.0), growth=0.0)
    bumped = collapse.collapse_rate(GRID, 6.0 + 12.0 * (GRID > 40), growth=0.0)
    assert bumped - flat < 0.6                          # < 60 bp despite a +12pp tail
