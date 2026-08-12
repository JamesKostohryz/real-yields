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


def test_bootstrap_spot_reads_the_published_curve(tmp_path):
    import csv
    curve = tmp_path / "curve.csv"
    with open(curve, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tenor", "fwd_real_yield", "fwd_erp", "fwd_coe",
                    "spot_real_yield", "spot_erp", "spot_coe"])
        w.writerow([30, 3.61, 1.92, 5.53, 3.00, 3.3355, 6.3355])
    vintage = tmp_path / "vintage.csv"
    with open(vintage, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vintage", "date", "eff_tips_ry", "eff_erp", "eff_coe", "duration"])
        w.writerow(["2026-08-12", "2026-08-12", 2.563, 3.9252, 6.4882, 23.82])
    value, date = collapse.bootstrap_spot(30, curve_path=str(curve), vintage_path=str(vintage))
    assert abs(value - 3.3355) < 1e-9
    assert date == "2026-08-12"


def test_bootstrap_spot_missing_tenor_raises(tmp_path):
    import csv
    curve = tmp_path / "curve.csv"
    with open(curve, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tenor", "spot_erp"])
        w.writerow([1, 3.9])
    try:
        collapse.bootstrap_spot(30, curve_path=str(curve), vintage_path=str(tmp_path / "nope.csv"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bootstrap_spot_missing_curve_file_raises(tmp_path):
    try:
        collapse.bootstrap_spot(30, curve_path=str(tmp_path / "nope.csv"),
                                vintage_path=str(tmp_path / "also_nope.csv"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_bootstrap_spot_vintage_is_best_effort(tmp_path):
    # vintage file missing -> still returns the value, with vintage=None
    import csv
    curve = tmp_path / "curve.csv"
    with open(curve, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tenor", "spot_erp"])
        w.writerow([30, 3.3355])
    value, date = collapse.bootstrap_spot(
        30, curve_path=str(curve), vintage_path=str(tmp_path / "nope.csv"))
    assert abs(value - 3.3355) < 1e-9
    assert date is None
