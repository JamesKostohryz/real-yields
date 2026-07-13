"""Offline tests for the per-rating credit grid construction."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import credit

GRID = np.arange(1, 31, dtype=float)
IG_KNOTS = [(2, 0.5), (4, 0.7), (6, 0.9), (8.5, 1.1), (12.5, 1.3), (20, 1.5)]
ANCHORS = {"AA": 0.6, "A": 1.0, "BBB": 1.5}
TSY = [(1, 4.0), (2, 4.1), (5, 4.3), (10, 4.6), (20, 4.9), (30, 5.0)]
REAL_FWD = np.linspace(1.5, 2.8, 30)


def g():
    return credit.build_from_knots(GRID, IG_KNOTS, 1.0, ANCHORS, TSY, REAL_FWD)


def test_shape_scales_to_rating_level():
    d = g()
    shape10 = np.interp(10, [k for k, _ in IG_KNOTS], [v for _, v in IG_KNOTS])
    # overall = 1.0, so A (anchor 1.0) reproduces the raw IG shape
    assert abs(d.loc[10, "spread_A"] - shape10) < 1e-9
    # BBB (1.5) = shape * 1.5 ; AA (0.6) = shape * 0.6
    assert abs(d.loc[10, "spread_BBB"] - shape10 * 1.5) < 1e-9
    assert abs(d.loc[10, "spread_AA"] - shape10 * 0.6) < 1e-9


def test_rating_ordering_everywhere():
    d = g()
    assert (d["spread_AA"] < d["spread_A"]).all()
    assert (d["spread_A"] < d["spread_BBB"]).all()
    assert (d["spread_AA"] > 0).all()


def test_real_cost_of_debt_is_sum():
    d = g()
    for r in ["AA", "A", "BBB"]:
        assert np.max(np.abs(d[f"real_cod_{r}"] - (d["real_fwd"] + d[f"spread_{r}"]))) < 1e-12


def test_flat_extrapolation_beyond_last_knot():
    d = g()
    assert abs(d.loc[30, "ig_index_spread"] - 1.5) < 1e-9   # held flat past 20y
    assert abs(d.loc[1, "ig_index_spread"] - 0.5) < 1e-9    # held flat before 2y


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    d = g()
    print(d[["treasury_nominal", "ig_index_spread", "spread_AA", "spread_A",
             "spread_BBB", "real_cod_BBB"]].loc[[1, 2, 5, 10, 20, 30]].round(3).to_string())
