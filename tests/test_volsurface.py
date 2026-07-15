"""Tests for the v2 vol-term-structure assembly and market-ERP wiring (pure parts)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import credit, volsurface as vs

GRID = np.arange(1, 151, dtype=float)
AN = {"AAA": 0.55, "AA": 0.70, "A": 1.05, "BBB": 1.42, "BB": 2.60, "B": 4.20, "CCC": 9.00}
IG = [(2, 0.55), (4, 0.65), (6, 0.75), (8.5, 0.85), (12.5, 0.95), (20, 1.05)]
TSY = [(1, 4.3), (2, 4.2), (5, 4.2), (10, 4.5), (20, 4.9), (30, 4.8)]


def _cg():
    return credit.build_from_knots(GRID, IG, 0.90, AN, TSY, np.linspace(1.6, 3.2, len(GRID)))


def test_assemble_sorts_and_drops_bad_points():
    ts = vs.assemble_vol_ts({"VIX1Y": 20.0, "VIXCLS": 17.0, "VXVCLS": 0.0, "junk": 9})
    assert ts == [(30 / 365.0, 17.0), (1.0, 20.0)]        # sorted by tenor, zeros/unknowns dropped


def test_floor_is_bond_risk_plus_wedge():
    cg = _cg()
    f0 = vs.floor_from_credit_grid(cg, wedge=0.0)
    f1 = vs.floor_from_credit_grid(cg, wedge=1.0)
    assert np.isclose(f1 - f0, 1.0)                        # wedge adds directly
    assert f0 > 0                                          # bond risk premium positive


def test_build_v2_needs_two_points_and_blends_to_150y():
    cg = _cg()
    floor = vs.floor_from_credit_grid(cg, wedge=1.0)
    vols = {"VIXCLS": 17.0, "VXVCLS": 18.5, "VIX1Y": 20.0, "CME5Y": 21.5}
    df, ts = vs.build_v2_market_erp(GRID, vols, floor, converge_year=30.0)
    assert len(ts) == 4 and ts[-1][0] == 5.0              # 5y observed front
    assert np.isclose(df["market_erp"].loc[5.0], 21.5 ** 2 / 100)
    assert abs(df["market_erp"].loc[30.0] - floor) < 1e-9  # converged to floor by 30y
    assert abs(df["market_erp"].loc[150.0] - floor) < 1e-9  # flat to 150y


def test_build_v2_rejects_too_few_points():
    cg = _cg()
    try:
        vs.build_v2_market_erp(GRID, {"VIXCLS": 17.0}, vs.floor_from_credit_grid(cg))
        assert False, "should have raised"
    except ValueError:
        pass
