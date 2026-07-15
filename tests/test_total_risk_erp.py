"""Tests for the total-risk (Sharpe) single-name ERP with a VIX-term-structure market ERP."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import credit, total_risk_erp as tr

GRID = np.arange(1, 101, dtype=float)
IDX_VOL = [(0.08, 17.0), (0.25, 18.5), (0.5, 19.5), (1.0, 20.0), (3.0, 21.0), (5.0, 21.5)]
AN = {"AAA": 0.55, "AA": 0.70, "A": 1.05, "BBB": 1.42, "BB": 2.60, "B": 4.20, "CCC": 9.00}
IG = [(2, 0.55), (4, 0.65), (6, 0.75), (8.5, 0.85), (12.5, 0.95), (20, 1.05)]
TSY = [(1, 4.3), (2, 4.2), (5, 4.2), (10, 4.5), (20, 4.9), (30, 4.8)]


def _cg():
    return credit.build_from_knots(GRID, IG, 0.90, AN, TSY, np.linspace(1.6, 3.0, len(GRID)))


def test_market_erp_uses_the_vol_curve_then_glides_to_floor():
    floor = 2.0
    m = tr.build_market_erp_curve(GRID, IDX_VOL, floor)["market_erp"]
    assert np.isclose(m.loc[1.0], 20.0 ** 2 / 100.0)        # 1y = Martin of the 1y vol
    assert np.isclose(m.loc[5.0], 21.5 ** 2 / 100.0)        # 5y = Martin of the 5y vol (observed)
    assert m.loc[10.0] > floor and m.loc[100.0] < m.loc[10.0]  # glides down after
    assert abs(m.loc[100.0] - floor) < 0.05                 # reaches floor deep out


def test_single_name_never_below_market():
    cg = _cg()
    # a LOW-vol name (below the index) must still not price below the market
    stock_vol = [(0.08, 12.0), (0.5, 13.0), (1.0, 14.0), (5.0, 15.0)]   # < index everywhere
    R = tr.build_risk_ratio(GRID, stock_vol, IDX_VOL, "A", cg, "A")["R"]
    assert np.all(R >= 1.0 - 1e-12)                         # floored at 1
    m = tr.build_market_erp_curve(GRID, IDX_VOL, 2.0)["market_erp"].to_numpy()
    out = tr.single_name_erp(GRID, m, R.to_numpy())
    assert np.all(out["idiosyncratic"].to_numpy() >= -1e-12)          # never negative
    assert np.all(out["single_name_erp"].to_numpy() >= m - 1e-9)      # never below market


def test_higher_vol_name_prices_higher():
    cg = _cg()
    lo = [(0.08, 15.0), (1.0, 16.0), (5.0, 16.0)]
    hi = [(0.08, 30.0), (1.0, 32.0), (5.0, 33.0)]
    Rlo = tr.build_risk_ratio(GRID, lo, IDX_VOL, "BBB", cg, "B")["R"]
    Rhi = tr.build_risk_ratio(GRID, hi, IDX_VOL, "BBB", cg, "B")["R"]
    assert Rhi.loc[1.0] > Rlo.loc[1.0] >= 1.0              # low-vol name floors at the market
    assert np.isclose(Rhi.loc[1.0], 32.0 / 20.0)           # front is the vol ratio


def test_idiosyncratic_is_market_erp_times_R_minus_1():
    m = np.full_like(GRID, 3.0)
    R = np.linspace(1.0, 4.0, len(GRID))
    out = tr.single_name_erp(GRID, m, R)
    assert np.allclose(out["idiosyncratic"], 3.0 * (R - 1.0))
    assert np.allclose(out["single_name_erp"], out["market_erp"] + out["idiosyncratic"])


def test_elevator_lifts_R_toward_distress_past_ory():
    cg = _cg()
    sv = [(0.08, 22.0), (1.0, 23.0), (5.0, 23.0)]
    R = tr.build_risk_ratio(GRID, sv, IDX_VOL, "BBB", cg, "B", r_distress=6.0)["R"]  # ORY 40
    assert np.isclose(R.loc[20.0], R.loc[35.0])            # flat through maturity
    assert R.loc[60.0] > R.loc[40.0]                      # rises past ORY
    assert R.loc[95.0] <= 6.0 + 1e-9                      # bounded by the distressed level


def test_blended_market_erp_converges_slower_than_fast_glide():
    b = tr.build_market_erp_blended(GRID, IDX_VOL, 2.0, converge_year=30.0)["market_erp"]
    f = tr.build_market_erp_curve(GRID, IDX_VOL, 2.0)["market_erp"]
    assert np.isclose(b.loc[5.0], 21.5 ** 2 / 100)            # observed at 5y
    assert b.loc[15.0] > f.loc[15.0] and b.loc[20.0] > f.loc[20.0]  # stays elevated
    assert abs(b.loc[30.0] - 2.0) < 1e-9                      # fully bond by 30y
    assert np.all(np.diff(b.loc[5.0:30.0].to_numpy()) <= 1e-9)   # monotone down


def test_assemble_coe_v2_shape_and_invariants():
    G = np.arange(1, 151, dtype=float)
    rf = np.interp(G, [1, 10, 30, 150], [1.6, 2.4, 2.9, 2.9])
    mkt = tr.build_market_erp_blended(G, IDX_VOL, 1.57, converge_year=30.0)["market_erp"].to_numpy()
    coe = tr.assemble_coe_v2(G, rf, mkt, [(1.0, 30.0)], [(1.0, 20.0)], "BBB", "B")  # ORY 40
    assert list(coe.columns) == ["real_rf", "market_erp", "idiosyncratic", "company_erp", "real_coe"]
    assert np.all(coe["idiosyncratic"].to_numpy() >= -1e-9)               # never negative
    assert np.all(coe["company_erp"].to_numpy() >= coe["market_erp"].to_numpy() - 1e-9)  # >= market
    assert coe["real_coe"].loc[100.0] > coe["real_coe"].loc[40.0]         # obsolescence tail rises
    assert np.allclose(coe["real_coe"], coe["real_rf"] + coe["company_erp"])  # additive identity
