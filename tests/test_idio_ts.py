"""Tests for the unified idiosyncratic term (options front -> relative-bond -> elevator)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import credit, idio_ts

GRID = np.arange(1, 101, dtype=float)
AN = {"AAA": 0.55, "AA": 0.70, "A": 1.05, "BBB": 1.42, "BB": 2.10, "B": 3.40, "CCC": 7.50}
IG = [(2, 0.55), (4, 0.65), (6, 0.75), (8.5, 0.85), (12.5, 0.95), (20, 1.05)]
TSY = [(1, 4.3), (2, 4.2), (5, 4.2), (10, 4.5), (20, 4.9), (30, 4.8)]


def _cg():
    return credit.build_from_knots(GRID, IG, 0.90, AN, TSY, np.linspace(1.6, 2.9, len(GRID)))


def test_options_own_the_front_two_years():
    d = idio_ts.build_idiosyncratic_for_name(GRID, _cg(), "BBB", 1.5, "B", market_rating="A")
    assert np.allclose(d["idiosyncratic"].loc[:2.0].to_numpy(), 1.5)   # flat options front


def test_fades_to_relative_bond_multiple():
    cg = _cg()
    d = idio_ts.build_idiosyncratic_for_name(GRID, cg, "BBB", 1.5, "B",
                                             market_rating="A", M_b=1.5)
    # by ~6y (past fade_end, before ORY) idio == M_b*(issuer - market)
    exp = 1.5 * (float(np.interp(6, GRID, cg["spread_BBB"].to_numpy()))
                 - float(np.interp(6, GRID, cg["spread_A"].to_numpy())))
    assert abs(float(d["idiosyncratic"].loc[6.0]) - exp) < 1e-9


def test_tighter_than_market_is_negative():
    # an AA name vs an A market -> relative spread negative -> idio floor negative
    d = idio_ts.build_idiosyncratic_for_name(GRID, _cg(), "AA", 0.6, "A", market_rating="A")
    assert d["idiosyncratic"].loc[6.0] < 0.0


def test_rides_the_elevator_past_ory():
    d = idio_ts.build_idiosyncratic_for_name(GRID, _cg(), "BBB", 1.5, "B",
                                             market_rating="A")   # ORY=40
    i = d["idiosyncratic"]
    assert np.isclose(i.loc[40.0], i.loc[20.0])      # flat through maturity, up to ORY
    assert i.loc[60.0] > i.loc[45.0] > i.loc[40.0]   # rises just past ORY (the elevator)


def test_multiple_scales_the_bond_region_only():
    cg = _cg()
    a = idio_ts.build_idiosyncratic_for_name(GRID, cg, "BBB", 1.5, "B", M_b=1.0)["idiosyncratic"]
    b = idio_ts.build_idiosyncratic_for_name(GRID, cg, "BBB", 1.5, "B", M_b=2.0)["idiosyncratic"]
    assert np.isclose(a.loc[1.0], b.loc[1.0])        # front (options) unaffected by M_b
    assert b.loc[10.0] > a.loc[10.0] > 0             # bond region scales with M_b
