"""Tests for the VIX-term-structure market ERP front (options-front / bond-level)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import erp_ts

GRID = np.arange(1, 31, dtype=float)
# a normal-ish contango vol curve (short vol below long vol)
VOL_TS = [(0.025, 15.0), (0.083, 17.0), (0.25, 18.5), (0.5, 19.5), (1.0, 20.0)]
FLOOR = 1.2


def test_front_follows_martin_from_the_vol_curve():
    erp = erp_ts.build_market_erp_ts(GRID, VOL_TS, FLOOR, front=1.0, joint=5.0)
    # at 1y the ERP equals the Martin bound of the 1y implied vol
    assert np.isclose(erp["market_erp"].loc[1.0], 20.0 ** 2 / 100.0)


def test_level_is_bonds_from_the_joint_outward():
    erp = erp_ts.build_market_erp_ts(GRID, VOL_TS, FLOOR, front=1.0, joint=5.0)
    m = erp["market_erp"]
    assert np.allclose(m.loc[5.0:].to_numpy(), FLOOR)     # 5y+ is exactly the floor
    assert m.loc[3.0] > FLOOR and m.loc[3.0] < m.loc[1.0]  # transition sits between


def test_options_level_cannot_leak_past_the_joint():
    """James's requirement: a spike in the options level must not raise the ERP
    anywhere at/after the joint — bonds own that level."""
    calm = [(0.025, 15.0), (0.083, 17.0), (0.25, 18.5), (0.5, 19.5), (1.0, 20.0)]
    spike = [(0.025, 55.0), (0.083, 50.0), (0.25, 42.0), (0.5, 36.0), (1.0, 32.0)]
    e_calm = erp_ts.build_market_erp_ts(GRID, calm, FLOOR, joint=5.0)["market_erp"]
    e_spike = erp_ts.build_market_erp_ts(GRID, spike, FLOOR, joint=5.0)["market_erp"]
    # front differs a lot...
    assert e_spike.loc[1.0] > e_calm.loc[1.0] + 5.0
    # ...but from the joint on, identical (bond-anchored, options can't reach)
    assert np.allclose(e_spike.loc[5.0:].to_numpy(), e_calm.loc[5.0:].to_numpy())


def test_martin_conversion():
    assert np.isclose(erp_ts.martin_erp_pct(18.0), 3.24)   # VIX 18 -> 3.24%
