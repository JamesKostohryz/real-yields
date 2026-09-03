"""Tests for the VIX-term-structure MARKET equity risk premium.

AMENDED 2026-09-02. Four tests were removed with the code they covered: the single-name
(total-risk / constant-Sharpe) construction -- build_risk_ratio(), single_name_erp() and the
elevator ramp that lifted the risk ratio toward distress past ORY. James ruled that there is ONE
approved method for a company's idiosyncratic risk premium, the four-block risk score in
aeg-valuation/idio/, and that the retired ones must not be referred to anywhere.

The removed tests asserted, among other things, that the idiosyncratic term is "never negative"
and that a single name "never prices below the market". Those were properties of the retired
construction and are NOT properties of the approved one, which is an increment centred on zero:
14 of the 16 onboarded names carry a DISCOUNT of about -1pp. Anyone reinstating a test like that
here would be re-asserting the thing the ruling overturned.

What is left is the market ERP itself, which is untouched, plus the surviving two-leg
assemble_coe_v2(). See asfp/total_risk_erp.py's docstring.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import total_risk_erp as tr

GRID = np.arange(1, 101, dtype=float)
IDX_VOL = [(0.08, 17.0), (0.25, 18.5), (0.5, 19.5), (1.0, 20.0), (3.0, 21.0), (5.0, 21.5)]
def test_market_erp_uses_the_vol_curve_then_glides_to_floor():
    floor = 2.0
    m = tr.build_market_erp_curve(GRID, IDX_VOL, floor)["market_erp"]
    assert np.isclose(m.loc[1.0], 20.0 ** 2 / 100.0)        # 1y = Martin of the 1y vol
    # FORWARD convention, settled 2026-08-19 -- see the note in tests/test_volsurface.py.
    # Note that the 1y assertion above passes unchanged: forward and spot coincide at the first
    # tenor and diverge after it, which is exactly why this defect sat here looking like a small
    # numerical disagreement rather than a units error.
    assert np.isclose(m.loc[1.0:5.0].mean(), 21.5 ** 2 / 100.0)   # spot at 5y = mean of forwards
    assert m.loc[10.0] > floor and m.loc[100.0] < m.loc[10.0]  # glides down after
    assert abs(m.loc[100.0] - floor) < 0.05                 # reaches floor deep out


def test_blended_market_erp_converges_slower_than_fast_glide():
    b = tr.build_market_erp_blended(GRID, IDX_VOL, 2.0, converge_year=30.0)["market_erp"]
    f = tr.build_market_erp_curve(GRID, IDX_VOL, 2.0)["market_erp"]
    assert np.isclose(b.loc[1.0:5.0].mean(), 21.5 ** 2 / 100)   # spot at 5y = mean of forwards
    assert b.loc[15.0] > f.loc[15.0] and b.loc[20.0] > f.loc[20.0]  # stays elevated
    assert abs(b.loc[30.0] - 2.0) < 1e-9                      # fully bond by 30y
    assert np.all(np.diff(b.loc[5.0:30.0].to_numpy()) <= 1e-9)   # monotone down


def test_assemble_coe_v2_publishes_the_two_house_view_legs_and_nothing_else():
    """The engine reads real_rf and market_erp from this table. It must carry no company-specific
    column: a company's idiosyncratic premium comes from aeg-valuation/idio/, added inside the
    engine, and a second one published here is how five constructions of one quantity came to
    exist at once (register item A6)."""
    G = np.arange(1, 151, dtype=float)
    rf = np.interp(G, [1, 10, 30, 150], [1.6, 2.4, 2.9, 2.9])
    mkt = tr.build_market_erp_blended(G, IDX_VOL, 1.57, converge_year=30.0)["market_erp"].to_numpy()
    coe = tr.assemble_coe_v2(G, rf, mkt, [(1.0, 30.0)], [(1.0, 20.0)], "BBB", "B")
    assert list(coe.columns) == ["real_rf", "market_erp"]
    for retired in ("idiosyncratic", "company_erp", "real_coe", "single_name_erp"):
        assert retired not in coe.columns, (
            "%s is a retired single-name column; see asfp/total_risk_erp.py" % retired)
    assert np.allclose(coe["real_rf"].to_numpy(), rf)
    assert np.allclose(coe["market_erp"].to_numpy(), mkt)
    assert len(coe) == len(G)


def test_the_retired_single_name_constructors_are_gone():
    """A named guard, so a future session cannot reintroduce them by accident and find the suite
    still green. James, 2026-09-02: the retired methods "should not be in the engine or referred
    to by the engine in any way"."""
    for gone in ("build_risk_ratio", "single_name_erp"):
        assert not hasattr(tr, gone), (
            "%s was retired on 2026-09-02 -- there is ONE approved idiosyncratic method and it "
            "is not in this repository" % gone)
