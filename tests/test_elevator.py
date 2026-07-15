"""Offline tests for the Merton Elevator (obsolescence cost-of-equity premium)."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import credit, elevator as ev

GRID = np.arange(1, 101, dtype=float)                       # 1..100y
IG = [(2, 0.55), (4, 0.65), (6, 0.75), (8.5, 0.85), (12.5, 0.95), (20, 1.05)]
AN = {"AAA": 0.45, "AA": 0.55, "A": 0.80, "BBB": 1.15, "BB": 2.10, "B": 3.40, "CCC": 7.50}
TSY = [(1, 4.3), (2, 4.2), (5, 4.2), (10, 4.5), (20, 4.9), (30, 4.8)]


def _cg():
    return credit.build_from_knots(GRID, IG, 0.90, AN, TSY, np.linspace(1.6, 2.9, len(GRID)))


# ------------------------------------------------------------------ the ramp p(t)
def test_progress_zero_before_onset_and_plateau_after():
    t = GRID
    p = ev.progress(t, ory=40.0, W=20.0)
    assert np.all(p[t <= 40] == 0.0)                       # nothing before onset
    assert np.all((p >= 0) & (p <= 1))                     # bounded
    assert np.isclose(p[t == 60][0], 1.0)                  # reaches 1 at H+W
    assert np.all(np.isclose(p[t >= 60], 1.0))             # plateau after
    rising = p[(t > 40) & (t < 60)]
    assert np.all(np.diff(rising) > 0)                     # strictly increasing between


def test_onset_override_is_a_pure_horizontal_shift():
    # James's rule 3: changing H just translates the same shape by that many years.
    t = np.linspace(0, 120, 1201)
    base = ev.progress(t, ory=40.0, W=25.0)
    shifted = ev.progress(t, ory=50.0, W=25.0)               # onset 10y later
    # shifted(t) must equal base(t-10) everywhere
    base_at_t_minus_10 = ev.progress(t - 10.0, ory=40.0, W=25.0)
    assert np.max(np.abs(shifted - base_at_t_minus_10)) < 1e-12


# ------------------------------------------------------------------ the elevator
def test_elevator_converges_to_common_floor():
    cg = _cg()
    floor = ev.floor_curve_from_grid(cg, "CCC", cushion_bp=500.0)
    start = cg["spread_BBB"].to_numpy()
    tab = ev.build_elevator(GRID, start, floor, ory=30.0, W=20.0, M=1.75)
    # before onset: no widening; at/after H+W: bond spread sits on the floor
    assert np.allclose(tab["bond_obs_increment"].to_numpy()[GRID <= 30], 0.0)
    tail = GRID >= 50
    assert np.allclose(tab["bond_spread"].to_numpy()[tail], floor[tail], atol=1e-9)
    # premium is exactly M x the incremental widening, and non-decreasing
    incr = tab["bond_obs_increment"].to_numpy()
    assert np.allclose(tab["obs_equity_premium"].to_numpy(), 1.75 * incr)
    assert np.all(np.diff(incr) >= -1e-12)


def test_lower_rating_reaches_floor_sooner():
    # fixed rate => fewer notches to fall => shorter descent. BBB before AAA (cat C).
    w_bbb = ev.derive_width("BBB", "CCC", rate=0.30)
    w_aaa = ev.derive_width("AAA", "CCC", rate=0.30)
    assert w_bbb < w_aaa
    assert np.isclose(w_bbb, 3 / 0.30) and np.isclose(w_aaa, 6 / 0.30)
    # a name already at/below the floor gets no elevator
    assert ev.derive_width("CCC", "CCC", rate=0.30) == 0.0


def test_category_speed_and_onset_ordering():
    cg = _cg()
    exp = ev.elevator_for_category(GRID, cg, "BBB", "C")     # onset 30, fast
    dur = ev.elevator_for_category(GRID, cg, "BBB", "A")     # onset 50, slow
    # at 60y the exposed name has ballooned well past the durable one
    e60 = exp["obs_equity_premium"].loc[60.0]
    d60 = dur["obs_equity_premium"].loc[60.0]
    assert e60 > d60 > 0.0
    # exposed onsets earlier: positive premium by 40y where durable is still zero
    assert exp["obs_equity_premium"].loc[40.0] > 0.0
    assert dur["obs_equity_premium"].loc[40.0] == 0.0


def test_onset_override_delays_the_premium():
    cg = _cg()
    base = ev.elevator_for_category(GRID, cg, "BBB", "B")                 # H=40
    late = ev.elevator_for_category(GRID, cg, "BBB", "B", ory_override=50)  # H=50
    # later onset => less (or equal) premium at any given horizon
    assert np.all(late["obs_equity_premium"].to_numpy()
                  <= base["obs_equity_premium"].to_numpy() + 1e-12)
    assert late["obs_equity_premium"].loc[45.0] < base["obs_equity_premium"].loc[45.0]


def test_equity_multiple_scales_linearly():
    cg = _cg()
    e1 = ev.elevator_for_category(GRID, cg, "BBB", "C", M=1.0)["obs_equity_premium"].to_numpy()
    e2 = ev.elevator_for_category(GRID, cg, "BBB", "C", M=2.0)["obs_equity_premium"].to_numpy()
    m = e1 > 1e-9
    assert np.allclose(e2[m] / e1[m], 2.0)


# ------------------------------------------------------------------ COE integration
def test_augment_coe_preserves_additive_identity():
    # a minimal COE components frame that satisfies the engine's identity
    n = len(GRID)
    coe = pd.DataFrame({
        "real_rf": np.full(n, 1.8), "market_erp": np.full(n, 2.2),
        "credit_relative": np.full(n, 0.3), "idiosyncratic": np.linspace(0.2, 1.4, n),
    }, index=GRID)
    coe["company_erp"] = coe["market_erp"] + coe["credit_relative"] + coe["idiosyncratic"]
    coe["real_coe"] = coe["real_rf"] + coe["company_erp"]

    obs = ev.elevator_for_category(GRID, _cg(), "BBB", "C")["obs_equity_premium"].to_numpy()
    aug = ev.augment_coe(coe, obs)

    lhs = (aug["real_rf"] + aug["market_erp"] + aug["credit_relative"]
           + aug["idiosyncratic"] + aug["obsolescence"])
    assert np.max(np.abs(lhs - aug["real_coe"])) < 1e-12          # identity holds w/ new term
    # company_erp absorbed the obsolescence term too
    assert np.allclose(aug["company_erp"],
                       coe["company_erp"].to_numpy() + obs)
    # and COE actually curls up in the tail where the base was flat
    assert aug["real_coe"].loc[70.0] > aug["real_coe"].loc[30.0]
