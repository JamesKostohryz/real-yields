"""Tests for the plateau-preset MECHANISM in build_erp_daily.py (landed 2026-08-12,
AEG-ERP-TASK6-BUILD-SPEC-2026-08-12.md sec.4; re-pointed 2026-09-16 when the plateau became
credit-anchored).

These test the machinery -- the blend weight, the ramp, the clamp, the ordering -- not the
plateau's VALUE. They therefore run on the gate's own credit anchor (build_erp_daily.GATE),
which reproduces the pre-2026-09-16 preset-B plateau of 2.40 exactly, so a change to the
machinery stays distinguishable from a change to the market reading. The VALUES live in
tests/test_plateau_credit_anchor.py, against the live credit grid.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_erp_daily as m


def _run(preset):
    return m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE, **m.GATE, preset=preset)


def _plateau(preset):
    return m.plateau_from_credit(m.GATE_BBB, preset)


def test_default_preset_is_B():
    assert m.PLATEAU_DEFAULT == "B"
    r = m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE, **m.GATE)
    assert r["preset"] == "B"
    assert abs(r["preset_pure_risk"] - _plateau("B")) < 1e-9


def test_unknown_preset_raises():
    try:
        m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE, **m.GATE, preset="D")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_front_end_is_dominated_by_the_option_value_not_the_preset():
    # plateau_w(T<=3) == 0, so the preset is NEVER directly blended into the front end.
    # A small (few-bp) difference across presets still shows up at the front, because
    # the per-tenor snapshot uses fey_out (the day's UPDATED fair-EY state), and fey_out
    # = 0.7*fey_in + 0.3*eff_coe already depends on which preset drove eff_coe -- that
    # feedback predates this change (the file'''s own comment: "term-structure snapshot
    # uses the UPDATED fair_ey"). The invariant this test actually holds the code to:
    # that feedback is second-order (a few bp), not the preset leaking in directly
    # (which would show up as points, not basis points).
    a, b, c = _run("A"), _run("B"), _run("C")
    assert m.plateau_w(1) == 0.0 and m.plateau_w(2) == 0.0 and m.plateau_w(3) == 0.0
    for i in range(3):   # tenors 1, 2, 3
        assert abs(a["spot_erp"][i] - b["spot_erp"][i]) < 0.05
        assert abs(b["spot_erp"][i] - c["spot_erp"][i]) < 0.05


def test_long_end_orders_by_preset_and_ramps_in():
    # At T=30 the three presets must be cleanly separated and ordered BY THEIR PLATEAUS,
    # since each preset's pure-risk value dominates the blend by then (plateau_w(30) == 1.0).
    # THE DIRECTION FLIPPED ON 2026-09-16: it used to be A > B > C and is now C > B > A,
    # because James inverted the letters. The assertion reads the order out of the constants
    # so that it tests the mechanism rather than restating the convention.
    r = {p: _run(p) for p in ("A", "B", "C")}
    order = sorted(r, key=lambda p: _plateau(p))               # low plateau -> high
    got = [r[p]["spot_erp"][29] for p in order]
    assert got == sorted(got), f"year-30 ERP must order with the plateau {order}: {got}"
    assert len(set(got)) == 3, f"presets are not cleanly separated at T=30: {got}"
    # AND THE YEAR-30 POINT IS THE PLATEAU, EXACTLY, not merely near it. plateau_w(30) == 1.0,
    # so the blend is full weight and the published floor is an identity:
    #     spot_erp[30] == max(corp_prem, plateau + rate_response) + cost
    # This replaced a "within 0.5pp of the plateau" check on 2026-09-16. That check was loose
    # enough to pass while the corp_prem clamp was silently binding -- which it does here, on
    # preset A at the gate's deliberately low anchor -- and loose enough to pass if the cost
    # premium were added twice, which is the one mistake the credit-anchor spec singles out.
    for name in ("A", "B", "C"):
        d = r[name]["decomposition"]
        expect = max(d["corp_prem_floor"], _plateau(name) + d["rate_response"]) + d["cost_premium"]
        assert abs(r[name]["spot_erp"][29] - expect) < 1e-9, (
            f"{name}: year-30 ERP {r[name]['spot_erp'][29]} != "
            f"max(corp_prem {d['corp_prem_floor']}, plateau {_plateau(name)} + Rresp "
            f"{d['rate_response']}) + cost {d['cost_premium']} = {expect}")
        assert abs(d["erp_plateau"] - _plateau(name)) < 1e-12


def test_plateau_weight_shape():
    assert m.plateau_w(1) == 0.0
    assert m.plateau_w(3) == 0.0
    assert m.plateau_w(30) == 1.0
    assert m.plateau_w(60) == 1.0   # flat past 30, matching gdecay/gap_decay convention
    assert 0.0 < m.plateau_w(15) < 1.0
    # monotonically non-decreasing
    ws = [m.plateau_w(t) for t in range(1, 31)]
    assert all(ws[i] <= ws[i + 1] + 1e-12 for i in range(len(ws) - 1))


def test_floor_still_clamps_underneath_every_preset():
    # a preset value below the live floor should never publish below the floor.
    # (Not reachable with today's presets/floor, so this exercises the clamp directly.)
    r = m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE,
                      fey_in=m.JUNE_STATE["fey_in"], D_in=m.JUNE_STATE["D_in"],
                      cost=m.JUNE_STATE["cost"], corp_prem=50.0, preset="C",
                      bbb_spread_30y=m.GATE_BBB)
    assert min(r["spot_erp"]) >= 50.0 - 1e-9
