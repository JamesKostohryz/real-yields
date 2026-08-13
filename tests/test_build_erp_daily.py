"""Tests for the plateau-preset mechanism in build_erp_daily.py (landed 2026-08-12,
AEG-ERP-TASK6-BUILD-SPEC-2026-08-12.md sec.4)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_erp_daily as m


def _run(preset):
    return m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE, **m.JUNE_STATE, preset=preset)


def test_default_preset_is_B():
    assert m.PLATEAU_DEFAULT == "B"
    r = m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE, **m.JUNE_STATE)
    assert r["preset"] == "B"
    assert abs(r["preset_pure_risk"] - m.PLATEAU_PRESETS["B"]) < 1e-9


def test_unknown_preset_raises():
    try:
        m.build_asof(m.JUNE_TIPS, m.JUNE_NORM_EY, m.VS_JUNE, **m.JUNE_STATE, preset="D")
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
    # at T=30 the three presets must be cleanly separated and ordered A > B > C
    # (matching PLATEAU_PRESETS' own ordering), since each preset's pure-risk value
    # dominates the blend by then (plateau_w(30) == 1.0).
    a, b, c = _run("A"), _run("B"), _run("C")
    assert a["spot_erp"][29] > b["spot_erp"][29] > c["spot_erp"][29]
    # and each sits close to its own preset value (blend is full weight, only Rc added on top)
    for r, name in [(a, "A"), (b, "B"), (c, "C")]:
        target = m.PLATEAU_PRESETS[name]
        assert abs(r["spot_erp"][29] - target) < 0.5, f"{name}: {r['spot_erp'][29]} vs target {target}"


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
                      cost=m.JUNE_STATE["cost"], corp_prem=50.0, preset="C")
    assert min(r["spot_erp"]) >= 50.0 - 1e-9
