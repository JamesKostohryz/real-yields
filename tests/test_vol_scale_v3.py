"""Tests for vol_scale_v3.py -- the VIX1Y-primary vs construction (session 13, 2026-08-18).
Network tests are marked and skip gracefully if offline; the pure-math tests need no network."""
import math
import sys
import os
import unittest.mock as mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vol_scale_v3 as v3


# ---------------------------------------------------------------- soft clip (pure math)
def test_soft_clip_identity_inside_knees():
    for raw in [0.70, 0.80, 0.9348, 1.00, 1.30, 1.55]:
        assert abs(v3.soft_clip_scalar(raw) - raw) < 1e-9


def test_soft_clip_matches_reference_table():
    # transcribed from AEG-Project/tools/volscale_clip_bounds_analysis.py section [8] output
    ref = {0.9348: 0.9348, 1.80: 1.7916, 2.00: 1.9571, 2.37: 2.1769, 3.00: 2.3621, 5.00: 2.4824}
    for raw, expect in ref.items():
        got = v3.soft_clip_scalar(raw)
        assert abs(got - expect) < 5e-4, (raw, got, expect)


def test_soft_clip_slope_is_one_at_each_knee():
    eps = 1e-6
    for knee in (v3.VS_KNEE_LO, v3.VS_KNEE_HI):
        d = (v3.soft_clip_scalar(knee * math.exp(eps)) - v3.soft_clip_scalar(knee * math.exp(-eps))) / (2 * eps * knee)
        assert abs(d - 1.0) < 1e-3


def test_soft_clip_strictly_monotone():
    xs = np.exp(np.linspace(math.log(0.01), math.log(100), 5000))
    ys = v3.soft_clip(xs)
    assert np.all(np.diff(ys) > 0)


def test_soft_clip_never_exceeds_asymptote():
    for raw in [1e-6, 1e-3, 1e3, 1e6]:
        s = v3.soft_clip_scalar(raw)
        assert v3.VS_LO <= s <= v3.VS_HI


# ---------------------------------------------------------------- guard rail
def test_guard_rail_passes_small_move():
    ok, pct, _ = v3.guard_rail_check(22.9)
    assert ok and abs(pct - (22.9 / 22.62 - 1)) < 1e-9


def test_guard_rail_trips_large_move():
    ok, pct, _ = v3.guard_rail_check(25.0)
    assert not ok and pct > 0.05


# ---------------------------------------------------------------- six-tier chain (mocked, no network)
def test_tier1_used_when_all_agree():
    with mock.patch.object(v3, "fetch_cboe_history_csv", return_value=("2026-08-18", 22.94)), \
         mock.patch.object(v3, "fetch_cboe_delayed_json", return_value=("2026-08-18", 22.94)), \
         mock.patch.object(v3, "fetch_yahoo_chart", return_value=(None, 22.94)):
        r = v3.fetch_vix1y_sixtier(asof_date="2026-08-18")
    assert r["tier_used"] == 1 and not r["alarm"] and r["value"] == 22.94


def test_cross_check_alarm_on_disagreement():
    with mock.patch.object(v3, "fetch_cboe_history_csv", return_value=("2026-08-18", 22.9)), \
         mock.patch.object(v3, "fetch_cboe_delayed_json", return_value=("2026-08-18", 24.5)), \
         mock.patch.object(v3, "fetch_yahoo_chart", side_effect=Exception("down")):
        r = v3.fetch_vix1y_sixtier(asof_date="2026-08-18")
    assert r["tier_used"] == "alarm" and r["value"] is None and r["alarm"]


def test_tier4_rebuild_when_1_to_3_fail():
    with mock.patch.object(v3, "fetch_cboe_history_csv", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_cboe_delayed_json", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_yahoo_chart", side_effect=Exception("down")):
        r = v3.fetch_vix1y_sixtier(asof_date="2026-08-18", vix_val=15.19, vix3m_val=19.27,
                                    vix6m_val=21.37, last_known_good=22.75,
                                    last_known_good_date="2026-08-17")
    assert r["tier_used"] == 4 and r["alarm"] and r["value"] is not None


def test_tier5_hold_stale_when_no_short_tenors_supplied():
    with mock.patch.object(v3, "fetch_cboe_history_csv", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_cboe_delayed_json", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_yahoo_chart", side_effect=Exception("down")):
        r = v3.fetch_vix1y_sixtier(asof_date="2026-08-18", last_known_good=22.75,
                                    last_known_good_date="2026-08-17")
    assert r["tier_used"] == 5 and r["value"] == 22.75 and r["alarm"]


def test_tier6_refuses_after_stale_threshold():
    with mock.patch.object(v3, "fetch_cboe_history_csv", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_cboe_delayed_json", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_yahoo_chart", side_effect=Exception("down")):
        r = v3.fetch_vix1y_sixtier(asof_date="2026-08-18", last_known_good=22.75,
                                    last_known_good_date="2026-08-14")  # 4 days stale
    assert r["tier_used"] == "refuse" and r["value"] is None


def test_all_tiers_fail_no_fallback_refuses():
    with mock.patch.object(v3, "fetch_cboe_history_csv", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_cboe_delayed_json", side_effect=Exception("down")), \
         mock.patch.object(v3, "fetch_yahoo_chart", side_effect=Exception("down")):
        r = v3.fetch_vix1y_sixtier(asof_date="2026-08-18")
    assert r["tier_used"] == "refuse" and r["value"] is None


# ---------------------------------------------------------------- end to end (pure math, no network)
def test_vol_scale_from_vix1y_end_to_end():
    vs = v3.vol_scale_from_vix1y(22.94, median=22.62)
    assert abs(vs - 22.94 / 22.62) < 1e-9   # inside the knees, so this is exact identity


def test_legacy_scalar_path_is_bit_identical():
    """The June scalar reference is SUPERSEDED as the published number but retained as a
    back-compatibility assertion: if it moves, the ENGINE moved, not the input."""
    import build_erp_daily as b
    assert b.VS_JUNE == 0.9348
    r = b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, b.VS_JUNE, **b.JUNE_STATE)
    assert abs(r["eff_erp"] - 3.396) < 0.01 and abs(r["eff_coe"] - 5.765) < 0.01  # re-baselined 2026-09-04 to match build_erp_daily.JUNE_EFF (PCHIP landing); see LANDED-ERP-Three-Preset-Publish-2026-09-04.md


# ============================================================================================
# THE SEVEN MANDATED SELF-TESTS FOR THE 2026-08-18 vs(T) LANDING.
# These are the tests James required to be IN THE TEST FILE, not merely run once in a tool.
# Standing rule, obeyed in order: the feasibility floor runs FIRST, against any pinned pair,
# before anything else is measured.
# ============================================================================================
import csv                                                            # noqa: E402

_CBOE = "/tmp/cb_VIX1Y.csv"


def _record():
    """The CBOE VIX1Y history if this machine has the pull; otherwise a spanning synthetic set
    so the property tests still run everywhere. Tests that need the real record say so."""
    if os.path.exists(_CBOE):
        out = []
        for r in csv.DictReader(open(_CBOE)):
            try:
                out.append(float(r["CLOSE"]))
            except (ValueError, KeyError):
                pass
        if out:
            return np.array(out), True
    return np.linspace(9.0, 60.0, 400), False


def _unclipped(vix1y):
    th = (v3.VS_THETA_VOL / 100.0) ** 2
    d1 = float(v3.vs_damp(1.0, v3.VS_KAPPA))
    dT = v3.vs_damp(np.minimum(v3.VS_TENORS, v3.VS_FREEZE_T), v3.VS_KAPPA)
    v0 = th + ((float(vix1y) / 100.0) ** 2 - th) / d1
    v0n = th + ((v3.VIX1Y_MEDIAN / 100.0) ** 2 - th) / d1
    return np.sqrt(th + (v0 - th) * dT) / np.sqrt(th + (v0n - th) * dT)


# ---- [1] FEASIBILITY FLOOR -- RUN THIS BEFORE MEASURING ANYTHING ELSE, FOREVER --------------
def test_1_feasibility_floor_value():
    """100*sqrt(theta*(1-d(1))) = 11.95 for the pinned pair kappa=0.325, theta=31.25."""
    assert abs(v3.feasibility_floor() - 11.9503) < 1e-3


def test_1_feasibility_floor_catches_the_old_defective_pair():
    """Regression guard on the check itself: the superseded (kappa=0.75, theta=30) pair must
    still be reported as having a floor of 16.34, which is what made 149 real days
    unrepresentable. If this ever stops failing, the check has been broken."""
    assert abs(v3.feasibility_floor(theta_vol=30.0, kappa=0.75) - 16.34) < 0.01


def test_1_every_input_clears_the_floor():
    x, real = _record()
    floor = v3.feasibility_floor()
    eff = np.array([v3.VIX1Y_MEDIAN * v3.vol_scale_from_vix1y(v) for v in x])
    assert (eff >= floor - 1e-9).all(), f"min effective VIX1Y {eff.min()} below floor {floor}"


def test_1_clip_asymptote_is_tied_to_the_floor():
    """The defect this landing repairs: the clip must not be able to produce a state the
    construction cannot represent. Enforced at import; asserted here so it stays enforced."""
    floor, eff_lo = v3.assert_clip_floor_consistent()
    assert eff_lo >= floor - 1e-9
    assert v3.VS_LO > 0.40, "lower asymptote regressed to the pre-repair value"


def test_1_clip_floor_tie_fires_on_an_inconsistent_pair():
    import pytest
    with pytest.raises(AssertionError):
        v3.assert_clip_floor_consistent(lo=0.40)


# ---- [2] NORMAL VIX1Y IN -> vs(T) = 1.000000 FLAT, MAX DEVIATION EXACTLY ZERO ---------------
def test_2_normal_market_gives_flat_unity():
    c = v3.vol_scale_curve_from_vix1y(v3.VIX1Y_MEDIAN)
    assert float(np.abs(c - 1.0).max()) == 0.0


# ---- [3] vs(1y) == soft_clip(VIX1Y/median) TO MACHINE PRECISION ON EVERY DAY ----------------
def test_3_year_one_equals_the_scalar_on_every_day():
    x, _ = _record()
    worst = max(abs(v3.vol_scale_curve_from_vix1y(v)[0] - v3.vol_scale_from_vix1y(v)) for v in x)
    assert worst < 1e-12, worst


# ---- [4] vs(1y) IS THE EXTREME ELEMENT OF THE UNCLIPPED VECTOR ------------------------------
def test_4_year_one_is_the_extreme_element():
    """This is what makes clipping the ANCHOR sufficient: bounding vs(1y) bounds every tenor,
    so the clip never has to touch tenors 2-30 and never flattens the term structure."""
    x, _ = _record()
    for v in x:
        u = _unclipped(v)
        if v >= v3.VIX1Y_MEDIAN:
            assert u[0] >= u.max() - 1e-12, v
        else:
            assert u[0] <= u.min() + 1e-12, v


# ---- [5] vs STAYS STRICTLY INSIDE THE ASYMPTOTES AT EVERY DAY AND TENOR ---------------------
def test_5_inside_asymptotes_in_sample():
    x, _ = _record()
    for v in x:
        c = v3.vol_scale_curve_from_vix1y(v)
        assert c.min() > v3.VS_LO and c.max() < v3.VS_HI, v


def test_5_inside_asymptotes_far_out_of_sample():
    """The clip exists for days that have not happened; measuring it only in sample is
    standing suspicion #2. Also pins the anchor clip's saturation behaviour: it must stay
    SLOPED at extreme inputs, not collapse flat at the cap the way elementwise clipping does."""
    for v in (1e-6, 0.5, 2.0, 5.0, 150.0, 500.0, 1e4):
        c = v3.vol_scale_curve_from_vix1y(v)
        assert c.min() > v3.VS_LO and c.max() < v3.VS_HI, v
    hot = v3.vol_scale_curve_from_vix1y(150.0)
    assert hot[0] - hot[29] > 0.5, "anchor clip degenerated to a flat capped curve"


def test_5_monotone_toward_normal_with_tenor():
    """Mean reversion, stated as a property: a curve above normal must decay toward 1 with
    tenor and one below normal must rise toward it. If this inverts, the sign is wrong."""
    hi = v3.vol_scale_curve_from_vix1y(45.86)
    lo = v3.vol_scale_curve_from_vix1y(13.31)
    assert np.all(np.diff(hi[:10]) < 0) and hi[9] > 1.0
    assert np.all(np.diff(lo[:10]) > 0) and lo[9] < 1.0


def test_5_frozen_past_year_ten():
    c = v3.vol_scale_curve_from_vix1y(45.86)
    assert float(np.abs(c[9:] - c[9]).max()) == 0.0


# ---- [6] PRESET INVARIANCE: THE vs-TO-eff_erp SENSITIVITY IS IDENTICAL UNDER A, B AND C -----
def test_6_preset_invariance_to_nine_decimals():
    import build_erp_daily as b
    for v_from, v_to in ((b.VS_JUNE, b.VS_AUG),
                         (0.90, 1.10),
                         (1.00, list(v3.vol_scale_curve_from_vix1y(45.86)))):
        d = [b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, v_to, preset=p, **b.JUNE_STATE)["eff_erp"]
             - b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, v_from, preset=p, **b.JUNE_STATE)["eff_erp"]
             for p in ("A", "B", "C")]
        assert max(d) - min(d) < 5e-10, d


def test_6_presets_themselves_are_untouched():
    """James confirmed twice that nothing in this landing touches the presets."""
    import build_erp_daily as b
    assert b.PLATEAU_PRESETS == {"A": 3.35, "B": 2.40, "C": 2.05}
    assert b.PLATEAU_DEFAULT == "B"
    assert b.C == 7.5 and b.VARP == 3.0 and b.VOLNORM == 13.0
    assert [round(b.gdecay(t), 4) for t in (1, 10, 20, 30)] == [1.12, 1.0, 0.9, 0.85]
    assert [round(b.plateau_w(t), 4) for t in (1, 3, 10, 20, 30)] == [0.0, 0.0, 0.35, 0.75, 1.0]
    assert round(b.rvbase(1), 4) == 0.195 and round(b.rvbase(30), 4) == 0.108


# ---- [7] THE LANDED ACCEPTANCE REFERENCE, AND THE ENGINE'S EXACT BACKWARD COMPATIBILITY -----
def test_7_landed_reference_reproduces():
    import build_erp_daily as b
    r = b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, b.VS_AUG, **b.JUNE_STATE)
    # Re-baselined 2026-09-04 to match build_erp_daily.AUG_EFF (PCHIP landing); this file's copy
    # of the acceptance reference had gone stale relative to run_gate()'s own copy, caught by
    # ci-on-push running this file's pure-math tests for the first time. See
    # LANDED-ERP-Three-Preset-Publish-2026-09-04.md.
    assert abs(r["eff_tips"] - 2.370) < 0.01
    assert abs(r["eff_erp"] - 3.468) < 0.01
    assert abs(r["eff_coe"] - 5.837) < 0.01
    assert max(abs(r["spot_coe"][i] - b.SPOT_COE_REF_AUG[i]) for i in range(30)) < 0.01


def test_7_embedded_vector_matches_the_live_construction():
    """The gate embeds VS_AUG as a literal so it stays hermetic. This is the check that the
    literal is what vol_scale_v3 actually produces -- otherwise the gate could drift away from
    the code it is meant to be gating, which is standing suspicion #1 exactly."""
    import build_erp_daily as b
    live = v3.vol_scale_curve_from_vix1y(b.VS_AUG_VIX1Y)
    assert float(np.abs(np.array(b.VS_AUG) - live).max()) < 1e-10


def test_7_constant_vector_equals_scalar_exactly():
    import build_erp_daily as b
    for vsv in (0.6, 0.9348, 1.4, 2.2):
        for p in ("A", "B", "C"):
            a = b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, vsv, preset=p, **b.JUNE_STATE)
            c = b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, np.full(30, vsv), preset=p, **b.JUNE_STATE)
            assert a["eff_erp"] == c["eff_erp"]
            assert max(abs(a["spot_coe"][i] - c["spot_coe"][i]) for i in range(30)) == 0.0


def test_7_wrong_length_vector_is_rejected():
    import pytest
    import build_erp_daily as b
    for bad in (np.full(29, 1.0), np.full(31, 1.0)):
        with pytest.raises(ValueError):
            b.build_asof(b.JUNE_TIPS, b.JUNE_NORM_EY, bad, **b.JUNE_STATE)


def test_7_hermetic_gate_runs_green():
    import build_erp_daily as b
    b.run_gate()


# ---- the re-anchor now actually calls vol_scale_v3, which was the point of the landing ------
def test_reanchor_returns_the_term_structure():
    import run_erp_daily as R
    with mock.patch.object(v3, "fetch_vix1y_sixtier",
                           return_value=dict(value=22.94, tier_used=1, source="mock",
                                             alarm=False, stale_days=0, cross_check=None,
                                             message="mock")):
        out = R.refresh_vol_scale("2026-08-18", log=lambda *a, **k: None)
    assert len(out["vs"]) == 30
    assert abs(out["vs_1y"] - 22.94 / 22.62) < 1e-9
    assert abs(out["vs"][0] - out["vs_1y"]) < 1e-12
