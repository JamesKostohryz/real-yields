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


def test_committed_june_reference_untouched():
    """The whole point: this module must NOT change the committed acceptance gate. Verified
    at the build_erp_daily.py level (VS_JUNE=0.9348 is untouched, unrelated to this module),
    but assert here too that vol_scale_v3 does not import/mutate anything from
    build_erp_daily at import time."""
    import build_erp_daily as b
    assert b.VS_JUNE == 0.9348
