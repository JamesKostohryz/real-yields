"""
tests/test_reanchor.py — hermetic tests for the automated monthly re-anchor.

Fully offline: FRED and the VIX1Y source chain are both stubbed. Nothing here touches the
network, so this runs as a gate BEFORE the live re-anchor is allowed to fetch anything.

The interesting tests are the last two. `test_no_input_is_inert` is the reason this file exists:
this project's signature failure is a correct value sitting inert while every gate reports
success, and it has now happened ten times. A suite that only checks arithmetic cannot see it,
because the arithmetic was always right. So the suite asserts INFLUENCE — perturb each input,
demand the published number move.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_erp_daily as BED       # noqa: E402
import reanchor as RA               # noqa: E402
import run_erp_daily as RR          # noqa: E402
import vol_scale_v3 as V3           # noqa: E402


# Real June inputs; the July figures are illustrative and only exercise the second replay step.
FAKE_FRED = {
    "2026-06-30": ({5: 1.885, 10: 2.204, 20: 2.745, 30: 2.73}, 3.83, 7450.03),
    "2026-07-31": ({5: 1.92, 10: 2.25, 20: 2.79, 30: 2.78}, 3.86, 7610.00),
}


def _fake_fred(asof, api_key=None):
    return FAKE_FRED[asof]


def _fake_vs(vix=22.94, tier=1):
    def f(asof_date, last_known_good=None, last_known_good_date=None,
          vix_val=None, vix3m_val=None, vix6m_val=None, log=print):
        return dict(vs=[float(x) for x in V3.vol_scale_curve_from_vix1y(vix)],
                    vs_1y=V3.vol_scale_from_vix1y(vix), vix1y=vix, value=vix,
                    tier_used=tier, source="STUB", alarm=False, message="stub",
                    stale_days=None, cross_check={})
    return f


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(RR, "fetch_daily_inputs", _fake_fred)
    monkeypatch.setattr(RR, "refresh_vol_scale", _fake_vs())


@pytest.fixture
def repo():
    """A throwaway copy of just the files the re-anchor reads."""
    d = tempfile.mkdtemp()
    shutil.copy(os.path.join(ROOT, "ERP_HELD_STATE_2026-06.json"), d)
    os.makedirs(os.path.join(d, "outputs"))
    os.makedirs(os.path.join(d, "history"))
    shutil.copy(os.path.join(ROOT, "outputs", "market_credit_latest.csv"),
                os.path.join(d, "outputs", "market_credit_latest.csv"))
    shutil.copy(os.path.join(ROOT, "outputs", "curve_latest.csv"),
                os.path.join(d, "outputs", "curve_latest.csv"))
    shutil.copy(os.path.join(ROOT, "history", "real_yield_curve_v3_MASTER.csv"),
                os.path.join(d, "history", "real_yield_curve_v3_MASTER.csv"))
    return d


# ------------------------------------------------------------------ the cost terminal rule

def test_cost_keeps_gliding_then_flatlines_at_the_floor():
    """James, 2026-08-19: the glide keeps running at its own rate until it reaches 0.25 and
    flatlines there. So nothing happens at 2026.5; the floor engages in mid-2032."""
    assert RA.cost_for(dt.date(2026, 8, 19)) == pytest.approx(0.4946, abs=1e-3)
    assert RA.cost_for(dt.date(2030, 1, 1)) == pytest.approx(0.3532, abs=1e-3)
    assert RA.cost_for(dt.date(2032, 1, 1)) > RA.COST_FLOOR
    assert RA.cost_for(dt.date(2033, 1, 1)) == pytest.approx(RA.COST_FLOOR)
    assert RA.cost_for(dt.date(2050, 1, 1)) == pytest.approx(RA.COST_FLOOR)


def test_breakeven1y_is_the_market_series_not_expected_inflation():
    """James's ruling 2026-08-19. The two are different quantities, not two estimates of one:
    the market breakeven carries the inflation risk premium and the TIPS liquidity premium, and
    a real discount rate needs the yield an investor can actually transact at. Guarding by
    ABSENCE, because the failure mode is a plausible-looking expected-inflation source creeping
    back in."""
    assert RA.BE1Y_CURVE_FILE.endswith("curve_latest.csv")
    for gone in ("BE1Y_WEDGE", "EXPINF1Y_ANCHOR", "BE1Y_SERIES", "BE1Y_ANCHOR"):
        assert not hasattr(RA, gone), (
            f"{gone} is back: breakeven1y has drifted to an expected-inflation construction")


def test_the_unfloored_glide_really_does_go_negative():
    """The floor is not decoration. Without it the overlay crosses zero in 2038."""
    assert BED.cost_of_year(2026.5) == pytest.approx(0.50, abs=1e-9)
    assert BED.cost_of_year(2040.0) < 0.0


# ------------------------------------------------------------------ the state machine walk

def test_one_replay_step_per_missed_month_not_one_per_reanchor():
    assert RA.months_to_replay("2026-06", (2026, 8)) == [dt.date(2026, 6, 30), dt.date(2026, 7, 31)]
    assert len(RA.months_to_replay("2026-07", (2026, 8))) == 1
    assert RA.months_to_replay("2026-12", (2027, 2))[0] == dt.date(2026, 12, 31)


def test_month_end_skips_weekends():
    assert RA.last_business_day(2026, 5).weekday() < 5
    assert RA.last_business_day(2026, 8).weekday() < 5


# ------------------------------------------------------------------ the happy path

def test_green_run_writes_a_vs_vector_and_the_resolver_picks_it_up(repo):
    res = RA.reanchor(root=repo, asof="2026-08-19")
    assert res["status"] == "written"
    st = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-08.json")))
    assert st["anchor_vintage"] == "2026-08"
    # THE POINT OF THE WHOLE EXERCISE: a 30-long vs(T), not June's scalar.
    assert isinstance(st["vs"], list) and len(st["vs"]) == 30
    assert st["cost"] == pytest.approx(0.4946, abs=1e-3)   # still gliding; floor is 0.25
    assert len(st["derivation"]["month_ends_replayed"]) == 2


def test_rerunning_the_same_month_is_a_noop(repo):
    RA.reanchor(root=repo, asof="2026-08-19")
    assert RA.reanchor(root=repo, asof="2026-08-19")["status"] == "noop"


def test_carried_inputs_are_carried_exactly(repo):
    prior = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-06.json")))
    RA.reanchor(root=repo, asof="2026-08-19")
    st = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-08.json")))
    for k in ("normalized_X4", "cpi_factor"):
        assert st[k] == prior[k], f"{k} must be carried unchanged until it has a live source"


def test_breakeven1y_comes_off_the_market_curve(repo):
    """RETIRED AND INVERTED 2026-08-19. This test used to assert the OPPOSITE -- that the curve
    file is never read -- which was the right guard while the target was the June plug and the
    wrong guard once the ruling settled the basis. Recorded rather than quietly rewritten: the
    68bp gap it was built to defend is the inflation risk premium plus the TIPS liquidity
    premium at one year, and that BELONGS in a market discount rate."""
    RA.reanchor(root=repo, asof="2026-08-19")
    st = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-08.json")))
    curve = pd.read_csv(os.path.join(repo, "outputs", "curve_latest.csv"))
    want = float(curve.loc[curve["maturity"] == 1.0, "breakeven"].iloc[0])
    assert st["breakeven1y"] == pytest.approx(want, abs=1e-6)


def test_a_missing_curve_file_falls_back_to_the_carry_loudly(repo):
    """Written, published, and AMBER. Refusing over one input would keep vs(T) out of
    production; going quiet would be the defect this module exists to prevent."""
    os.remove(os.path.join(repo, "outputs", "curve_latest.csv"))
    res = RA.reanchor(root=repo, asof="2026-08-19")
    assert res["status"] == "written"
    assert any("breakeven1y CARRIED" in m for m in res["amber"])


def test_a_front_constructed_point_is_flagged_every_month(repo):
    """No TIPS matures inside a year, so the 1-year real yield is extrapolated. That is a
    standing property of this tenor, so it is reported every month rather than accepted once."""
    res = RA.reanchor(root=repo, asof="2026-08-19")
    assert any("front-CONSTRUCTED" in m for m in res["amber"])


# ------------------------------------------------------------------ RED

def test_a_stale_vix_tier_is_refused(repo, monkeypatch):
    monkeypatch.setattr(RR, "refresh_vol_scale", _fake_vs(tier=5))
    with pytest.raises(RA.ReanchorRefused):
        RA.reanchor(root=repo, asof="2026-08-19")
    assert not os.path.exists(os.path.join(repo, "ERP_HELD_STATE_2026-08.json"))


def test_a_stale_credit_grid_is_refused(repo):
    p = os.path.join(repo, "outputs", "market_credit_latest.csv")
    old = (dt.datetime.now() - dt.timedelta(days=40)).timestamp()
    os.utime(p, (old, old))
    with pytest.raises(RA.ReanchorRefused):
        RA.reanchor(root=repo, asof="2026-08-19")


def test_an_out_of_band_input_is_refused(repo):
    prior = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-06.json")))
    with pytest.raises(RA.ReanchorRefused):
        RA.apply_guards(dict(vs_1y=9.9, corp_prem=1.0, breakeven1y=2.0, cost=0.5,
                             fey_in=6.0, D_in=25.0), prior, log=lambda *_: None)


def test_a_big_but_plausible_move_is_amber_not_red(repo):
    prior = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-06.json")))
    amber = RA.apply_guards(dict(vs_1y=0.9348, corp_prem=3.5, breakeven1y=2.76, cost=0.5,
                                 fey_in=6.02, D_in=24.72), prior, log=lambda *_: None)
    assert len(amber) == 1 and "corp_prem" in amber[0]


# ------------------------------------------------------------------ the one that matters

# Measured 2026-08-19. normalized_X4 and cpi_factor DO reach the published number, but only
# just: the plateau presets landed 2026-08-12 blend the normalized-earnings term to zero weight
# at 30 years and the effective rate is duration-collapsed at ~25 years, so almost all the
# weight sits where they have already been blended out. They are listed here with the measured
# sensitivity beside them so that if the presets ever change, this exemption becomes a visible
# lie a reader can catch rather than an omission nobody can see.
KNOWN_LOW_INFLUENCE = {"normalized_X4": 4.4, "cpi_factor": 2.2}   # bp per 10% / 5% bump


def test_no_input_is_inert(repo):
    RA.reanchor(root=repo, asof="2026-08-19")
    base = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-08.json")))
    reals, nom, sp = FAKE_FRED["2026-07-31"]

    def coe(state):
        real, ney = RR.construct_legs(state, reals, nom, sp)
        return BED.build_asof(real, ney, state["vs"], state["fey_in"], state["D_in"],
                              state["cost"], state["corp_prem"])["eff_coe"]

    b = coe(base)
    bumps = [("vs", 1.10), ("fey_in", 1.05), ("D_in", 1.05), ("cost", 1.20),
             ("breakeven1y", 1.10), ("normalized_X4", 1.10), ("cpi_factor", 1.05)]
    for key, mult in bumps:
        s = json.loads(json.dumps(base))
        s[key] = [x * mult for x in s[key]] if isinstance(s[key], list) else s[key] * mult
        moved_bp = abs(coe(s) - b) * 100
        floor_bp = 0.5 if key in KNOWN_LOW_INFLUENCE else 1.0
        assert moved_bp > floor_bp, (
            f"{key} moved the published eff_coe by only {moved_bp:.4f}bp — it is inert or "
            f"near-inert. Either it is not wired, or it belongs in KNOWN_LOW_INFLUENCE with "
            f"its measured sensitivity written down.")


def test_corp_prem_is_a_floor_so_it_only_bites_when_it_binds(repo):
    """corp_prem is the one input that is SUPPOSED to be inert most of the time — it is a floor
    under the ERP, and today's ERP sits far above it. Asserting it moves the number would be
    wrong; asserting it moves the number WHEN IT BINDS is the real test."""
    RA.reanchor(root=repo, asof="2026-08-19")
    base = json.load(open(os.path.join(repo, "ERP_HELD_STATE_2026-08.json")))
    reals, nom, sp = FAKE_FRED["2026-07-31"]

    def coe(cp):
        s = dict(base, corp_prem=cp)
        real, ney = RR.construct_legs(s, reals, nom, sp)
        return BED.build_asof(real, ney, s["vs"], s["fey_in"], s["D_in"], s["cost"], cp)["eff_coe"]

    assert coe(1.04) == pytest.approx(coe(1.50), abs=1e-9), "the floor should not bind today"
    assert coe(6.00) > coe(1.04) + 0.5, "the floor must bite once it is raised above the ERP"
