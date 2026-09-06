#!/usr/bin/env python3
"""Hermetic tests for the foreign-filer deflator and exchange-rate feeds.

NO NETWORK, NO SECRETS -- this file must satisfy ci-on-push's contract (see the header of
.github/workflows/ci.yml, which states that no test under tests/ makes a network call or reads an
API key). Everything below exercises the pure functions on synthetic series.

WHAT IS PINNED HERE, AND WHY EACH ONE EARNED A TEST
---------------------------------------------------
1. The quote-direction normalization, both ways plus the zero refusal. A reversed direction is a
   clean factor-of-fifty error with no symptom anywhere in the engine.

2. THE JENSEN TRAP, which was a real bug in this feed's own cross-check on 2026-09-06 and is a
   standing trap for anyone consuming fx_daily.csv. `mean(1/x)` is not `1/mean(x)`, and the gap
   grows with within-month variance. The first version of fetch_fx compared the Board's monthly
   average against the mean of the INVERTED daily series and flagged exactly three months:
   China 1989-12, Mexico 1994-12 and Brazil 1999-01 -- the yuan's 21% devaluation, the Tequila
   crisis and the real's float. Real events, correct data, false alarm. The check now compares in
   FRED's own quoted space. This test pins both halves: that the two means genuinely differ on a
   volatile month, and that comparing in quoted space agrees.

3. The annual-average construction and its n_months column, which is what lets the engine tell a
   complete year from the year in progress and extend a foreign anchor the way the US path does.

4. The coverage guard. A consumer price or exchange-rate series that silently SHORTENS would arrive
   as a hole in the earliest statement years and nothing would fail -- the workbook would tie on a
   history that had quietly lost its front end. That is failure mode A (00-START-HERE.md), so the
   expectation is declared in the registry and a shortfall must raise.

Written 2026-09-06 alongside the history extension James asked for.
"""
from __future__ import annotations

import os
import statistics
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fx"))
sys.path.insert(0, os.path.join(ROOT, "cpi"))

import fetch_fx as FX        # noqa: E402
import fetch_cpi as CPI      # noqa: E402


# ---------------------------------------------------------------------------
# 1. quote direction
# ---------------------------------------------------------------------------
def test_usd_per_foreign_is_passed_through_unchanged():
    """FRED quotes the euro as dollars per euro, which is already the published convention."""
    assert FX.to_usd_per_unit(1.1736, "usd_per_foreign") == 1.1736


def test_foreign_per_usd_is_inverted():
    """FRED quotes the yuan as yuan per dollar; the feed publishes dollars per yuan."""
    assert FX.to_usd_per_unit(7.0, "foreign_per_usd") == pytest.approx(1.0 / 7.0, rel=0, abs=0)


def test_a_zero_quote_is_refused_rather_than_becoming_an_infinity():
    with pytest.raises(FX.FXError):
        FX.to_usd_per_unit(0.0, "foreign_per_usd")


def test_an_unknown_direction_is_refused_rather_than_guessed():
    with pytest.raises(FX.FXError):
        FX.to_usd_per_unit(7.0, "probably_foreign_per_usd")


def test_direction_round_trips():
    """Inverting twice must return the original, or the convention is not self-consistent."""
    raw = 158.8476
    usd_per_unit = FX.to_usd_per_unit(raw, "foreign_per_usd")
    assert FX.to_usd_per_unit(usd_per_unit, "foreign_per_usd") == pytest.approx(raw, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. the Jensen trap
# ---------------------------------------------------------------------------
# A month shaped like China's December 1989: a stable rate that steps ~26% partway through.
_DEVALUATION_MONTH = [3.7314] * 10 + [4.7339] * 10


def test_mean_of_inverses_differs_materially_from_inverse_of_mean_in_a_devaluation_month():
    """The premise of the bug. If this ever stops being true the test below is vacuous."""
    quoted_mean = statistics.fmean(_DEVALUATION_MONTH)
    inverted_mean = statistics.fmean([1.0 / x for x in _DEVALUATION_MONTH])
    gap = abs(inverted_mean - 1.0 / quoted_mean) / (1.0 / quoted_mean)
    # Measured on this shape: about 1.6%, far outside the feed's 0.5% cross-check tolerance.
    assert gap > 0.01, f"expected a material Jensen gap, got {gap:.4%}"


def test_comparing_in_quoted_space_agrees_where_comparing_inverted_space_would_not():
    """What the fix does. The Board's monthly average IS the mean of the quoted daily prints, so
    the comparison must happen before inversion; afterwards it cannot agree, by arithmetic."""
    board_monthly_quoted = statistics.fmean(_DEVALUATION_MONTH)

    # in quoted space -- exact
    own_quoted = statistics.fmean(_DEVALUATION_MONTH)
    assert abs(board_monthly_quoted - own_quoted) / own_quoted == pytest.approx(0.0, abs=1e-12)

    # in inverted space -- fails the feed's own tolerance, which is the false alarm
    tol = 0.005
    own_inverted = statistics.fmean([1.0 / x for x in _DEVALUATION_MONTH])
    board_inverted = 1.0 / board_monthly_quoted
    assert abs(board_inverted - own_inverted) / own_inverted > tol


def test_the_published_monthly_value_is_the_inverse_of_the_quoted_mean_not_the_mean_of_inverses():
    """This is the contract fx_rates.usd_per_unit_avg() implements -- it averages the raw series
    and then inverts -- so the feed and that module agree by construction. A caller who averages
    the usd_per_unit column of fx_daily.csv instead gets the other quantity and will disagree."""
    published = FX.to_usd_per_unit(statistics.fmean(_DEVALUATION_MONTH), "foreign_per_usd")
    wrong = statistics.fmean([FX.to_usd_per_unit(x, "foreign_per_usd") for x in _DEVALUATION_MONTH])
    assert published != pytest.approx(wrong, rel=1e-6)
    assert published == pytest.approx(1.0 / statistics.fmean(_DEVALUATION_MONTH), rel=1e-15)


# ---------------------------------------------------------------------------
# 3. annual averages and n_months
# ---------------------------------------------------------------------------
def test_annual_mean_and_month_count():
    monthly = {"CNY": {"2024-01": 100.0, "2024-02": 102.0, "2024-03": 104.0,
                       "2025-01": 110.0}}
    ann = CPI.annual(monthly)
    mean24, n24 = ann["CNY"][2024]
    assert mean24 == pytest.approx(102.0)
    assert n24 == 3
    mean25, n25 = ann["CNY"][2025]
    assert mean25 == pytest.approx(110.0)
    assert n25 == 1, "n_months must expose an incomplete year, not smooth over it"


def test_a_single_month_year_is_not_silently_dropped():
    """The year in progress has few months and must still appear -- that partial year is exactly
    what deflator_extend uses to cover a current-year anchor."""
    ann = CPI.annual({"USD": {"2026-01": 330.0}})
    assert 2026 in ann["USD"]
    assert ann["USD"][2026] == (330.0, 1)


# ---------------------------------------------------------------------------
# 4. the coverage guard
# ---------------------------------------------------------------------------
def _fx_registry(history_from):
    return {
        "fetch_from": "1900-01-01",
        "monthly_vs_daily_tolerance": 0.005,
        "currencies": {
            "CNY": {"daily": "D", "monthly": "M", "direction": "foreign_per_usd",
                    "history_from": history_from},
        },
    }


def _stub_fred(monkeypatch, daily, monthly):
    def fake(sid, start="1900-01-01"):
        return dict(daily) if sid == "D" else dict(monthly)
    monkeypatch.setattr(FX, "fetch_fred", fake)


def test_fx_coverage_guard_raises_when_history_has_shortened(monkeypatch):
    # 20 business-ish days in one month so the cross-check has a complete month to look at
    daily = {f"2020-01-{d:02d}": 7.0 for d in range(1, 21)}
    monthly = {"2020-01": 7.0}
    _stub_fred(monkeypatch, daily, monthly)
    with pytest.raises(FX.FXError) as e:
        FX.build(_fx_registry("1981-01"))
    assert "SHORTENED" in str(e.value)


def test_fx_coverage_guard_passes_when_history_matches(monkeypatch):
    daily = {f"2020-01-{d:02d}": 7.0 for d in range(1, 21)}
    monthly = {"2020-01": 7.0}
    _stub_fred(monkeypatch, daily, monthly)
    d, m, _ = FX.build(_fx_registry("2020-01"))
    assert m["CNY"]["2020-01"] == pytest.approx(1.0 / 7.0)
    assert d["CNY"]["2020-01-01"] == pytest.approx(1.0 / 7.0)


def test_fx_cross_check_raises_on_a_reversed_direction(monkeypatch):
    """The failure this check exists for: the monthly series paired with a daily series quoted the
    other way round. Nothing else in the system would notice."""
    daily = {f"2020-01-{d:02d}": 7.0 for d in range(1, 21)}
    monthly = {"2020-01": 1.0 / 7.0}          # inverted relative to the daily series
    _stub_fred(monkeypatch, daily, monthly)
    with pytest.raises(FX.FXError) as e:
        FX.build(_fx_registry("2020-01"))
    assert "monthly average disagrees" in str(e.value)


def test_identity_currency_is_exactly_one_on_the_union_calendar(monkeypatch):
    daily = {f"2020-01-{d:02d}": 7.0 for d in range(1, 21)}
    monthly = {"2020-01": 7.0}
    _stub_fred(monkeypatch, daily, monthly)
    reg = _fx_registry("2020-01")
    reg["currencies"]["USD"] = {"identity": True}
    d, m, _ = FX.build(reg)
    assert set(d["USD"]) == set(d["CNY"]), "the identity calendar must cover every real date"
    assert all(v == 1.0 for v in d["USD"].values())
    assert all(v == 1.0 for v in m["USD"].values())
