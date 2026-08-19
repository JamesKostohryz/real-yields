"""Offline tests for the v2 live-data wiring: single-name IV term structure, the
long-dated index extension, the vol-curve merge/dedup, and effective collapse.

The yfinance-touching functions are exercised with synthetic Ticker stand-ins so the
PARSING/assembly logic is covered without any network."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import volsurface as vs, company as comp, collapse as col


# ----------------------------- synthetic yfinance ---------------------------------
class _Chain:
    def __init__(self, calls, puts):
        self.calls, self.puts = calls, puts


class _FakeTicker:
    """Minimal yfinance.Ticker stand-in with a downward-sloping IV term structure."""
    def __init__(self, price=100.0, iv_by_days=None):
        self.price = price
        self._iv = iv_by_days or {30: 0.34, 90: 0.31, 182: 0.29,
                                  365: 0.27, 545: 0.255, 730: 0.24}
        self.fast_info = {"last_price": price}

    @property
    def options(self):
        # expiry date strings at each offered horizon
        import datetime as dt
        base = dt.date(2026, 7, 15)
        return [(base + dt.timedelta(days=d)).isoformat() for d in self._iv]

    def option_chain(self, exp):
        import datetime as dt
        d = (dt.date.fromisoformat(exp) - dt.date(2026, 7, 15)).days
        iv = min(self._iv.items(), key=lambda kv: abs(kv[0] - d))[1]
        strikes = self.price * np.array([0.95, 0.98, 1.0, 1.02, 1.05])
        leg = pd.DataFrame({"strike": strikes,
                            "impliedVolatility": np.full(len(strikes), iv)})
        return _Chain(leg.copy(), leg.copy())


def _patch_today(monkeypatch):
    """_atm_iv measures days-to-expiry from 'today'; pin it to the fixtures' base."""
    import datetime as _dt

    class _D(_dt.date):
        @classmethod
        def today(cls):
            return _dt.date(2026, 7, 15)
    monkeypatch.setattr(comp._dt if hasattr(comp, "_dt") else _dt, "date", _D, raising=False)


# ----------------------------- single-name IV term structure ----------------------
def test_equity_vol_ts_is_a_sorted_point_curve(monkeypatch):
    import datetime as _dt

    class _D(_dt.date):
        @classmethod
        def today(cls):
            return _dt.date(2026, 7, 15)
    monkeypatch.setattr(_dt, "date", _D)

    tk = _FakeTicker()
    ts = comp.fetch_equity_vol_ts(tk, tk.price)
    assert len(ts) >= 4
    tens = [t for t, _ in ts]
    assert tens == sorted(tens)                       # sorted by tenor
    assert all(5.0 < v < 60.0 for _, v in ts)         # in VOL POINTS, plausible
    # downward-sloping fixture -> front vol > back vol
    assert ts[0][1] > ts[-1][1]


def test_equity_vol_ts_falls_back_to_flat_point(monkeypatch):
    class _Empty(_FakeTicker):
        @property
        def options(self):
            return []
    ts = comp.fetch_equity_vol_ts(_Empty(), 100.0, fallback_vol=0.30)
    assert ts == [(1.0, 30.0)]                         # single flat 1y point from fallback


# ----------------------------- long-dated index extension -------------------------
def test_fetch_index_vol_ts_yf_uses_chain(monkeypatch):
    import datetime as _dt

    class _D(_dt.date):
        @classmethod
        def today(cls):
            return _dt.date(2026, 7, 15)
    monkeypatch.setattr(_dt, "date", _D)

    fake = _FakeTicker(price=5000.0,
                       iv_by_days={365: 0.19, 545: 0.195, 730: 0.20, 1095: 0.205})

    class _YF:
        Ticker = staticmethod(lambda sym: fake)
    monkeypatch.setitem(sys.modules, "yfinance", _YF)

    ts = vs.fetch_index_vol_ts_yf(days_list=(365, 545, 730, 1095))
    assert [t for t, _ in ts] == [1.0, 1.4932, 2.0, 3.0]
    assert all(15.0 < v < 25.0 for _, v in ts)         # vol points


def test_cme_hook_is_non_fatal_and_returns_list():
    assert vs.fetch_cme_settlement_vols() == []        # hook returns [] until wired


# ----------------------------- vol-curve merge / dedup ----------------------------
def test_assemble_merges_named_and_long_dated():
    named = {"VIXCLS": 17.0, "VXVCLS": 18.5, "VIX1Y": 20.0}
    extra = [(1.4932, 19.5), (2.0, 19.0), (3.0, 18.5)]
    ts = vs.assemble_vol_ts(named, extra_ts=extra)
    tens = [round(t, 3) for t, _ in ts]
    assert tens == sorted(tens)
    assert 2.0 in tens and 3.0 in tens                 # long-dated points present
    assert ts[-1][0] == 3.0                            # observed front now reaches 3y


def test_assemble_dedup_prefers_short_end_on_tenor_clash():
    named = {"VIX1Y": 20.0}                            # 1.0y from a clean index
    extra = [(1.0, 25.0)]                              # a clashing 1y LEAPS point
    ts = vs.assemble_vol_ts(named, extra_ts=extra)
    assert ts == [(1.0, 20.0)]                         # named short-end wins


def test_extension_pushes_observed_front_out():
    named = {"VIXCLS": 17.0, "VXVCLS": 18.5, "VIX1Y": 20.0}
    long = [(2.0, 19.5), (3.0, 19.0)]
    grid = np.arange(1, 151, dtype=float)
    d_no, ts_no = vs.build_v2_market_erp(grid, named, 2.0, converge_year=30.0)
    d_yes, ts_yes = vs.build_v2_market_erp(grid, named, 2.0, converge_year=30.0, extra_ts=long)
    # without the extension the observed front ends at 1y; with it, at 3y
    assert ts_no[-1][0] == 1.0 and ts_yes[-1][0] == 3.0
    # inside the extended window the ERP is the OBSERVED option value (Martin of the
    # 3y vol), replacing the split-the-distance extrapolation used without it
    # FORWARD convention -- see the note in tests/test_volsurface.py. The spot Martin value of
    # the 3y vol is recovered as the MEAN of the forwards out to 3y, not read off the 3y cell.
    assert np.isclose(d_yes["market_erp"].loc[1.0:3.0].mean(), 19.0 ** 2 / 100)
    assert not np.isclose(d_yes["market_erp"].loc[3.0], d_no["market_erp"].loc[3.0])


# ----------------------------- effective collapse ---------------------------------
# NOTE 2026-08-12 (AEG-ERP-Collapse-Function-AUDIT): the previous version of this test
# asserted eff_rf + eff_mkt + eff_idio == eff_coe, but eff_mkt and eff_idio are DEFINED
# as differences that telescope back to eff_coe -- the assertion is true for any numbers
# whatsoever, including nonsense from a broken collapse_rate, and validated nothing.
# Replaced with three real tests of collapse_rate's actual repricing behaviour, run
# through the same rf/mkt/idio decomposition datasources.py and run_company.py use.

def test_effective_collapse_flat_curve_reprices_to_itself():
    # A flat curve on each leg must collapse to THAT flat level, not merely satisfy the
    # additive identity (which holds even when the collapse is wrong).
    grid = np.arange(1, 151, dtype=float)
    rf = np.full_like(grid, 2.0)
    mkt = np.full_like(grid, 3.5)
    idio = np.full_like(grid, 1.0)
    coe = rf + mkt + idio
    eff_coe = col.collapse_rate(grid, coe, growth=2.0)
    eff_rf = col.collapse_rate(grid, rf, growth=2.0)
    eff_rfmkt = col.collapse_rate(grid, rf + mkt, growth=2.0)
    eff_mkt = eff_rfmkt - eff_rf
    eff_idio = (eff_coe - eff_rf) - eff_mkt
    assert abs(eff_rf - 2.0) < 1e-4
    assert abs(eff_mkt - 3.5) < 1e-4
    assert abs(eff_idio - 1.0) < 1e-4
    assert abs(eff_coe - 6.5) < 1e-4


def test_effective_collapse_single_terminal_cashflow_matches_bootstrap():
    # A single cash flow at the final tenor T prices only the cumulative discount factor
    # to T, so the collapsed flat rate must exactly reproduce the curve's own T-year spot
    # -- the parameter-free "bootstrap" reading (AEG-ERP-Collapse-Function-AUDIT object d)
    # -- independent of the growth/profile default.
    grid = np.arange(1, 31, dtype=float)
    rf = np.interp(grid, [1, 10, 30], [1.6, 2.4, 3.0])
    T = grid[-1]
    cf = np.zeros_like(grid); cf[-1] = 1.0
    eff_rf = col.collapse_rate(grid, rf, cashflows=cf)
    spot_rf_T = (float(np.prod(1 + rf / 100.0)) ** (1.0 / T) - 1.0) * 100.0
    assert abs(eff_rf - spot_rf_T) < 1e-6


def test_effective_collapse_bounded_by_curve_range():
    grid = np.arange(1, 151, dtype=float)
    rf = np.interp(grid, [1, 10, 30, 150], [1.6, 2.4, 2.9, 2.9])
    mkt = np.interp(grid, [1, 5, 30, 150], [4.0, 4.4, 3.0, 3.0])
    idio = np.interp(grid, [1, 40, 150], [1.5, 1.5, 6.0])          # elevator-like tail
    coe = rf + mkt + idio
    eff_coe = col.collapse_rate(grid, coe, growth=2.0)
    eff_rf = col.collapse_rate(grid, rf, growth=2.0)
    assert coe.min() - 1e-6 <= eff_coe <= coe.max() + 1e-6
    assert rf.min() - 1e-6 <= eff_rf <= rf.max() + 1e-6
