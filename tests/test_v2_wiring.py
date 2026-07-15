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
    assert np.isclose(d_yes["market_erp"].loc[3.0], 19.0 ** 2 / 100)
    assert not np.isclose(d_yes["market_erp"].loc[3.0], d_no["market_erp"].loc[3.0])


# ----------------------------- effective collapse ---------------------------------
def test_effective_collapse_additive_and_bounded():
    grid = np.arange(1, 151, dtype=float)
    rf = np.interp(grid, [1, 10, 30, 150], [1.6, 2.4, 2.9, 2.9])
    mkt = np.interp(grid, [1, 5, 30, 150], [4.0, 4.4, 3.0, 3.0])
    idio = np.interp(grid, [1, 40, 150], [1.5, 1.5, 6.0])          # elevator-like tail
    coe = rf + mkt + idio
    eff_coe = col.collapse_rate(grid, coe, growth=2.0)
    eff_rf = col.collapse_rate(grid, rf, growth=2.0)
    eff_rfmkt = col.collapse_rate(grid, rf + mkt, growth=2.0)
    eff_mkt = eff_rfmkt - eff_rf
    eff_company = eff_coe - eff_rf
    eff_idio = eff_company - eff_mkt
    # additive by construction
    assert abs((eff_rf + eff_mkt + eff_idio) - eff_coe) < 1e-9
    # each effective rate sits inside the curve's own range (PV-weighted average)
    assert coe.min() - 1e-6 <= eff_coe <= coe.max() + 1e-6
    assert eff_company > eff_mkt                                    # idio is positive
