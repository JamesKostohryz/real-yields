"""
Equity-index implied-vol TERM STRUCTURE for the market-ERP front (v2).

Assembles a vol curve from whatever sources are reachable at runtime, then hands it
to total_risk_erp.build_market_erp_blended. Sources, in order of reliability:

  FRED  VIXCLS (30d), VXVCLS (3-month)          — the guaranteed base
  CBOE  VIX9D, VIX6M, VIX1Y                      — best-effort term-structure indices
  CME   long-dated SPX/E-mini settlement ATM IV  — best-effort ~5y extension (hook)

Everything degrades gracefully: with only the two FRED points the blend still works
(it just hands off to bonds sooner); each extra point pushes the observed front out.
The pure pieces are unit-tested; the network fetches run on the CI runner (open
network) and are wrapped so a failure never breaks the weekly job.
"""
from __future__ import annotations

import numpy as np

# nominal tenor (years) of each source series
TENORS = {
    "VIX9D": 9 / 365.0, "VIXCLS": 30 / 365.0, "VXVCLS": 93 / 365.0,
    "VIX6M": 0.5, "VIX1Y": 1.0, "CME5Y": 5.0,
}


def assemble_vol_ts(points, extra_ts=None):
    """points: {source: vol_points} keyed by the named short-end series; extra_ts: an
    optional list of explicit [(tenor_years, vol_points), …] long-dated points (SPX/SPY
    LEAPS, CME settlements). Returns one sorted, tenor-deduped curve; on a tenor clash
    the named short-end point wins (its index is cleaner). Drops non-positive/unknown."""
    ts = [(TENORS[k], float(v)) for k, v in points.items()
          if k in TENORS and v is not None and float(v) > 0]
    for t, v in (extra_ts or []):
        if t is not None and v is not None and float(v) > 0:
            ts.append((round(float(t), 4), float(v)))
    ts.sort()
    # dedupe by tenor (keep the first = shortest-source-wins after the stable sort)
    seen, dedup = set(), []
    for t, v in ts:
        key = round(t, 3)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((t, v))
    return dedup


def fetch_fred_vols(fred_key, fetch_fred_latest):
    """The reliable base: VIXCLS (30d) and VXVCLS (3-month) via the existing FRED reader."""
    out = {}
    for sid in ("VIXCLS", "VXVCLS"):
        try:
            v, _ = fetch_fred_latest(fred_key, sid)
            if v and v > 0:
                out[sid] = float(v)
        except Exception:
            pass
    return out


def fetch_cboe_vols(timeout=15):
    """Best-effort CBOE term-structure indices (VIX9D, VIX6M, VIX1Y) from the public
    daily-price CSVs. Returns {} if unreachable — verify endpoints at runtime."""
    out = {}
    base = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
    try:
        import requests
        for k in ("VIX9D", "VIX6M", "VIX1Y"):
            try:
                r = requests.get(base.format(k), timeout=timeout)
                if not r.ok:
                    continue
                rows = [ln for ln in r.text.strip().splitlines() if ln]
                cells = rows[-1].split(",")               # last row = latest; CLOSE is last col
                val = float(cells[-1])
                if 3.0 < val < 200.0:
                    out[k] = val
            except Exception:
                pass
    except Exception:
        pass
    return out


# calendar-day horizons for the long-dated index extension (1y..3y)
INDEX_TS_DAYS = (365, 545, 730, 1095)


def fetch_index_vol_ts_yf(days_list=INDEX_TS_DAYS, symbols=("^SPX", "SPY"), timeout=None):
    """Long-dated index ATM implied-vol TERM STRUCTURE via option chains (the reliable
    'futures/LEAPS' workhorse): [(years, vol_points), …] out to ~3y.

    Tries each symbol in order (^SPX cash index first, SPY ETF as the liquid fallback)
    and takes a robust near-the-money ATM IV at each target horizon via the same reader
    used for single names. This is what pushes the OBSERVED market-ERP front past the
    1y VIX index out toward the 2-3y region before the bond blend takes over. Returns []
    if unreachable — the ERP then just hands off to bonds sooner. CI-runner only."""
    from . import company as comp
    for sym in symbols:
        try:
            import yfinance as yf
            tk = yf.Ticker(sym)
            fast = tk.fast_info
            price = float(fast.get("last_price") or fast.get("lastPrice") or 0.0)
            if price <= 0:
                continue
            pts = []
            for d in days_list:
                try:
                    iv = comp._atm_iv(tk, price, target_days=int(d))
                except Exception:
                    iv = None
                if iv is not None and 0.05 <= iv <= 2.0:
                    pts.append((round(d / 365.0, 4), round(iv * 100.0, 3)))
            if pts:
                pts.sort()
                return pts
        except Exception:
            continue
    return []


def fetch_cme_settlement_vols(disc_rate_pct=4.5, timeout=20, log=print):
    """Long-dated ATM implied vols from CME ES option settlements (quarterly expiries to
    ~5y), extending the observed front past the ~3y LEAPS reach. Returns
    [(years, vol_points), …] or [] on any failure. Delegates to asfp.cme, which is
    heavily logged and hard-validated so a wrong endpoint/format degrades safely."""
    try:
        from . import cme
        return cme.fetch_cme_settlement_vols(disc_rate_pct=disc_rate_pct,
                                             timeout=timeout, log=log)
    except Exception as e:
        log(f"  cme: skipped (non-fatal): {e}")
        return []


def floor_from_credit_grid(cg, wedge=1.0, lgd=0.60, hazard=0.30, liquidity=0.30, tenor=30.0):
    """Market-ERP floor (%) = bond risk premium + equity convergence premium.
    bond risk premium = IG index spread − expected loss − liquidity haircut."""
    ig = float(np.interp(tenor, cg.index.to_numpy(), cg["ig_index_spread"].to_numpy()))
    bond_rp = ig - lgd * hazard - liquidity
    return bond_rp + wedge


def build_v2_market_erp(grid, vols, floor, converge_year=30.0, extra_ts=None):
    """Assemble the full vol curve (named short-end `vols` + optional long-dated
    `extra_ts` LEAPS/CME points) and build the blended market ERP (percent) on `grid`.
    Returns (DataFrame, vol_ts). Requires >= 2 vol points."""
    from . import total_risk_erp as tr
    vol_ts = assemble_vol_ts(vols, extra_ts=extra_ts)
    if len(vol_ts) < 2:
        raise ValueError(f"need >=2 vol points, got {vol_ts}")
    df = tr.build_market_erp_blended(grid, vol_ts, floor, converge_year=converge_year)
    return df, vol_ts
