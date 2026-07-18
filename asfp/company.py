"""
Per-company data for the cost-of-equity overlay.

Runs in the ticker-triggered job: given a ticker, pull fundamentals (leverage)
and the option chain (asset vol + idiosyncratic variance) via yfinance, and
emit company_<ticker>.csv for the Sheet.

The two load-bearing numbers — economic leverage and the Merton pass-through k —
are pure functions, unit-tested offline against AT&T.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.stats import norm
    _NCDF = norm.cdf
except Exception:                       # tiny fallback if scipy is absent
    import math
    _NCDF = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ----------------------------------------------------------------- leverage
def economic_leverage(total_debt, cash, pensions, minority, market_equity):
    """Economic NFO basis (matches the model's λ₀ / L).

    NFO = (total debt incl. capitalized leases − cash) + underfunded pensions
          + minority interest.  L = NFO/(NFO+E),  λ₀ = NFO/E.
    All arguments in the same currency units.
    """
    nfo = (total_debt - cash) + pensions + minority
    L = nfo / (nfo + market_equity)
    lam0 = nfo / market_equity
    return dict(nfo=nfo, market_equity=market_equity, L=L, lambda0=lam0)


# ----------------------------------------------------------------- Merton k
def merton_omega(L, sigma_V, T=10.0, r=0.0):
    """Equity elasticity Ω_E = N(d1)/(1−L), equity as a call on assets.
    L = D/V (debt/asset value); σ_V = asset volatility."""
    L = min(max(L, 1e-6), 0.999)
    d1 = (-np.log(L) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    return _NCDF(d1) / (1.0 - L)


def merton_k(base_k, L_i, sigma_V_i, L_mkt, sigma_V_mkt, T=10.0, r=0.0):
    """Per-name credit→equity pass-through, scaled off the market average."""
    return base_k * merton_omega(L_i, sigma_V_i, T, r) / merton_omega(L_mkt, sigma_V_mkt, T, r)


# ------------------------------------------------------- asset vol from options
def asset_vol_from_equity(equity_vol, L):
    """De-lever the equity (option-implied) vol to an asset vol: σ_V ≈ σ_E·(1−L)."""
    return equity_vol * (1.0 - L)


def pick_equity_vol(iv, rv, lo=0.05, hi=2.0, default=0.25):
    """Choose the equity vol from an option-implied read `iv` and a realized read `rv`,
    guarding hard against degenerate quotes. Prefer a plausible IV; else a plausible
    realized vol; else a sane default. A near-zero/stale quote (e.g. 0.2%) or an
    implausibly high one is NEVER selected — that would collapse the risk ratio to 1
    and silently zero out the idiosyncratic premium."""
    if iv is not None and lo <= iv <= hi:
        return float(iv)
    if rv is not None and lo <= rv <= hi:
        return float(rv)
    return float(default)


def idiosyncratic_variance(equity_var, market_var, avg_correlation):
    """Firm-specific variance the market can't diversify away.
    Martin–Wagner: the stock's own risk-neutral variance minus the average
    stock's; the average ≈ market_var / avg_correlation. IDIO add-on ~ ½ of it.
    """
    avg_stock_var = market_var / max(avg_correlation, 1e-3)
    idio = 0.5 * (equity_var - avg_stock_var)
    return max(idio, 0.0)


# --------------------------------------------------------- yfinance pulls (runner)
def fetch_company(ticker, avg_correlation=0.35):
    """Pull fundamentals + options via yfinance and assemble the company inputs.
    Runs in the job (needs network + yfinance). Returns a dict."""
    import yfinance as yf
    tk = yf.Ticker(ticker)

    bs = tk.balance_sheet                       # most-recent column = latest FY
    def bget(*names):
        for n in names:
            if n in bs.index:
                v = bs.loc[n].dropna()
                if len(v):
                    return float(v.iloc[0])
        return 0.0
    total_debt = bget("Total Debt")
    cash = bget("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    pensions = bget("Non Current Pension And Other Postretirement Benefit Plans",
                    "Pensionand Other Post Retirement Benefit Plans Current")
    minority = bget("Minority Interest")

    fast = tk.fast_info
    price = float(fast.get("last_price") or fast.get("lastPrice"))
    shares = float(fast.get("shares") or bget("Share Issued"))
    market_equity = price * shares

    lev = economic_leverage(total_debt, cash, pensions, minority, market_equity)

    # equity vol: option-implied ATM IV when the chain is fresh/plausible,
    # else realized vol from ~1y of prices (robust to stale/after-hours option
    # quotes that come back zero — e.g. a liquid name reading ~0% by mistake).
    iv = _atm_iv(tk, price, target_days=365)
    rv = _realized_vol(tk)
    equity_vol = pick_equity_vol(iv, rv)
    sigma_V = asset_vol_from_equity(equity_vol, lev["L"])

    # v2: the single-name option-implied vol TERM STRUCTURE (1m..2y) that the
    # risk ratio R_i(t) needs at the front. Falls back to the flat 1y point (as a
    # single-element curve) when the long-dated chain is thin. In vol POINTS.
    equity_vol_ts = fetch_equity_vol_ts(tk, price, fallback_vol=equity_vol)

    return dict(ticker=ticker, price=price, **lev,
                equity_vol=equity_vol, sigma_V=sigma_V,
                equity_vol_ts=equity_vol_ts,
                avg_correlation=avg_correlation)


def fetch_smile(tk, price, target_days=365, band=(0.70, 1.30)):
    """OTM implied-vol SMILE around the money for the expiry nearest `target_days`:
    (strikes_ascending, ivs_decimal, forward≈price) or None. Puts below the forward,
    calls above (the OTM side each), filtered to `band` moneyness and plausible IV.
    Used by the skew diagnostic; best-effort, CI-runner only."""
    import datetime as _dt
    try:
        exps = tk.options
    except Exception:
        return None
    if not exps:
        return None
    exp = min(exps, key=lambda e: abs((_dt.date.fromisoformat(e) - _dt.date.today()).days - target_days))
    try:
        chain = tk.option_chain(exp)
    except Exception:
        return None
    F = float(price)
    lo, hi = band[0] * F, band[1] * F
    pts = {}
    for leg, side in ((chain.puts, "p"), (chain.calls, "c")):
        try:
            leg = leg.dropna(subset=["impliedVolatility"])
        except Exception:
            continue
        for K, iv in zip(leg["strike"].astype(float), leg["impliedVolatility"].astype(float)):
            if not (0.03 <= iv <= 2.0):
                continue
            if side == "p" and lo <= K < F:
                pts[float(K)] = float(iv)              # OTM put
            elif side == "c" and F <= K <= hi:
                pts.setdefault(float(K), float(iv))    # OTM call
    if len(pts) < 5:
        return None
    ks = sorted(pts)
    return ks, [pts[k] for k in ks], F


def fetch_smiles(tk, price, days_list=(182, 365, 730, 1095, 1825), band=(0.60, 1.50)):
    """Multi-tenor option SMILES for the skew-ERP engine: {tenor_years: (strikes, ivs, F)}
    at each horizon in `days_list` for which a plausible smile exists. Single names usually
    reach ~1-2y; the index (SPX/SPY LEAPS + CME) reaches ~3-5y. Skips tenors with no chain."""
    out = {}
    for d in days_list:
        sm = fetch_smile(tk, price, target_days=int(d), band=band)
        if sm:
            out[round(d / 365.0, 4)] = sm
    return out


def realized_skew(ticker, lookback_years=15):
    """Physical (realized) skew for the φ dial: annualized down-semivariance minus
    up-semivariance of monthly log returns, in percent — the realized analog of the
    option-implied corridor. Returns dict(down, up, corridor, n) in percent variance, or
    None. φ ≈ realized_corridor / implied_corridor, estimated per name on the runner."""
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(period=f"{int(lookback_years)}y", interval="1mo")
        c = h["Close"].dropna()
        if len(c) < 36:
            return None
        r = np.log(c / c.shift(1)).dropna().to_numpy()
    except Exception:
        return None
    mu = float(np.mean(r))
    dn = r[r < mu] - mu
    up = r[r >= mu] - mu
    # semivariances, annualized (×12 for monthly), in percent
    down = float(np.sum(dn * dn) / len(r) * 12.0 * 100.0)
    upv = float(np.sum(up * up) / len(r) * 12.0 * 100.0)
    return {"down": round(down, 3), "up": round(upv, 3),
            "corridor": round(down - upv, 3), "n": len(r)}


def skew_diag(ticker, target_days=365):
    """Skew diagnostic for one ticker: pull the smile, run the corridor down/up variance
    split, return dict(ticker, atm, k_down, k_up, k_var, skew, n) in annual variance, or
    None. Non-breaking — pure measurement, nothing in the model depends on it."""
    import yfinance as yf
    from . import skew as sk
    tk = yf.Ticker(ticker)
    try:
        fast = tk.fast_info
        price = float(fast.get("last_price") or fast.get("lastPrice") or 0.0)
    except Exception:
        price = 0.0
    if price <= 0:
        return None
    sm = fetch_smile(tk, price, target_days)
    if not sm:
        return None
    ks, ivs, F = sm
    d = sk.skew_price(ks, ivs, F, target_days / 365.0)
    d.update(ticker=ticker, atm=float(np.interp(F, ks, ivs)), n=len(ks))
    return d


# tenors (calendar days) at which we sample the single-name IV term structure
EQUITY_TS_DAYS = (30, 90, 182, 365, 545, 730)


def fetch_equity_vol_ts(tk, price, days_list=EQUITY_TS_DAYS, fallback_vol=None):
    """Single-name ATM implied-vol TERM STRUCTURE: [(years, vol_points), …].

    Samples the option chain at several target horizons (default 1m..2y), taking a
    robust near-the-money ATM IV at each (via `_atm_iv`). Points that come back
    empty/implausible are dropped. Returns vol in POINTS (e.g. 27.5 = 27.5%), sorted
    by tenor. If nothing survives, returns a single flat point at 1y from
    `fallback_vol` (the realized-vol-hardened equity_vol) so the caller always has a
    usable curve. Runs on the CI runner (needs yfinance); wrapped non-fatally."""
    out = []
    for d in days_list:
        try:
            iv = _atm_iv(tk, price, target_days=int(d))
        except Exception:
            iv = None
        if iv is not None and 0.05 <= iv <= 2.0:
            out.append((round(d / 365.0, 4), round(iv * 100.0, 3)))
    out.sort()
    if not out:
        fv = fallback_vol if (fallback_vol and 0.05 <= fallback_vol <= 2.0) else 0.25
        out = [(1.0, round(fv * 100.0, 3))]
    return out


def _atm_iv(tk, price, target_days=365, band=(0.02, 3.0), window=0.15, n=4):
    """At-the-money implied vol near `target_days`, hardened against bad quotes.

    Drops NaN / zero / implausible IVs (outside `band`), keeps only strikes within
    `window` moneyness of spot, and medians the nearest `n` per leg. Returns None
    if nothing plausible survives (caller then falls back to realized vol)."""
    import datetime as _dt
    try:
        exps = tk.options
    except Exception:
        return None
    if not exps:
        return None
    def days(e):
        return abs((_dt.date.fromisoformat(e) - _dt.date.today()).days - target_days)
    exp = min(exps, key=days)
    try:
        chain = tk.option_chain(exp)
    except Exception:
        return None
    ivs = []
    for leg in (chain.calls, chain.puts):
        leg = leg.dropna(subset=["impliedVolatility"]).copy()
        if not len(leg):
            continue
        iv = leg["impliedVolatility"]
        leg = leg[(iv >= band[0]) & (iv <= band[1])
                  & ((leg["strike"] - price).abs() <= window * price)]   # near the money
        if not len(leg):
            continue
        leg = leg.assign(dist=(leg["strike"] - price).abs()).sort_values("dist")
        ivs.extend(leg["impliedVolatility"].head(n).tolist())
    return float(np.median(ivs)) if ivs else None


def _realized_vol(tk, lookback="1y"):
    """Annualized realized vol from ~1y of daily closes. Always available and
    independent of option-quote freshness — the fallback for the ATM-IV read."""
    try:
        h = tk.history(period=lookback, interval="1d")
        close = h["Close"].dropna()
        if len(close) < 30:
            return None
        rets = np.log(close / close.shift(1)).dropna()
        return float(rets.std() * np.sqrt(252.0))
    except Exception:
        return None


# a broad, liquid large-cap basket standing in for "the average stock" — used to
# MEASURE the average single-stock variance (Martin-Wagner) instead of assuming it.
DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "TSLA",
    "JPM", "BAC", "GS", "V", "MA", "JNJ", "UNH", "PFE", "MRK", "ABBV",
    "PG", "KO", "PEP", "WMT", "MCD", "HD", "NKE", "COST",
    "CAT", "BA", "HON", "GE", "XOM", "CVX", "DIS", "VZ", "ORCL", "CSCO",
]


def basket_avg_variance(tickers=None, min_names=12, default_vol=0.30):
    """Average risk-neutral variance of a large-cap basket — the 'average stock'
    variance for the idiosyncratic term. Per name: ~1y ATM implied vol (realized-vol
    fallback), robust to individual failures. Returns (avg_variance, n_used).
    Falls back to default_vol**2 if too few names succeed."""
    import time
    import yfinance as yf
    tickers = tickers or DEFAULT_BASKET
    variances = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            fast = tk.fast_info
            price = float(fast.get("last_price") or fast.get("lastPrice") or 0.0)
            if price <= 0:
                continue
            iv = _atm_iv(tk, price, target_days=365)
            v = iv if (iv is not None and 0.05 <= iv <= 2.0) else _realized_vol(tk)
            if v is not None and 0.05 <= v <= 2.0:
                variances.append(v ** 2)
        except Exception:
            pass
        time.sleep(0.3)                       # gentle on Yahoo's rate limits
    if len(variances) >= min_names:
        return float(np.mean(variances)), len(variances)
    return float(default_vol ** 2), len(variances)
