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
    if iv is not None and 0.05 <= iv <= 2.0:
        equity_vol = iv
    elif rv is not None:
        equity_vol = rv
    else:
        equity_vol = iv or rv or 0.25
    sigma_V = asset_vol_from_equity(equity_vol, lev["L"])

    # v2: the single-name option-implied vol TERM STRUCTURE (1m..2y) that the
    # risk ratio R_i(t) needs at the front. Falls back to the flat 1y point (as a
    # single-element curve) when the long-dated chain is thin. In vol POINTS.
    equity_vol_ts = fetch_equity_vol_ts(tk, price, fallback_vol=equity_vol)

    return dict(ticker=ticker, price=price, **lev,
                equity_vol=equity_vol, sigma_V=sigma_V,
                equity_vol_ts=equity_vol_ts,
                avg_correlation=avg_correlation)


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
