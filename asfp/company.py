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

    # option-implied equity vol: near-the-money IV at ~1y expiration
    equity_vol = _atm_iv(tk, price, target_days=365) or 0.25
    sigma_V = asset_vol_from_equity(equity_vol, lev["L"])

    return dict(ticker=ticker, price=price, **lev,
                equity_vol=equity_vol, sigma_V=sigma_V,
                avg_correlation=avg_correlation)


def _atm_iv(tk, price, target_days=365):
    """Robust at-the-money implied-vol read near a target horizon."""
    import datetime as _dt
    try:
        exps = tk.options
    except Exception:
        return None
    if not exps:
        return None
    # pick the expiration closest to target_days out
    def days(e):
        return abs((_dt.date.fromisoformat(e) - _dt.date.today()).days - target_days)
    exp = min(exps, key=days)
    chain = tk.option_chain(exp)
    ivs = []
    for leg in (chain.calls, chain.puts):
        leg = leg.dropna(subset=["impliedVolatility"])
        if len(leg):
            leg = leg.assign(dist=(leg["strike"] - price).abs()).sort_values("dist")
            ivs.extend(leg["impliedVolatility"].head(3).tolist())    # 3 nearest strikes
    return float(np.median(ivs)) if ivs else None
