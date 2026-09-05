"""
Per-company data for the cost-of-equity overlay.

Runs in the ticker-triggered job: given a ticker, pull fundamentals (leverage)
and the option chain (asset vol + idiosyncratic variance) via yfinance, and
emit company_<ticker>.csv for the Sheet.

The two load-bearing numbers — economic leverage and the Merton pass-through k —
are pure functions, unit-tested offline against AT&T.
"""
from __future__ import annotations

import datetime as dt

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


# RETIRED 2026-09-02 -- idiosyncratic_variance(). The Martin-Wagner add-on,
# `max(0.5 * (equity_var - market_var / avg_correlation), 0.0)`. Superseded by the four-block risk
# score in aeg-valuation/idio/, the one approved method. Note what the `max(..., 0.0)` did: it
# floored the premium at zero for any company below average-stock variance, so LIN, PEP and WMT
# published exactly 0.000% at every tenor -- read as a data failure for weeks when it was the
# formula. The approved score has no floor at zero; it is an increment that may be negative.


# ----------------------------------------------------- descriptive company facts
def _company_facts(tk, fast):
    """Sector/industry/country/52-week range/market cap — descriptive facts, not
    model inputs. Nothing here is read by name into a valuation; added 2026-09-05
    for the SETUP app's 'at a glance' panel (and, eventually, the screener).

    fast_info already carries the 52-week range, market cap, exchange and currency
    for free — no extra network call. Only sector/industry/country/employees/name
    need the slower .info fetch, wrapped so a flaky or rate-limited call here can
    never take down the rest of the job: on failure this returns Nones for those
    five fields, never raises, and never guesses."""
    facts = {
        "cf_week52_high": fast.get("yearHigh"),
        "cf_week52_low": fast.get("yearLow"),
        "cf_market_cap": fast.get("marketCap"),
        "cf_currency": fast.get("currency"),
        "cf_exchange": fast.get("exchange"),
        "cf_sector": None, "cf_industry": None, "cf_country": None,
        "cf_employees": None, "cf_long_name": None,
    }
    try:
        info = tk.info
        facts["cf_sector"] = info.get("sector")
        facts["cf_industry"] = info.get("industry")
        facts["cf_country"] = info.get("country")
        facts["cf_employees"] = info.get("fullTimeEmployees")
        facts["cf_long_name"] = info.get("longName") or info.get("shortName")
    except Exception:
        pass   # descriptive only — never block the run over a slow/flaky .info call
    facts["cf_fetched_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return facts


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

    cfacts = _company_facts(tk, fast)

    return dict(ticker=ticker, price=price, **lev,
                book_total_debt=total_debt,   # reported book debt (par mark for bondless MVD)
                equity_vol=equity_vol, sigma_V=sigma_V,
                equity_vol_ts=equity_vol_ts,
                avg_correlation=avg_correlation,
                **cfacts)


# RETIRED 2026-09-02 -- fetch_smile(), fetch_smiles(), realized_skew() and skew_diag().
#
# Four functions, ~95 lines, all of them serving the skew-corridor construction: the option-smile
# pulls, the physical semivariance corridor for the phi dial, and the diagnostic that wrote
# skew_diag_<T>.csv. James ruled 2026-09-02 that there is ONE approved method for a company's
# idiosyncratic risk premium -- the four-block risk score in aeg-valuation/idio/ -- and that the
# superseded ones "should not be referred to anywhere". asfp/skew.py and asfp/erp_engine.py went
# with them.
#
# TWO THINGS THAT SHOULD OUTLIVE THE CODE.
#
# The corridor was attributed to James inside asfp/skew.py's own docstring ("Principle (James):
# you only demand compensation for the ASYMMETRY"). He had never seen it. A section of the agreed
# workflow design was written on that attribution and ruled that the published premium should move
# to it. One misattributed docstring, load-bearing for weeks.
#
# skew_diag()'s `atm` -- np.interp(F, ks, ivs) off a thin smile -- is where register item B1's
# 3.13% at-the-money implied volatility came from, identical to two decimals across HD, LIN, PEP
# and WMT. It never reached a valuation, and it never even reached `equity_vol`: pick_equity_vol()
# above rejects anything below lo=0.05 and falls back to realized vol. B1 is closed by this
# deletion. The standing protection is the refusal register in WORKFLOW-DESIGN-2026-09-01.md 1.3,
# which fires on an ATM implied volatility identical across unrelated tickers to two decimals.

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


# RETIRED 2026-09-02 -- basket_avg_variance(). It measured the "average stock" variance for the
# Martin-Wagner idiosyncratic term and wrote outputs/market_micro_latest.csv. That term is retired,
# the file is no longer written, and nothing else read either. DEFAULT_BASKET and _atm_iv above
# remain: _atm_iv still serves fetch_equity_vol_ts, which feeds the market-ERP path.
