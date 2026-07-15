"""
Company debt analytics (roadmap bundle A1 + A2).

From an issuer's bond list (as exported from TradingView, which carries price,
coupon, maturity, YTW, amount outstanding and rating) compute:

  A1  market value of the company's debt          = sum(price * amount outstanding)
  A2  portfolio yield to maturity (combined-CF IRR),
      market-value-weighted modified duration,
      market-value-weighted coupon ("par-ish" number),
      market-value-weighted YTW (quick cross-check).

Design notes
------------
* Bonds are priced/aggregated on a semiannual coupon convention (US corporates).
* The portfolio YTM is the single rate that prices ALL bonds' combined cash flows
  at their total market value — more rigorous than averaging individual YTWs.
* Pure functions; unit-tested by repricing each bond at its own YTW and checking
  the model price reproduces the quoted price.

All rates are DECIMAL fractions here (0.0582 = 5.82%), matching the export.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- parsing
_UNIT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_CUSIP_RE = re.compile(r"^([A-Z0-9]{9,10})")     # leading CUSIP glued to the name


def parse_amount(x):
    """'1.25 B USD' -> 1.25e9 ; '550 M USD' -> 5.5e8 ; '—'/'' -> nan."""
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("", "—", "-", "nan", "None"):
        return np.nan
    m = re.match(r"([\d.]+)\s*([KMBT]?)", s, re.I)
    if not m:
        return np.nan
    val = float(m.group(1))
    unit = m.group(2).upper()
    return val * _UNIT.get(unit, 1.0)


def _num(x):
    """Coerce a numeric cell to float. Handles decimals (0.0582), percent strings
    ('5.82%' -> 0.0582, '115.25%' -> 1.1525), and thousands commas. This makes the
    parser agnostic to whether the source exports raw values or formatted percents
    (Google Sheets CSV export vs the gviz endpoint)."""
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("", "—", "-", "nan", "None"):
        return np.nan
    pct = s.endswith("%")
    s = s.replace("%", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return v / 100.0 if pct else v


def _locate_header(df):
    """Find the bond header (row/columns containing a 'Symbol…' cell) whether it
    sits in df.columns (clean CSV) or in a data row (header=None or a sheet with a
    TICKER cell above the block). Returns (headers_list, body_df) positionally
    indexed, or (None, None)."""
    cols = [str(c).strip() for c in df.columns]
    if any(c.lower().startswith("symbol") for c in cols):
        body = df.copy()
        body.columns = range(body.shape[1])
        return cols, body.reset_index(drop=True)
    for i in range(min(len(df), 40)):
        row = [str(v).strip() for v in df.iloc[i].tolist()]
        if any(v.lower().startswith("symbol") for v in row):
            body = df.iloc[i + 1:].copy()
            body.columns = range(body.shape[1])
            return row, body.reset_index(drop=True)
    return None, None


_EMPTY_COLS = ["cusip", "description", "ytw", "price_frac", "coupon", "maturity",
               "years", "outstanding", "face", "sp_rating", "issuer"]


def parse_tradingview_bonds(df, asof="2026-07-12"):
    """Normalize an issuer's TradingView bond block into a clean frame.

    Robust to: clean CSV headers, doubled headers ('Symbol Symbol'), a TICKER cell
    above the block, repeated header rows, and percent-formatted values.
    Returns one row per bond with numeric ytw/price_frac/coupon as decimals.
    """
    df = df.copy()
    headers, body = _locate_header(df)
    if headers is None or body is None or len(body) == 0:
        return pd.DataFrame(columns=_EMPTY_COLS)

    def pos(*names):
        for n in names:
            for j, h in enumerate(headers):
                if str(h).strip().lower().startswith(n.lower()):
                    return j
        return None

    def colvals(*names):
        j = pos(*names)
        if j is None or j >= body.shape[1]:
            return pd.Series([np.nan] * len(body))
        return body.iloc[:, j].reset_index(drop=True)

    # drop any repeated header rows inside the body
    sym = colvals("symbol").astype(str)
    keep = ~sym.str.strip().str.lower().str.startswith("symbol")
    body = body[keep.values].reset_index(drop=True)

    sym = colvals("symbol").astype(str)
    cusip = sym.str.extract(_CUSIP_RE, expand=False)
    asof_ts = pd.Timestamp(asof)
    maturity = pd.to_datetime(colvals("maturity"), errors="coerce")
    years = (maturity - asof_ts).dt.days / 365.25

    out = pd.DataFrame({
        "cusip": cusip.values,
        "description": sym.str.replace(_CUSIP_RE, "", regex=True).str.strip().values,
        "ytw": colvals("ytw").map(_num).values,
        "price_frac": colvals("price").map(_num).values,
        "coupon": colvals("coupon").map(_num).values,
        "maturity": maturity.values,
        "years": years.values,
        "outstanding": colvals("outstanding").map(parse_amount).values,
        "face": colvals("face").map(parse_amount).values,
        "sp_rating": colvals("s&p", "sp", "s & p").astype(str).str.strip().values,
        "issuer": colvals("issuer").astype(str).str.strip().values,
    })
    out = out[(out["years"] > 0.05) & out["ytw"].notna()
              & out["price_frac"].notna() & out["coupon"].notna()]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------- pricing
def _cashflows(years, coupon, freq=2, notional=1.0):
    """Semiannual cash-flow (times, amounts) for a bullet bond, from maturity back."""
    n = max(int(round(years * freq)), 1)
    times = np.array([years - k / freq for k in range(n)][::-1])
    times = times[times > 1e-6]
    cpn = coupon / freq * notional
    amts = np.full(len(times), cpn)
    amts[-1] += notional                       # principal at maturity
    return times, amts


def price_bond(years, coupon, y, freq=2, notional=1.0):
    """Present value of a bond's cash flows discounted at annual yield y."""
    t, a = _cashflows(years, coupon, freq, notional)
    return float(np.sum(a / (1.0 + y / freq) ** (freq * t)))


def modified_duration(years, coupon, y, freq=2):
    """Numerical modified duration: -(1/P) dP/dy via central difference."""
    h = 1e-4
    p0 = price_bond(years, coupon, y, freq)
    pu = price_bond(years, coupon, y + h, freq)
    pd_ = price_bond(years, coupon, y - h, freq)
    return -(pu - pd_) / (2 * h) / p0


# ----------------------------------------------------------------- aggregates
def market_value(bonds):
    """A1: per-bond and total market value of debt (price * amount outstanding)."""
    b = bonds.copy()
    b["market_value"] = b["price_frac"] * b["outstanding"]
    total = float(b["market_value"].sum(skipna=True))
    return b, total


def portfolio_ytm(bonds, freq=2):
    """A2: single IRR pricing ALL bonds' combined cash flows at total market value.

    Bonds without an amount outstanding are dropped from the weighting (can't
    scale their notional); everything else is scaled to actual notional.
    """
    b = bonds.dropna(subset=["outstanding"]).copy()
    if b.empty:
        return np.nan
    # combine ALL bonds' cash flows on their exact times (no grid snapping),
    # and solve the single IRR that prices them at total market value.
    times_l, flows_l = [], []
    total_mv = 0.0
    for _, r in b.iterrows():
        t, a = _cashflows(r["years"], r["coupon"], freq, notional=r["outstanding"])
        times_l.append(t)
        flows_l.append(a)
        total_mv += r["price_frac"] * r["outstanding"]
    times = np.concatenate(times_l)
    flows = np.concatenate(flows_l)

    def npv(y):
        return np.sum(flows / (1.0 + y / freq) ** (freq * times)) - total_mv

    lo, hi = -0.5, 1.0
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(200):                         # bisection
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def portfolio_summary(bonds, freq=2):
    """A1 + A2 headline numbers for one issuer."""
    b, mv_total = market_value(bonds)
    w = b["market_value"]
    havemv = w.notna() & (w > 0)
    wsum = w[havemv].sum()

    def wavg(col):
        return float(np.average(b.loc[havemv, col], weights=w[havemv])) if wsum else np.nan

    mdur = b.apply(lambda r: modified_duration(r["years"], r["coupon"], r["ytw"]), axis=1)
    b = b.assign(mod_duration=mdur)
    port_mdur = float(np.average(mdur[havemv], weights=w[havemv])) if wsum else np.nan

    return {
        "n_bonds": int(len(b)),
        "n_weighted": int(havemv.sum()),
        "market_value_debt": mv_total,
        "wavg_coupon": wavg("coupon"),
        "wavg_ytw": wavg("ytw"),
        "portfolio_ytm": portfolio_ytm(bonds, freq),
        "wavg_mod_duration": port_mdur,
        "wavg_years": wavg("years"),
    }, b
