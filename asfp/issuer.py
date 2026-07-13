"""
Per-company assembly (the ticker job's core).

Given the market grids (produced by the weekly job) and an issuer's bond list,
produce every per-company output the downstream valuation engine and the
diagnostic chart consume:

  cod_<T>.csv         issuer REAL cost of debt, forward by tenor  (+ _annual)
  coe_<T>.csv         COE components: real_rf, market_erp,
                      credit_relative, idiosyncratic, ...          (+ _annual)
  company_<T>.csv     fundamentals + market_value_of_debt + portfolio analytics
  <T>_rating_fan.png  the rating-fan diagnostic chart

Everything here is pure/injectable so the whole output-production path is tested
offline; run_company.py adds only the live FRED/yfinance reads.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import credit, coe, units, company as comp, debt_analytics as da

# market-average calibration for the Merton pass-through (mirrors COE template)
MARKET = dict(base_k=2.0, L_mkt=0.33, sigma_V_mkt=0.22, T=10.0, r=0.0,
              lgd=0.60, p=0.5)


# ------------------------------------------------------------ rating & offset
def modal_rating(bonds):
    """Issuer's modal coarse S&P rating from its bonds (fallback BBB)."""
    from .charts import _coarse_rating
    cs = [c for c in bonds["sp_rating"].map(_coarse_rating) if c]
    return max(set(cs), key=cs.count) if cs else "BBB"


def fit_offset(cg, bonds, rating):
    """Multiplicative spread offset fitted from the issuer's own bonds, robust to
    distressed/subordinated outliers (e.g. subsidiary paper) via a MAD filter.

    offset = median( bond_spread / rating_curve_spread ), outliers beyond
    3 robust-sigma dropped. Returns (offset, n_used, n_excluded).
    """
    ten = cg.index.to_numpy()
    tsy = np.interp(bonds["years"], ten, cg["treasury_nominal"].to_numpy())
    rs = np.interp(bonds["years"], ten, cg[f"spread_{rating}"].to_numpy())
    bs = bonds["ytw"].to_numpy() * 100.0 - tsy          # bond spread over Treasury
    ratio = np.where(rs > 0, bs / rs, np.nan)
    valid = np.isfinite(ratio) & (bs > 0)
    r = ratio[valid]
    if r.size == 0:
        return 1.0, 0, 0
    if r.size >= 4:
        med = np.median(r)
        mad = np.median(np.abs(r - med)) or 1e-9
        keep = np.abs(r - med) <= 3 * 1.4826 * mad
    else:
        keep = np.ones(r.size, bool)
    off = float(np.median(r[keep])) if keep.any() else float(np.median(r))
    return off, int(keep.sum()), int(r.size - keep.sum())


# ------------------------------------------------------------ cost of debt
def build_cost_of_debt(cg, bonds=None, rating=None):
    """Issuer real cost-of-debt curve + metadata. Pure-rating fallback (offset=1)
    when no bonds are supplied."""
    if rating is None:
        rating = modal_rating(bonds) if bonds is not None and len(bonds) else "BBB"
    if f"spread_{rating}" not in cg.columns:
        rating = "BBB" if "spread_BBB" in cg.columns else rating
    if bonds is not None and len(bonds):
        offset, n_used, n_excl = fit_offset(cg, bonds, rating)
    else:
        offset, n_used, n_excl = 1.0, 0, 0
    cod = credit.issuer_real_cod(cg, rating, offset)
    return cod, dict(rating=rating, offset=offset, n_used=n_used, n_excluded=n_excl)


# ------------------------------------------------------------ cost of equity
def build_coe(grid, real_rf, market_erp, market_ig_spread, issuer_spread,
              rating, fund, vix, avg_stock_var=None, params=None):
    """COE components DataFrame (real_rf, market_erp, credit_relative,
    idiosyncratic, company_erp, real_coe) plus the k / idio_anchor used.

    If `avg_stock_var` (the measured average-stock variance) is supplied, the
    idiosyncratic term uses it directly; otherwise it falls back to the
    VIX + fixed-correlation proxy."""
    p = dict(MARKET); p.update(params or {})
    k = comp.merton_k(p["base_k"], fund["L"], fund["sigma_V"],
                      p["L_mkt"], p["sigma_V_mkt"], p["T"], p["r"])
    if avg_stock_var is not None:
        idio_anchor = coe.idio_anchor_from_variance(fund["equity_vol"], avg_stock_var)
    else:
        idio_anchor = coe.idio_anchor_from_options(fund["equity_vol"], vix,
                                                   fund.get("avg_correlation", 0.35))
    df = coe.assemble_coe(grid, real_rf, market_erp, market_ig_spread,
                          issuer_spread, rating, k, idio_anchor,
                          p=p["p"], lgd=p["lgd"])
    return df, dict(k=k, idio_anchor=idio_anchor)


# ------------------------------------------------------------ full assembly
def assemble(ticker, cg, real_rf, market_erp, vix, fund, bonds=None,
             rating=None, params=None, avg_stock_var=None):
    """Compute every per-company table (no I/O). Returns a dict of DataFrames
    and a meta dict. `cg` is the market credit grid (index tenor)."""
    grid = cg.index.to_numpy()
    cod, cmeta = build_cost_of_debt(cg, bonds, rating)
    issuer_spread = cod["spread"].to_numpy()
    market_ig = cg["ig_index_spread"].to_numpy()

    coe_df, emeta = build_coe(grid, real_rf, market_erp, market_ig,
                              issuer_spread, cmeta["rating"], fund, vix,
                              avg_stock_var=avg_stock_var, params=params)

    # market value of debt + portfolio analytics (if bonds present)
    if bonds is not None and len(bonds):
        summ, _ = da.portfolio_summary(bonds)
    else:
        summ = {}

    # annual-decimal variants
    rating = cmeta["rating"]
    cod_annual = pd.DataFrame({
        "tenor": grid,
        "real_cod": units.annualize_rate(cod["real_cod"].to_numpy()),
        "spread": units.to_decimal(cod["spread"].to_numpy()),
        "rating": cod["rating"].to_numpy(),
        "offset": cod["offset"].to_numpy(),
        f"real_cod_{rating}": units.annualize_rate(
            cod[f"real_cod_{rating}"].to_numpy()),
    }).set_index("tenor")
    ann = units.coe_annual_components(coe_df["real_rf"], coe_df["market_erp"],
                                      coe_df["credit_relative"], coe_df["idiosyncratic"])
    coe_annual = pd.DataFrame({"tenor": grid, **ann}).set_index("tenor")

    meta = dict(ticker=ticker, **cmeta, **emeta,
                market_value_of_debt=summ.get("market_value_debt"),
                portfolio_ytm=summ.get("portfolio_ytm"),
                wavg_mod_duration=summ.get("wavg_mod_duration"),
                wavg_coupon=summ.get("wavg_coupon"),
                wavg_years=summ.get("wavg_years"))
    return dict(cod=cod, cod_annual=cod_annual, coe=coe_df, coe_annual=coe_annual,
                summary=summ), meta


def write_outputs(outdir, ticker, tables, meta, fund):
    """Write all per-company CSVs. Chart is written separately (needs bonds+cg)."""
    os.makedirs(outdir, exist_ok=True)
    t = ticker.upper()
    tables["cod"].round(4).to_csv(f"{outdir}/cod_{t}.csv")
    tables["cod_annual"].round(6).to_csv(f"{outdir}/cod_{t}_annual.csv")
    tables["coe"].round(4).to_csv(f"{outdir}/coe_{t}.csv")
    tables["coe_annual"].round(6).to_csv(f"{outdir}/coe_{t}_annual.csv")

    # company_<T>.csv: fundamentals + debt analytics, long field,value form
    order = ["ticker", "price", "market_equity", "nfo", "L", "lambda0",
             "equity_vol", "sigma_V", "avg_correlation"]
    rows = [{"field": k, "value": fund[k]} for k in order if k in fund]
    for k in ["market_value_of_debt", "portfolio_ytm", "wavg_mod_duration",
              "wavg_coupon", "wavg_years", "rating", "offset"]:
        if meta.get(k) is not None:
            rows.append({"field": k, "value": meta[k]})
    pd.DataFrame(rows).to_csv(f"{outdir}/company_{t}.csv", index=False)
    return [f"cod_{t}.csv", f"cod_{t}_annual.csv", f"coe_{t}.csv",
            f"coe_{t}_annual.csv", f"company_{t}.csv"]
