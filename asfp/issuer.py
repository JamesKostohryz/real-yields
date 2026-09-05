"""
Per-company assembly (the ticker job's core).

Given the market grids (produced by the weekly job) and an issuer's bond list,
produce every per-company output the downstream valuation engine and the
diagnostic chart consume:

  cod_<T>.csv         issuer REAL cost of debt, forward by tenor  (+ _annual)
  company_<T>.csv     fundamentals + market_value_of_debt + portfolio analytics
  <T>_rating_fan.png  the rating-fan diagnostic chart

Everything here is pure/injectable so the whole output-production path is tested
offline; run_company.py adds only the live FRED/yfinance reads.

RETIRED 2026-09-02 -- `coe_<T>.csv` / `coe_<T>_annual.csv` AND THE MODULE BEHIND THEM.

James ruled that there is exactly ONE approved method for a company's idiosyncratic risk premium
-- the four-block risk score in `aeg-valuation/idio/` -- that every other method is superseded, and
that the retired ones "should not be referred to anywhere." `asfp/coe.py` was one of them: a
Martin-Wagner total-variance anchor, `max(0.5*(equity_var - avg_stock_var), 0.0) * (t/30)^p`,
floored at zero and hung off a single yfinance volatility scalar.

Two things are worth knowing before anyone reconstructs it. It **never reached a valuation** --
`aeg-valuation`'s `rate_feed.py` reads `coe_v2_<T>_latest_annual.csv`, not these files, and it now
reads only the `real_rf` and `market_erp` columns of that. And the `max(..., 0.0)` made the premium
STRUCTURALLY ZERO for any company below average-stock variance, which is why LIN, PEP and WMT
published 0.000% at every tenor -- read for weeks as a data failure when it was the formula.

The Merton pass-through `k` went with it: its only consumer was `assemble_coe`'s `credit_relative`
term, which the v2 feed had already dropped ("leverage/credit stay inside the engine via MM
re-lever"). Working: `aeg-project/docs/engine/A6-Cost-Of-Equity-FINDINGS-2026-09-02.md`.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import credit, units, debt_analytics as da


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


# ------------------------------------------------------------ full assembly
# `build_coe` and the MARKET calibration dict were removed here on 2026-09-02 -- see the module
# docstring. `real_rf`, `market_erp`, `vix` and `avg_stock_var` remain in this signature and are
# now unused: the callers still pass them, and changing four call sites for cosmetics would be a
# wider edit than a retirement warrants. They are the seam through which the retired construction
# entered; leave them empty rather than re-wiring anything into them.
def assemble(ticker, cg, real_rf, market_erp, vix, fund, bonds=None,
             rating=None, params=None, avg_stock_var=None):
    """Compute every per-company table (no I/O). Returns a dict of DataFrames
    and a meta dict. `cg` is the market credit grid (index tenor)."""
    grid = cg.index.to_numpy()
    cod, cmeta = build_cost_of_debt(cg, bonds, rating)

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
    # market value of debt. Bonded: book debt marked to the mean traded price of the issuer's
    # own bond curve (mvd_basis=book-scaled). BONDLESS: no traded bonds to mark, so par-mark the
    # reported book total debt (mvd_basis=book-par) — bank/term/securitization debt sits near par,
    # so book ~= market. Mirrors the synthetic-cod bondless logic (COCKPIT 2026-07-29).
    mvd = summ.get("market_value_debt")
    mvd_basis = "book-scaled" if mvd is not None else None
    if mvd is None:
        bd = fund.get("book_total_debt")
        if bd is not None and float(bd) > 0:
            mvd = float(bd)
            mvd_basis = "book-par"
    meta = dict(ticker=ticker, **cmeta,
                market_value_of_debt=mvd, mvd_basis=mvd_basis,
                portfolio_ytm=summ.get("portfolio_ytm"),
                wavg_mod_duration=summ.get("wavg_mod_duration"),
                wavg_coupon=summ.get("wavg_coupon"),
                wavg_years=summ.get("wavg_years"))
    return dict(cod=cod, cod_annual=cod_annual, summary=summ), meta


def write_outputs(outdir, ticker, tables, meta, fund):
    """Write all per-company CSVs. Chart is written separately (needs bonds+cg)."""
    os.makedirs(outdir, exist_ok=True)
    t = ticker.upper()
    tables["cod"].round(4).to_csv(f"{outdir}/cod_{t}.csv")
    # annual files: publish at 9 dp so the additive-identity rounding residual
    # (~1e-9) stays far inside the valuation engine's 1e-6 fail-loud tolerance.
    tables["cod_annual"].round(9).to_csv(f"{outdir}/cod_{t}_annual.csv")

    # company_<T>.csv: fundamentals + debt analytics, long field,value form
    #
    # `equity_vol`, `sigma_V` and `avg_correlation` were dropped from this list on 2026-09-02.
    # They existed to feed the retired Martin-Wagner anchor and the retired Merton pass-through,
    # and they appear NOWHERE in aeg-valuation -- `rate_feed.load_company()` reads
    # `market_value_of_debt` and the four debt-analytics fields by NAME, so removing them cannot
    # move a valuation. `equity_vol` was also the field register item B1 was assumed to poison;
    # it never was (company.pick_equity_vol's lo=0.05 guard rejects the 3.13% quote outright), but
    # publishing a volatility that nothing consumes invites the next reader to consume it.
    order = ["ticker", "price", "market_equity", "nfo", "L", "lambda0"]
    rows = [{"field": k, "value": fund[k]} for k in order if k in fund]
    for k in ["market_value_of_debt", "portfolio_ytm", "wavg_mod_duration",
              "wavg_coupon", "wavg_years", "rating", "offset"]:
        if meta.get(k) is not None:
            rows.append({"field": k, "value": meta[k]})
    # PROVENANCE: the feed publishes no per-bond amount outstanding, so the bond list
    # is notional-weighted by allocating the issuer's REPORTED book total debt evenly
    # across the observed bonds (refresh_bonds.eodhd_book_total_debt). Market value of
    # debt is therefore book debt marked to the mean traded price of the issuer's own
    # curve -- an approximation, NOT issue-level truth. Tag it so no downstream
    # consumer mistakes it for the latter. Upgrade path: a real 10-K/XBRL schedule.
    if meta.get("market_value_of_debt") is not None:
        rows.append({"field": "mvd_basis", "value": meta.get("mvd_basis") or "book-scaled"})
    pd.DataFrame(rows).to_csv(f"{outdir}/company_{t}.csv", index=False)
    written = [f"cod_{t}.csv", f"cod_{t}_annual.csv", f"company_{t}.csv"]

    # company_facts_<T>.csv: descriptive facts ONLY -- sector/industry/country/52-week
    # range/market cap/employees. Kept SEPARATE from company_<T>.csv on purpose: that
    # file is model inputs read by NAME into a valuation (market_value_of_debt, the
    # debt-analytics fields); this one is display-only, for SETUP's "at a glance"
    # panel (and, later, the screener), and nothing in it should ever be read into a
    # calculation. Added 2026-09-05. Soft: a field yfinance didn't return comes back
    # as an empty cell, never a guess -- see company._company_facts.
    cf_order = ["cf_long_name", "cf_sector", "cf_industry", "cf_country",
                "cf_exchange", "cf_currency", "cf_market_cap",
                "cf_week52_high", "cf_week52_low", "cf_employees", "cf_fetched_utc"]
    if any(fund.get(k) is not None for k in cf_order):
        cf_rows = [{"field": k[3:], "value": fund.get(k)} for k in cf_order]
        pd.DataFrame(cf_rows).to_csv(f"{outdir}/company_facts_{t}.csv", index=False)
        written.append(f"company_facts_{t}.csv")
    return written
