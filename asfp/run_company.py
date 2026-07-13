"""
Ticker-triggered job. Given a ticker, produce every per-company output:

    outputs/cod_<T>.csv   / cod_<T>_annual.csv     real cost of debt by tenor
    outputs/coe_<T>.csv   / coe_<T>_annual.csv     COE components by tenor
    outputs/company_<T>.csv                        fundamentals + MV of debt
    outputs/bonds_used_<T>.csv                     the exact bonds this run used
    outputs/<T>_rating_fan.png                     rating-fan diagnostic

Bond data is pulled LIVE from a shared Google Sheet (env BONDS_SHEET_ID) — one
reusable tab holding a TICKER cell and a pasted bond block. Nothing is uploaded to
the repo. Fallbacks: committed bonds/<T>.csv, then the pure-rating curve.

The ticker comes from the workflow input if given, otherwise from the sheet's
TICKER cell.

Run:  python -m asfp.run_company [TICKER]
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

from . import issuer, debt_analytics as da, sheets

OUTDIR = "outputs"
GRID = np.arange(1, 31, dtype=float)


def _load_market():
    cg = pd.read_csv(f"{OUTDIR}/market_credit_latest.csv").set_index("tenor")
    cg = cg.reindex(GRID).interpolate().bfill().ffill()
    cur = pd.read_csv(f"{OUTDIR}/curve_latest.csv").set_index("maturity")
    real_rf = np.interp(GRID, cur.index.to_numpy(), cur["real_fwd1y"].to_numpy())
    erp = pd.read_csv(f"{OUTDIR}/erp_market_latest.csv").set_index("tenor")
    erp = erp.reindex(GRID).interpolate().bfill().ffill()
    market_erp = erp["market_erp"].to_numpy()
    a_mkt = float(erp["a_mkt"].iloc[0]) if "a_mkt" in erp else (18.0 ** 2) / 100.0
    vix = float(np.sqrt(a_mkt * 100.0))
    return cg, real_rf, market_erp, vix


def _load_committed_bonds(ticker):
    path = f"bonds/{ticker.upper()}.csv"
    if not os.path.exists(path):
        return None
    b = da.parse_tradingview_bonds(pd.read_csv(path))
    return b if len(b) else None


def _issuer_matches(bonds, ticker):
    """Loose sanity check: does the bonds' Issuer text look like this ticker's co?"""
    if bonds is None or "issuer" not in bonds:
        return True
    names = " ".join(bonds["issuer"].dropna().astype(str).str.lower().tolist())
    hint = {"T": "at&t", "AAPL": "apple", "HD": "home depot", "KO": "coca",
            "CLX": "clorox", "HSY": "hershey", "SJM": "smucker"}.get(ticker.upper())
    return (hint in names) if hint else True


def main():
    arg_ticker = (os.environ.get("TICKER")
                  or (sys.argv[1] if len(sys.argv) > 1 else "")).strip().upper()

    if not os.path.exists(f"{OUTDIR}/market_credit_latest.csv"):
        raise SystemExit("market grids missing — run the weekly job first.")

    # --- bonds + ticker: live Google Sheet first, then committed file ---
    sheet_id = os.environ.get("BONDS_SHEET_ID", "").strip()
    bonds, sheet_ticker = (sheets.bonds_and_ticker(sheet_id) if sheet_id else (None, None))
    ticker = arg_ticker or (sheet_ticker or "")
    if not ticker:
        raise SystemExit("No ticker (workflow input empty and no TICKER cell in the sheet).")
    ticker = ticker.upper()

    source = "google-sheet"
    if bonds is None:
        bonds = _load_committed_bonds(ticker)
        source = "committed-file" if bonds is not None else "none (pure-rating)"
    print(f"ticker={ticker}  bonds source={source}  "
          f"n_bonds={0 if bonds is None else len(bonds)}")
    if bonds is not None and not _issuer_matches(bonds, ticker):
        print(f"  ** WARNING: bonds' Issuer column does not look like {ticker} — "
              f"check the sheet holds {ticker}'s bonds, not another company's.")

    cg, real_rf, market_erp, vix = _load_market()

    from . import company as comp                       # yfinance import deferred
    fund = comp.fetch_company(ticker)

    tables, meta = issuer.assemble(ticker, cg, real_rf, market_erp, vix, fund, bonds)
    written = issuer.write_outputs(OUTDIR, ticker, tables, meta, fund)

    # archive the exact bonds this run used (audit trail; no manual tab-keeping)
    if bonds is not None:
        bonds.to_csv(f"{OUTDIR}/bonds_used_{ticker}.csv", index=False)
        written.append(f"bonds_used_{ticker}.csv")
        from . import charts
        chart = f"{OUTDIR}/{ticker}_rating_fan.png"
        res = charts.rating_fan_chart(cg, bonds, ticker, chart)
        written.append(os.path.basename(chart))
        print(f"  chart: modal={res['modal_rating']} offset=x{res['offset']:.2f} "
              f"flagged={res['n_flagged']}")

    print(f"OK {ticker}: wrote {', '.join(written)}")
    print(f"  rating={meta['rating']} offset=x{meta['offset']:.3f} "
          f"k={meta['k']:.2f} idio_anchor={meta['idio_anchor']:.2f}%")
    if meta.get("market_value_of_debt"):
        print(f"  MV(debt)=${meta['market_value_of_debt']/1e9:.1f}B "
              f"portYTM={meta['portfolio_ytm']*100:.2f}% "
              f"modDur={meta['wavg_mod_duration']:.1f}y")


if __name__ == "__main__":
    main()
