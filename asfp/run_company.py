"""
Ticker-triggered job. Given a ticker, produce every per-company output:

    outputs/cod_<T>.csv   / cod_<T>_annual.csv     real cost of debt by tenor
    outputs/coe_<T>.csv   / coe_<T>_annual.csv     COE components by tenor
    outputs/company_<T>.csv                        fundamentals + MV of debt
    outputs/<T>_rating_fan.png                     rating-fan diagnostic

Reads the market grids the weekly job already committed (curve_latest,
erp_market_latest, market_credit_latest), the issuer's committed bond list
(bonds/<T>.csv, optional — pure-rating fallback if absent), and live company
fundamentals + options via yfinance.

Run:  python -m asfp.run_company T
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

from . import issuer, debt_analytics as da

OUTDIR = "outputs"
GRID = np.arange(1, 31, dtype=float)


def _load_market():
    """Load the committed market grids into (cg, real_rf, market_erp, vix)."""
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


def _load_bonds(ticker):
    path = f"bonds/{ticker.upper()}.csv"
    if not os.path.exists(path):
        return None
    b = da.parse_tradingview_bonds(pd.read_csv(path))
    return b if len(b) else None


def main():
    ticker = (os.environ.get("TICKER")
              or (sys.argv[1] if len(sys.argv) > 1 else "")).strip().upper()
    if not ticker:
        raise SystemExit("No ticker provided (env TICKER or argv).")

    if not os.path.exists(f"{OUTDIR}/market_credit_latest.csv"):
        raise SystemExit("market grids missing — run the weekly job first.")

    cg, real_rf, market_erp, vix = _load_market()
    bonds = _load_bonds(ticker)

    from . import company as comp                       # yfinance import deferred
    fund = comp.fetch_company(ticker)

    tables, meta = issuer.assemble(ticker, cg, real_rf, market_erp, vix, fund, bonds)
    written = issuer.write_outputs(OUTDIR, ticker, tables, meta, fund)

    if bonds is not None:
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
