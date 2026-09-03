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
import datetime as dt
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

    # `avg_stock_var` was read from outputs/market_micro_latest.csv here until 2026-09-02. Its
    # ONLY consumer was the retired Martin-Wagner anchor (asfp/coe.py), which is gone; the file
    # is no longer written. See asfp/issuer.py's docstring.
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

    # freshness stamp: lets the Google Sheet show WHEN these numbers were generated
    # and WHAT bonds fed them, so a stale IMPORTDATA cache is obvious at a glance.
    now = dt.datetime.utcnow()
    stamp = [
        {"field": "ticker", "value": ticker},
        {"field": "generated_utc", "value": now.strftime("%Y-%m-%d %H:%M UTC")},
        {"field": "generated_iso", "value": now.isoformat(timespec="seconds") + "Z"},
        {"field": "bonds_source", "value": source},
        {"field": "n_bonds", "value": 0 if bonds is None else len(bonds)},
        {"field": "run_id", "value": os.environ.get("GITHUB_RUN_ID", "local")},
        {"field": "git_sha", "value": os.environ.get("GITHUB_SHA", "")[:7]},
        # THE DURABILITY JUDGMENT, RECORDED 2026-08-20. OBS_CATEGORY picks the obsolescence
        # elevator preset and therefore lands directly in coe_v2_<T>_latest_annual.csv -- the
        # cost-of-equity curve the AEG engine discounts with. It defaults to "B" and, until
        # this line, was written down NOWHERE. Every company's published cost of equity
        # embedded a durability judgment that could not be read back, compared, or reused.
        #
        # It has to be readable for a scheduled refresh to exist at all: re-running a company
        # without knowing its category would silently re-decide it, which is worse than not
        # refreshing. aeg-valuation's rate-side refresh reads these two fields.
        {"field": "obs_category",
         "value": os.environ.get("OBS_CATEGORY", "B").strip().upper()[:1] or "B"},
        {"field": "ory_override", "value": (os.environ.get("ORY_OVERRIDE") or "").strip()},
    ]
    pd.DataFrame(stamp).to_csv(f"{OUTDIR}/run_stamp_{ticker}.csv", index=False)
    written.append(f"run_stamp_{ticker}.csv")

    # --- v2 (non-breaking): total-risk single-name COE to 150y, from the VIX-curve
    # market ERP. New files coe_v2_<T>_latest(.csv/_annual.csv); existing outputs
    # untouched until the engine cuts over. Needs the weekly market_erp_v2 file. ---
    try:
        from . import total_risk_erp as trv
        me2p = f"{OUTDIR}/market_erp_v2_latest.csv"
        if not os.path.exists(me2p):
            print("  coe v2 skipped: market_erp_v2_latest.csv missing (run weekly job first)")
        else:
            GV = np.arange(1, 151, dtype=float)
            me2 = pd.read_csv(me2p).set_index("tenor").reindex(GV).interpolate().ffill().bfill()
            mkt2 = me2["market_erp"].to_numpy()
            cur = pd.read_csv(f"{OUTDIR}/curve_latest.csv").set_index("maturity")
            rf2 = np.interp(GV, cur.index.to_numpy(), cur["real_fwd1y"].to_numpy())   # flat past 30y

            # single-name option-implied vol TERM STRUCTURE (1m..2y); flat 1y fallback
            stock_vol_ts = fund.get("equity_vol_ts") or [(1.0, float(fund.get("equity_vol", 0.25)) * 100.0)]
            # index vol term structure for the risk ratio R = σ_i/σ_mkt: use the OBSERVED
            # (spot) index vols the weekly job published, so R is a clean vol ratio. (Do NOT
            # invert the market ERP here — it is now a FORWARD/marginal variance, so its
            # implied vol is a forward vol and would mismatch the stock's spot vols.)
            ivp = f"{OUTDIR}/index_vol_ts_latest.csv"
            if os.path.exists(ivp):
                _iv = pd.read_csv(ivp)
                index_vol_ts = list(zip(_iv["tenor"].astype(float).tolist(),
                                        _iv["index_vol"].astype(float).tolist()))
            else:   # degraded fallback: flat 1y index vol from the front market ERP (spot approx)
                index_vol_ts = [(1.0, float(np.sqrt(max(np.interp(1.0, GV, mkt2), 0.01) * 100.0)))]
            category = os.environ.get("OBS_CATEGORY", "B").strip().upper()[:1] or "B"  # Phase 4: from sheet
            ory_ov = os.environ.get("ORY_OVERRIDE")
            ory_ov = float(ory_ov) if ory_ov else None
            coe2 = trv.assemble_coe_v2(GV, rf2, mkt2, stock_vol_ts, index_vol_ts,
                                       meta["rating"], category, ory_override=ory_ov)
            coe2.round(4).to_csv(f"{OUTDIR}/coe_v2_{ticker}_latest.csv")
            # THREE COLUMNS, NOT SIX, SINCE 2026-09-02.
            #
            # This file used to carry `idiosyncratic`, `company_erp` and `real_coe` as well. All
            # three were the retired single-name construction (see asfp/total_risk_erp.py's
            # docstring), and `aeg-valuation` never read any of them -- `rate_feed.load_coe()`
            # takes `real_rf` and `market_erp` and nothing else. Publishing the other three put a
            # `real_coe` on a public surface that disagreed with the rate the valuation actually
            # discounted at (6.87% against 6.2169% for AMCR), which is most of register item A6.
            #
            # THE ORDER MATTERED AND IT HAS BEEN OBSERVED: `rate_feed.load_coe()` stopped
            # REQUIRING those columns in aeg-valuation b22d5f1, which landed BEFORE this. Dropping
            # them first would have made the engine refuse every ticker.
            #
            # The file itself must survive. It is the engine's market-ERP feed and the target
            # apply_erp_overlay.py rewrites.
            rfp, mep = (coe2[c].to_numpy() for c in ("real_rf", "market_erp"))
            l0 = np.expm1(rfp / 100); l1 = np.expm1((rfp + mep) / 100)
            pd.DataFrame({"tenor": GV, "real_rf": l0, "market_erp": l1 - l0}
                         ).set_index("tenor").round(9).to_csv(f"{OUTDIR}/coe_v2_{ticker}_latest_annual.csv")
            written += [f"coe_v2_{ticker}_latest.csv", f"coe_v2_{ticker}_latest_annual.csv"]

            # RETIRED 2026-09-02 -- coe_v2_<T>_effective.csv and _effective_annual.csv.
            #
            # A ~75-line block stood here that collapsed the whole term structure to a single
            # cash-flow-PV-weighted rate, the equity analogue of a bond's YTM, and published five
            # fields including `idiosyncratic`, `company_erp` and `real_coe`. It is deleted, and
            # the two files it wrote are deleted with it.
            #
            # WHY, PRECISELY. The collapse arithmetic was sound. What was not sound is what
            # apply_erp_overlay.py then did to the output: it overwrote `real_rf` and `market_erp`
            # with the ERP engine's Decision-B state-machine reading while leaving `idiosyncratic`
            # as this block's YTM collapse of a different curve representation, and ADDED THE TWO
            # TOGETHER. Its own `methodology_note` field said so in as many words. The addition
            # passes its assertion because the assertion is arithmetic; the objects are not
            # commensurable. For AMCR this file published a `real_coe` of 11.873% while the
            # valuation discounted at 6.2169% -- an eleven-fold spread in the idiosyncratic leg,
            # register item A6.
            #
            # Nothing read it. Not the engine, not the screener, not the Dashboard. It existed to
            # be looked at, and what it showed was wrong. The retirement of the single-name
            # construction removes the leg it was collapsing anyway.
            r1 = stock_vol_ts[0][1] / max(index_vol_ts[0][1], 1e-6)
            print(f"  coe v2: R(front)={r1:.2f} cat={category} "
                  f"stock_vol_pts={len(stock_vol_ts)} obs_to={stock_vol_ts[-1][0]:.2f}y "
                  f"(house-view legs only: real_rf + market_erp; the company premium is the "
                  f"four-block score inside aeg-valuation)")
            print(f"  coe v2: market_erp(1y)={coe2['market_erp'].loc[1]:.2f}% "
                  f"market_erp(30y)={coe2['market_erp'].loc[30]:.2f}%")
    except Exception as e:
        print(f"  coe v2 skipped (non-fatal): {e}")

    # RETIRED 2026-09-02 -- the two SKEW blocks that stood here.
    #
    # The first wrote `skew_diag_<T>.csv` (a corridor down/up variance split against the name's
    # own Martin ATM variance); the second wrote `skew_erp_<T>.csv`, `skew_erp_<T>_realized.csv`
    # and called `erp_engine.skew_erp_curve`, labelled "final engine". Both are gone, with
    # `asfp/skew.py`, `asfp/erp_engine.py`, and `company.skew_diag` / `realized_skew` /
    # `fetch_smile` / `fetch_smiles`.
    #
    # THE SECOND BLOCK IS THE ONE THAT COST THE MOST. `asfp/skew.py`'s docstring opens "Principle
    # (James): you only demand compensation for the ASYMMETRY" -- it attributes the method to
    # James by name. On 2026-09-02, shown it: "I don't even know what the 'skew corridor' is." A
    # whole section of the agreed workflow design (WORKFLOW-DESIGN-2026-09-01.md 1.2) was written
    # on that attribution and ruled that the published premium should MOVE to this corridor. It
    # has been struck. Measured before it was: the switch would have raised the cost of equity in
    # 416 of 450 name-tenor rows, median +0.96pp, almost all of it the removal of the discount the
    # approved score gives a large-cap defensive -- because the corridor is an absolute premium
    # floored at zero and cannot produce a discount at all.
    #
    # `skew_diag_<T>.csv` is also where register item B1's 3.13% sentinel lived. It reached this
    # diagnostic and the retired Martin-Wagner curve, and nothing else; B1 is closed by this
    # deletion.

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
    # `k` and `idio_anchor` were printed here until 2026-09-02. Both belonged to the retired
    # Martin-Wagner / Merton-pass-through construction in asfp/coe.py and no longer exist.
    print(f"  rating={meta['rating']} offset=x{meta['offset']:.3f}")
    if meta.get("market_value_of_debt"):
        _pytm = meta.get("portfolio_ytm"); _mdur = meta.get("wavg_mod_duration")
        _ytm_s = f"{_pytm*100:.2f}%" if _pytm is not None else "n/a"
        _dur_s = f"{_mdur:.1f}y" if _mdur is not None else "n/a"
        print(f"  MV(debt)=${meta['market_value_of_debt']/1e9:.1f}B "
              f"portYTM={_ytm_s} modDur={_dur_s}  ({meta.get('mvd_basis','')})")


if __name__ == "__main__":
    main()
