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

    # measured average-stock variance (idiosyncratic term); None -> fixed-corr fallback
    avg_stock_var = None
    mp = f"{OUTDIR}/market_micro_latest.csv"
    if os.path.exists(mp):
        try:
            m = pd.read_csv(mp).set_index("field")["value"]
            avg_stock_var = float(m.loc["avg_stock_var"])
        except Exception:
            avg_stock_var = None
    return cg, real_rf, market_erp, vix, avg_stock_var


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

    cg, real_rf, market_erp, vix, avg_stock_var = _load_market()

    from . import company as comp                       # yfinance import deferred
    fund = comp.fetch_company(ticker)

    tables, meta = issuer.assemble(ticker, cg, real_rf, market_erp, vix, fund, bonds,
                                   avg_stock_var=avg_stock_var)
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
    ]
    pd.DataFrame(stamp).to_csv(f"{OUTDIR}/run_stamp_{ticker}.csv", index=False)
    written.append(f"run_stamp_{ticker}.csv")

    # --- v2 (non-breaking): total-risk single-name COE to 150y, from the VIX-curve
    # market ERP. New files coe_v2_<T>_latest(.csv/_annual.csv); existing outputs
    # untouched until the engine cuts over. Needs the weekly market_erp_v2 file. ---
    try:
        from . import total_risk_erp as trv, units
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
            # exact additive annual-decimal variant (marginal compounding)
            rfp, mep, idp = (coe2[c].to_numpy() for c in ("real_rf", "market_erp", "idiosyncratic"))
            l0 = np.expm1(rfp / 100); l1 = np.expm1((rfp + mep) / 100); l2 = np.expm1((rfp + mep + idp) / 100)
            pd.DataFrame({"tenor": GV, "real_rf": l0, "market_erp": l1 - l0,
                          "idiosyncratic": l2 - l1, "company_erp": l2 - l0, "real_coe": l2}
                         ).set_index("tenor").round(9).to_csv(f"{OUTDIR}/coe_v2_{ticker}_latest_annual.csv")
            written += [f"coe_v2_{ticker}_latest.csv", f"coe_v2_{ticker}_latest_annual.csv"]

            # --- EFFECTIVE (collapsed) ERP: the whole term structure summarized as one
            # cash-flow-PV-weighted rate, the equity analogue of a bond's YTM. Collapse
            # nested cumulative curves so the effective pieces still add up. ---
            from . import collapse as col
            growth = float(os.environ.get("COE_CF_GROWTH", "2.0"))    # real CF growth for the weights
            coe_curve = coe2["real_coe"].to_numpy()
            rf_curve = coe2["real_rf"].to_numpy()
            rfmkt_curve = rf_curve + coe2["market_erp"].to_numpy()
            eff_coe = col.collapse_rate(GV, coe_curve, growth=growth)
            eff_rf = col.collapse_rate(GV, rf_curve, growth=growth)
            eff_rfmkt = col.collapse_rate(GV, rfmkt_curve, growth=growth)
            eff_mkt = eff_rfmkt - eff_rf
            eff_company = eff_coe - eff_rf
            eff_idio = eff_company - eff_mkt
            eff_rows = [
                ("real_rf", round(eff_rf, 4)),
                ("market_erp", round(eff_mkt, 4)),
                ("idiosyncratic", round(eff_idio, 4)),
                ("company_erp", round(eff_company, 4)),
                ("real_coe", round(eff_coe, 4)),
                ("cf_growth", round(growth, 3)),
            ]
            pd.DataFrame(eff_rows, columns=["field", "value_pct"]).to_csv(
                f"{OUTDIR}/coe_v2_{ticker}_effective.csv", index=False)
            # annual-decimal companion (rates via exp(cc/100)-1; premia as marginal steps)
            e0 = np.expm1(eff_rf / 100); e1 = np.expm1((eff_rf + eff_mkt) / 100)
            e2 = np.expm1((eff_rf + eff_mkt + eff_idio) / 100)
            pd.DataFrame([
                ("real_rf", round(e0, 6)), ("market_erp", round(e1 - e0, 6)),
                ("idiosyncratic", round(e2 - e1, 6)), ("company_erp", round(e2 - e0, 6)),
                ("real_coe", round(e2, 6)),
            ], columns=["field", "value_decimal"]).to_csv(
                f"{OUTDIR}/coe_v2_{ticker}_effective_annual.csv", index=False)
            written += [f"coe_v2_{ticker}_effective.csv", f"coe_v2_{ticker}_effective_annual.csv"]

            r1 = stock_vol_ts[0][1] / max(index_vol_ts[0][1], 1e-6)
            print(f"  coe v2: R(front)={r1:.2f} cat={category} "
                  f"stock_vol_pts={len(stock_vol_ts)} obs_to={stock_vol_ts[-1][0]:.2f}y "
                  f"coe(1y)={coe2['real_coe'].loc[1]:.2f}% coe(100y)={coe2['real_coe'].loc[100]:.2f}%")
            print(f"  coe v2 EFFECTIVE: real_coe={eff_coe:.2f}%  company_erp={eff_company:.2f}%  "
                  f"(mkt={eff_mkt:.2f} + idio={eff_idio:.2f}) over rf={eff_rf:.2f}%")
    except Exception as e:
        print(f"  coe v2 skipped (non-fatal): {e}")

    # --- non-fatal DIAGNOSTIC: skew-priced ERP for this name vs the index, next to the
    # current variance-based number. Pure measurement; nothing consumes it. Lets us see
    # real skew results before deciding whether to adopt the approach. ---
    try:
        from . import company as comp
        dn = comp.skew_diag(ticker)
        if dn:
            var_erp = dn["atm"] ** 2 * 100                       # Martin(ATM), name's own
            mkt_skew = None
            msp = f"{OUTDIR}/market_skew_diag.csv"
            if os.path.exists(msp):
                mkt_skew = float(pd.read_csv(msp)["skew_erp"].iloc[0])
            row = {"ticker": ticker, "atm_vol": round(dn["atm"] * 100, 2),
                   "k_down_var": round(dn["k_down"] * 100, 3), "k_up_var": round(dn["k_up"] * 100, 3),
                   "skew_erp": round(dn["skew"] * 100, 3),
                   "variance_erp_ownvol": round(var_erp, 3),
                   "market_skew_erp": mkt_skew, "n_strikes": dn["n"]}
            pd.DataFrame([row]).to_csv(f"{OUTDIR}/skew_diag_{ticker}.csv", index=False)
            written.append(f"skew_diag_{ticker}.csv")
            mk = f" mkt_skew={mkt_skew:.2f}%" if mkt_skew is not None else ""
            print(f"  SKEW DIAG {ticker}: skew_erp={dn['skew']*100:.2f}%  "
                  f"variance_erp(ownvol)={var_erp:.2f}%  "
                  f"(down={dn['k_down']*100:.2f} up={dn['k_up']*100:.2f}, n={dn['n']}){mk}")
    except Exception as e:
        print(f"  skew diagnostic skipped (non-fatal): {e}")

    # --- non-fatal: SKEW-PRICED ERP term structure for this name (final engine).
    # Corridor off the name's own multi-tenor smiles; single names compress vs the index. ---
    try:
        from . import company as comp, erp_engine as ee
        import yfinance as _yf
        _tk = _yf.Ticker(ticker); _fi = _tk.fast_info
        _px = float(_fi.get("last_price") or _fi.get("lastPrice") or 0.0)
        smiles = comp.fetch_smiles(_tk, _px) if _px > 0 else {}
        if len(smiles) >= 2:
            GS = np.arange(1, 31, dtype=float)
            curve = ee.skew_erp_curve(smiles, GS, phi=1.0)
            curve.round(4).to_csv(f"{OUTDIR}/skew_erp_{ticker}.csv")
            written.append(f"skew_erp_{ticker}.csv")
            eff = ee.effective_erp(curve)
            rs = comp.realized_skew(ticker)
            if rs:
                pd.DataFrame([{"field": k, "value": v} for k, v in rs.items()]
                             ).to_csv(f"{OUTDIR}/skew_erp_{ticker}_realized.csv", index=False)
                written.append(f"skew_erp_{ticker}_realized.csv")
            print(f"  skew-ERP {ticker}: tenors={sorted(smiles)} eff={eff:.2f}% "
                  f"1y={curve['erp'].loc[1]:.2f} 5y={curve['erp'].loc[5]:.2f}"
                  + (f"  realized_corridor={rs['corridor']:.2f}" if rs else ""))
    except Exception as e:
        print(f"  skew-ERP {ticker} skipped (non-fatal): {e}")

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
