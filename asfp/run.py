"""
Weekly entry point. Fetches the latest data, builds the four curves and the
one-year forward strip, and writes CSVs that a Google Sheet pulls in via
IMPORTDATA. Also appends a point-in-time history row.

Run locally or in CI:  python -m asfp.run   (needs env var FRED_API_KEY)
"""
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd

from . import datasources as ds
from . import model
from . import curves
from . import credit
from . import erp
from . import units

OUTDIR = "outputs"
GRID = np.arange(1, 31, dtype=float)

# as_of lives on the headline tab (kept out of the curve rows so Google Sheets
# doesn't render a date serial in every row)
CURVE_COLS = ["maturity", "nominal", "real", "breakeven",
              "exp_inflation", "phi", "nominal_fwd1y", "real_fwd1y",
              "breakeven_fwd1y", "exp_inflation_fwd1y", "reliability", "provenance"]

# --- v1 default calibration (refined later against history) ---
CALIB = dict(tau_lo=2.0, tau_hi=20.0, front_floor=-0.15,
             tail_damp=0.5, seam_window=1.0, nowcast_blend_to=2.0)


def main():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY environment variable is not set.")

    nom = ds.latest_gsw_params(ds.fetch_text(ds.GSW_NOMINAL_URL))
    real = ds.latest_gsw_params(ds.fetch_text(ds.GSW_REAL_URL))
    cf, cf_date = ds.fetch_expinf(key)

    pk = lambda d: {k: d[k] for k in ["b0", "b1", "b2", "b3", "t1", "t2"]}
    nominal_base = curves.svensson_zero(GRID, **pk(nom))
    real_base = curves.svensson_zero(GRID, **pk(real))

    # Roll the weekly GSW curves forward to today using the daily FRED moves
    # (nominal DGS + real DFII) since each series' GSW as-of date. Non-fatal:
    # if the roll fails, fall back to the plain weekly GSW curve.
    as_of = max(nom["date"], real["date"])
    nominal, real_gsw = nominal_base, real_base
    try:
        ndm, ndv, nlt = ds.fetch_curve_delta(key, ds.DGS_MAP, nom["date"])
        rdm, rdv, rlt = ds.fetch_curve_delta(key, ds.DFII_MAP, real["date"])
        if ndm:
            nominal = model.roll_forward(nominal_base, GRID, ndm, ndv)
        if rdm:
            real_gsw = model.roll_forward(real_base, GRID, rdm, rdv, front_fill_below=5)
        as_of = max(d for d in [nlt, rlt, as_of] if d)
        print(f"rolled forward to {as_of} "
              f"(nominal pts={len(ndm)}, real pts={len(rdm)})")
    except Exception as e:
        print(f"roll-forward skipped (non-fatal, using weekly GSW): {e}")

    # v1: front expected-inflation anchor = Cleveland Fed 1y (nowcast override later)
    df = model.build_cross_section(nominal, real_gsw, cf,
                                   nowcast_1yr=float(cf[0]),
                                   calib=CALIB, grid=GRID)
    os.makedirs(OUTDIR, exist_ok=True)

    curve = df.reset_index()[CURVE_COLS].round(4)
    curve.to_csv(f"{OUTDIR}/curve_latest.csv", index=False)

    # annual-decimal variant for the valuation engine (rates via exp(cc/100)-1)
    ann_cols = ["real", "real_fwd1y", "exp_inflation", "exp_inflation_fwd1y",
                "breakeven", "breakeven_fwd1y", "nominal", "nominal_fwd1y"]
    curve_ann = curve[["maturity"] + ann_cols].copy()
    for c in ann_cols:
        curve_ann[c] = units.annualize_rate(curve_ann[c])
    # the valuation engine keys per-horizon on `tenor`; publish under that name
    curve_ann = curve_ann.rename(columns={"maturity": "tenor"})
    curve_ann.round(6).to_csv(f"{OUTDIR}/curve_latest_annual.csv", index=False)

    # display-ready summary (human labels, grouped) for the Summary tab
    hp = model.headline_points(df)
    g = lambda k: round(hp[k], 3)
    display_rows = [
        ("As of", as_of),
        ("", ""),
        ("REAL YIELDS (%)", ""),
        ("5-Year Real", g("real_5y")),
        ("10-Year Real", g("real_10y")),
        ("30-Year Real", g("real_30y")),
        ("5y5y Forward Real", g("5y5y_real")),
        ("", ""),
        ("BREAKEVEN INFLATION (%)", ""),
        ("5-Year Breakeven", g("breakeven_5y")),
        ("10-Year Breakeven", g("breakeven_10y")),
        ("30-Year Breakeven", g("breakeven_30y")),
        ("5y5y Forward Breakeven", g("5y5y_breakeven")),
        ("", ""),
        ("EXPECTED INFLATION (%)", ""),
        ("10-Year Expected Inflation", g("exp_infl_10y")),
        ("5y5y Forward Expected Inflation", g("5y5y_exp_infl")),
        ("", ""),
        ("CURVE SLOPES (pp)", ""),
        ("Real 2s10s", g("slope_real_2s10s")),
        ("Nominal 2s10s", g("slope_nominal_2s10s")),
    ]
    pd.DataFrame(display_rows, columns=["Metric", "Value"]).to_csv(
        f"{OUTDIR}/headline_latest.csv", index=False)

    # lean history-of-key-points feed (one row per as-of date) for trend charts
    hist_keys = ["nominal_5y", "nominal_10y", "nominal_30y",
                 "real_5y", "real_10y", "real_30y", "5y5y_real",
                 "breakeven_5y", "breakeven_10y", "breakeven_30y", "5y5y_breakeven",
                 "exp_infl_5y", "exp_infl_10y", "exp_infl_30y",
                 "slope_real_2s10s", "slope_nominal_2s10s"]
    hist_row = {"as_of": as_of, **{k: round(hp[k], 4) for k in hist_keys}}
    hhfile = f"{OUTDIR}/headline_history.csv"
    if os.path.exists(hhfile):
        h = pd.read_csv(hhfile)
        h = h[h["as_of"].astype(str) != str(as_of)]      # one row per date
        h = pd.concat([h, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        h = pd.DataFrame([hist_row])
    h.sort_values("as_of").to_csv(hhfile, index=False)

    # point-in-time history (git commits provide the vintage trail)
    histfile = f"{OUTDIR}/history.csv"
    hist = curve.copy()
    hist.insert(0, "as_of", as_of)
    hist.insert(0, "run_utc", dt.datetime.utcnow().isoformat(timespec="seconds"))
    hist.to_csv(histfile, mode="a", header=not os.path.exists(histfile), index=False)

    # --- non-fatal validation vs FRED's published series -------------------
    # confirms our breakevens/real match the official numbers, so phi (their
    # difference vs Cleveland Fed expected inflation) is trustworthy.
    try:
        idx = df.index.to_numpy()
        zf = lambda col, t: float(np.interp(t, idx, df[col].to_numpy()))
        checks = [
            ("breakeven_5y", "T5YIE", zf("breakeven", 5)),
            ("breakeven_10y", "T10YIE", zf("breakeven", 10)),
            ("breakeven_5y5y_fwd", "T5YIFR", hp["5y5y_breakeven"]),
            ("real_5y", "DFII5", zf("real", 5)),
            ("real_10y", "DFII10", zf("real", 10)),
            ("real_30y", "DFII30", zf("real", 30)),
        ]
        rows = []
        for metric, sid, model_val in checks:
            fred_val, fdate = ds.fetch_fred_latest(key, sid)
            if fred_val is None:
                continue
            rows.append(dict(metric=metric, model=round(model_val, 4),
                             fred=round(fred_val, 4),
                             diff_bp=round((model_val - fred_val) * 100, 1),
                             fred_date=fdate))
        if rows:
            pd.DataFrame(rows).to_csv(f"{OUTDIR}/validation_latest.csv", index=False)
            print("validation vs FRED written")
    except Exception as e:
        print(f"validation step skipped (non-fatal): {e}")

    # --- non-fatal: aggregate per-rating credit grid (cost-of-debt backbone) ---
    try:
        real_fwd = df["real_fwd1y"].to_numpy()
        cg = credit.build_credit_grid(key, GRID, real_fwd)
        cg.round(4).to_csv(f"{OUTDIR}/market_credit_latest.csv")
        print("market credit grid written")

        # market equity-risk-premium grid (Martin/VIX anchor + credit floor)
        eg = erp.build_from_fred(key, GRID, cg["ig_index_spread"].to_numpy())
        eg.round(4).to_csv(f"{OUTDIR}/erp_market_latest.csv")
        print(f"market ERP grid written (VIX={eg.attrs.get('vix')})")

        # annual-decimal variants for the valuation engine
        cg_ann = cg.copy()
        for c in [col for col in cg_ann.columns if col.startswith("real_")]:
            cg_ann[c] = units.annualize_rate(cg_ann[c])
        cg_ann.round(6).to_csv(f"{OUTDIR}/market_credit_latest_annual.csv")
        eg_ann = eg.copy()
        eg_ann["market_erp"] = units.to_decimal(eg_ann["market_erp"])
        eg_ann[["market_erp"]].round(6).to_csv(f"{OUTDIR}/erp_market_latest_annual.csv")

        # --- v2 (non-breaking): market ERP from the VIX TERM STRUCTURE, blended to
        # the bond floor over 5-30y, published to 150y. New files only; the existing
        # erp_market_latest.csv above is untouched until the engine cuts over. ---
        try:
            from . import volsurface as vs
            GRID_V2 = np.arange(1, 151, dtype=float)
            # short-end VIX term structure: FRED (VIX 30d, VIX3M) + CBOE (9d, 6m, 1y)
            vols = vs.fetch_fred_vols(key, ds.fetch_fred_latest)   # reliable base
            vols.update(vs.fetch_cboe_vols())                      # best-effort term structure
            # long-dated extension: SPX/SPY LEAPS to ~3y, then a best-effort CME hook.
            # This is the "futures options" reach that keeps the ERP options-driven past 1y.
            extra_ts = list(vs.fetch_index_vol_ts_yf())
            extra_ts += list(vs.fetch_cme_settlement_vols())
            floor_v2 = vs.floor_from_credit_grid(cg, wedge=1.0)
            me2, vol_ts = vs.build_v2_market_erp(GRID_V2, vols, floor_v2,
                                                 converge_year=30.0, extra_ts=extra_ts)
            me2.round(4).to_csv(f"{OUTDIR}/market_erp_v2_latest.csv")
            me2a = me2.copy(); me2a["market_erp"] = units.to_decimal(me2a["market_erp"])
            me2a.round(6).to_csv(f"{OUTDIR}/market_erp_v2_latest_annual.csv")
            # publish the exact vol term structure that fed the ERP (audit + the Sheet)
            pd.DataFrame(vol_ts, columns=["tenor", "index_vol"]).round(3).to_csv(
                f"{OUTDIR}/index_vol_ts_latest.csv", index=False)
            obs_max = max(t for t, _ in vol_ts)
            print(f"market ERP v2 written: {len(vol_ts)} vol pts (obs to {obs_max:.2f}y; "
                  f"short={sorted(vols)}, long={len(extra_ts)}), floor={floor_v2:.2f}%, "
                  f"1y={me2['market_erp'].loc[1]:.2f} 5y={me2['market_erp'].loc[5]:.2f} "
                  f"30y={me2['market_erp'].loc[30]:.2f}")
        except Exception as e:
            print(f"market ERP v2 skipped (non-fatal): {e}")
    except Exception as e:
        print(f"credit/erp grid skipped (non-fatal): {e}")

    # --- non-fatal: measured average-stock variance (idiosyncratic term input) ---
    # replaces the fixed-correlation assumption with a live large-cap basket average.
    try:
        from . import company as comp
        avg_var, n = comp.basket_avg_variance()
        pd.DataFrame([
            {"field": "avg_stock_var", "value": round(avg_var, 6)},
            {"field": "avg_stock_vol", "value": round(avg_var ** 0.5, 4)},
            {"field": "n_names", "value": n},
        ]).to_csv(f"{OUTDIR}/market_micro_latest.csv", index=False)
        print(f"avg-stock variance written: {avg_var:.4f} (vol {avg_var**0.5:.1%}, n={n})")
    except Exception as e:
        print(f"avg-stock variance skipped (non-fatal): {e}")

    print(f"OK  as_of={as_of}  cf_date={cf_date}  rows={len(curve)}")


if __name__ == "__main__":
    main()
