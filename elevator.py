"""
Weekly rate-infrastructure job (STANDALONE entry point).

Builds the four term structures — nominal, real, breakeven, expected inflation —
and their one-year-forward strips on a 1..30y grid, and writes the CSVs a Google
Sheet (IMPORTDATA) or Excel (Power Query) pulls from GitHub. Self-contained: uses
only datasources / curves / model / units.

Pipeline:
  1. Fetch the Fed GSW Svensson params (nominal feds200628, real feds200805).
  2. Fetch Cleveland Fed expected inflation (FRED EXPINF1..30YR).
  3. Evaluate Svensson zero curves, roll them forward to today with daily FRED
     moves (DGS nominal, DFII real).
  4. build_cross_section: expected inflation = Cleveland Fed (1y nowcast blended
     to ~2y); real = GSW in the trusted 2-20y band, constructed elsewhere from
     nominal − (expected inflation + extrapolated risk premium phi); light
     penalized smoothing; one-year forwards; provenance flags.
  5. Write curve_latest.csv (+ annual-decimal variant), a headline summary,
     history feeds, and a non-fatal validation vs FRED's published series.

Run:  FRED_API_KEY=xxxx  python -m asfp.run_curves
"""
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd

from . import datasources as ds
from . import curves
from . import model
from . import units

OUTDIR = "outputs"
GRID = np.arange(1, 31, dtype=float)

CURVE_COLS = ["maturity", "nominal", "real", "breakeven",
              "exp_inflation", "phi", "nominal_fwd1y", "real_fwd1y",
              "breakeven_fwd1y", "exp_inflation_fwd1y", "reliability", "provenance"]

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

    # roll the weekly GSW curves forward to today with daily FRED moves (non-fatal)
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
        print(f"rolled forward to {as_of} (nominal pts={len(ndm)}, real pts={len(rdm)})")
    except Exception as e:
        print(f"roll-forward skipped (non-fatal, using weekly GSW): {e}")

    df = model.build_cross_section(nominal, real_gsw, cf, nowcast_1yr=float(cf[0]),
                                   calib=CALIB, grid=GRID)
    os.makedirs(OUTDIR, exist_ok=True)

    curve = df.reset_index()[CURVE_COLS].round(4)
    curve.to_csv(f"{OUTDIR}/curve_latest.csv", index=False)

    # annual-decimal variant (rates via exp(cc/100)-1) for downstream consumers
    ann_cols = ["real", "real_fwd1y", "exp_inflation", "exp_inflation_fwd1y",
                "breakeven", "breakeven_fwd1y", "nominal", "nominal_fwd1y"]
    curve_ann = curve[["maturity"] + ann_cols].copy()
    for c in ann_cols:
        curve_ann[c] = units.annualize_rate(curve_ann[c])
    curve_ann = curve_ann.rename(columns={"maturity": "tenor"})
    curve_ann.round(6).to_csv(f"{OUTDIR}/curve_latest_annual.csv", index=False)

    # display-ready headline summary
    hp = model.headline_points(df)
    g = lambda k: round(hp[k], 3)
    display_rows = [
        ("As of", as_of), ("", ""),
        ("REAL YIELDS (%)", ""),
        ("5-Year Real", g("real_5y")), ("10-Year Real", g("real_10y")),
        ("30-Year Real", g("real_30y")), ("5y5y Forward Real", g("5y5y_real")), ("", ""),
        ("BREAKEVEN INFLATION (%)", ""),
        ("5-Year Breakeven", g("breakeven_5y")), ("10-Year Breakeven", g("breakeven_10y")),
        ("30-Year Breakeven", g("breakeven_30y")),
        ("5y5y Forward Breakeven", g("5y5y_breakeven")), ("", ""),
        ("EXPECTED INFLATION (%)", ""),
        ("10-Year Expected Inflation", g("exp_infl_10y")),
        ("5y5y Forward Expected Inflation", g("5y5y_exp_infl")), ("", ""),
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
        h = h[h["as_of"].astype(str) != str(as_of)]
        h = pd.concat([h, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        h = pd.DataFrame([hist_row])
    h.sort_values("as_of").to_csv(hhfile, index=False)

    # point-in-time vintage log (git commits provide the trail)
    histfile = f"{OUTDIR}/history.csv"
    hist = curve.copy()
    hist.insert(0, "as_of", as_of)
    hist.insert(0, "run_utc", dt.datetime.utcnow().isoformat(timespec="seconds"))
    hist.to_csv(histfile, mode="a", header=not os.path.exists(histfile), index=False)

    # --- non-fatal validation vs FRED's published series -------------------
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

    print(f"OK  as_of={as_of}  cf_date={cf_date}  rows={len(curve)}")


if __name__ == "__main__":
    main()
