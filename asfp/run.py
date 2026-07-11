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
    # v1: front expected-inflation anchor = Cleveland Fed 1y (nowcast override is a later enhancement)
    df = model.build_cross_section(pk(nom), pk(real), cf,
                                   nowcast_1yr=float(cf[0]),
                                   calib=CALIB, grid=GRID)

    as_of = max(nom["date"], real["date"])
    os.makedirs(OUTDIR, exist_ok=True)

    curve = df.reset_index()[CURVE_COLS].round(4)
    curve.to_csv(f"{OUTDIR}/curve_latest.csv", index=False)

    # headline tab carries the as-of date + the watched summary points
    hp = model.headline_points(df)
    pd.DataFrame(
        [{"item": "as_of", "value": as_of}]
        + [{"item": k, "value": round(v, 4)} for k, v in hp.items()]
    ).to_csv(f"{OUTDIR}/headline_latest.csv", index=False)

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

    print(f"OK  as_of={as_of}  cf_date={cf_date}  rows={len(curve)}")


if __name__ == "__main__":
    main()
