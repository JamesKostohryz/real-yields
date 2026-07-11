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

CURVE_COLS = ["as_of", "maturity", "nominal", "real", "breakeven",
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

    out = df.reset_index()
    out.insert(0, "as_of", as_of)
    curve = out[CURVE_COLS].round(4)
    curve.to_csv(f"{OUTDIR}/curve_latest.csv", index=False)

    hp = model.headline_points(df)
    pd.DataFrame([{"as_of": as_of, **{k: round(v, 4) for k, v in hp.items()}}]) \
        .to_csv(f"{OUTDIR}/headline_latest.csv", index=False)

    # point-in-time history (git commits provide the vintage trail)
    histfile = f"{OUTDIR}/history.csv"
    hist = curve.copy()
    hist.insert(0, "run_utc", dt.datetime.utcnow().isoformat(timespec="seconds"))
    hist.to_csv(histfile, mode="a", header=not os.path.exists(histfile), index=False)

    print(f"OK  as_of={as_of}  cf_date={cf_date}  rows={len(curve)}")


if __name__ == "__main__":
    main()
