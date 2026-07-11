"""
ASFP cross-sectional curve builder (weekly live engine).

Given, for a single as-of date:
  * nominal GSW Svensson parameters      (valid 1-30y)
  * real   GSW Svensson parameters        (trusted ~2-20y)
  * Cleveland Fed expected inflation       at 1..30y (percent)
  * a near-term (1y) expected-inflation nowcast
  * calibration settings (tau_lo, tau_hi, front floor, tail damping, seam window)

produce the four term structures (nominal, real, breakeven, expected inflation),
the premium spread phi, one-year forwards for each, discount factors, and
provenance flags.

Design note
-----------
This is the *cross-sectional* construction appropriate to weekly recompute.
Correctness properties it preserves:
  - expected inflation = Cleveland Fed, held between monthly releases (sticky);
  - between releases, market moves flow into phi (premium), NOT into expected
    inflation;
  - the unobserved front real yield moves with fresh nominal, which carries the
    policy-rate signal — the economically dominant front driver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import curves


def _smootherstep(x):
    """0..1 smooth ramp (C2) for x in [0,1]; clipped outside."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def reliability_weight(taus, tau_lo, tau_hi, window):
    """Weight ~1 inside [tau_lo, tau_hi], tapering to 0 outside over `window`."""
    taus = np.asarray(taus, dtype=float)
    w = np.ones_like(taus)
    # lower taper: 0 at tau_lo-window, 1 at tau_lo
    lo = _smootherstep((taus - (tau_lo - window)) / window)
    # upper taper: 1 at tau_hi, 0 at tau_hi+window
    hi = 1.0 - _smootherstep((taus - tau_hi) / window)
    w = np.minimum(lo, hi)
    return np.clip(w, 0.0, 1.0)


def whittaker_smooth(y, w, lam):
    """Penalized smoother: minimize sum w_i (z_i - y_i)^2 + lam * sum (D2 z)^2.
    High w = trust the point (fit closely); low w = allow more smoothing.
    `lam` sets overall smoothing strength (penalizes curvature)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    A = np.diag(w) + lam * (D.T @ D)
    return np.linalg.solve(A, w * y)


def _fidelity_profile(grid):
    """Per-maturity trust weights for smoothing: hold the 5-18y core tightly,
    smooth GSW's noisy short end (2-4y) and the model-constructed 1y and 20-30y
    more."""
    grid = np.asarray(grid, dtype=float)
    w = np.ones_like(grid)
    for i, m in enumerate(grid):
        if m <= 1:
            w[i] = 0.25
        elif m <= 4:
            w[i] = 0.45
        elif m <= 18:
            w[i] = 1.0
        elif m <= 20:
            w[i] = 0.70
        else:
            w[i] = 0.35
    return w


def extrapolate_phi(taus, phi_obs_func, tau_lo, tau_hi, front_floor, tail_damp):
    """Extrapolated premium spread phi_hat on `taus`.

    Inside [tau_lo, tau_hi] returns the observed phi. Front pulls toward
    `front_floor` (typically small negative, the liquidity floor). Back damps
    the near-boundary slope toward flat.
    """
    taus = np.asarray(taus, dtype=float)
    phi_lo = float(phi_obs_func(tau_lo))
    phi_hi = float(phi_obs_func(tau_hi))
    # slope of observed phi just inside the upper boundary (for damped tail)
    h = 0.5
    slope_hi = (phi_hi - float(phi_obs_func(tau_hi - h))) / h

    out = np.empty_like(taus)
    for i, t in enumerate(taus):
        if t < tau_lo:
            # linear from floor at tau=0 to phi_lo at tau=tau_lo
            out[i] = front_floor + (phi_lo - front_floor) * (t / tau_lo)
        elif t > tau_hi:
            out[i] = phi_hi + tail_damp * slope_hi * (t - tau_hi)
        else:
            out[i] = float(phi_obs_func(t))
    return out


def build_cross_section(nominal_params, real_params, cf_expinf, nowcast_1yr,
                        calib=None, grid=None):
    """Return a DataFrame indexed by maturity (years) with the four curves,
    phi, forwards, discount factors and provenance flags.

    Parameters
    ----------
    nominal_params, real_params : dicts with keys b0,b1,b2,b3,t1,t2 (percent)
    cf_expinf : array length len(grid) of Cleveland Fed expected inflation (pct)
                aligned to `grid` maturities
    nowcast_1yr : float, near-term 1y expected inflation (pct)
    calib : dict of calibration settings (see defaults below)
    grid  : maturities in years (default 1..30 integer)
    """
    if grid is None:
        grid = np.arange(1, 31, dtype=float)
    grid = np.asarray(grid, dtype=float)

    c = dict(tau_lo=2.0, tau_hi=20.0, front_floor=-0.15,
             tail_damp=0.5, seam_window=1.0, nowcast_blend_to=2.0,
             smooth=True, smooth_lambda=30.0)
    if calib:
        c.update(calib)

    # --- nominal (observed everywhere) ---
    nominal = curves.svensson_zero(grid, **nominal_params)

    # --- real from GSW (trusted in band), and observed breakeven ---
    real_gsw = curves.svensson_zero(grid, **real_params)
    bei_gsw = nominal - real_gsw            # GSW breakeven where trusted

    # --- expected inflation: Cleveland Fed, with nowcast override at the front
    cf = np.asarray(cf_expinf, dtype=float).copy()
    exp_infl = cf.copy()
    blend_to = c["nowcast_blend_to"]
    for i, t in enumerate(grid):
        if t <= blend_to:
            # blend nowcast (at t<=1) into CF by t=blend_to
            wt = _smootherstep((t - 1.0) / max(blend_to - 1.0, 1e-9)) if blend_to > 1 else 1.0
            anchor = nowcast_1yr if t <= 1.0 else nowcast_1yr
            exp_infl[i] = (1 - wt) * anchor + wt * cf[i] if t >= 1.0 else nowcast_1yr

    # --- observed phi in band (breakeven - expected inflation) ---
    # build an interpolator for observed phi over the trusted band
    band_mask = (grid >= c["tau_lo"]) & (grid <= c["tau_hi"])
    phi_obs_band = (bei_gsw - exp_infl)[band_mask]
    band_taus = grid[band_mask]

    def phi_obs_func(t):
        return float(np.interp(t, band_taus, phi_obs_band))

    phi_hat = extrapolate_phi(grid, phi_obs_func, c["tau_lo"], c["tau_hi"],
                              c["front_floor"], c["tail_damp"])

    # --- constructed breakeven & real from CF + phi_hat ---
    bei_constructed = exp_infl + phi_hat
    real_constructed = nominal - bei_constructed

    # --- blend GSW real (trusted band) with constructed real (front/back) ---
    w = reliability_weight(grid, c["tau_lo"], c["tau_hi"], c["seam_window"])
    real = w * real_gsw + (1 - w) * real_constructed

    # light penalized smoothing to temper GSW's noisy short end and the seams,
    # so the forward strip (which amplifies point-to-point noise) stays clean
    if c.get("smooth", True):
        real = whittaker_smooth(real, _fidelity_profile(grid), c["smooth_lambda"])

    breakeven = nominal - real
    # keep phi consistent with the (smoothed) real curve
    phi = breakeven - exp_infl

    # --- provenance ---
    prov = np.where(band_mask, "observed",
                    np.where(grid < c["tau_lo"], "front-constructed", "back-constructed"))

    df = pd.DataFrame({
        "maturity": grid,
        "nominal": nominal,
        "real": real,
        "breakeven": breakeven,
        "exp_inflation": exp_infl,
        "phi": phi,
        "reliability": w,
        "provenance": prov,
    }).set_index("maturity")

    # --- forwards (one-year) for each rate curve ---
    for col in ["nominal", "real", "breakeven", "exp_inflation"]:
        z = df[col].to_numpy()
        df[f"{col}_fwd1y"] = curves.one_year_forwards(z, short_rate=z[0])

    # --- discount factors ---
    df["disc_nominal"] = curves.discount_factor(grid, df["nominal"].to_numpy())
    df["disc_real"] = curves.discount_factor(grid, df["real"].to_numpy())

    return df


def headline_points(df):
    """Standard watched summary points from a built cross-section."""
    def z(col, t):
        return float(np.interp(t, df.index.to_numpy(), df[col].to_numpy()))

    def fwd(col, a, b):
        za, zb = z(col, a), z(col, b)
        return (b * zb - a * za) / (b - a)

    return {
        "real_5y": z("real", 5), "real_10y": z("real", 10), "real_30y": z("real", 30),
        "breakeven_10y": z("breakeven", 10),
        "exp_infl_10y": z("exp_inflation", 10),
        "5y5y_breakeven": fwd("breakeven", 5, 10),
        "5y5y_exp_infl": fwd("exp_inflation", 5, 10),
        "5y5y_real": fwd("real", 5, 10),
        "slope_real_2s10s": z("real", 10) - z("real", 2),
        "slope_nominal_2s10s": z("nominal", 10) - z("nominal", 2),
    }
