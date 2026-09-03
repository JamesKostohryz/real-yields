"""
The MARKET equity risk premium from the VIX term structure. Percent (cc).

The MARKET ERP uses the observed variance term structure at the front (Martin: ERP ≈ implied
variance) and glides over ~20 years to a floor = bond risk premium + equity convergence premium.
Pure/injectable; live vol/curve reads happen in the job.

RETIRED 2026-09-02 -- THE SINGLE-NAME (total-risk / constant-Sharpe) CONSTRUCTION.

This module used to carry a second thing: a per-company idiosyncratic premium built as

    single_name_ERP(t) = market_ERP(t) × R_i(t),   R_i(t) ≥ 1
    idiosyncratic_i(t) = market_ERP(t) × (R_i(t) − 1) ≥ 0

with R_i the name's own option-implied vol ratio, held flat past the observed horizon and lifted
toward a distressed level by the Merton elevator past ORY. `build_risk_ratio()` and
`single_name_erp()` implemented it and are gone.

James ruled 2026-09-02 that there is ONE approved method for a company's idiosyncratic premium --
the four-block risk score in `aeg-valuation/idio/` -- and that every other method is retired and
must not be referred to anywhere. This was one of five constructions of that single quantity that
were live simultaneously; it produced the `idiosyncratic` column of `coe_v2_<T>_latest_annual.csv`
(3.566% at tenor 1 for AMCR) and the `coe_v2_<T>_effective*` files (5.868%), against the +0.5179pp
the valuation actually used. **None of them ever reached a valuation.** `aeg-valuation`'s
`rate_feed.load_coe()` reads only `real_rf` and `market_erp` from this file, and has since
`b22d5f1`.

Note the shape of the disagreement, because it is the reason the ruling exists: this construction
floors the premium at zero and can only ADD to the market ERP, while the approved score is an
increment CENTRED on zero -- 14 of the 16 onboarded names take a discount of about −1pp. The two
cannot be reconciled by calibration; they are different objects.

`assemble_coe_v2()` SURVIVES and is deliberately kept: the engine reads its `real_rf` and
`market_erp` columns, and `coe_v2_<T>_latest_annual.csv` must go on being published.
Working: `aeg-project/docs/engine/A6-Cost-Of-Equity-FINDINGS-2026-09-02.md`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def martin_pct(vol_points):
    """Martin ERP (percent) from an implied vol in vol points: ERP ≈ variance.

    NOTE this is a SPOT / cumulative measure: an implied vol σ(T) describes the
    AVERAGE variance over [0,T], so martin(σ(T)) is the average ERP to horizon T,
    not the marginal ERP earned in year T. Use forward_erp() to get the per-year
    (marginal) premium the term-structure model actually discounts with."""
    v = np.asarray(vol_points, dtype=float)
    return v * v / 100.0


def forward_erp(grid, spot_erp):
    """Convert a SPOT ERP term structure into the FORWARD (marginal) ERP per year.

    A T-year implied vol prices the *cumulative* variance over [0,T] — like a 2-year
    option premium covering both years. The premium attributable to year T alone is
    the marginal (forward) variance, exactly analogous to a one-year forward rate on
    a yield curve:

        cumulative variance to t :  C(t) = spot_erp(t) · t
        forward ERP in year t    :  f(t) = [C(t) − C(t−1)] / Δt          (f(1)=spot(1))

    So the pieces telescope: Σ_{s≤t} f(s) = spot_erp(t) · t. This is the object the
    COE model must use, because the risk-free leg is a one-year forward (real_fwd1y)
    and cash flows are discounted year-by-year — the ERP for each year must likewise
    be that year's marginal premium, not the average-to-date. Floored at 0 (a period's
    variance cannot be negative), which also tames a downward vol kink at the front.

    On a rising vol curve the forward sits ABOVE the spot (later years cost more on the
    margin); on a falling/backwardated curve it sits BELOW (the "second year is cheaper"
    case) — both handled correctly."""
    grid = np.asarray(grid, dtype=float)
    spot = np.asarray(spot_erp, dtype=float)
    cum = spot * grid                                   # cumulative variance to each tenor
    fwd = np.empty_like(cum)
    fwd[0] = spot[0]                                    # year-1 forward == spot(1)
    dt = np.diff(grid)
    fwd[1:] = (cum[1:] - cum[:-1]) / np.where(dt == 0, 1.0, dt)
    return np.maximum(fwd, 0.0)


# The largest one-year FORWARD vol, as a multiple of the spot vol at the preceding tenor, that
# an index vol term structure may imply before the input is treated as a bad quote rather than a
# market view. 1.5 is a declared judgement, not a measurement, and it is deliberately loose: a
# genuine vol-term-structure steepening of 50% in a single year is already extreme.
#
# WHY THIS EXISTS. On 2026-08-20 outputs/index_vol_ts_latest.csv carried 18.493 at 1y, 19.166 at
# 1.5y, 17.75 at 2y and 21.768 at 3y — one thin three-year SPX LEAPS quote. forward_erp() is
# correct and does exactly what it says: C(2) = 3.1506 x 2 = 6.3013, C(3) = 4.7385 x 3 = 14.2155,
# forward(3) = 7.914. That is a 28.1% forward vol for year three against a 17.75% two-year spot,
# and it published market_erp_v2 as 3.42% at 1y, 2.88% at 2y and 7.91% at 3y, holding near 7.8%
# out to seven years. The daily ERP overlay masks it for the discount rate, but it does NOT touch
# the idiosyncratic column, so coe_v2_MSFT_effective.csv published idio 5.87% and real_coe 11.90%.
#
# The floor at zero in forward_erp() already handles a DOWNWARD kink. Nothing handled an upward
# one, because a large positive forward variance is arithmetically legitimate — which is exactly
# why it needed a data-quality check rather than a change to the maths.
MAX_FORWARD_VOL_RATIO = 1.5


def vol_ts_quality(ts, max_ratio=MAX_FORWARD_VOL_RATIO):
    """Flag tenors of an index vol term structure whose implied one-year FORWARD vol is
    implausible against the preceding spot. Returns (clean_ts, rejected), never raises.

    Reports rather than smooths: the offending POINT is dropped, the rest of the observed curve
    is kept, and the drop is returned so a caller can print it. Silently interpolating over a bad
    quote would leave a plausible-looking curve nobody could audit.
    """
    ts = sorted(((float(t), float(v)) for t, v in ts), key=lambda x: x[0])
    clean, rejected = [], []
    for i, (t, v) in enumerate(ts):
        if i == 0 or not clean:
            clean.append((t, v))
            continue
        pt, pv = clean[-1]
        dt = t - pt
        if dt <= 0:
            rejected.append((t, v, "non-increasing tenor"))
            continue
        cum_prev = (pv * pv / 100.0) * pt
        cum_here = (v * v / 100.0) * t
        fwd_var = (cum_here - cum_prev) / dt              # ERP points per year
        fwd_vol = (max(fwd_var, 0.0) * 100.0) ** 0.5      # back to vol points
        if fwd_vol > max_ratio * pv:
            rejected.append((t, v, "implied %.1f%% forward vol against %.1f%% spot at %.2fy "
                                   "(limit %.1fx)" % (fwd_vol, pv, pt, max_ratio)))
            continue
        clean.append((t, v))
    return clean, rejected


def _interp_ts(grid, ts):
    """Interpolate a term structure [(tenor_years, value), …] onto grid, flat beyond ends."""
    ten = np.array([t for t, _ in ts], float)
    val = np.array([v for _, v in ts], float)
    o = np.argsort(ten)
    return np.interp(grid, ten[o], val[o]), float(ten[o][-1])


# ------------------------------------------------------------ market ERP (vol curve)
def build_market_erp_curve(grid, index_vol_ts, floor, glide_half_life=5.0):
    """Market ERP (percent): Martin from the observed index vol TERM STRUCTURE out to
    its last tenor, then a glide to `floor` (~20y). `index_vol_ts` = [(yrs, vol_pts)…]
    e.g. [(0.08,17),(0.25,18.5),(0.5,19.5),(1,20),(3,21),(5,21.5)]."""
    grid = np.asarray(grid, float)
    # Data-quality guard on the SCRAPED input, applied here so every caller is protected.
    # The forward construction below is correct; a single thin long-dated LEAPS quote is not.
    index_vol_ts, _rejected = vol_ts_quality(index_vol_ts)
    for _t, _v, _why in _rejected:
        print("  [vol-ts] REJECTED index vol %.3f at tenor %.3fy: %s" % (_v, _t, _why))
    vol, obs_max = _interp_ts(grid, index_vol_ts)
    erp_obs = forward_erp(grid, martin_pct(vol))        # marginal (per-year) ERP, not spot
    erp_at_max = float(np.interp(obs_max, grid, erp_obs))
    out = np.where(
        grid <= obs_max,
        erp_obs,
        floor + (erp_at_max - floor) * 0.5 ** ((grid - obs_max) / glide_half_life),
    )
    return pd.DataFrame({"tenor": grid, "market_erp": out}).set_index("tenor")


def build_market_erp_blended(grid, index_vol_ts, floor, converge_year=30.0):
    """Market ERP that does not converge to bonds too fast — the futures-options market
    says the equity premium persists past 5y. Observed options set 0..obs_max; beyond,
    weight-average an EQUITY view (flat at the obs_max level — options persistence)
    against a BOND view (glide to `floor`), with equity weight 1→0 from obs_max to
    `converge_year`. From converge_year on it is the floor (the elevator takes the tail).
    """
    grid = np.asarray(grid, float)
    # Data-quality guard on the SCRAPED input, applied here so every caller is protected.
    # The forward construction below is correct; a single thin long-dated LEAPS quote is not.
    index_vol_ts, _rejected = vol_ts_quality(index_vol_ts)
    for _t, _v, _why in _rejected:
        print("  [vol-ts] REJECTED index vol %.3f at tenor %.3fy: %s" % (_v, _t, _why))
    vol, obs_max = _interp_ts(grid, index_vol_ts)
    erp_obs = forward_erp(grid, martin_pct(vol))        # marginal (per-year) ERP, not spot
    e5 = float(np.interp(obs_max, grid, erp_obs))
    out = np.empty_like(grid)
    span = max(converge_year - obs_max, 1e-6)
    for i, t in enumerate(grid):
        if t <= obs_max:
            out[i] = erp_obs[i]                      # observed options
        elif t >= converge_year:
            out[i] = floor                           # fully bond-anchored
        else:
            w = (converge_year - t) / span           # equity weight 1 -> 0
            e_bond = floor + (e5 - floor) * w        # bond convergence
            out[i] = w * e5 + (1.0 - w) * e_bond     # split the distance
    return pd.DataFrame({"tenor": grid, "market_erp": out}).set_index("tenor")


# ------------------------------------------------------------ assembly
# `build_risk_ratio()` and `single_name_erp()` stood here until 2026-09-02. They were the retired
# single-name construction; see the module docstring. Do not reinstate them, and do not add a
# company-specific term to the table below: a company's idiosyncratic premium comes from ONE
# place, `aeg-valuation/idio/company_curve_v2.py`, and it is added inside the engine.
def assemble_coe_v2(grid, real_rf, market_erp, stock_vol_ts=None, index_vol_ts=None,
                    issuer_rating=None, category=None, ory_override=None,
                    r_distress=None):
    """The two HOUSE-VIEW legs of the real cost of equity, by tenor. Columns:
    real_rf, market_erp. Nothing company-specific.

    The vol / rating / category / ORY arguments are RETAINED AND IGNORED so the two callers keep
    working unchanged; they were the inputs to the retired risk-ratio construction. They are kept
    rather than deleted because removing them turns a retirement into a call-site refactor across
    `asfp/run_company.py` and `datasources.py` for no gain -- and because a reader who finds them
    here should find the reason with them rather than wonder what was lost."""
    return pd.DataFrame({"tenor": np.asarray(grid, float),
                         "real_rf": np.asarray(real_rf, float),
                         "market_erp": np.asarray(market_erp, float)}).set_index("tenor")
