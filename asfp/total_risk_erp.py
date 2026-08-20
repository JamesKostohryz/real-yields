"""
Total-risk (constant-Sharpe) single-name ERP, with a VIX-term-structure market ERP.

Principle (James): a single, undiversified stock is never LESS risky than the
diversified market, so its ERP is never below the market's. We enforce this by
pricing every name at the market's Sharpe ratio applied to the name's OWN total
risk:

    single_name_ERP(t) = market_ERP(t) × R_i(t),     R_i(t) ≥ 1
    idiosyncratic_i(t)  = market_ERP(t) × (R_i(t) − 1)   ≥ 0   (additive-contract form)

R_i(t) = the name's total risk ÷ the market's total risk, across the horizon:
  • front (0..obs_max): observed option-implied vol ratio  σ_i(t)/σ_mkt(t)  — using the
    whole vol TERM STRUCTURE (VIX9D…VIX1Y, extended toward ~5y via long-dated SPX/
    E-mini settlement IVs), not a single spot number.
  • maturity: hold the front ratio (persistence), floored at 1.
  • tail (past ORY): the Merton-elevator ramp lifts R toward a distressed level as the
    firm heads to junk — obsolescence expressed as an exploding risk ratio, not an
    arbitrary additive multiple.

The MARKET ERP itself uses the observed variance term structure at the front (Martin:
ERP ≈ implied variance) and glides over ~20 years to a floor = bond risk premium +
equity convergence premium.

All percent (cc). Pure/injectable; live vol/curve reads happen in the job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import elevator as ev

DEFAULT_R_DISTRESS = 6.0        # risk ratio of a near-death firm (~110% vol / ~18% mkt)


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


# ------------------------------------------------------------ risk ratio R_i(t)
def build_risk_ratio(grid, stock_vol_ts, index_vol_ts, issuer_rating, cg, category,
                     ory_override=None, r_distress=DEFAULT_R_DISTRESS):
    """R_i(t) ≥ 1: vol ratio at the front, held through maturity, lifted toward
    `r_distress` past ORY by the elevator ramp."""
    grid = np.asarray(grid, float)
    sv, s_max = _interp_ts(grid, stock_vol_ts)
    iv, i_max = _interp_ts(grid, index_vol_ts)
    obs_max = min(s_max, i_max)
    ratio = sv / np.maximum(iv, 1e-6)
    # hold the last observed ratio flat through maturity
    r_at_max = float(np.interp(obs_max, grid, ratio))
    r_base = np.where(grid <= obs_max, ratio, r_at_max)
    r_base = np.maximum(r_base, 1.0)                         # never below the market

    # tail: elevator ramp lifts R from its maturity level toward the distressed level
    preset = ev.CATEGORY_PRESETS[category]
    ory = float(preset["ory"] if ory_override is None else ory_override)
    W = ev.derive_width(issuer_rating, preset["floor"], preset["rate"])
    p = ev.progress(grid, ory, W)
    R = r_base + p * (r_distress - r_at_max)
    R = np.maximum(R, 1.0)
    return pd.DataFrame({"tenor": grid, "R": R, "r_base": r_base, "p_elevator": p}).set_index("tenor")


# ------------------------------------------------------------ assembly
def single_name_erp(grid, market_erp, R):
    """single_name_ERP = market_ERP × R ; idiosyncratic = market_ERP × (R−1) ≥ 0."""
    market_erp = np.asarray(market_erp, float)
    R = np.asarray(R, float)
    idio = market_erp * (R - 1.0)
    return pd.DataFrame({"tenor": np.asarray(grid, float),
                         "market_erp": market_erp,
                         "idiosyncratic": idio,
                         "single_name_erp": market_erp * R}).set_index("tenor")


def assemble_coe_v2(grid, real_rf, market_erp, stock_vol_ts, index_vol_ts,
                    issuer_rating, category, ory_override=None,
                    r_distress=DEFAULT_R_DISTRESS):
    """Full single-name real COE table (v2): real_rf + market_ERP×R, with the
    additive-contract idiosyncratic = market_ERP×(R−1) ≥ 0. Columns:
    real_rf, market_erp, idiosyncratic, single_name_erp (=company_erp), real_coe."""
    R = build_risk_ratio(grid, stock_vol_ts, index_vol_ts, issuer_rating, None,
                         category, ory_override, r_distress)["R"].to_numpy()
    out = single_name_erp(grid, market_erp, R)
    out.insert(0, "real_rf", np.asarray(real_rf, float))
    out = out.rename(columns={"single_name_erp": "company_erp"})
    out["real_coe"] = out["real_rf"].to_numpy() + out["company_erp"].to_numpy()
    return out
