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
    """Martin ERP (percent) from an implied vol in vol points: ERP ≈ variance."""
    v = np.asarray(vol_points, dtype=float)
    return v * v / 100.0


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
    vol, obs_max = _interp_ts(grid, index_vol_ts)
    erp_obs = martin_pct(vol)
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
    vol, obs_max = _interp_ts(grid, index_vol_ts)
    erp_obs = martin_pct(vol)
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
