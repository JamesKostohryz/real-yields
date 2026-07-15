"""
Market equity-risk-premium term structure (generic, pipeline layer).

Martin (2017) bound: the market equity premium ~ the market's risk-neutral
variance, which is exactly what VIX measures. So the near-term anchor is

    A_mkt (%) = VIX^2 / 100          # VIX=18 -> 3.24%

The market ERP then mean-reverts toward a long-run FLOOR set by the corporate
BOND risk premium (the aggregate credit spread stripped of expected loss and a
liquidity/technicals deduction, plus a small tail residual):

    market_credit_RP(t) = ig_spread(t) - LGD * hazard      # expected loss stripped
    FLOOR               = market_credit_RP(30) - liquidity_deduction + tail
    market_ERP(t)       = FLOOR + (A_mkt - FLOOR) * 0.5^((t-1)/half_life)

All values in percent, consistent with the other grids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import datasources as ds

VIX_SERIES = "VIXCLS"     # 30-day (the anchor)
VXV_SERIES = "VXVCLS"     # 3-month (optional slope check)

# blended IG cumulative default hazard, annualized %, to strip expected loss
# from the market spread and leave the pure credit RISK premium.
IG_ANNUAL_HAZARD = 0.30

DEFAULTS = dict(half_life=8.0, liquidity_deduction=0.30, tail=0.50,
                lgd=0.60, ig_hazard=IG_ANNUAL_HAZARD)


def build_market_erp(grid, ig_spread, a_mkt, calib=None):
    """Pure construction. ig_spread: aggregate IG index spread (%) on `grid`;
    a_mkt: near-term market ERP anchor (%). Returns a DataFrame indexed by tenor."""
    c = dict(DEFAULTS)
    if calib:
        c.update(calib)
    grid = np.asarray(grid, dtype=float)
    ig_spread = np.asarray(ig_spread, dtype=float)

    credit_rp = ig_spread - c["lgd"] * c["ig_hazard"]
    floor = float(credit_rp[-1] - c["liquidity_deduction"] + c["tail"])
    decay = 0.5 ** ((grid - 1.0) / c["half_life"])
    market_erp = floor + (a_mkt - floor) * decay

    return pd.DataFrame({
        "tenor": grid,
        "market_erp": market_erp,
        "market_credit_rp": credit_rp,
        "floor": floor,
        "a_mkt": a_mkt,
    }).set_index("tenor")


def build_from_fred(api_key, grid, ig_spread, calib=None):
    """Fetch VIX, form the Martin anchor, and build the market ERP grid."""
    vix, vdate = ds.fetch_fred_latest(api_key, VIX_SERIES)
    if vix is None:
        raise RuntimeError("erp: VIX (VIXCLS) returned no data")
    a_mkt = (vix ** 2) / 100.0          # % ERP anchor from risk-neutral variance
    grid_df = build_market_erp(grid, ig_spread, a_mkt, calib)
    grid_df.attrs["vix"] = vix
    grid_df.attrs["vix_date"] = vdate
    return grid_df
