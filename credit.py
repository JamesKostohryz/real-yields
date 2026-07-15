"""
Market ERP front from the OBSERVED implied-variance term structure (prototype).

Replaces the single-VIX-point anchor + parametric decay with a construction that
uses the whole VIX term structure (VIX9D, VIX, VIX3M, VIX6M, VIX1Y) for the SHAPE
of the front, while BONDS own the LEVEL from the joint outward. The design point,
per James: the options level must NOT leak into the 5y+ region — that level is the
bond risk-premium floor, full stop. Options only move the 0–1y front.

Three regions along the horizon:
  0 .. front (~1y)   : ERP = Martin bound from the observed vol term structure
  front .. joint(~5y): smooth reversion from the 1y options level DOWN to the floor
  joint .. end       : ERP = bond floor (bond-anchored level; options can't reach here)

Martin (2017): the equity premium ≈ the risk-neutral variance, so
    ERP%(t) = vol%(t)**2 / 100      (VIX 17 -> 2.89%).

Pure/injectable — the live VIX-term-structure and bond-floor reads happen in the job.
All values in percent, consistent with the other grids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def martin_erp_pct(vol_pct):
    """Martin lower-bound ERP (percent) from an implied vol in vol points."""
    v = np.asarray(vol_pct, dtype=float)
    return v * v / 100.0


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def build_market_erp_ts(grid, vol_ts, floor, front=1.0, joint=5.0):
    """Market ERP term structure (percent) with options-front / bond-level split.

    grid   : horizons (years), e.g. 1..30 (or 1..100)
    vol_ts : sequence of (tenor_years, vol_points) from the VIX term structure,
             e.g. [(0.025, 15), (0.083, 17), (0.25, 18), (0.5, 19), (1.0, 20)]
    floor  : the bond-anchored long-run ERP level (percent) — sets 5y+ entirely
    front  : horizon (years) out to which options are used directly (~1y)
    joint  : horizon (years) by which the ERP has fully reverted to `floor` (~5y)

    Returns a DataFrame indexed by tenor with market_erp (percent).
    """
    grid = np.asarray(grid, dtype=float)
    ten = np.array([t for t, _ in vol_ts], dtype=float)
    vol = np.array([v for _, v in vol_ts], dtype=float)
    order = np.argsort(ten)
    ten, vol = ten[order], vol[order]

    # Martin ERP across the observed front; flat-extrapolate the 1y value to `front`
    martin_front = martin_erp_pct(np.interp(grid, ten, vol))       # only used for t<=front
    martin_at_front = float(martin_erp_pct(np.interp(front, ten, vol)))

    erp = np.empty_like(grid)
    for i, t in enumerate(grid):
        if t <= front:
            erp[i] = martin_front[i]                               # OPTIONS own the front
        elif t >= joint:
            erp[i] = floor                                         # BONDS own 5y+
        else:
            u = (t - front) / (joint - front)
            w = 1.0 - _smoothstep(u)                               # 1 at front -> 0 at joint
            erp[i] = w * martin_at_front + (1.0 - w) * floor       # reversion to bonds
    return pd.DataFrame({"tenor": grid, "market_erp": erp}).set_index("tenor")
