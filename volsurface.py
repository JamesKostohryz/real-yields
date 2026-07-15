"""
The Merton Elevator — firm-specific obsolescence premium for the cost of equity.

Obsolescence is modelled as a gradual ride DOWN through the credit rating floors:
past an analyst-set onset horizon H, the firm's effective spread rises from its own
rating curve toward a common distressed "junk floor", and equity — junior and
residual — bears a MULTIPLE of that widening. The result is a cost-of-equity term
structure that curls up at long horizons, which (in the steady-state AEG engine,
with RORE pinned at COE(H)) generates negative abnormal earnings growth in the tail.

Design record: see ERP_obsolescence_design_spec.md (v2). Three orthogonal controls:

  * WHERE YOU BOARD  — today's rating -> the starting spread curve `start_curve`.
  * HOW FAST / WHEN   — durability CATEGORY -> onset H and descent width W
                        (analyst-set, NOT the rating: obsolescence is a different
                        axis from current credit quality).
  * WHERE IT BOTTOMS  — a common distressed `floor_curve` (junk spread + cushion).

Everything here is pure/injectable (percent, cc — the pipeline's internal unit), so
the whole path is testable offline. Wiring into the 1..100y weekly/company job is a
separate, coordinated step (the live grid is 1..30 today).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# coarse rating ladder, strongest -> weakest (matches credit.py's spread_<R> columns)
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_INDEX = {r: i for i, r in enumerate(RATING_ORDER)}

# default per-category parameters (all calibratable — spec §9). Onset H in YEARS;
# `rate` is the descent speed in RATING NOTCHES PER YEAR; `floor` is the rating the
# elevator bottoms toward (+ a cushion). Descent width W is DERIVED per name as
# (notches from start rating to floor) / rate — so a name with fewer notches to fall
# reaches the floor sooner on its own ("faster for everybody else"), while onset H
# still just shifts the whole ramp horizontally.
# `cushion` (bp) is added on top of the floor rating's spread — reserved for names
# that truly ride toward distress. A DURABLE name only drifts a fraction of a notch,
# so it carries NO cushion; its defaults reproduce the ~0.5% 30->70y widening measured
# on AT&T (a durable BBB). Exposed names carry a real distress cushion.
# Defaults are illustrative placeholders — calibration is spec §9.
# Every name eventually reaches JUNK (a common CCC floor + cushion) — categories
# differ only in WHEN. `rate` (notches/yr) sets the descent speed; a BBB (3 notches to
# CCC) reaching junk in ~15y implies rate 0.20. Durable is pinned slow so it reproduces
# AT&T's tiny measured 30->70y widening (it only reaches junk far past 100y — but it
# IS on the path). All magnitudes/timing are calibration knobs (spec §9).
# Onset H and rate chosen so a BBB name (3 notches to CCC) reaches the common junk
# floor at: exposed ~40y, moderate ~92y, durable ~150y (arrival = H + 3/rate). Each
# plateaus from its arrival year on. Quoted arrivals are for a BBB start; a higher
# rating has more notches to fall and arrives later, a lower one sooner. Knobs — §9.
CATEGORY_PRESETS = {
    "A": dict(ory=50.0, rate=0.030, floor="CCC", cushion=300.0, label="durable"),
    "B": dict(ory=40.0, rate=0.060, floor="CCC", cushion=300.0, label="moderate"),
    "C": dict(ory=30.0, rate=0.300, floor="CCC", cushion=300.0, label="exposed"),
}

DEFAULT_MULTIPLE = 1.5           # M: equity bears ~1.5x the bond obsolescence widening
DEFAULT_CUSHION_BP = 0.0         # per-category cushion overrides this in the presets


# --------------------------------------------------------------- the universal shape
def progress(t, ory, W):
    """The universal elevator ramp p(t) in [0, 1].

    0 for t <= ORY (before onset), a smooth S (smoothstep: flat-accelerate-decelerate)
    up to 1 at t = ORY + W, then a plateau at 1. Changing ORY translates the whole ramp
    horizontally (the onset-override property); W sets how fast the descent runs.
    ORY = Obsolescence Risk Year (the onset).
    """
    t = np.asarray(t, dtype=float)
    W = float(W)
    if W <= 0:
        return (t > ory).astype(float)
    u = np.clip((t - float(ory)) / W, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)      # smoothstep: p(0)=0, p(1)=1, flat both ends


# --------------------------------------------------------------- floor construction
def floor_curve_from_grid(cg, floor_rating, cushion_bp=DEFAULT_CUSHION_BP):
    """Common distressed floor (percent): the floor rating's spread curve + a cushion.
    `cg` is the market credit grid (index tenor, columns spread_<RATING>)."""
    col = f"spread_{floor_rating}"
    if col not in cg.columns:
        raise KeyError(f"credit grid has no {col}")
    return cg[col].to_numpy() + cushion_bp / 100.0


# --------------------------------------------------------------- the elevator itself
def build_elevator(grid, start_curve, floor_curve, ory, W, M=DEFAULT_MULTIPLE):
    """Full obsolescence term structure for one name+category (all percent, cc).
    `ory` = Obsolescence Risk Year (onset); W = descent width (years).

    Returns a DataFrame indexed by tenor with:
      p                    the ramp, 0..1
      bond_spread          issuer spread riding the elevator (start -> floor)
      bond_obs_increment   the incremental widening vs. the start curve (>= 0)
      obs_equity_premium   M x the increment — the addition to the cost of equity
    """
    grid = np.asarray(grid, dtype=float)
    start_curve = np.asarray(start_curve, dtype=float)
    floor_curve = np.asarray(floor_curve, dtype=float)
    p = progress(grid, ory, W)
    incr = p * (floor_curve - start_curve)          # = bond_spread - start_curve
    incr = np.maximum(incr, 0.0)                     # elevator only ever widens
    return pd.DataFrame({
        "tenor": grid,
        "p": p,
        "bond_spread": start_curve + incr,
        "bond_obs_increment": incr,
        "obs_equity_premium": float(M) * incr,
    }).set_index("tenor")


def derive_width(start_rating, floor_rating, rate):
    """Descent width W (years) = notches from start to floor / rate. Names already
    at/below the floor have no elevator (W -> 0)."""
    notches = RATING_INDEX[floor_rating] - RATING_INDEX[start_rating]
    if notches <= 0 or rate <= 0:
        return 0.0
    return notches / float(rate)


def elevator_for_category(grid, cg, start_rating, category, ory_override=None,
                          M=DEFAULT_MULTIPLE, cushion_bp=DEFAULT_CUSHION_BP):
    """Convenience: build the elevator from the market credit grid for a rating +
    a durability category, with an optional analyst ORY (onset) override in years.
    The descent width is derived from the rating's distance to the category floor."""
    if category not in CATEGORY_PRESETS:
        raise KeyError(f"unknown category {category!r}; use one of {list(CATEGORY_PRESETS)}")
    preset = CATEGORY_PRESETS[category]
    ory = float(preset["ory"] if ory_override is None else ory_override)
    W = derive_width(start_rating, preset["floor"], preset["rate"])
    cushion = preset.get("cushion", cushion_bp)
    start_col = f"spread_{start_rating}"
    if start_col not in cg.columns:
        raise KeyError(f"credit grid has no {start_col}")
    start_curve = cg[start_col].to_numpy()
    floor = floor_curve_from_grid(cg, preset["floor"], cushion)
    tab = build_elevator(grid, start_curve, floor, ory, W, M)
    tab.attrs.update(category=category, label=preset["label"], ory=ory, W=W,
                     floor_rating=preset["floor"], start_rating=start_rating, M=M)
    return tab


# --------------------------------------------------------------- COE integration
def augment_coe(coe_df, obs_equity_premium):
    """Add the obsolescence component to a COE components frame while PRESERVING the
    additive identity the valuation engine hard-checks.

    Adds an `obsolescence` column and folds it into company_erp and real_coe, so
        real_rf + market_erp + credit_relative + idiosyncratic + obsolescence == real_coe
    still holds exactly. (The engine must add `obsolescence` to its identity check
    when it consumes the 100y term structure — part of the coordinated horizon step.)
    """
    out = coe_df.copy()
    obs = np.asarray(obs_equity_premium, dtype=float)
    out["obsolescence"] = obs
    if "company_erp" in out:
        out["company_erp"] = out["company_erp"].to_numpy() + obs
    out["real_coe"] = out["real_coe"].to_numpy() + obs
    return out
