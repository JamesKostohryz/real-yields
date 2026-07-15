"""
The idiosyncratic (firm-specific, non-diversifiable) component of the single-name ERP.

Added to — or subtracted from — the (untouched) market ERP. Three regions:

  0 .. options_end (~2y) : the stock's OWN option-implied idiosyncratic premium
                           ½·(stock variance − average-stock variance), the front.
  options_end .. fade_end (~4y): a quick fade from the options level down to the
                           bond-relative floor.
  fade_end .. onward     : M_b · (issuer spread − market spread) — the issuer's
                           credit spread RELATIVE to the market's representative
                           rating, times a multiple. Flat through maturity; then it
                           rides the Merton elevator UP as the issuer's spread widens
                           toward junk past ORY. (A name tighter than the market is
                           NEGATIVE here.)

So past the front, the idiosyncratic premium is one relation — M_b × relative bond
risk — and the obsolescence elevator is simply what that relation does after ORY, not
a separate component. `M_b` is a plain multiple (default 1.5), deliberately NOT the
Merton leverage factor, so it does not double-count the engine's own leverage.

All percent (cc). Pure/injectable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import elevator as ev

DEFAULT_MB = 1.5


def build_idiosyncratic(grid, options_anchor, issuer_spread_path, market_spread,
                        M_b=DEFAULT_MB, options_end=2.0, fade_end=4.0):
    """Idiosyncratic premium term structure (percent).

    options_anchor      : the front option-implied idiosyncratic premium (scalar %,
                          e.g. ½·(stock var − avg stock var)·100). May be negative.
    issuer_spread_path  : issuer spread by tenor (%) — flat in maturity, rising past
                          ORY via the elevator (pass the elevator's bond_spread).
    market_spread       : market representative-rating spread by tenor (%).
    """
    grid = np.asarray(grid, dtype=float)
    issuer_spread_path = np.asarray(issuer_spread_path, dtype=float)
    market_spread = np.asarray(market_spread, dtype=float)

    bond_idio = M_b * (issuer_spread_path - market_spread)      # relative-bond floor (can be < 0)

    idio = np.empty_like(grid)
    for i, t in enumerate(grid):
        if t <= options_end:
            idio[i] = options_anchor                            # OPTIONS own the front (~2y)
        elif t >= fade_end:
            idio[i] = bond_idio[i]                              # BONDS own it from here (+ elevator)
        else:
            u = (t - options_end) / (fade_end - options_end)
            w = u * u * (3.0 - 2.0 * u)                         # smoothstep fade
            idio[i] = (1.0 - w) * options_anchor + w * bond_idio[i]
    return pd.DataFrame({"tenor": grid, "idiosyncratic": idio,
                         "relative_bond_floor": bond_idio}).set_index("tenor")


def build_idiosyncratic_for_name(grid, cg, start_rating, options_anchor, category,
                                 market_rating="A", M_b=DEFAULT_MB,
                                 ory_override=None, cushion_bp=ev.DEFAULT_CUSHION_BP,
                                 options_end=2.0, fade_end=4.0):
    """Convenience: issuer rides the elevator toward junk past ORY; the idiosyncratic
    premium fades from the options front to M_b·(issuer − market) and rides up with it."""
    preset = ev.CATEGORY_PRESETS[category]
    ory = float(preset["ory"] if ory_override is None else ory_override)
    W = ev.derive_width(start_rating, preset["floor"], preset["rate"])
    issuer_flat = cg[f"spread_{start_rating}"].to_numpy()
    junk_floor = ev.floor_curve_from_grid(cg, preset["floor"], preset.get("cushion", cushion_bp))
    elev = ev.build_elevator(grid, issuer_flat, junk_floor, ory, W, M=1.0)  # M=1: want the SPREAD path
    issuer_spread_path = elev["bond_spread"].to_numpy()
    market_spread = cg[f"spread_{market_rating}"].to_numpy()
    out = build_idiosyncratic(grid, options_anchor, issuer_spread_path, market_spread,
                              M_b=M_b, options_end=options_end, fade_end=fade_end)
    out.attrs.update(ory=ory, W=W, M_b=M_b, market_rating=market_rating,
                     start_rating=start_rating, category=category)
    return out
