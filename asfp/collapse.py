"""
Collapse a discount-rate TERM STRUCTURE to a single equivalent rate — the equity
analogue of a bond's yield to maturity.

A bond's YTM is the one flat rate that reprices all its cash flows. Same idea here:
given a cost-of-equity term structure (a per-year rate by horizon) and a cash-flow
profile, find the single flat rate that gives the identical present value. Report that
as the name's "single COE", and single ERP = single COE − single risk-free.

Discounting uses FORWARD/marginal rates: DF(t) = ∏_{s≤t} 1/(1+rate_s). The collapsed
number is therefore a cash-flow-PV-weighted average of the curve — dominated by the
near-to-middle horizons (where the discounted cash flows are biggest), lightly pulled
by the tail, exactly like a bond's YTM weights its nearer coupons more.
"""
from __future__ import annotations

import numpy as np


def _pv_termstructure(grid, rate_pct, cashflows):
    df = 1.0 / np.cumprod(1.0 + np.asarray(rate_pct, float) / 100.0)
    return float(np.sum(cashflows * df))


def _pv_flat(grid, f_pct, cashflows):
    df = 1.0 / (1.0 + f_pct / 100.0) ** np.asarray(grid, float)
    return float(np.sum(cashflows * df))


def collapse_rate(grid, rate_pct, cashflows=None, growth=2.0,
                  lo=-5.0, hi=60.0, tol=1e-8):
    """Single flat rate (%) repricing `cashflows` identically to the term structure.

    grid       : horizons in years (e.g. 1..100)
    rate_pct   : the per-year rate term structure in percent (e.g. real COE)
    cashflows  : weight per horizon; default a real cash-flow stream growing at
                 `growth`%/yr — i.e. (1+growth/100)^t — a stand-in for equity's
                 growing payout. Pass your own profile to match a specific name.
    """
    grid = np.asarray(grid, dtype=float)
    if cashflows is None:
        cashflows = (1.0 + growth / 100.0) ** grid
    cashflows = np.asarray(cashflows, dtype=float)
    target = _pv_termstructure(grid, rate_pct, cashflows)
    # PV is monotically decreasing in the flat rate -> bisection
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _pv_flat(grid, mid, cashflows) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def collapse_coe_and_erp(grid, coe_pct, rf_pct, cashflows=None, growth=2.0):
    """Return (single_coe, single_rf, single_erp) — the term structure summarized as
    one number, and its premium over the collapsed risk-free."""
    coe1 = collapse_rate(grid, coe_pct, cashflows, growth)
    rf1 = collapse_rate(grid, rf_pct, cashflows, growth)
    return coe1, rf1, coe1 - rf1


def bootstrap_spot(tenor, *, curve_path="history/TODAY_forward_curve_latest.csv",
                   vintage_path="history/ERP_effective_latest.csv", field="spot_erp"):
    """The PARAMETER-FREE reading for a market-level figure with no company forecast
    behind it (object (d) in AEG-ERP-Collapse-Function-AUDIT-2026-08-12.md section 3;
    Task 2 spec section 5). This is NOT a collapse -- no cash-flow profile, no growth
    assumption, nothing to choose. It is just the published SPOT rate at `tenor`,
    already bootstrapped from the one-year forwards (verified to machine precision
    against a from-scratch bootstrap in the audit, section 2).

    Use this instead of running collapse_rate() with a synthetic growth profile
    whenever there is no company (and therefore no real cash-flow profile) behind the
    number -- e.g. quoting "the market's effective ERP" on its own. A per-company
    effective rate still belongs through collapse_rate() with that company's own
    profile (see asfp/run_company.py and aeg_schedule_feed.py).

    Returns (value, vintage_date) so a disclosure line always carries its own
    freshness -- "re-read the live curve, don't quote this number from memory" is a
    standing instruction, not a one-time caveat. Raises FileNotFoundError /
    ValueError rather than returning a stale or guessed number.
    """
    import csv
    with open(curve_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    match = [r for r in rows if float(r["tenor"]) == float(tenor)]
    if not match:
        raise ValueError(f"{curve_path}: no row at tenor {tenor}")
    if field not in match[0]:
        raise ValueError(f"{curve_path}: no column '{field}' (have {list(match[0])})")
    value = float(match[0][field])

    vintage = None
    try:
        with open(vintage_path, newline="") as fh:
            vrows = list(csv.DictReader(fh))
        if vrows and "date" in vrows[0]:
            vintage = vrows[0]["date"]
    except (FileNotFoundError, OSError):
        pass                                          # vintage stamp is best-effort
    return value, vintage
