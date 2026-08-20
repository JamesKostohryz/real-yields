#!/usr/bin/env python3
"""
tests_vol_ts_quality.py — one thin LEAPS quote must not become a 7.9% equity risk premium.

WHAT HAPPENED. On 2026-08-20 outputs/index_vol_ts_latest.csv carried 18.493 at 1y, 19.166 at
1.5y, 17.75 at 2y and 21.768 at 3y — ATM implied vols scraped live from SPX/SPY option chains,
where the three-year point is one thin LEAPS quote. `forward_erp()` differences cumulative
variance and is CORRECT: C(2) = 6.3013, C(3) = 14.2155, forward(3) = 7.914. That is a 28.1%
forward vol for year three against a 17.75% two-year spot.

It published market_erp_v2 as 3.42 / 2.88 / **7.91** at 1/2/3 years, holding near 7.8% to seven
years. The daily ERP overlay masks it for the discount rate. It does NOT touch the idiosyncratic
column, so coe_v2_MSFT_effective.csv published idio 5.87% and real_coe 11.90%.

The floor at zero in forward_erp() already caught a DOWNWARD kink. Nothing caught an upward one,
because a large positive forward variance is arithmetically legitimate — which is exactly why
this belongs in a data-quality check on the input and not in the maths.

A guard that never fires and a guard that always fires look identical from the outside, so both
directions are asserted here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from asfp.total_risk_erp import vol_ts_quality, forward_erp, martin_pct, build_market_erp_blended

_p = _f = 0
def ok(c, m):
    global _p, _f
    globals().__setitem__("_p", _p + 1) if c else globals().__setitem__("_f", _f + 1)
    print(("  PASS " if c else "  FAIL ") + m)

REAL = [(0.025, 13.59), (0.082, 15.19), (0.255, 19.04), (0.5, 21.37),
        (1.0, 18.493), (1.493, 19.166), (2.0, 17.75), (3.0, 21.768)]

# --- IT FIRES on the real defect, and on exactly one point
clean, rej = vol_ts_quality(REAL)
ok(len(rej) == 1 and abs(rej[0][0] - 3.0) < 1e-9,
   "the 3-year quote is rejected, and it alone (%d rejected)" % len(rej))
ok(len(clean) == 7, "the other seven observed points are KEPT, not smoothed away")
ok("forward vol" in rej[0][2], "the rejection says why, in vol terms a reader can check")

# --- and the published curve stops being absurd
g = np.arange(1, 9, dtype=float)
ten = np.array([t for t, _ in REAL]); val = np.array([v for _, v in REAL])
before = forward_erp(g, martin_pct(np.interp(g, ten, val)))
tenc = np.array([t for t, _ in clean]); valc = np.array([v for _, v in clean])
after = forward_erp(g, martin_pct(np.interp(g, tenc, valc)))
ok(before[2] > 7.5, "BEFORE: year-3 forward ERP is %.2f%% — the defect reproduces" % before[2])
ok(after[2] < 4.0, "AFTER:  year-3 forward ERP is %.2f%% — sane" % after[2])
ok(max(after) < 5.0, "no tenor is left above 5%% (max %.2f%%)" % max(after))

# --- IT DOES NOT FIRE on an ordinary upward-sloping vol curve
NORMAL = [(0.25, 16.0), (0.5, 17.0), (1.0, 18.0), (2.0, 19.0), (3.0, 20.0), (5.0, 21.0)]
c2, r2 = vol_ts_quality(NORMAL)
ok(not r2, "an ordinary rising vol term structure is untouched (%d rejected)" % len(r2))
ok(len(c2) == len(NORMAL), "every point survives")

# --- and it does not fire on a BACKWARDATED curve either: the zero floor already handles that
BACKWARD = [(0.25, 24.0), (1.0, 20.0), (2.0, 18.0), (3.0, 17.0)]
c3, r3 = vol_ts_quality(BACKWARD)
ok(not r3, "a falling vol term structure is untouched — the existing zero floor owns that case")

# --- the guard is wired into the builder, not merely available
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    df = build_market_erp_blended(np.arange(1, 31, dtype=float), REAL, floor=1.04)
ok("REJECTED index vol" in buf.getvalue(), "build_market_erp_blended applies it and SAYS so")
ok(float(df["market_erp"].values[2]) < 4.0,
   "the built curve's year-3 point is %.2f%%, not 7.91%%" % df["market_erp"].values[2])

print("\n%d passed, %d failed" % (_p, _f))
raise SystemExit(1 if _f else 0)
