#!/usr/bin/env python3
"""test_overlay_units.py -- the `_annual` files must be ANNUAL-COMPOUNDED, not cc.

WHY. Until 2026-08-22 `overlay_curve` and `overlay_coe_termstructure` wrote `cc / 100` into files
whose contract (aeg-valuation rate_feed.py) is `annual = exp(cc/100) - 1`. `asfp/run.py` builds
those files correctly and the overlay undid it. The third function in the same module,
`overlay_effective`, had the conversion right all along -- so one file carried two conventions and
nothing ever compared them.

It understated every real rate (1.8bp at tenor 1, 4.6bp at 30) and therefore OVERSTATED every
valuation the engine produced. Caught by James noticing the screen quote a 30-year real yield of
3.03% when FRED's DFII30 read 2.95%.

A RATE compounds; a PREMIUM does not. real-yields' own units.py draws that line; this pins both
sides of it. Runs under pytest and standalone.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import apply_erp_overlay as O                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_rate_compounds():
    assert abs(O._to_annual("real", 3.03) - math.expm1(0.0303)) < 1e-15
    assert abs(O._to_annual("real_fwd1y", 3.58) - math.expm1(0.0358)) < 1e-15
    assert abs(O._to_annual("real_rf", 2.94) - math.expm1(0.0294)) < 1e-15


def test_a_premium_does_not_compound():
    """A premium is an additive spread on a base rate; the engine's decomposition requires the
    pieces to SUM to the annual total, so compounding it would fix one error with a smaller one."""
    assert abs(O._to_annual("market_erp", 3.3908) - 0.033908) < 1e-15


def test_the_old_code_understated_every_real_rate():
    """The direction matters: cc/100 < exp(cc/100)-1, so the bug made rates too LOW and every
    valuation too HIGH."""
    assert O._to_annual("real", 3.03) > 3.03 / 100.0


def _curve_files():
    h = os.path.join(ROOT, "history", "TODAY_forward_curve_latest.csv")
    c = os.path.join(ROOT, "outputs", "curve_latest_annual.csv")
    if not (os.path.exists(h) and os.path.exists(c)):
        return None, None
    return ({int(float(r["tenor"])): r for r in csv.DictReader(open(h))},
            {int(float(r["tenor"])): r for r in csv.DictReader(open(c))})


def test_published_curve_is_annualised():
    H, C = _curve_files()
    if H is None:
        return
    bad = [t for t in C if t in H
           and abs(float(C[t]["real"]) - math.expm1(float(H[t]["spot_real_yield"]) / 100)) > 1e-6]
    assert not bad, f"tenors not annualised from the history curve: {bad[:5]}"


def test_no_tenor_carries_the_raw_cc_value():
    """The bug, stated as the thing that must never be true again."""
    H, C = _curve_files()
    if H is None:
        return
    raw = [t for t in C if t in H
           and abs(float(C[t]["real"]) - float(H[t]["spot_real_yield"]) / 100) < 1e-9]
    assert not raw, f"raw cc written into an _annual file at tenors {raw[:5]}"


def test_fisher_identity_holds():
    _, C = _curve_files()
    if C is None:
        return
    bad = [t for t in C
           if abs(float(C[t]["nominal"])
                  - ((1 + float(C[t]["real"])) * (1 + float(C[t]["breakeven"])) - 1)) > 1e-9]
    assert not bad, f"nominal != (1+real)(1+breakeven)-1 at tenors {bad[:5]}"


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                fails.append(name)
    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
    sys.exit(1 if fails else 0)
