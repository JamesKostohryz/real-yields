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

2026-09-02: `overlay_effective` is RETIRED (there is one approved idiosyncratic method and it is
not in this repository), so the sentence above about it having "the conversion right all along" is
history. The retirement guards that replaced tests/test_apply_erp_overlay_effective.py are at the
bottom of this file.
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


# ---------------------------------------------------------------- A6b retirement guards
# These replaced tests/test_apply_erp_overlay_effective.py, which tested `overlay_effective` and
# its `methodology_note`. That note was the clearest statement anywhere of what register item A6
# was about: it recorded, in the published file, that real_rf/market_erp and `idiosyncratic` came
# from different methodologies on different curve representations -- and then the function ADDED
# them and asserted the sum. For AMCR the result was a published real cost of equity of 11.873%
# against the 6.2169% the valuation used.


def test_overlay_effective_is_retired():
    """James, 2026-09-02: one approved idiosyncratic method, the rest "should not be referred to
    anywhere". A named guard so this cannot be reintroduced quietly."""
    for gone in ("overlay_effective", "METHODOLOGY_NOTE"):
        assert not hasattr(O, gone), (
            "%s was retired on 2026-09-02 with coe_v2_<T>_effective*.csv" % gone)


def test_the_config_no_longer_maps_the_effective_files():
    import json
    cfg = json.load(open(os.path.join(ROOT, "history", "ERP_OVERLAY.json")))
    assert "coe_v2_effective" not in cfg.get("mapping", {})
    assert set(cfg["mapping"]) == {"curve_latest_annual", "coe_v2_latest_annual"}


def test_the_coe_v2_annual_files_carry_only_the_two_house_view_legs():
    """The engine reads real_rf and market_erp. `idiosyncratic`, `company_erp` and `real_coe` were
    dropped on 2026-09-02 -- AFTER aeg-valuation b22d5f1 stopped requiring them, which is the order
    that mattered: dropping them first would have made the engine refuse every ticker."""
    import glob
    found = sorted(glob.glob(os.path.join(ROOT, "outputs", "coe_v2_*_latest_annual.csv")))
    if not found:
        return
    for path in found:
        cols = next(iter(csv.reader(open(path))))
        assert cols[:3] == ["tenor", "real_rf", "market_erp"], (path, cols)
        for retired in ("idiosyncratic", "company_erp", "real_coe"):
            assert retired not in cols, "%s still published in %s" % (retired, path)


def test_no_retired_artifact_is_still_published():
    import glob
    for pat in ("coe_v2_*_effective*.csv", "skew_erp_*.csv", "skew_diag_*.csv",
                "market_skew_*.csv", "market_micro_latest.csv"):
        hits = glob.glob(os.path.join(ROOT, "outputs", pat))
        assert not hits, "retired artifact still in outputs/: %s" % [os.path.basename(h)
                                                                    for h in hits]


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
