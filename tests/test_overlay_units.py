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

2026-09-04: AND THE 2026-08-22 FIX WAS ITSELF WRONG, IN THE OPPOSITE DIRECTION. It is true that an
`_annual` file must be annual-compounded. It is not true that `exp(y/100)-1` is how you get there
from THIS file. That formula belongs to asfp/run_curves.py, which evaluates the Fed's GSW Svensson
curves and IS continuously compounded. This overlay reads history/TODAY_forward_curve_latest.csv,
built by build_erp_daily from FRED DFII/DGS constant-maturity par yields quoted SEMIANNUALLY, with
forwards bootstrapped as (1+z)^t. Two paths, one conversion, applied to the wrong one.

The correction is a basis change, (1 + y/2)^2 - 1, not an exponential. It moves every real rate
DOWN by 1.0bp at tenor 1 and 2.3bp at tenor 30, and 3.3bp on the one-year forward at tenor 30 --
the leg Valuation!B20 capitalizes at -- which RAISES every valuation by about 0.54%. The tests
below now pin the quoted basis, the direction, and the requirement James stated when he ruled it:
the published number must still tie to FRED by eye.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import apply_erp_overlay as O                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _series(spot_by_tenor, fwd_by_tenor=None):
    """A minimal `curve` dict of the shape load_curve() returns, for the unit tests below."""
    fwd_by_tenor = fwd_by_tenor or spot_by_tenor
    return {t: {O.REAL_SPOT_SOURCE: spot_by_tenor[t],
                O.REAL_FWD_SOURCE: fwd_by_tenor[t],
                "fwd_erp": 3.3908} for t in range(1, 31)}


def test_a_rate_converts_off_treasurys_quoted_basis_not_off_cc():
    """THE 2026-09-04 CORRECTION. The ERP daily file's rate columns are Treasury constant-maturity
    par yields, quoted semiannually. They were never continuously compounded, so exp(y/100)-1 --
    which the 2026-08-22 fix applied -- over-compounds them by roughly y^2/4."""
    assert abs(O.annual_from_quoted(2.98) - ((1 + 0.0298 / 2) ** 2 - 1)) < 1e-15
    assert O.annual_from_quoted(2.98) < math.expm1(0.0298)          # the error that was there
    assert O.annual_from_quoted(2.98) > 0.0298                      # and it is still a conversion
    assert abs(O.annual_from_quoted(2.98) - 0.030022010) < 1e-9     # 2.9800% quoted = 3.0022% p.a.


def test_the_overstatement_that_was_published_is_pinned_by_size():
    """Direction and size both matter: expm1 made every real rate too HIGH, which made every
    valuation too LOW. 2.3bp at tenor 30 on the spot curve."""
    over = (math.expm1(0.0298) - O.annual_from_quoted(2.98)) * 1e4
    assert 2.0 < over < 2.6, over


def test_a_premium_does_not_convert_at_all():
    """A premium is an additive spread on a base rate; the engine's decomposition requires the
    pieces to SUM to the annual total, so compounding it would fix one error with a smaller one."""
    curve = _series({t: 2.98 for t in range(1, 31)})
    v = O._annual_value("market_erp", "fwd_erp", 30, curve, O.real_annual_series(curve))
    assert abs(v - 0.033908) < 1e-15


def test_the_forward_is_rebootstrapped_from_the_converted_spot():
    """Not converted column-wise. The published fwd_real_yield was bootstrapped from the QUOTED
    spot curve, so the overlay must convert the spots and re-derive -- which is also what the
    workbook does in Market Data rows 22/24 from the spot curve it is handed."""
    spot = {t: 2.0 + 0.03 * t for t in range(1, 31)}
    s, f = O.real_annual_series(_series(spot))
    assert abs(f[1] - s[1]) < 1e-15
    for t in range(2, 31):
        expect = (1 + s[t]) ** t / (1 + s[t - 1]) ** (t - 1) - 1
        assert abs(f[t] - expect) < 1e-15


def test_an_unknown_rate_source_is_refused_rather_than_guessed():
    """Both the 2026-08-22 and the 2026-09-04 defects were a guess about an input's compounding
    basis. A new mapping must fail loudly instead of inheriting one."""
    curve = _series({t: 2.98 for t in range(1, 31)})
    try:
        O._annual_value("real", "some_new_column", 30, curve, O.real_annual_series(curve))
    except O.OverlayError:
        return
    raise AssertionError("an unmapped rate source was converted on an assumed basis")


def _curve_files():
    h = os.path.join(ROOT, "history", "TODAY_forward_curve_latest.csv")
    c = os.path.join(ROOT, "outputs", "curve_latest_annual.csv")
    if not (os.path.exists(h) and os.path.exists(c)):
        return None, None
    return ({int(float(r["tenor"])): r for r in csv.DictReader(open(h))},
            {int(float(r["tenor"])): r for r in csv.DictReader(open(c))})


def test_published_curve_is_annualised_off_the_quoted_basis():
    H, C = _curve_files()
    if H is None:
        return
    bad = [t for t in C if t in H
           and abs(float(C[t]["real"])
                   - O.annual_from_quoted(float(H[t]["spot_real_yield"]))) > 1e-6]
    assert not bad, f"tenors not converted from the history curve: {bad[:5]}"


def test_no_tenor_carries_the_expm1_value():
    """The 2026-09-04 bug, stated as the thing that must never be true again. Treasury's DFII
    yields are not continuously compounded and must never be exponentiated as if they were."""
    H, C = _curve_files()
    if H is None:
        return
    bad = [t for t in C if t in H
           and abs(float(C[t]["real"])
                   - math.expm1(float(H[t]["spot_real_yield"]) / 100)) < 1e-9]
    assert not bad, f"expm1 of a quoted par yield written into an _annual file at {bad[:5]}"


def test_the_published_curve_still_ties_to_treasury_by_eye():
    """James's requirement, 2026-09-04: a reader must be able to check the published real rate
    against FRED without help. The conversion is a basis change, not a different rate, so the
    published figure stays within 3bp of Treasury's own quote at every tenor."""
    H, C = _curve_files()
    if H is None:
        return
    bad = [t for t in C if t in H
           and abs(float(C[t]["real"]) * 100 - float(H[t]["spot_real_yield"])) > 0.03]
    assert not bad, f"published real drifts more than 3bp from Treasury's quote at {bad[:5]}"


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
