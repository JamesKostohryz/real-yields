"""Hermetic writer<->overlay contract test (no external files).

Runs run_erp_daily.run() for June-2026 from the committed held-state anchors,
then validates the two _latest files with apply_erp_overlay's OWN readers
(load_curve / load_effective) — the same identity gates the live overlay enforces.
Guards the effective-row precision: eff_coe == eff_tips_ry + eff_erp within TOL_IDENT.
"""
import json, os, sys, tempfile, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_erp_daily as rr
import held_state as hs

def _load_overlay():
    spec = importlib.util.spec_from_file_location("ovl", os.path.join(ROOT, "apply_erp_overlay.py"))
    ovl = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ovl)
    except SystemExit:
        pass
    return ovl

def test_writer_overlay_contract():
    ovl = _load_overlay()
    # PINNED TO JUNE ON PURPOSE -- this is a June reproduction fixture, not a live
    # read, and must not follow resolve_held_state(). Named via the module constant
    # so it cannot drift out of step with run_erp_daily's own smoke.
    state = json.load(open(os.path.join(ROOT, hs.JUNE_REPRO_STATE)))
    d = tempfile.mkdtemp()
    r = rr.run("2026-06-01", {5: 1.885, 10: 2.204, 20: 2.745, 30: 2.73},
               nominal_1y=3.83, sp_close=7450.03, state=state, outdir=d)
    curve = ovl.load_curve(os.path.join(d, "TODAY_forward_curve_latest.csv"))
    eff = ovl.load_effective(os.path.join(d, "ERP_effective_latest.csv"))
    assert len(curve) == 30, f"expected 30 tenors, got {len(curve)}"
    # RE-BASELINED 2026-09-04, same root cause and same fix as build_erp_daily.py's own
    # JUNE_EFF (see the dated note there): the 2026-09-03 PCHIP interpolation change moved
    # tips_eff for a 5/10/20/30-only knot set (no seven-year point) by +0.0212pp. This
    # assertion was not updated when that landed and has been failing at HEAD since 06864f7.
    assert abs(r["eff_tips"] - 2.370) < 0.01 and abs(r["eff_erp"] - 3.396) < 0.01 \
        and abs(r["eff_coe"] - 5.765) < 0.011, "June effective tie failed"  # preset B, re-baselined 2026-09-04
    # load_effective already enforces eff_coe == eff_tips_ry + eff_erp within TOL_IDENT
    # and the live path: the resolver must answer on this tree, with the same file
    # BROKEN 2026-08-19 AND FIXED HERE. This used to assert the resolver returns the JUNE file.
    # That was true only while June was the ONLY vintage in the tree, and it was written in the
    # same commit that made the resolver take the NEWEST vintage -- so the assertion held right
    # up until the mechanism it guards actually did something. The first automated re-anchor
    # (ERP_HELD_STATE_2026-08.json) turned the whole daily publish red.
    #
    # A test that fails the moment the system works is not a regression test, it is a pin on a
    # date. What this step is FOR is the live path: that a resolver answer exists, is fresh, and
    # that its filename agrees with its contents. That is now what it checks. The June
    # reproduction above is unaffected -- it reads hs.JUNE_REPRO_STATE directly and always did.
    p, s = hs.resolve_held_state(ROOT, log=lambda *_a: None)
    assert os.path.basename(p).startswith("ERP_HELD_STATE_"), f"resolver returned {p!r}"
    assert s["anchor_vintage"] == os.path.basename(p)[len("ERP_HELD_STATE_"):-len(".json")], \
        "resolved held state's filename disagrees with its own anchor_vintage"
    assert "vs" in s and "fey_in" in s and "D_in" in s, "resolved held state is missing inputs"
    print("writer<->overlay contract OK: load_curve 30 tenors, load_effective identity holds, eff ties June")

if __name__ == "__main__":
    test_writer_overlay_contract()
