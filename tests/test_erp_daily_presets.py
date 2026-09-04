"""Hermetic test for run_erp_daily.run_all_presets() -- added 2026-09-04 alongside the
three-preset daily publish (docs/engine/PASTE-THIS-next-SETUP-app-session.md open item 3).

Confirms, all offline:
  1. Preset B written by run_all_presets() is BYTE-IDENTICAL to what plain run() writes.
     The existing writer<->overlay contract (tests/test_erp_daily_writer.py) and everything
     downstream of it (apply_erp_overlay.py, aeg-valuation's outputs/) is completely
     unaffected by this landing -- this is the test that proves that, not just an assertion
     of intent.
  2. Preset A and C files are written, validate against apply_erp_overlay's OWN readers (the
     same identity gates the live overlay enforces on B every day), and their effective cost
     of equity orders with the plateau (A's 3.35pp pure-risk plateau is the highest of the
     three, C's 2.05pp the lowest) -- read from PLATEAU_PRESETS itself rather than assumed,
     so a future re-numbering cannot silently invert this check.
"""
import json, os, sys, tempfile, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_erp_daily as rr
import held_state as hs
from build_erp_daily import PLATEAU_PRESETS, PLATEAU_DEFAULT


def _load_overlay():
    spec = importlib.util.spec_from_file_location("ovl", os.path.join(ROOT, "apply_erp_overlay.py"))
    ovl = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ovl)
    except SystemExit:
        pass
    return ovl


def test_preset_b_unchanged_by_run_all_presets():
    state = json.load(open(os.path.join(ROOT, hs.JUNE_REPRO_STATE)))
    knots = {5: 1.885, 10: 2.204, 20: 2.745, 30: 2.73}
    d_single, d_all = tempfile.mkdtemp(), tempfile.mkdtemp()
    rr.run("2026-06-01", knots, nominal_1y=3.83, sp_close=7450.03, state=state, outdir=d_single)
    rr.run_all_presets("2026-06-01", knots, nominal_1y=3.83, sp_close=7450.03, state=state, outdir=d_all)
    for fname in ("TODAY_forward_curve_latest.csv", "ERP_effective_latest.csv"):
        a = open(os.path.join(d_single, fname)).read()
        b = open(os.path.join(d_all, fname)).read()
        assert a == b, f"{fname}: run_all_presets' preset-B file differs byte-for-byte from run()'s"
    print("preset B, written by run_all_presets(), is byte-identical to plain run() -- "
          "no behavior change for the existing overlay contract")


def test_all_three_presets_publish_and_validate():
    ovl = _load_overlay()
    state = json.load(open(os.path.join(ROOT, hs.JUNE_REPRO_STATE)))
    knots = {5: 1.885, 10: 2.204, 20: 2.745, 30: 2.73}
    d = tempfile.mkdtemp()
    results = rr.run_all_presets("2026-06-01", knots, nominal_1y=3.83, sp_close=7450.03,
                                  state=state, outdir=d)
    assert set(results) == set(PLATEAU_PRESETS), f"expected presets {set(PLATEAU_PRESETS)}, got {set(results)}"

    eff_by_preset = {}
    for preset in PLATEAU_PRESETS:
        suffix = "" if preset == PLATEAU_DEFAULT else f"_{preset}"
        curve_path = os.path.join(d, f"TODAY_forward_curve_latest{suffix}.csv")
        eff_path = os.path.join(d, f"ERP_effective_latest{suffix}.csv")
        assert os.path.exists(curve_path), f"preset {preset}: {curve_path} was not written"
        assert os.path.exists(eff_path), f"preset {preset}: {eff_path} was not written"
        curve = ovl.load_curve(curve_path)     # raises OverlayError if the 30-tenor identity breaks
        eff = ovl.load_effective(eff_path)     # raises OverlayError if eff_coe != eff_tips_ry + eff_erp
        assert len(curve) == 30, f"preset {preset}: expected 30 tenors, got {len(curve)}"
        eff_by_preset[preset] = eff

    # A's pure-risk plateau (3.35) is the highest of the three, C's (2.05) the lowest -- the
    # long end of the curve, and therefore the effective COE, must order the same way.
    order = sorted(PLATEAU_PRESETS, key=lambda p: PLATEAU_PRESETS[p])          # low -> high plateau
    coe = [eff_by_preset[p]["eff_coe"] for p in order]
    assert coe == sorted(coe), (
        f"effective COE must order with the plateau ({order}); got {list(zip(order, coe))}")
    print(f"three presets published and validated: "
          f"{[(p, round(eff_by_preset[p]['eff_coe'], 4)) for p in PLATEAU_PRESETS]}")
    print(f"  plateau ordering holds: {order} -> COE {coe}")


if __name__ == "__main__":
    test_preset_b_unchanged_by_run_all_presets()
    test_all_three_presets_publish_and_validate()
    print("ALL PRESET TESTS PASSED")
