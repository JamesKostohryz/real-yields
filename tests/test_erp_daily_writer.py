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
    state = json.load(open(os.path.join(ROOT, "ERP_HELD_STATE_2026-06.json")))
    d = tempfile.mkdtemp()
    r = rr.run("2026-06-01", {5: 1.885, 10: 2.204, 20: 2.745, 30: 2.73},
               nominal_1y=3.83, sp_close=7450.03, state=state, outdir=d)
    curve = ovl.load_curve(os.path.join(d, "TODAY_forward_curve_latest.csv"))
    eff = ovl.load_effective(os.path.join(d, "ERP_effective_latest.csv"))
    assert len(curve) == 30, f"expected 30 tenors, got {len(curve)}"
    assert abs(r["eff_tips"] - 2.349) < 0.01 and abs(r["eff_erp"] - 3.887) < 0.01 \
        and abs(r["eff_coe"] - 6.236) < 0.011, "June effective tie failed"
    # load_effective already enforces eff_coe == eff_tips_ry + eff_erp within TOL_IDENT
    print("writer<->overlay contract OK: load_curve 30 tenors, load_effective identity holds, eff ties June")

if __name__ == "__main__":
    test_writer_overlay_contract()
