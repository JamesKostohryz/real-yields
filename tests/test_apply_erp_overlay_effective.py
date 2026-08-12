"""Tests for apply_erp_overlay.py::overlay_effective — the Task 5 methodology_note
addition (AEG-ERP-Collapse-Function-AUDIT-2026-08-12.md section 5). overlay_effective
mixes two different collapse methodologies (Decision-B state-machine real_rf/market_erp,
and a cash-flow-PV YTM idiosyncratic) into one additive real_coe/company_erp; this note
must be present, and must not duplicate across repeated runs (the overlay runs daily)."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import apply_erp_overlay as aeo


def _write_effective_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["field", "value_pct"])
        w.writeheader()
        for f, v in rows:
            w.writerow({"field": f, "value_pct": v})


def _read_dict(path):
    with open(path, newline="") as fh:
        return {r["field"]: r["value_pct"] for r in csv.DictReader(fh)}


def test_overlay_adds_methodology_note(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs")
    _write_effective_csv("outputs/coe_v2_ZZZ_effective.csv", [
        ("real_rf", "2.7236"), ("market_erp", "1.2148"), ("idiosyncratic", "0.30"),
        ("company_erp", "1.5148"), ("real_coe", "4.2384"), ("cf_growth", "2.0"),
        ("profile_basis", "ZZZ_aeg_schedule dps_real, 30y"),
    ])
    eff = {"eff_tips_ry": 2.563, "eff_erp": 3.9252, "eff_coe": 6.4882}
    mapping = {"real_rf": "eff_tips_ry", "market_erp": "eff_erp"}

    out = aeo.overlay_effective("ZZZ", eff, mapping)
    d = _read_dict("outputs/coe_v2_ZZZ_effective.csv")

    assert "methodology_note" in d
    assert "Decision-B state-machine" in d["methodology_note"]
    # real_rf/market_erp overwritten to the state-machine basis
    assert float(d["real_rf"]) == 2.563
    assert float(d["market_erp"]) == 3.9252
    # idiosyncratic NEVER touched
    assert float(d["idiosyncratic"]) == 0.30
    # profile_basis (Task 2/3) survives untouched
    assert d["profile_basis"] == "ZZZ_aeg_schedule dps_real, 30y"
    assert any("mixed-methodology" in line for line in out)


def test_overlay_note_does_not_duplicate_on_repeated_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs")
    _write_effective_csv("outputs/coe_v2_ZZZ_effective.csv", [
        ("real_rf", "2.7236"), ("market_erp", "1.2148"), ("idiosyncratic", "0.30"),
        ("company_erp", "1.5148"), ("real_coe", "4.2384"),
    ])
    eff = {"eff_tips_ry": 2.563, "eff_erp": 3.9252, "eff_coe": 6.4882}
    mapping = {"real_rf": "eff_tips_ry", "market_erp": "eff_erp"}

    aeo.overlay_effective("ZZZ", eff, mapping)
    aeo.overlay_effective("ZZZ", eff, mapping)     # simulate the next day's run
    with open("outputs/coe_v2_ZZZ_effective.csv", newline="") as fh:
        fields = [r["field"] for r in csv.DictReader(fh)]
    assert fields.count("methodology_note") == 1
