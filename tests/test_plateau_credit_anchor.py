"""THE CREDIT-ANCHORED MARKET-ERP PLATEAU. James's rulings, 2026-09-16.

Spec: aeg-project docs/SPEC-Market-ERP-Plateau-Credit-Anchor-2026-09-16.md, section 8 --
six proof obligations, one test (or one pair) each. GATED: this change moves every tenor from
year 3 out and therefore moves valuations.

    BBB 30-year spread, as published            1.225
  + RISK premium add-on      A 1.25 / B 2.00 / C 2.75
  = EQUITY RISK PREMIUM plateau  A 2.475 / B 3.225 / C 3.975
  + EQUITY COST PREMIUM (cost_of_year glide)   +0.493      SEPARATE, added by the engine
  = EQUITY PREMIUM plateau       A 2.968 / B 3.718 / C 4.468

These run against the LIVE credit grid and the 2026-09 held state. tests/test_build_erp_daily.py
tests the machinery on the gate's fixed anchor; this file tests the values.
"""
import csv, json, os, sys, tempfile, inspect

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_erp_daily as BED
import run_erp_daily as RR
from asfp import volsurface as VS

STATE_PATH = os.path.join(ROOT, "ERP_HELD_STATE_2026-09.json")
CURVE_2026_09_16 = os.path.join(ROOT, "history", "TODAY_forward_curve_latest.csv")
NORM_EY_2026_09_16 = 3.0818      # solved out of the published curve; see _inputs' docstring


def _state():
    with open(STATE_PATH) as fh:
        return json.load(fh)


def _inputs():
    """The 2026-09-16 inputs, recovered from the vintage the repository actually published.

    The real curve is read back at its own knots off history/TODAY_forward_curve_latest.csv and
    re-interpolated, which is exact: PCHIP through points lying ON its own output returns that
    output. With norm_ey = 3.0818 this reproduces the committed preset-B curve to 7.5e-5 pp
    across all thirty tenors and the committed effective triple (2.7633 / 3.3244 / 6.0877) to
    four decimals -- asserted in test_1 below, so the harness is proved before it is used.
    """
    with open(CURVE_2026_09_16) as fh:
        rows = list(csv.DictReader(fh))
    spot_real = [float(r["spot_real_yield"]) for r in rows]
    return {k: spot_real[k - 1] for k in (1, 5, 7, 10, 20, 30)}, rows


def _run(preset="B", bbb=None, **kw):
    s = _state()
    knots, _ = _inputs()
    return BED.build_asof(knots, NORM_EY_2026_09_16, s["vs"], s["fey_in"], s["D_in"],
                          kw.pop("cost", s["cost"]), kw.pop("corp_prem", s["corp_prem"]),
                          preset=preset,
                          bbb_spread_30y=(RR.credit_anchor(s) if bbb is None else bbb), **kw)


# ============================================================================================
# 1. REPRODUCE THE ACCEPTANCE TABLE FROM THE ENGINE ITSELF, NOT FROM A REIMPLEMENTATION
# ============================================================================================
def test_1a_the_harness_reproduces_the_published_vintage_before_it_is_trusted():
    """A harness that has not been shown to reproduce what was actually published can prove
    nothing about what a change to it does. So: this exact call, at the OLD plateau, must give
    back the committed 2026-09-16 numbers."""
    r = _run(bbb=BED.GATE_BBB)       # GATE_BBB + addon B = 2.40, the plateau that vintage used
    with open(CURVE_2026_09_16) as fh:
        rows = list(csv.DictReader(fh))
    assert max(abs(r["spot_erp"][i] - float(rows[i]["spot_erp"])) for i in range(30)) < 1e-3
    assert max(abs(r["spot_coe"][i] - float(rows[i]["spot_coe"])) for i in range(30)) < 1e-3
    assert round(r["eff_tips"], 4) == 2.7633
    assert round(r["eff_erp"], 4) == 3.3244
    assert round(r["eff_coe"], 4) == 6.0877


# The spec's section 7 table, VERBATIM, and then what the engine says.
#
# THE PLATEAU AND FLOOR COLUMNS REPRODUCE EXACTLY. They are the change. The spec's EFFECTIVE
# columns do not, and the spec says how it got them: it "reconstructed" the construction by
# solving the normalized earnings yield out of the published curve. That reproduces the CURVE
# (it does, to 0.0001pp -- test_1a) but not the duration-collapsed EFFECTIVE, which build_asof
# computes with the incoming monthly state (fey_in, D_in) and a duration weighting the
# reconstruction did not carry. The engine is the authority, not the document: the spec's
# effective column is wrong by -0.21 to -0.46pp and its risk-free effective by +0.34pp
# (3.1071 against the engine's own 2.7633). Recorded here rather than quietly substituted.
SPEC_S7 = {          # preset: (plateau, year30 floor, spec's eff ERP, spec's real COE)
    "A": (2.475, 2.739, 2.9301, 6.0372),
    "B": (3.225, 3.489, 3.5199, 6.6270),
    "C": (3.975, 4.239, 4.0639, 7.1710),
}
ENGINE_S7 = {        # preset: (plateau, year30 floor, eff ERP, real COE, market multiple)
    "A": (2.475, 2.7392, 3.3695, 6.1328, 16.31),
    "B": (3.225, 3.4892, 3.8209, 6.5842, 15.19),
    "C": (3.975, 4.2392, 4.2723, 7.0356, 14.21),
}


@pytest.mark.parametrize("preset", ["A", "B", "C"])
def test_1b_plateau_and_floor_reproduce_the_spec_exactly(preset):
    r = _run(preset)
    plateau, floor = SPEC_S7[preset][0], SPEC_S7[preset][1]
    assert abs(r["preset_pure_risk"] - plateau) < 1e-9, "the plateau is not what the spec ruled"
    assert abs(r["spot_erp"][29] - floor) < 5e-4, "the year-30 floor is not what the spec measured"


@pytest.mark.parametrize("preset", ["A", "B", "C"])
def test_1c_the_engines_own_effective_numbers_are_pinned(preset):
    """Pinned so that a later change to the collapse shows up here. These are the ENGINE's,
    not the spec's -- see the note above SPEC_S7."""
    r = _run(preset)
    plateau, floor, erp, coe, mult = ENGINE_S7[preset]
    assert round(r["eff_erp"], 4) == erp
    assert round(r["eff_coe"], 4) == coe
    assert round(100.0 / r["eff_coe"], 2) == mult


def test_1d_the_specs_effective_column_is_not_reproducible_and_that_is_recorded():
    """A guard that cannot fail is not a guard, and neither is a discrepancy left in prose.
    If a later change ever DOES make the engine agree with the spec's effective column, this
    fails and the note above SPEC_S7 has to be rewritten rather than silently outlived."""
    for preset in ("A", "B", "C"):
        r = _run(preset)
        assert abs(r["eff_erp"] - SPEC_S7[preset][2]) > 0.1, (
            f"{preset}: the engine now agrees with the spec's effective ERP; the recorded "
            f"discrepancy is stale")


# ============================================================================================
# 2. THE NETTING PARAMETERS ARE SHARED, NOT COPIED -- BY IDENTITY, NOT BY VALUE
# ============================================================================================
def test_2_netting_parameters_are_one_object_not_two_copies():
    """A value check passes just as happily on a second copy that has not been edited yet.
    Two copies of 0.18/0.30 in two files is the drift this project spent 2026-08-31 fixing."""
    sig = inspect.signature(VS.floor_from_credit_grid).parameters
    assert sig["lgd"].default is VS.LGD
    assert sig["hazard"].default is VS.HAZARD
    assert sig["liquidity"].default is VS.LIQUIDITY
    # and build_erp_daily reads THOSE objects rather than restating them
    src = open(os.path.join(ROOT, "build_erp_daily.py"), encoding="utf-8").read()
    assert "from asfp.volsurface import LGD, HAZARD, LIQUIDITY" in src
    d = _run("B")["decomposition"]
    assert d["net_basis_expected_loss"] is not None
    assert abs(d["net_basis_expected_loss"] - VS.LGD * VS.HAZARD) < 1e-15
    assert abs(d["net_basis_liquidity"] - VS.LIQUIDITY) < 1e-15
    # no second copy of the constants anywhere in build_erp_daily
    for lit in ("0.60", "0.18"):
        assert f"lgd = {lit}" not in src and f"lgd={lit}" not in src


# ============================================================================================
# 3. THE DECOMPOSITION ADDS UP, LINE BY LINE, ON THE PUBLISHED FILES ALONE
# ============================================================================================
def test_3_the_decomposition_reconstructs_from_the_written_files():
    s = _state()
    knots, _ = _inputs()
    d = tempfile.mkdtemp()
    # Drive the REAL writer path, construct_legs included, by inverting its own two rules
    # rather than bypassing them: real_1y = nominal_1y - breakeven1y, and
    # norm_ey = 100 * normalized_X4 * cpi_factor / sp_close.
    nominal_1y = knots[1] + float(s["breakeven1y"])
    sp_close = 100.0 * float(s["normalized_X4"]) * float(s["cpi_factor"]) / NORM_EY_2026_09_16
    RR.run_all_presets("2026-09-16", knots, nominal_1y=nominal_1y, sp_close=sp_close,
                       state=s, outdir=d)
    for preset in BED.PLATEAU_ADDONS:
        suffix = "" if preset == BED.PLATEAU_DEFAULT else f"_{preset}"
        with open(os.path.join(d, f"ERP_effective_latest{suffix}.csv")) as fh:
            row = list(csv.DictReader(fh))[0]
        f = lambda k: float(row[k])
        assert row["preset"] == preset
        assert row["credit_anchor_rating"] == "BBB"
        # spread + add-on = the EQUITY RISK PREMIUM plateau
        assert abs(f("bbb_spread_30y") + f("risk_addon") - f("erp_plateau")) < 1e-9
        # + the cost premium = the EQUITY PREMIUM plateau (James's identity)
        assert abs(f("erp_plateau") + f("cost_premium") - f("equity_premium_plateau")) < 1e-9
        # + the rate response = the published year-30 spot ERP, which is the floor
        assert abs(f("equity_premium_plateau") + f("rate_response") - f("year30_spot_erp")) < 1e-6
        # and the recorded-only net basis reconstructs from the shared constants
        assert abs(f("erp_plateau") - f("net_basis_expected_loss") - f("net_basis_liquidity")
                   - f("net_basis_erp_plateau")) < 1e-9
        # the year-30 point of the published CURVE is the same number
        with open(os.path.join(d, f"TODAY_forward_curve_latest{suffix}.csv")) as fh:
            curve = list(csv.DictReader(fh))
        assert abs(float(curve[29]["spot_erp"]) - f("year30_spot_erp")) < 1e-3


# ============================================================================================
# 4. THE COST PREMIUM APPEARS EXACTLY ONCE
# ============================================================================================
def test_4a_cost_premium_appears_exactly_once_in_the_published_total():
    """The plateau slot carries BBB + risk_addon and NOTHING ELSE, because build_asof adds the
    cost premium itself at every tenor. Writing BBB + risk_addon + cost into the slot would
    look plausible, would match James's stated identity if read carelessly, and would add
    ~49bp to every company's cost of equity."""
    r = _run("B")
    d = r["decomposition"]
    assert not d["corp_prem_binds"], "this test is about the unclamped path"
    assert abs(d["erp_plateau"] - (d["bbb_spread_30y"] + d["risk_addon"])) < 1e-12, (
        "the cost premium has been folded into the plateau slot")
    # and the published floor carries it exactly once, not twice
    once = d["erp_plateau"] + d["rate_response"] + d["cost_premium"]
    twice = once + d["cost_premium"]
    assert abs(r["spot_erp"][29] - once) < 1e-6
    assert abs(r["spot_erp"][29] - twice) > 0.4, (
        f"the published floor is consistent with the cost premium counted TWICE "
        f"({d['cost_premium']}pp added twice)")


def test_4b_the_corp_prem_clamp_double_counts_the_cost_premium_KNOWN_DEFECT():
    """PINS AN OPEN DEFECT SO THAT FIXING IT IS VISIBLE. Spec section 6, on the register.

    `corp_prem` is built by floor_from_credit_grid(..., wedge=0.50) and is a TOTAL: 0.52 of
    pure credit risk plus a 0.50 cost wedge. The engine compares it against a PURE-RISK
    expression and then adds cost on top -- so whenever the clamp binds, the cost premium is
    counted twice. It binds in none of the measured live scenarios, so it is latent.

    This test drives the clamp deliberately and asserts the double-count IS present. When the
    defect is fixed -- clamp the pure-risk expression against the pure-risk floor (0.52), then
    add cost once -- this test FAILS, and that is the point: it is how the fix announces
    itself instead of slipping in unnoticed."""
    binding = 12.0                                  # far above any plateau; the clamp must bite
    r = _run("B", corp_prem=binding)
    d = r["decomposition"]
    assert d["corp_prem_binds"], "the clamp did not bind; this test proves nothing"
    assert abs(r["spot_erp"][29] - (binding + d["cost_premium"])) < 1e-9, (
        "the clamped floor is no longer corp_prem + cost -- section 6 may have been fixed; "
        "if so, invert this test and close the register item")


# ============================================================================================
# 5. PRESET ORDERING FOLLOWS THE PLATEAU, READ FROM THE CONSTANTS
# ============================================================================================
def test_5_ordering_follows_the_plateau_and_the_letters_are_inverted():
    bbb = RR.credit_anchor(_state())
    plats = BED.plateau_presets(bbb)
    assert plats["A"] < plats["B"] < plats["C"], (
        "A must now be the LOWEST premium and C the highest -- James inverted the letters on "
        "2026-09-16, and every pack written before that date means the OLD letter")
    assert BED.PLATEAU_ADDONS == {"A": 1.25, "B": 2.00, "C": 2.75}
    assert BED.PLATEAU_DEFAULT == "B"
    order = sorted(plats, key=lambda p: plats[p])
    floors = [_run(p)["spot_erp"][29] for p in order]
    coes = [_run(p)["eff_coe"] for p in order]
    assert floors == sorted(floors) and coes == sorted(coes)
    # the retired constants must be GONE, not merely unused: a stale import has to fail loudly
    assert not hasattr(BED, "PLATEAU_PRESETS")


# ============================================================================================
# 6. cost_of_year AND ITS 0.25 FLOOR ARE UNTOUCHED
# ============================================================================================
def test_6_cost_of_year_and_its_floor_are_untouched():
    """This spec rules on the cost premium's PRESENTATION -- a separate line, never folded in
    -- and on nothing else about it. reanchor.py's glide and floor must be bit-identical."""
    import reanchor as RA
    assert RA.COST_FLOOR == 0.25
    assert BED.cost_of_year(1995) == 1.5
    assert BED.cost_of_year(2026.5) == 0.5
    assert round(BED.cost_of_year(2026.71), 6) == 0.491325
    assert round(BED.cost_of_year(2032), 6) == 0.267297
    assert BED.cost_of_year(2040) < 0.25          # the raw glide goes below; the FLOOR catches it
    assert RA.cost_for(__import__("datetime").date(2040, 1, 1)) == 0.25
