"""Offline tests for the per-company assembly (cod + coe + MV of debt)."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import credit, issuer

GRID = np.arange(1, 31, dtype=float)
IG = [(2, 0.55), (4, 0.65), (6, 0.75), (8.5, 0.85), (12.5, 0.95), (20, 1.05)]
AN = {"AAA": 0.45, "AA": 0.55, "A": 0.80, "BBB": 1.15, "BB": 2.10, "B": 3.40, "CCC": 7.50}
TSY = [(1, 4.3), (2, 4.2), (5, 4.2), (10, 4.5), (20, 4.9), (30, 4.8)]


def _cg():
    return credit.build_from_knots(GRID, IG, 0.90, AN, TSY, np.linspace(1.6, 2.6, 30))


def _fund():
    return dict(ticker="X", price=20.0, market_equity=100e9, nfo=80e9,
                L=0.44, lambda0=0.8, equity_vol=0.22, sigma_V=0.12,
                avg_correlation=0.35)


def test_fit_offset_excludes_distressed():
    cg = _cg()
    ten, tsy, sp = cg.index.to_numpy(), cg["treasury_nominal"], cg["spread_BBB"]
    # clean BBB bonds at ~1.0x the BBB curve + two distressed at ~5x
    rows = []
    for yr in [3, 5, 7, 10, 15, 20]:
        t = np.interp(yr, ten, tsy); s = np.interp(yr, ten, sp)
        rows.append(dict(years=yr, ytw=(t + s) / 100.0, sp_rating="BBB"))
    for yr in [12, 14]:
        t = np.interp(yr, ten, tsy); s = np.interp(yr, ten, sp)
        rows.append(dict(years=yr, ytw=(t + 5 * s) / 100.0, sp_rating="BBB"))
    bonds = pd.DataFrame(rows)
    off, n_used, n_excl = issuer.fit_offset(cg, bonds, "BBB")
    assert abs(off - 1.0) < 0.05          # clean fit ~1.0 despite distressed pair
    assert n_excl >= 2                    # the two distressed dropped


def test_cost_of_debt_fallback_vs_bonds():
    cg = _cg()
    cod0, m0 = issuer.build_cost_of_debt(cg, bonds=None, rating="A")
    assert m0["offset"] == 1.0            # pure-rating fallback
    assert np.allclose(cod0["real_cod"], cg["real_fwd"] + cg["spread_A"])


def test_assemble_produces_all_components():
    cg = _cg()
    real_rf = cg["real_fwd"].to_numpy()
    market_erp = 3.2 * 0.5 ** ((GRID - 1) / 8.0) + 1.0
    tables, meta = issuer.assemble("X", cg, real_rf, market_erp, vix=18.0,
                                   fund=_fund(), bonds=None, rating="BBB")
    coe = tables["coe"]
    for col in ["real_rf", "market_erp", "credit_relative", "idiosyncratic",
                "company_erp", "real_coe"]:
        assert col in coe.columns
    # real_coe == real_rf + the three premia (additive, cc space)
    recomposed = (coe["real_rf"] + coe["market_erp"] + coe["credit_relative"]
                  + coe["idiosyncratic"])
    assert np.max(np.abs(recomposed - coe["real_coe"])) < 1e-9
    # annual decomposition sums to annual real_coe
    ca = tables["coe_annual"]
    s = ca["real_rf"] + ca["market_erp"] + ca["credit_relative"] + ca["idiosyncratic"]
    assert np.max(np.abs(s - ca["real_coe"])) < 1e-12
    assert meta["k"] > 0 and meta["rating"] == "BBB"
