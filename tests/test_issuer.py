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


def test_assemble_produces_the_cost_of_debt_and_nothing_idiosyncratic():
    """RETIRED 2026-09-02. This test used to assert that `assemble` returns a `coe` table with
    real_rf / market_erp / credit_relative / idiosyncratic / company_erp / real_coe, and that the
    four legs sum to real_coe. All of that came from asfp/coe.py's Martin-Wagner anchor and the
    Merton credit pass-through, which are retired: James ruled 2026-09-02 that a company's
    idiosyncratic risk premium has exactly ONE approved method, the four-block risk score in
    aeg-valuation/idio/, and that the rest must not be referred to anywhere.

    Note the identity the old test checked. `real_coe == real_rf + market_erp + credit_relative +
    idiosyncratic` is true by construction the moment you write both sides -- assemble_coe()
    computed the sum and then the test asserted it. It could not have failed, and it could not
    have detected that the idiosyncratic leg was structurally zero for every defensive name,
    which it was."""
    cg = _cg()
    real_rf = cg["real_fwd"].to_numpy()
    market_erp = 3.2 * 0.5 ** ((GRID - 1) / 8.0) + 1.0
    tables, meta = issuer.assemble("X", cg, real_rf, market_erp, vix=18.0,
                                   fund=_fund(), bonds=None, rating="BBB")
    assert set(tables) == {"cod", "cod_annual", "summary"}
    for retired in ("coe", "coe_annual"):
        assert retired not in tables, (
            "%s was retired on 2026-09-02 with asfp/coe.py" % retired)
    for retired in ("k", "idio_anchor"):
        assert retired not in meta, (
            "meta[%r] belonged to the retired construction" % retired)
    assert meta["rating"] == "BBB"


def test_annual_files_match_engine_contract():
    """The valuation engine binds to exact columns in the *_annual files and
    fail-hard on drift. Lock the cod annual header so we can't break it.

    The coe_<T>_annual.csv half of this lock is gone with the file (2026-09-02). The engine's
    cost-of-equity contract is now `coe_v2_<T>_latest_annual.csv` carrying tenor / real_rf /
    market_erp, and it is locked in tests/test_total_risk_erp.py instead."""
    cg = _cg()
    real_rf = cg["real_fwd"].to_numpy()
    market_erp = 3.2 * 0.5 ** ((GRID - 1) / 8.0) + 1.0
    tables, meta = issuer.assemble("T", cg, real_rf, market_erp, vix=18.0,
                                   fund=_fund(), bonds=None, rating="BBB")
    cod_cols = list(tables["cod_annual"].reset_index().columns)
    assert cod_cols == ["tenor", "real_cod", "spread", "rating", "offset",
                        "real_cod_BBB"]


def test_company_csv_no_longer_publishes_the_retired_volatility_fields():
    """`equity_vol`, `sigma_V` and `avg_correlation` were dropped from company_<T>.csv on
    2026-09-02. They fed the retired anchor and the retired Merton pass-through, and they appear
    NOWHERE in aeg-valuation -- rate_feed.load_company() reads market_value_of_debt and the debt
    analytics BY NAME, so this cannot move a valuation. `equity_vol` is also the field register
    item B1 was assumed to poison, and never did."""
    import os, tempfile, csv
    cg = _cg()
    real_rf = cg["real_fwd"].to_numpy()
    market_erp = 3.2 * 0.5 ** ((GRID - 1) / 8.0) + 1.0
    fund = _fund()
    tables, meta = issuer.assemble("X", cg, real_rf, market_erp, vix=18.0,
                                   fund=fund, bonds=None, rating="BBB")
    d = tempfile.mkdtemp()
    issuer.write_outputs(d, "X", tables, meta, fund)
    with open(os.path.join(d, "company_X.csv")) as fh:
        fields = [r["field"] for r in csv.DictReader(fh)]
    for retired in ("equity_vol", "sigma_V", "avg_correlation"):
        assert retired not in fields, "%s is retired and must not be published" % retired
    assert "market_value_of_debt" in fields or "ticker" in fields
    written = issuer.write_outputs(d, "X", tables, meta, fund)
    assert not any("coe_" in n for n in written), (
        "coe_<T>.csv / coe_<T>_annual.csv are retired and must not be written")
