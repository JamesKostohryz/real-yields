"""
Unit conversions for the downstream valuation engine.

The rate pipeline works in CONTINUOUSLY-COMPOUNDED PERCENT (2.11 = 2.11% cc),
because that is the GSW convention and because in cc space the term structures
are additive (real = nominal - breakeven, real_coe = real_rf + erp + ... ).

The valuation engine works in ANNUAL-COMPOUNDED DECIMAL FRACTIONS (0.0211), with
forwards via (1+x)^t and a (1+n)/(1+i)-1 deflator. These helpers publish `_annual`
decimal variants so the engine needs zero conversion and carries zero convention
risk.

Two conversion kinds:
  * a RATE/YIELD (real_rf, exp_inflation, breakeven, real cost of debt):
        annual_decimal = exp(cc%/100) - 1                        [annualize_rate]
  * a PREMIUM built additively on top of a base rate (the ERP and its COE
    components): its exact annual-decimal contribution is the marginal step in the
    compounded build-up, so the pieces still SUM to the annual total. That is what
    coe_annual_components does; a plain %/100 is within ~1-5 bp of it and is fine
    if the engine prefers a standalone number.
"""
from __future__ import annotations

import numpy as np


def annualize_rate(cc_percent):
    """cc percent -> annual-compounded decimal fraction.  2.11 -> 0.02132."""
    return np.expm1(np.asarray(cc_percent, dtype=float) / 100.0)


def to_decimal(percent):
    """plain percent -> decimal fraction (for premia/spreads).  3.24 -> 0.0324."""
    return np.asarray(percent, dtype=float) / 100.0


def coe_annual_components(real_rf, market_erp, credit_relative, idiosyncratic):
    """Exact annual-decimal decomposition of the real cost of equity.

    Given the four additive cc-percent terms, return each term's annual-decimal
    marginal contribution in the compounded build-up
        rf -> +erp -> +credit -> +idio
    so that  real_rf + market_erp + credit_relative + idiosyncratic == real_coe
    holds EXACTLY in annual-decimal space (the pieces sum to the annual total).

    The engine can then keep market_erp + idiosyncratic and drop credit_relative
    (it owns leverage) and still add cleanly on top of real_rf.
    """
    rf = np.asarray(real_rf, dtype=float)
    erp = np.asarray(market_erp, dtype=float)
    cr = np.asarray(credit_relative, dtype=float)
    idio = np.asarray(idiosyncratic, dtype=float)

    l0 = np.expm1(rf / 100.0)
    l1 = np.expm1((rf + erp) / 100.0)
    l2 = np.expm1((rf + erp + cr) / 100.0)
    l3 = np.expm1((rf + erp + cr + idio) / 100.0)
    return {
        "real_rf": l0,
        "market_erp": l1 - l0,
        "credit_relative": l2 - l1,
        "idiosyncratic": l3 - l2,
        "company_erp": l3 - l0,
        "real_coe": l3,
    }


def annualize_curve(df, rate_cols=(), premium_cols=()):
    """Return a copy of `df` with `_annual` decimal columns for the named columns.
    rate_cols use annualize_rate; premium_cols use to_decimal."""
    out = df.copy()
    for c in rate_cols:
        if c in out:
            out[c] = annualize_rate(out[c])
    for c in premium_cols:
        if c in out:
            out[c] = to_decimal(out[c])
    return out
