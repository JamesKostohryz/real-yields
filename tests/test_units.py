"""Tests for cc%->annual-decimal conversions and the COE annual decomposition."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import units


def test_annualize_rate_matches_expm1():
    assert abs(units.annualize_rate(2.11) - (np.exp(0.0211) - 1)) < 1e-12
    assert abs(units.annualize_rate(0.0) - 0.0) < 1e-12
    # cc is always <= its own simple %; annual-decimal of 5% cc ~ 0.05127
    assert abs(units.annualize_rate(5.0) - 0.051271) < 1e-5


def test_to_decimal():
    assert abs(units.to_decimal(3.24) - 0.0324) < 1e-12


def test_coe_components_sum_to_total_exactly():
    rf = np.array([2.0, 2.5, 3.0])
    erp = np.array([3.2, 2.4, 1.6])
    cr = np.array([0.2, 0.1, -0.1])
    idio = np.array([0.5, 0.4, 0.3])
    d = units.coe_annual_components(rf, erp, cr, idio)
    recomposed = d["real_rf"] + d["market_erp"] + d["credit_relative"] + d["idiosyncratic"]
    assert np.max(np.abs(recomposed - d["real_coe"])) < 1e-12       # additive, exact
    # keeping erp+idio and dropping credit_relative == real_coe minus the credit term
    keep = d["real_rf"] + d["market_erp"] + d["idiosyncratic"]
    assert np.max(np.abs(keep - (d["real_coe"] - d["credit_relative"]))) < 1e-12


def test_coe_components_close_to_simple_decimal():
    # the exact marginal ERP contribution is within a few bp of a plain %/100
    rf = np.array([2.0]); erp = np.array([3.24]); z = np.array([0.0])
    d = units.coe_annual_components(rf, erp, z, z)
    assert abs(d["market_erp"][0] - 0.0324) < 0.0015               # within ~15 bp
