"""Offline tests for the company debt-analytics engine (roadmap A1/A2)."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import debt_analytics as da


def test_parse_amount_units():
    assert da.parse_amount("1.25 B USD") == 1.25e9
    assert da.parse_amount("550 M USD") == 5.5e8
    assert da.parse_amount("18.29 M USD") == 18.29e6
    assert np.isnan(da.parse_amount("—"))


def test_price_at_par_when_yield_equals_coupon():
    # a bond priced at yield == coupon should sit at par (1.0)
    assert abs(da.price_bond(10.0, 0.05, 0.05) - 1.0) < 1e-6
    assert abs(da.price_bond(3.0, 0.04, 0.04) - 1.0) < 1e-6


def test_discount_below_par_premium_above():
    assert da.price_bond(10, 0.03, 0.05) < 1.0      # coupon < yield -> discount
    assert da.price_bond(10, 0.07, 0.05) > 1.0      # coupon > yield -> premium


def test_modified_duration_positive_and_monotone():
    d5 = da.modified_duration(5, 0.05, 0.05)
    d20 = da.modified_duration(20, 0.05, 0.05)
    assert 0 < d5 < d20                              # longer bond, more duration


def test_repricing_property_random_bonds():
    # price each bond at its own yield, then recover the yield -> consistency
    rng = np.random.default_rng(0)
    for _ in range(25):
        yrs = float(rng.uniform(1, 30)); cpn = float(rng.uniform(0.01, 0.08))
        y = float(rng.uniform(0.02, 0.09))
        p = da.price_bond(yrs, cpn, y)
        assert p > 0
        # market value = price * notional; single-bond portfolio YTM ~= y
        bonds = pd.DataFrame([dict(years=yrs, coupon=cpn, ytw=y, price_frac=p,
                                   outstanding=1e9)])
        assert abs(da.portfolio_ytm(bonds) - y) < 1e-3


def test_portfolio_summary_shapes():
    bonds = pd.DataFrame([
        dict(years=5, coupon=0.04, ytw=0.05, price_frac=da.price_bond(5, .04, .05),
             outstanding=1e9),
        dict(years=20, coupon=0.05, ytw=0.055, price_frac=da.price_bond(20, .05, .055),
             outstanding=2e9),
    ])
    s, b = da.portfolio_summary(bonds)
    assert s["n_bonds"] == 2
    assert s["market_value_debt"] > 0
    assert 0.04 < s["portfolio_ytm"] < 0.06
    assert s["wavg_years"] > 5
