"""
Cost-of-equity assembly (per company).

    real_COE(t) = real_rf(t) + market_ERP(t) + credit_relative(t) + idiosyncratic(t)

  credit_relative(t) = k * [ issuer_creditRP(t) - market_creditRP(t) ]     (Merton-scaled)
  creditRP           = spread - LGD * hazard(rating)                        (expected loss stripped)
  idiosyncratic(t)   = idio_anchor * (t/30)^p                               (compounds with horizon)

The idiosyncratic anchor is read from the firm's OWN options (Martin-Wagner),
not judgment: a defensive, low-vol name earns little; a high-vol name earns more.
All rates in percent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# approximate historical annualized default hazard by rating (%), for stripping
# expected loss out of the credit spread to leave the pure credit RISK premium.
HAZARD = {"AAA": 0.00, "AA": 0.02, "A": 0.06, "BBB": 0.24, "BB": 1.00, "B": 4.00}
MARKET_IG_HAZARD = 0.10          # blended A/BBB investment-grade index


def assemble_coe(grid, real_rf, market_erp, market_ig_spread, issuer_spread,
                 issuer_rating, k, idio_anchor, p=0.5, lgd=0.60):
    """Return a DataFrame of the four real curves and the ERP decomposition."""
    grid = np.asarray(grid, dtype=float)
    real_rf = np.asarray(real_rf, dtype=float)
    market_erp = np.asarray(market_erp, dtype=float)
    market_ig_spread = np.asarray(market_ig_spread, dtype=float)
    issuer_spread = np.asarray(issuer_spread, dtype=float)

    h_i = HAZARD.get(str(issuer_rating).upper(), 0.24)
    issuer_rp = issuer_spread - lgd * h_i
    market_rp = market_ig_spread - lgd * MARKET_IG_HAZARD
    credit_rel = k * (issuer_rp - market_rp)
    idio = idio_anchor * (grid / 30.0) ** p

    company_erp = market_erp + credit_rel + idio
    real_coe = real_rf + company_erp

    return pd.DataFrame({
        "tenor": grid,
        "real_rf": real_rf,
        "market_erp": market_erp,
        "credit_relative": credit_rel,
        "idiosyncratic": idio,
        "company_erp": company_erp,
        "real_coe": real_coe,
    }).set_index("tenor")


def idio_anchor_from_options(equity_vol, vix, avg_correlation=0.35):
    """Martin-Wagner idiosyncratic premium (% ) at the long horizon:
       ½ * (stock's own risk-neutral variance - the average stock's variance),
       floored at zero. avg stock variance ~ market variance / avg correlation.
    """
    equity_var = equity_vol ** 2
    market_var = (vix / 100.0) ** 2
    avg_stock_var = market_var / max(avg_correlation, 1e-3)
    idio = 0.5 * (equity_var - avg_stock_var)
    return max(idio, 0.0) * 100.0        # decimal variance -> percent premium
