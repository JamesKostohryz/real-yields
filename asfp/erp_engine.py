"""
Skew-priced ERP engine (final architecture).

Principle (James): the fair equity risk premium compensates only for the ASYMMETRY —
a symmetric bet (equal up/down) has zero expected excess return and earns nothing. The
premium is the price of the skew: the corridor **down-variance-swap minus up-variance-swap**,
read off the option surface (see skew.py). Symmetric smile → zero; left-skew → positive;
tail-weighted so the deep-put crash region carries it.

    ERP(security, T) = phi · corridor_marginal(security, T)  [+ liquidity floor]

  * corridor(T)      = K_down(T) − K_up(T), the risk-neutral asymmetry to horizon T.
  * corridor_marginal= the per-year (FORWARD) asymmetry — the same one-year-forward transform
    we apply to variance, so each year's ERP is that year's marginal skew, consistent with
    discounting year-by-year off the real forward rate.
  * phi ∈ (0,1]      = the ONE dial: the physical/risk-neutral skew ratio. phi=1 uses the full
    option-implied (risk-neutral) skew — what the market charges; phi<1 haircuts it toward the
    realized (physical) skew, on the view that crash insurance is somewhat overpriced. Estimable
    from implied-vs-realized skew history; default 1.
  * floor            = optional liquidity/undiversifiable base (deferred; default 0).

No pricing kernel: a fixed kernel prices the second moment (variance/semivariance) first-order
and would hand a symmetric bet a premium — which violates the philosophy. The corridor is the
third-moment (skew-swap) object that prices the asymmetry alone. Applied identically to the
index and to single names; single-stock smiles are flatter, so names get a smaller premium —
the compression that stops charging a name for its two-sided volatility.

All variances/premia in percent-per-year. Pure and injectable; live smiles read in the job.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from . import skew, total_risk_erp as tr


def _symmetric_band(strikes, ivs, F, band_logm):
    """Keep only strikes within a band SYMMETRIC in log-moneyness: |ln(K/F)| ≤ band_logm.
    Essential for a clean skew: if the put wing reaches further down (in log terms) than the
    call wing reaches up — which real chains do — the coverage asymmetry alone produces a
    spurious corridor. Restricting to a symmetric log band makes a symmetric smile price to
    zero and leaves only the true put-vs-call asymmetry."""
    ks, vs = [], []
    for K, v in zip(strikes, ivs):
        if abs(math.log(float(K) / F)) <= band_logm:
            ks.append(float(K)); vs.append(float(v))
    return ks, vs


def _net_skew(strikes, ivs, F, T, band_logm):
    """Baseline-net corridor skew at one tenor (decimal variance): the smile's
    (K_down − K_up) minus the same corridor computed on a FLAT smile at the ATM vol.
    Subtracting the flat baseline removes the variance-swap's lognormal geometry, so a
    symmetric-vol smile prices to ~0 and only the true put-vs-call asymmetry remains."""
    ks, vs = _symmetric_band(strikes, ivs, F, band_logm)
    if len(ks) < 5:
        return 0.0
    d, u = skew.corridor_variances(ks, vs, F, T)
    atm = float(np.interp(F, ks, vs))
    d0, u0 = skew.corridor_variances(ks, [atm] * len(ks), F, T)   # flat-vol baseline
    return (d - u) - (d0 - u0)


def corridor_term_structure(smiles, grid, band_logm=0.35):
    """smiles: {tenor_years: (strikes, ivs, forward)} → baseline-net corridor skew
    (percent, annual, cumulative) interpolated onto `grid`, flat past the last observed
    tenor. Each smile is first truncated to a log-symmetric band. `ivs` decimal."""
    tens = sorted(smiles)
    corr = [_net_skew(*smiles[T], T, band_logm) * 100.0 for T in tens]
    grid = np.asarray(grid, float)
    return np.interp(grid, tens, corr), float(tens[-1])


def skew_erp_curve(smiles, grid, phi=1.0, floor=0.0, forward=True):
    """Fair skew-priced ERP term structure (percent per year).

    Returns a DataFrame indexed by tenor with the (baseline-net) corridor skew and
    erp = phi·corridor + floor. `forward=True` uses the marginal (per-year) corridor."""
    grid = np.asarray(grid, float)
    corr_cum, obs_max = corridor_term_structure(smiles, grid)
    corr = tr.forward_erp(grid, corr_cum) if forward else corr_cum   # marginal per-year skew
    corr = np.maximum(corr, 0.0)
    erp = phi * corr + floor
    return pd.DataFrame({"tenor": grid, "corridor_skew": corr, "erp": erp}).set_index("tenor")


def effective_erp(erp_curve, growth=2.0):
    """Collapse an ERP term structure to a single cash-flow-weighted number (percent).
    Weights are a real cash-flow stream growing at `growth`%/yr, discounted by the ERP path —
    dominated by the near-to-middle horizons, like a bond's YTM weights nearer coupons."""
    grid = erp_curve.index.to_numpy(dtype=float)
    erp = erp_curve["erp"].to_numpy()
    cf = (1.0 + growth / 100.0) ** grid
    df = 1.0 / np.cumprod(1.0 + erp / 100.0)      # discount by the ERP path itself
    w = cf * df
    return float(np.sum(w * erp) / np.sum(w))
