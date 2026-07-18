"""
Skew-priced equity risk premium (corridor / semivariance variance swaps).

Principle (James): you only demand compensation for the ASYMMETRY. A symmetric
bet — equal upside and downside — has zero expected excess return, so it earns no
risk premium (only, later, a liquidity floor). The premium is the price of insuring
the downside-beyond-symmetric — i.e. the skew.

That price is replicable and model-light: the standard variance swap (the object VIX
is built from) is a 1/K²-weighted strip of OTM options. Split it at the forward:

    K_down = the PUT strip   (downside variance, replicable)
    K_up   = the CALL strip  (upside   variance, replicable)
    K_var  = K_down + K_up    (total variance — the Martin/VIX object)

    skew price = K_down − K_up          ← what it costs to harvest the skew

A perfectly symmetric smile gives K_down = K_up and a zero skew price. A left-skewed
equity smile gives K_down > K_up; that positive difference IS the fear premium.

    ERP(security) = (K_down − K_up)  [+ liquidity floor, applied elsewhere]

applied identically to the index and to single names. Because single-stock smiles are
flatter than the index smile (crash-correlation lives in the basket), single names get
a SMALLER skew price than the index — the compression that stops charging a name for its
two-sided volatility.

Pure and unit-tested. Strips are truncated to traded strikes by the caller to control the
deep-tail sensitivity of the 1/K² weighting.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist().cdf


def _bs(cp, F, K, T, vol):
    """Black-76 OTM option price (r=0, real terms), per unit notional."""
    if vol <= 0 or T <= 0:
        return max(F - K, 0.0) if cp == "c" else max(K - F, 0.0)
    srt = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / srt
    d2 = d1 - srt
    if cp == "c":
        return F * _N(d1) - K * _N(d2)
    return K * _N(-d2) - F * _N(-d1)


def corridor_variances(strikes, ivs, F, T):
    """Down- and up-variance-swap strikes (K_down, K_up) from an option smile.

    strikes, ivs : equal-length sequences (strikes ascending, implied vols in decimal).
    F, T         : forward and horizon (years). r assumed 0 (real terms).
    Uses the 1/K² OTM strip (puts below F, calls above), trapezoidal ΔK. Returns annual
    variances (e.g. 0.04 = 4% variance)."""
    ks = [float(k) for k in strikes]
    vs = [float(v) for v in ivs]
    n = len(ks)
    if n < 3:
        raise ValueError("need >=3 strikes")
    kd = ku = 0.0
    for i, (K, v) in enumerate(zip(ks, vs)):
        lo = ks[i - 1] if i > 0 else ks[i]
        hi = ks[i + 1] if i < n - 1 else ks[i]
        dK = (hi - lo) / 2.0 if 0 < i < n - 1 else (hi - lo)
        if dK <= 0:
            continue
        cp = "p" if K < F else "c"
        contrib = (dK / (K * K)) * _bs(cp, F, K, T, v)
        if K < F:
            kd += contrib
        else:
            ku += contrib
    return (2.0 / T) * kd, (2.0 / T) * ku


def skew_price(strikes, ivs, F, T):
    """Full decomposition. Returns dict(k_down, k_up, k_var, skew) in annual variance.
    `skew` = k_down − k_up is the price of the skew — the skew-based ERP (pre-floor)."""
    kd, ku = corridor_variances(strikes, ivs, F, T)
    return {"k_down": kd, "k_up": ku, "k_var": kd + ku, "skew": kd - ku}
