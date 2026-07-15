"""
Curve mathematics for the ASFP real-yield tool.

Conventions
-----------
* Maturities `n` are in years.
* Yields are in PERCENT (matching the GSW published files), continuously
  compounded, unless a function name says otherwise.
* The Svensson parameterization is exactly the one used by the Federal Reserve
  GSW datasets feds200628 (nominal) and feds200805 (TIPS/real):

      y(n) = b0
           + b1 * (1 - exp(-n/t1)) / (n/t1)
           + b2 * [ (1 - exp(-n/t1)) / (n/t1) - exp(-n/t1) ]
           + b3 * [ (1 - exp(-n/t2)) / (n/t2) - exp(-n/t2) ]

  with instantaneous forward

      f(n) = b0
           + b1 * exp(-n/t1)
           + b2 * (n/t1) * exp(-n/t1)
           + b3 * (n/t2) * exp(-n/t2)

These are pure functions with no I/O so they can be unit-tested offline.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def svensson_zero(n, b0, b1, b2, b3, t1, t2):
    """Continuously-compounded zero-coupon yield (percent) at maturity n (years).

    Handles the n -> 0 limit (y -> b0 + b1) without dividing by zero.
    """
    n = np.asarray(n, dtype=float)
    n_safe = np.where(n <= 0, _EPS, n)
    x1 = n_safe / t1
    x2 = n_safe / t2
    d1 = (1.0 - np.exp(-x1)) / x1                      # -> 1 as n->0
    term1 = d1
    term2 = d1 - np.exp(-x1)                           # -> 0 as n->0
    term3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)     # -> 0 as n->0
    y = b0 + b1 * term1 + b2 * term2 + b3 * term3
    # exact limit at n==0
    y = np.where(n <= 0, b0 + b1, y)
    return y


def svensson_inst_forward(n, b0, b1, b2, b3, t1, t2):
    """Instantaneous forward rate (percent) at horizon n (years)."""
    n = np.asarray(n, dtype=float)
    x1 = n / t1
    x2 = n / t2
    return (b0
            + b1 * np.exp(-x1)
            + b2 * x1 * np.exp(-x1)
            + b3 * x2 * np.exp(-x2))


def forward_from_zeros(mats, zeros, start, end):
    """Continuously-compounded forward yield (percent) between `start` and `end`
    years, given a zero curve sampled on `mats` (must include start and end,
    with y(0)=short rate available for start==0)."""
    mats = np.asarray(mats, dtype=float)
    zeros = np.asarray(zeros, dtype=float)
    y = {round(float(m), 6): float(z) for m, z in zip(mats, zeros)}
    ys = y[round(float(start), 6)]
    ye = y[round(float(end), 6)]
    if end == start:
        raise ValueError("start and end must differ")
    return (end * ye - start * ys) / (end - start)


def one_year_forwards(zeros_1_to_N, short_rate):
    """Strip of one-year forward rates f(k, k+1) for k = 0 .. N-1 (percent).

    Parameters
    ----------
    zeros_1_to_N : array of zero yields at maturities 1, 2, ..., N (percent, cc)
    short_rate   : the instantaneous short rate y(0) (percent) — used only for
                   f(0,1)=y(1) internally this is not needed, but kept for a
                   fully general (a,b) call site; f(0,1) reduces to y(1).
    Returns
    -------
    array length N: f(0,1), f(1,2), ..., f(N-1, N)
    """
    z = np.asarray(zeros_1_to_N, dtype=float)
    N = len(z)
    mats = np.arange(1, N + 1, dtype=float)
    y_prev = np.concatenate([[0.0], z[:-1]])           # y(0)=0 weight, y(1)..y(N-1)
    n_prev = np.concatenate([[0.0], mats[:-1]])        # 0,1,...,N-1
    # f(k,k+1) = (k+1)*y(k+1) - k*y(k)
    fwd = mats * z - n_prev * y_prev
    return fwd


def discount_factor(n, y_percent):
    """Zero-coupon discount factor P(n) = exp(-(y/100) * n)."""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y_percent, dtype=float)
    return np.exp(-(y / 100.0) * n)


def par_yield(mats, zeros, freq=2):
    """Par (coupon-equivalent) yield in percent at each maturity in `mats`,
    given cc zero yields `zeros` (percent). `freq` = coupon payments per year.

    par = freq * (1 - P(T)) / sum_{k=1..T*freq} P(k/freq)
    (returned as an annualized percentage).
    """
    mats = np.asarray(mats, dtype=float)
    zeros = np.asarray(zeros, dtype=float)
    zfun = {round(float(m), 6): float(z) for m, z in zip(mats, zeros)}
    out = []
    for T in mats:
        n_cpns = int(round(T * freq))
        times = np.array([(k + 1) / freq for k in range(n_cpns)])
        # need zeros at coupon dates; linear-interpolate on the provided grid
        z_at = np.interp(times, mats, zeros)
        dfs = discount_factor(times, z_at)
        annuity = dfs.sum()
        P_T = discount_factor(T, zfun[round(float(T), 6)])
        par = freq * (1.0 - P_T) / annuity * 100.0
        out.append(par)
    return np.array(out)
