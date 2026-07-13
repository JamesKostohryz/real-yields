"""End-to-end sanity test of the cross-sectional builder on synthetic data."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import model, curves

# Realistic-ish synthetic Svensson params (percent)
NOM = dict(b0=4.6, b1=-1.2, b2=-1.0, b3=2.0, t1=1.4, t2=9.0)
REAL = dict(b0=2.2, b1=-1.6, b2=-0.8, b3=1.5, t1=1.4, t2=9.0)
GRID = np.arange(1, 31, dtype=float)
CF = np.full(30, 2.30) + 0.01 * (GRID - 10)     # gently rising ~2.2->2.5
NOWCAST = 2.9


def build():
    return model.build_from_params(NOM, REAL, CF, NOWCAST,
                                   calib=dict(tau_hi=20.0, front_floor=-0.15))


def test_fisher_identity_holds():
    df = build()
    resid = df["nominal"] - (df["real"] + df["breakeven"])
    assert np.max(np.abs(resid)) < 1e-9


def test_in_band_real_matches_gsw():
    # with smoothing OFF, the trusted band must reproduce GSW real exactly
    df = model.build_from_params(NOM, REAL, CF, NOWCAST,
                                 calib=dict(tau_hi=20.0, front_floor=-0.15,
                                            smooth=False))
    band = (df.index.to_numpy() >= 2) & (df.index.to_numpy() <= 20)
    real_gsw = curves.svensson_zero(df.index.to_numpy(), **REAL)
    diff = (df["real"].to_numpy() - real_gsw)[band]
    assert np.max(np.abs(diff)) < 1e-6


def test_smoothing_reduces_front_wiggle_but_preserves_core():
    grid = np.arange(1, 31, dtype=float)
    real = curves.svensson_zero(grid, **REAL)
    noisy = real.copy()
    noisy[1] += 0.35          # inject a GSW-style spike at the 2y point
    w = model._fidelity_profile(grid)
    smoothed = model.whittaker_smooth(noisy, w, 30.0)

    # front forward roughness (2nd diff near the 2y) must drop substantially
    def front_rough(z):
        f = curves.one_year_forwards(z, z[0])
        return np.max(np.abs(np.diff(f[:5], 2)))
    assert front_rough(smoothed) < 0.5 * front_rough(noisy)
    # the 10y level must barely move (core is trusted)
    assert abs(smoothed[9] - real[9]) < 0.03


def test_expected_inflation_front_uses_nowcast():
    df = build()
    assert abs(df.loc[1.0, "exp_inflation"] - NOWCAST) < 1e-9
    assert abs(df.loc[2.0, "exp_inflation"] - CF[1]) < 1e-9   # back to CF by 2y


def test_forwards_telescope_all_curves():
    df = build()
    for col in ["nominal", "real", "breakeven", "exp_inflation"]:
        z = df[col].to_numpy()
        f = df[f"{col}_fwd1y"].to_numpy()
        for N in [1, 5, 10, 30]:
            assert abs(f[:N].sum() - N * z[N - 1]) < 1e-9


def test_no_seam_kink_in_real_forwards():
    df = build()
    # Exclude f(0,1): it is a spot yield, so the spot->forward step to f(1,2)
    # is genuine front steepness, not a seam artifact. Check the belly/back,
    # which spans both the 2y and 20y construction seams.
    f = df["real_fwd1y"].to_numpy()[1:]
    d2 = np.abs(np.diff(f, 2))
    assert np.max(d2) < 0.25, f"max 2nd-diff {np.max(d2):.3f} suggests a seam kink"


def test_provenance_labels():
    df = build()
    assert df.loc[1.0, "provenance"] == "front-constructed"
    assert df.loc[10.0, "provenance"] == "observed"
    assert df.loc[30.0, "provenance"] == "back-constructed"


def test_roll_forward_shifts_curve():
    grid = np.arange(1, 31, dtype=float)
    base = curves.svensson_zero(grid, **REAL)
    # daily DFII moves: +10bp at 5y, +8 at 10y, +5 at 20/30
    dmats = [5, 7, 10, 20, 30]
    dvals = [0.10, 0.09, 0.08, 0.05, 0.05]
    rolled = model.roll_forward(base, grid, dmats, dvals, front_fill_below=5)
    # at 10y the curve rose ~8bp
    assert abs((rolled[9] - base[9]) - 0.08) < 1e-9
    # below 5y the 5y delta is carried in (front moves with 5y+)
    assert abs((rolled[0] - base[0]) - 0.10) < 1e-9
    # a zero-delta roll is a no-op
    same = model.roll_forward(base, grid, dmats, [0]*5, front_fill_below=5)
    assert np.max(np.abs(same - base)) < 1e-12


if __name__ == "__main__":
    df = build()
    pd_opt = dict(float_format=lambda x: f"{x:7.3f}")
    import pandas as pd
    pd.set_option("display.width", 200)
    cols = ["nominal", "real", "breakeven", "exp_inflation", "phi",
            "real_fwd1y", "provenance"]
    print(df[cols].to_string())
    print("\nHeadline points:")
    for k, v in model.headline_points(df).items():
        print(f"  {k:22s} {v:7.3f}")
