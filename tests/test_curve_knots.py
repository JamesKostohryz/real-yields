"""Every published knot must survive into the curve, exactly.

Written 2026-09-03, after James compared the engine's thirty-year rate against FRED and found
it did not tie. Two things were wrong and neither had a test:

  * `build_asof` interpolated the real curve over a hard-coded [1, 5, 10, 20, 30] while
    `asfp.datasources.DFII_MAP` published five real tenors including a SEVEN-YEAR point. The
    seven-year knot was fetched by nobody and interpolated straight through. On the
    2026-09-03 curve the engine read 2.2580% there against Treasury's published 2.2700%.
  * The interpolant was linear, which kinks at every knot and slopes through a flat segment.

A published market rate that the engine quietly replaces with its own interpolation is the
project's first standing failure mode -- internally consistent, externally wrong, every gate
green. These tests are cheap and they close it.
"""
import json
import os

import numpy as np
import pytest

import build_erp_daily as BED
from asfp import datasources as ds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _state():
    with open(os.path.join(ROOT, "ERP_HELD_STATE_2026-06.json")) as fh:
        return json.load(fh)


def _build(real_knots, **kw):
    s = _state()
    return BED.build_asof(real_knots, 5.95, s["vs"], s["fey_in"], s["D_in"],
                          s["cost"], s["corp_prem"], **kw)


def test_every_supplied_knot_is_reproduced_exactly():
    """The whole point. A knot is an observed market rate; the curve must not move it."""
    knots = {1: 1.63, 5: 2.15, 7: 2.27, 10: 2.42, 20: 2.76, 30: 2.96}
    r = _build(knots)
    for tenor, published in knots.items():
        got = r["spot_real"][tenor - 1]
        assert abs(got - published) < 1e-9, (
            f"tenor {tenor}: curve says {got:.6f}, Treasury published {published:.4f}")


def test_the_seven_year_knot_is_not_interpolated_through():
    """The specific regression. Dropping the 7y and interpolating 5->10 misses it."""
    full = {1: 1.63, 5: 2.15, 7: 2.27, 10: 2.42, 20: 2.76, 30: 2.96}
    without = {k: v for k, v in full.items() if k != 7}
    assert abs(_build(full)["spot_real"][6] - 2.27) < 1e-9
    # and dropping it really does land somewhere else, so this test can fail
    assert abs(_build(without)["spot_real"][6] - 2.27) > 1e-4


def test_a_flat_segment_stays_flat():
    """PCHIP is shape-preserving. Treasury's nominal curve was flat 20->30 on 2026-09-03;
    an interpolant that slopes through a flat segment is inventing a term premium."""
    knots = {1: 2.00, 5: 2.40, 10: 2.60, 20: 2.90, 30: 2.90}
    seg = _build(knots)["spot_real"][19:30]
    assert np.allclose(seg, 2.90, atol=1e-9), f"flat segment drifted: {seg}"


def test_no_overshoot_between_knots():
    """A monotone interpolant never leaves the interval its own knots define."""
    knots = {1: 1.50, 5: 2.15, 7: 2.27, 10: 2.42, 20: 2.76, 30: 2.96}
    y = _build(knots)["spot_real"]
    assert y.min() >= min(knots.values()) - 1e-9
    assert y.max() <= max(knots.values()) + 1e-9
    assert np.all(np.diff(y) >= -1e-9), "monotone inputs produced a non-monotone curve"


def test_refuses_to_extrapolate_past_the_last_knot():
    """`outputs/curve_latest.csv` flags its own tenors 21-30 `back-constructed,
    reliability 0.0`. This path must die instead of doing that."""
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        _build({1: 1.63, 5: 2.15, 10: 2.42, 20: 2.76})       # no 30y knot


def test_the_fetcher_asks_for_every_tenor_the_map_publishes():
    """Source-level guard. The bug was a hard-coded subset in the CALLER, not in the map."""
    import inspect
    import run_erp_daily as RED
    src = inspect.getsource(RED.fetch_daily_inputs)
    assert "sorted(ds.DFII_MAP)" in src, (
        "fetch_daily_inputs must iterate DFII_MAP, not re-list a subset of its tenors")
    assert set(ds.DFII_MAP) == {5, 7, 10, 20, 30}
