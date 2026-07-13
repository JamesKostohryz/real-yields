"""
Aggregate credit market grid for the cost-of-debt tool.

Builds smooth per-rating spread curves (AA / A / BBB), 1-30y, from ICE BofA
indices on FRED, plus the nominal Treasury curve used to strip issuer spreads.

Design: the *shape* (term structure) comes from the overall IG spread-by-maturity
buckets; each rating's *level* comes from its index OAS. A rating curve is the IG
shape scaled to that rating's level (multiplicative, so it stays positive and
proportional). An individual issuer's curve is later this rating curve times a
level offset fitted from the issuer's own bonds.

All series are in percent, consistent with the yield curves elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import datasources as ds

# ICE BofA OAS by maturity bucket -> representative tenor (years)
IG_MATURITY = [(2.0, "BAMLC1A0C13Y"), (4.0, "BAMLC2A0C35Y"), (6.0, "BAMLC3A0C57Y"),
               (8.5, "BAMLC4A0C710Y"), (12.5, "BAMLC7A0C1015Y"), (20.0, "BAMLC8A0C15PY")]
IG_OVERALL = "BAMLC0A0CM"                       # overall IG index OAS (the level)

# Full rating family (ICE BofA OAS on FRED). Investment grade = C-series,
# high yield = H-series. Each rating's LEVEL comes from its index OAS; the
# term SHAPE is the IG maturity-bucket shape scaled multiplicatively (a mild
# approximation for HY, whose real curves are flatter/humped — flagged for the
# diagnostic-fan use, not used to reprice HY debt).
RATING_SERIES = {
    "AAA": "BAMLC0A1CAAA",     # US Corporate AAA
    "AA":  "BAMLC0A2CAA",      # US Corporate AA
    "A":   "BAMLC0A3CA",       # US Corporate A
    "BBB": "BAMLC0A4CBBB",     # US Corporate BBB
    "BB":  "BAMLH0A1HYBB",     # US High Yield BB
    "B":   "BAMLH0A2HYB",      # US High Yield B
    "CCC": "BAMLH0A3HYC",      # US High Yield CCC & lower
}
IG_RATING = {k: RATING_SERIES[k] for k in ("AA", "A", "BBB")}   # back-compat
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

TSY = [(1.0, "DGS1"), (2.0, "DGS2"), (3.0, "DGS3"), (5.0, "DGS5"),
       (7.0, "DGS7"), (10.0, "DGS10"), (20.0, "DGS20"), (30.0, "DGS30")]


def build_from_knots(grid, ig_mat_knots, ig_overall, rating_anchors, tsy_knots, real_fwd):
    """Pure construction (no I/O), so it can be unit-tested offline.

    ig_mat_knots : list[(tenor, spread%)] for the IG maturity buckets
    ig_overall   : overall IG index OAS (%) — the level the shape is scaled from
    rating_anchors : {rating: level%}
    tsy_knots    : list[(tenor, yield%)]
    real_fwd     : array of real 1y-forward yields aligned to `grid`
    Returns a DataFrame indexed by tenor.
    """
    grid = np.asarray(grid, dtype=float)
    real_fwd = np.asarray(real_fwd, dtype=float)

    mx = [t for t, _ in ig_mat_knots]
    my = [v for _, v in ig_mat_knots]
    ig_shape = np.interp(grid, mx, my)          # linear, flat beyond the ends

    tx = [t for t, _ in tsy_knots]
    ty = [v for _, v in tsy_knots]
    treasury = np.interp(grid, tx, ty)

    out = pd.DataFrame({"tenor": grid, "treasury_nominal": treasury,
                        "ig_index_spread": ig_shape, "real_fwd": real_fwd})
    for r, level in rating_anchors.items():
        scale = (level / ig_overall) if ig_overall else 1.0
        spread = ig_shape * scale
        out[f"spread_{r}"] = spread
        out[f"real_cod_{r}"] = real_fwd + spread          # rating real cost of debt
    return out.set_index("tenor")


def issuer_real_cod(cg, rating, offset=1.0):
    """Per-company REAL cost of debt (forward, by tenor) for the downstream
    valuation engine's cod_<ticker>.csv.

    Issuer curve = rating curve's spread x multiplicative offset (fitted from the
    issuer's own bonds; offset=1.0 is the pure-rating fallback) added to the real
    forward risk-free.  cg is the credit grid; `rating` in AAA..CCC.
    Returns a DataFrame indexed by tenor with the issuer curve and its rating
    fallback, all in percent (cc).
    """
    import pandas as pd
    real_fwd = cg["real_fwd"].to_numpy()
    spread = cg[f"spread_{rating}"].to_numpy() * float(offset)
    return pd.DataFrame({
        "tenor": cg.index.to_numpy(),
        "real_cod": real_fwd + spread,               # issuer (offset-adjusted)
        "spread": spread,
        "rating": rating,
        "offset": float(offset),
        f"real_cod_{rating}": real_fwd + cg[f"spread_{rating}"].to_numpy(),  # fallback
    }).set_index("tenor")


def build_credit_grid(api_key, grid, real_fwd, ratings=None):
    """Fetch the FRED series and build the per-rating grid.

    `ratings` selects which of the full AAA..CCC family to fetch (default: all).
    Ratings whose FRED series returns no data are skipped, so a temporary HY
    outage still leaves the IG grid intact.
    """
    series = RATING_SERIES if ratings is None else {r: RATING_SERIES[r] for r in ratings}
    ig_mat = [(t, ds.fetch_fred_latest(api_key, sid)[0]) for t, sid in IG_MATURITY]
    ig_overall = ds.fetch_fred_latest(api_key, IG_OVERALL)[0]
    anchors = {r: ds.fetch_fred_latest(api_key, sid)[0] for r, sid in series.items()}
    tsy = [(t, ds.fetch_fred_latest(api_key, sid)[0]) for t, sid in TSY]

    ig_mat = [(t, v) for t, v in ig_mat if v is not None]
    tsy = [(t, v) for t, v in tsy if v is not None]
    anchors = {r: v for r, v in anchors.items() if v is not None}
    if not ig_mat or not tsy or not anchors or ig_overall is None:
        raise RuntimeError("credit grid: one or more FRED series returned no data")
    return build_from_knots(grid, ig_mat, ig_overall, anchors, tsy, real_fwd)
