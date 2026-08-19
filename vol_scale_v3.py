"""
vol_scale_v3.py -- VIX1Y-primary volatility-scale (vs) construction for build_erp_daily.py.

Replaces vol_scale_from_shiller() (semi-deviation of realized S&P 500 returns, flat-clipped
to [0.8, 2.0]) as the METHOD used at each monthly re-anchor of the held `vs` input. Nothing in
this file changes the committed June-2026 acceptance reference (VS_JUNE=0.9348 in
build_erp_daily.py stays exactly as landed) -- this is new machinery for the NEXT re-anchor,
not a retroactive change to a published number.

Design, approved by James 2026-08-18 (AEG-Project/docs/SESSION-13-HANDOFF-2026-08-18.md and
the three round documents it links):

  (1) SOFT CLIP, replacing the flat [0.8, 2.0] stop. Tanh-in-log-space: identity between the
      knees, slope exactly 1.00 at each knee (no kink), the asymptote is approached but never
      reached. Knees [0.70, 1.55], asymptotes [0.40, 2.50]. Rationale and the 15-check
      verification live in AEG-Project/tools/volscale_clip_bounds_analysis.py and
      volscale_clip_bounds_VERIFY.py -- soft_clip() below is transcribed unchanged from there.

  (2) VIX1Y AS SOLE PRIMARY, normalized by a FIXED full-record median, not a rolling one (a
      rolling window chases the volatility cycle rather than averaging over it -- see the
      1937-38 / 1973-74 case study in AEG-Project/tools/volscale_clip_bounds_analysis.py
      section 3). The median is a dated constant, re-derived annually on a stated date, with a
      change log and a guard rail: a refresh that would move it more than 5% does NOT
      auto-adopt, it flags for review instead.

  (3) SIX-TIER SOURCE CHAIN, no semi-deviation fallback (switching to semi-deviation was wrong
      by 0.734 in vs units at the 95th percentile on the historical overlap -- worse than
      freezing the number for a full month at 0.221; see
      AEG-Project/docs/AEG-Market-VolScale-ROUND-2-RESULTS-2026-08-18.md section 2).
        tier 1  CBOE history CSV      (daily-close file; the current primary)
        tier 2  CBOE delayed-quote    (different infrastructure, same vendor)
        tier 3  Yahoo ^VIX1Y          (current value only -- no usable history)
        tier 4  rebuild from VIX/VIX3M/VIX6M via a pinned-kappa mean-reversion fit
        tier 5  hold last known-good value, stamp stale, raise an alarm
        tier 6  refuse to publish after N=3 stale trading days

STATUS: built and self-tested here; NOT yet wired into the monthly re-anchor process or any
automated job. Wiring this in (i.e., making it the thing that actually sets the next month's
`vs`) is a separate decision -- see the accompanying results doc.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from datetime import datetime, timedelta

import numpy as np

# ------------------------------------------------------------------ (1) soft clip
VS_LO, VS_HI = 0.40, 2.50          # asymptotes -- never reached
VS_KNEE_LO, VS_KNEE_HI = 0.70, 1.55  # identity region -- unchanged inside here


def soft_clip(x, lo=VS_LO, hi=VS_HI, knee_lo=VS_KNEE_LO, knee_hi=VS_KNEE_HI):
    """Monotone, C1, identity inside [knee_lo, knee_hi], asymptotic to (lo, hi) outside.
    Transcribed unchanged from AEG-Project/tools/volscale_clip_bounds_analysis.py, verified
    there by 15 independent checks (volscale_clip_bounds_VERIFY.py): slope exactly 1.00 at
    each knee, strictly monotone, arithmetically incapable of exceeding the asymptote."""
    x = np.asarray(x, dtype=float)
    out = np.array(x, dtype=float)
    lx = np.log(x)
    lk_hi, lk_lo, lhi, llo = math.log(knee_hi), math.log(knee_lo), math.log(hi), math.log(lo)
    up = lx > lk_hi
    H = lhi - lk_hi
    out[up] = np.exp(lk_hi + H * np.tanh((lx[up] - lk_hi) / H))
    dn = lx < lk_lo
    L = lk_lo - llo
    out[dn] = np.exp(lk_lo - L * np.tanh((lk_lo - lx[dn]) / L))
    return out


def soft_clip_scalar(x, lo=VS_LO, hi=VS_HI, knee_lo=VS_KNEE_LO, knee_hi=VS_KNEE_HI):
    """Scalar convenience wrapper -- what the daily/monthly caller actually wants."""
    return float(soft_clip(np.array([float(x)]), lo, hi, knee_lo, knee_hi)[0])


# ------------------------------------------------------------------ (2) fixed median + guard rail
# Dated constant. CHANGE LOG:
#   2026-08-18  22.62   full CBOE VIX1Y record 2007-01-03 .. 2026-08-17 (4,930 obs), verified
#                       independently against a fresh CBOE pull the same day. Moving-block
#                       bootstrap (1-year blocks) 95% interval [20.37, 24.96] (wider than the
#                       [21.6, 23.6] quoted in round 2 -- that interval was on a shorter/earlier
#                       pull; re-stated here honestly rather than silently tightened).
VIX1Y_MEDIAN = 22.62
VIX1Y_MEDIAN_ASOF = "2026-08-18"
VIX1Y_MEDIAN_CHANGE_LOG = [
    {"date": "2026-08-18", "median": 22.62, "n_obs": 4930,
     "record": "2007-01-03..2026-08-17", "note": "initial adoption, session 13"},
]


def guard_rail_check(new_median, old_median=VIX1Y_MEDIAN, threshold=0.05):
    """A scheduled annual refresh does NOT auto-adopt if it would move the constant by more
    than `threshold` (default 5%). Returns (accept: bool, pct_move: float, message: str)."""
    pct_move = (new_median / old_median) - 1.0
    if abs(pct_move) > threshold:
        return False, pct_move, (
            f"GUARD RAIL TRIPPED: proposed median {new_median:.3f} vs current {old_median:.3f} "
            f"is a {100*pct_move:+.1f}% move, exceeding the {100*threshold:.0f}% threshold. "
            f"Flagged for review, NOT auto-adopted.")
    return True, pct_move, (
        f"within tolerance: {new_median:.3f} vs {old_median:.3f} ({100*pct_move:+.1f}%), "
        f"auto-adopt permitted.")


def vol_scale_from_vix1y(vix1y_value, median=VIX1Y_MEDIAN):
    """The new vs construction: VIX1Y / fixed median, soft-clipped. This is what a monthly
    re-anchor should call once a validated VIX1Y value (see fetch_vix1y_sixtier below) is in
    hand."""
    return soft_clip_scalar(float(vix1y_value) / float(median))


# ------------------------------------------------------------------ (3) six-tier source chain
CBOE_HISTORY_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
CBOE_DELAYED_JSON = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{sym}.json"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/%5E{sym}"
CROSS_CHECK_TOLERANCE = 0.25    # vol points; disagreement beyond this is an alarm, not a coin flip
STALE_REFUSE_DAYS = 3           # tier 6: refuse to publish after this many stale trading days


def _http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_cboe_history_csv(sym="VIX1Y", timeout=15):
    """Tier 1. Returns (date_str, value) for the LAST ROW of the published history file, or
    raises on any failure (network, parse, empty)."""
    text = _http_get(CBOE_HISTORY_CSV.format(sym=sym), timeout=timeout)
    lines = [ln for ln in text.strip().splitlines() if ln]
    if len(lines) < 2:
        raise ValueError("CBOE history CSV: no data rows")
    cells = lines[-1].split(",")
    m, d, y = cells[0].split("/")
    date_str = f"{y}-{int(m):02d}-{int(d):02d}"
    val = float(cells[-1])
    if not (3.0 < val < 200.0):
        raise ValueError(f"CBOE history CSV: implausible value {val}")
    return date_str, val


def fetch_cboe_delayed_json(sym="VIX1Y", timeout=15):
    """Tier 2. Different CBOE infrastructure (delayed-quote endpoint, not the published file),
    so a failure of the file-publishing path specifically doesn't take this down too."""
    text = _http_get(CBOE_DELAYED_JSON.format(sym=sym), timeout=timeout)
    j = json.loads(text)
    val = float(j["data"]["current_price"])
    ts = j.get("timestamp", "")
    date_str = ts.split(" ")[0] if ts else None
    if not (3.0 < val < 200.0):
        raise ValueError(f"CBOE delayed JSON: implausible value {val}")
    return date_str, val


def fetch_yahoo_chart(sym="VIX1Y", timeout=15):
    """Tier 3. CURRENT VALUE ONLY -- confirmed in round 2 that every history range (1mo/1y/
    5y/max) returns a single point for this symbol. Live-value source only; cannot be used
    for the annual median recalibration and cannot be validated against CBOE over history,
    only day-by-day."""
    text = _http_get(YAHOO_CHART.format(sym=sym), timeout=timeout)
    j = json.loads(text)
    meta = j["chart"]["result"][0]["meta"]
    val = float(meta["regularMarketPrice"])
    if not (3.0 < val < 200.0):
        raise ValueError(f"Yahoo chart: implausible value {val}")
    return None, val   # no reliable date field for this degenerate history case


def integrated_var(T, theta, v0, kappa):
    """Heston/OU average expected variance over [0,T]. Same formula as
    AEG-Project/tools/vix_termstructure_and_sources.py::integrated_var."""
    T = float(T)
    kT = kappa * T
    damp = 1.0 if kT < 1e-8 else (1.0 - math.exp(-kT)) / kT
    return theta + (v0 - theta) * damp


def rebuild_vix1y_from_short_tenors(vix_val, vix3m_val, vix6m_val, kappa_grid=None):
    """Tier 4. Fit theta and v0 (kappa profiled over a grid, exactly as in
    vix_termstructure_and_sources.py::fit_one_factor) to the three SEPARATE CBOE index files
    (VIX 30d, VIX3M, VIX6M -- a VIX1Y-file-specific outage does not touch these) and read off
    the 1-year point. Round 2 measured this against the traded VIX1Y: median error 0.022,
    95th pct 0.081 in vs units -- about as accurate as holding two days stale, but responsive."""
    if kappa_grid is None:
        kappa_grid = np.exp(np.linspace(math.log(0.05), math.log(30.0), 260))
    Ts = np.array([30 / 365.0, 91 / 365.0, 182 / 365.0])
    vols = np.array([vix_val, vix3m_val, vix6m_val], dtype=float)
    y = (vols / 100.0) ** 2
    best = None
    for k in kappa_grid:
        kT = k * Ts
        d = (1.0 - np.exp(-kT)) / kT
        X = np.column_stack([1.0 - d, d])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        theta, v0 = beta
        if theta <= 1e-6 or v0 <= 1e-6:
            continue
        pred = 100.0 * np.sqrt(np.maximum(X @ beta, 1e-12))
        sse = float(np.sum((pred - vols) ** 2))
        if best is None or sse < best[0]:
            best = (sse, theta, v0, k)
    if best is None:
        raise ValueError("tier 4 rebuild: no valid (theta, v0) fit on this kappa grid")
    _, theta, v0, kappa = best
    v1y = 100.0 * math.sqrt(max(integrated_var(1.0, theta, v0, kappa), 1e-12))
    return v1y


def fetch_vix1y_sixtier(asof_date=None, last_known_good=None, last_known_good_date=None,
                         vix_val=None, vix3m_val=None, vix6m_val=None,
                         cross_check_tolerance=CROSS_CHECK_TOLERANCE,
                         stale_refuse_days=STALE_REFUSE_DAYS, log=print):
    """Walk the six-tier chain. Returns a dict:
        value        -- the VIX1Y value to use (or None if refusing to publish, tier 6)
        tier_used    -- 1..6, or 'refuse'
        source       -- human-readable source name
        alarm        -- bool, True if this result needs a human to look at it
        message      -- explanation
        stale_days   -- if tier 5, how many trading days stale
        cross_check  -- dict with tier1/tier2/tier3 raw reads if more than one answered, for
                         the disagreement-is-an-alarm rule

    `asof_date` is informational only (for logging/staleness bookkeeping); the live tiers
    return whatever is currently published. `vix_val/vix3m_val/vix6m_val` are required only
    if tiers 1-3 all fail and tier 4 is reached -- caller must supply the shorter-tenor CBOE
    reads (a VIX1Y-specific outage should not have taken these down)."""
    reads = {}
    for tier, name, fn in [(1, "CBOE history CSV", fetch_cboe_history_csv),
                            (2, "CBOE delayed-quote JSON", fetch_cboe_delayed_json),
                            (3, "Yahoo ^VIX1Y", fetch_yahoo_chart)]:
        try:
            date_str, val = fn()
            reads[tier] = (name, date_str, val)
            log(f"  tier {tier} ({name}): OK, value={val}")
        except Exception as e:
            log(f"  tier {tier} ({name}): FAILED ({e})")

    if reads:
        # cross-check: if more than one tier answered, they must agree within tolerance
        vals = [v for (_, _, v) in reads.values()]
        if len(vals) > 1 and (max(vals) - min(vals)) > cross_check_tolerance:
            return dict(value=None, tier_used="alarm", source="cross-check mismatch",
                        alarm=True, stale_days=0, cross_check=reads,
                        message=(f"ALARM: sources disagree beyond tolerance "
                                 f"({cross_check_tolerance} vol pts): {reads}. "
                                 f"Not auto-resolved -- needs review."))
        # use the lowest-numbered (most reliable) tier that answered
        best_tier = min(reads)
        name, date_str, val = reads[best_tier]
        return dict(value=val, tier_used=best_tier, source=name, alarm=False,
                     stale_days=0, cross_check=reads if len(reads) > 1 else None,
                     message=f"OK via tier {best_tier} ({name}) = {val}"
                             + (f", cross-checked against {len(reads)-1} other source(s)"
                                if len(reads) > 1 else ""))

    # tiers 1-3 all failed -> tier 4
    if vix_val is not None and vix3m_val is not None and vix6m_val is not None:
        try:
            val = rebuild_vix1y_from_short_tenors(vix_val, vix3m_val, vix6m_val)
            log(f"  tier 4 (rebuild from VIX/VIX3M/VIX6M): OK, value={val:.3f}")
            return dict(value=val, tier_used=4, source="rebuild from VIX/VIX3M/VIX6M",
                        alarm=True, stale_days=0, cross_check=None,
                        message=(f"tiers 1-3 all failed; rebuilt VIX1Y={val:.3f} from short "
                                 f"tenors (median historical error 0.022, 95th pct 0.081 vs "
                                 f"units). ALARM raised so a human confirms tiers 1-3 are "
                                 f"actually down, not silently degraded."))
        except Exception as e:
            log(f"  tier 4 (rebuild): FAILED ({e})")
    else:
        log("  tier 4 (rebuild): skipped, no VIX/VIX3M/VIX6M supplied")

    # tier 4 unavailable/failed -> tier 5 (hold last known good) or tier 6 (refuse)
    if last_known_good is None:
        return dict(value=None, tier_used="refuse", source=None, alarm=True, stale_days=None,
                     cross_check=None,
                     message="ALL TIERS FAILED and no last-known-good value supplied. "
                             "Refusing to publish.")
    stale_days = 0
    if last_known_good_date and asof_date:
        try:
            stale_days = (datetime.fromisoformat(asof_date)
                          - datetime.fromisoformat(last_known_good_date)).days
        except Exception:
            stale_days = None
    if stale_days is not None and stale_days >= stale_refuse_days:
        return dict(value=None, tier_used="refuse", source=None, alarm=True,
                     stale_days=stale_days, cross_check=None,
                     message=(f"tier 6: {stale_days} stale trading days >= "
                              f"{stale_refuse_days}-day refuse threshold. Refusing to "
                              f"publish a fresh valuation until the source is restored."))
    return dict(value=last_known_good, tier_used=5, source="hold last known-good",
                alarm=True, stale_days=stale_days, cross_check=None,
                message=(f"tier 5: holding last known-good value {last_known_good} "
                         f"({stale_days if stale_days is not None else '?'} stale day(s)). "
                         f"ALARM raised."))
