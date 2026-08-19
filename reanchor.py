#!/usr/bin/env python3
"""
reanchor.py — write the next ERP_HELD_STATE_YYYY-MM.json, automatically.

WHY THIS EXISTS. The ERP engine splits its inputs into fast (the real Treasury curve and the
S&P close, refreshed every weekday by erp-daily-close) and slow (eight numbers frozen in
ERP_HELD_STATE_<vintage>.json). Refreshing the slow eight — the "monthly re-anchor" — was a
manual step by design: `run_erp_daily.refresh_vol_scale()` deliberately stopped short of writing
the file so that a human could eyeball the volatility number before it became a published rate.

That design failed in the ordinary way. It was never run for July 2026 and never run for August
2026, because nobody knew it was a standing chore. The consequences as of 2026-08-19:

  * vs(T) — the VIX1Y-derived volatility term structure landed and approved 2026-08-18 — was
    inert. The state still carried June's SCALAR 0.9348. `build_asof` accepts either a scalar or
    a 30-vector, so nothing raised and no gate turned red; the engine simply went on charging a
    flat June number at every tenor. Worth 7.2 basis points on the published cost of equity.
  * The daily publish was on a path to hard-stop on 2026-11-01, when
    `held_state.resolve_held_state` would refuse a state more than MAX_STATE_AGE_MONTHS=4 old.

JAMES'S RULING, 2026-08-19, VERBATIM: "It should not require any human intervention. If the model
is good it is good. When making a valuation the analyst can over-ride." This module implements
that ruling. The analyst override he refers to already exists downstream and is untouched:
`repoint_rates.apply_erp_override()` forces a flat company ERP and preserves the four-method tie.

WHAT THIS MODULE DOES NOT DO. It does not relax the 4-month staleness guard in held_state.py.
That guard is correct. It is being made unreachable by making the re-anchor actually happen,
not by moving the wall further away.

--------------------------------------------------------------------------------------------
THE EIGHT SLOW INPUTS AND WHERE EACH COMES FROM
--------------------------------------------------------------------------------------------
  vs            30-long vs(T)  <- refresh_vol_scale(): six-tier VIX1Y chain, soft clip, fixed
                                  median 22.62. Code already existed; nothing called it.
  fey_in        fair-ey state  <- the prior state's fey_out, recomputed at the prior month end
  D_in          duration state <- the prior state's D_out, likewise
  corp_prem     ERP floor      <- floor_from_credit_grid(cg, wedge=0.50) on
                                  outputs/market_credit_latest.csv (rewritten every weekday)
  breakeven1y   1y "breakeven" <- EXPINF1YR (Cleveland Fed via FRED) + a held wedge. NOT the
                                  1-year row of outputs/curve_latest.csv — see
                                  derive_breakeven1y() for why that is 68bp wrong.
  cost          cost overlay   <- cost_of_year(), glide continues, FLOORED at 0.25 (mid-2032)
  normalized_X4 normalized EPS <- carried forward; the normalization job is not built yet
  cpi_factor    deflator       <- carried forward, same reason

fey_in and D_in are NOT fetched. The engine emits next month's values as fey_out and D_out on
every run; the re-anchor replays the prior month's final business day through build_asof and
reads them back. The state machine already closed on itself — nobody had written down that it
did, which is why two months of carry-forward were simply skipped rather than reported missing.

--------------------------------------------------------------------------------------------
THE THREE-TIER GUARD, WHICH IS THE POINT OF THE MODULE
--------------------------------------------------------------------------------------------
Removing the human does not remove the risk the human was there to catch; it relocates it into
code. "No human intervention" is a ruling about ROUTINE operation, not a licence to publish a
number nobody could ever have questioned.

  GREEN  every input inside its absolute band and inside its month-on-month move limit.
         Write, publish, say nothing.

  AMBER  inside the absolute band, past the move limit. WRITE AND PUBLISH ANYWAY — markets do
         move that far and suppressing a real move is worse than reporting it — but exit
         non-zero so the workflow run goes red and GitHub emails James the same morning.
         The number is never suppressed. Only the silence is.

  RED    do not write; keep the prior vintage; exit non-zero. Any of: the VIX1Y chain returning
         tier 5 (stale) or tier 6 (refusing); effective VIX1Y below the feasibility floor; a
         source file staler than MAX_SOURCE_AGE_DAYS; any NaN; any input outside its absolute
         band. The daily job keeps running on the previous month's state, which is precisely
         what the 4-month tolerance is runway for.

Spec: AEG-Project/docs/SPEC-Automated-Monthly-Reanchor-2026-08-19.md
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import os
import sys

import pandas as pd

import build_erp_daily as BED
import held_state as HS
import run_erp_daily as RR


# ============================================================================================
# (0) THE COST TERMINAL RULE
# ============================================================================================
# build_erp_daily.cost_of_year() is a glide path: 1.50% in 1995 arriving at exactly 0.50% at
# mid-2026. It has now arrived, and it was never given a terminal rule, so past 2026.5 it keeps
# falling — and because the exponent is 1.3 it falls at an ACCELERATING rate:
#
#       2026.50   0.5000        2035      0.1358
#       2026.63   0.4946        2038.03   0.0000  <- crosses zero
#       2030      0.3532        2040     -0.0899
#
# A cost overlay that goes negative is not a cost.
#
# JAMES'S RULING, 2026-08-19, VERBATIM: "The cost floor should move down at whatever rate it is
# not moving at until it reaches 0.25% and then should flatline from there."
#
# So the glide is NOT stopped at 0.50 -- it keeps running at its own rate and is floored at 0.25.
# The formula reaches 0.25 in mid-2032 and is flat thereafter:
#
#       2026.63 (today)  0.4946        2032.00  0.2673
#       2028             0.4377        2032.40  0.2500  <- floor engages
#       2030             0.3532        2035+    0.2500
#
# This is a floor, not a stop: `max(cost_of_year(yr), 0.25)`. Nothing special happens at 2026.5.
# The only thing being prevented is the descent through zero into a NEGATIVE cost overlay, which
# the unfloored formula does in 2038.
COST_FLOOR = 0.25

# ============================================================================================
# (1) GUARD BANDS
# ============================================================================================
# (absolute_lo, absolute_hi, max_absolute_move_vs_prior_vintage)
# Absolute breach -> RED (refuse). Move breach -> AMBER (write, then fail the run).
BANDS = {
    "vs_1y":       (0.5283, 2.50, 0.35),   # lo is the derived feasibility floor / median
    "corp_prem":   (0.0,    4.00, 1.00),
    "breakeven1y": (0.0,    6.00, 1.00),
    "cost":        (0.25,   1.50, 0.05),   # lo == COST_FLOOR; the glide runs down to it
    "fey_in":      (2.0,   12.00, 1.00),
    "D_in":        (12.0,  60.00, 5.00),
}

# A source file older than this is RED. The two source files are rewritten every weekday, so
# five calendar days covers a long weekend plus one failed run without tripping.
MAX_SOURCE_AGE_DAYS = 5


class ReanchorRefused(Exception):
    """RED. Nothing is written; the prior vintage stands."""


# ============================================================================================
# (2) DERIVATIONS
# ============================================================================================

def cost_for(asof: dt.date) -> float:
    """The cost overlay: the glide continues at its own rate, floored at COST_FLOOR."""
    yr = asof.year + (asof.timetuple().tm_yday - 1) / (366.0 if calendar.isleap(asof.year) else 365.0)
    return float(max(BED.cost_of_year(yr), COST_FLOOR))


def _check_source_age(path: str, asof: dt.date, log=print) -> None:
    if not os.path.exists(path):
        raise ReanchorRefused(f"source file missing: {path}")
    mtime = dt.date.fromtimestamp(os.path.getmtime(path))
    age = (asof - mtime).days
    if age > MAX_SOURCE_AGE_DAYS:
        raise ReanchorRefused(
            f"{os.path.basename(path)} is {age} days old (limit {MAX_SOURCE_AGE_DAYS}). The "
            f"weekday job that writes it has stopped; re-anchoring off it would freeze a stale "
            f"credit/curve read into a published rate.")
    log(f"  source {os.path.basename(path)}: {age}d old, OK")


def derive_corp_prem(root: str, asof: dt.date, log=print) -> float:
    """Market-ERP floor from the live credit grid. Method and every constant are fixed by
    ERP_HELD_STATE_2026-06.json's own corp_prem_derivation block, which ends: 'Re-derive at each
    monthly re-anchor rather than holding this value indefinitely -- the live credit spread
    moves.' This function is that instruction, executed."""
    from asfp import volsurface as VS
    path = os.path.join(root, "outputs", "market_credit_latest.csv")
    _check_source_age(path, asof, log=log)
    cg = pd.read_csv(path).set_index("tenor")
    val = float(VS.floor_from_credit_grid(cg, wedge=0.50))
    if math.isnan(val):
        raise ReanchorRefused("corp_prem derived as NaN from the credit grid")
    log(f"  corp_prem: {val:.4f} (floor_from_credit_grid, wedge=0.50)")
    return val


# ---------------------------------------------------------------------------- breakeven1y
# THE 1-YEAR ANCHOR PAIR. Both halves are observations, not choices, and both are recorded here
# so the wedge below is auditable rather than typed.
#
#   BE1Y_ANCHOR      2.76       ERP_HELD_STATE_2026-06.json. Verified to be the plug that makes
#                               nominal_1y - breakeven1y reproduce the monthly master's 1-year
#                               real yield: 3.83 - 1.069601 = 2.760399.
#   EXPINF1Y_ANCHOR  3.439483   Cleveland Fed 1-year expected inflation for the SAME vintage,
#                               history/expected_inflation_termstructure_v2.csv, 2026-06-30.
#
# WEDGE = BE1Y_ANCHOR - EXPINF1Y_ANCHOR = -0.679483. Economically this is the 1-year inflation
# risk premium plus the TIPS liquidity effect: the gap between the breakeven the curve model
# wants at one year and pure expected inflation. It is a slow structural quantity. Expected
# inflation is not.
BE1Y_ANCHOR = 2.76
EXPINF1Y_ANCHOR = 3.439483376669958
BE1Y_WEDGE = BE1Y_ANCHOR - EXPINF1Y_ANCHOR


def derive_breakeven1y(root: str, prior: dict, asof: dt.date, log=print) -> tuple[float, list]:
    """DERIVED FROM FRED. breakeven1y = EXPINF1YR + BE1Y_WEDGE. Returns (value, amber).

    WHY NOT THE OBVIOUS SOURCE. `breakeven1y` looks like an inflation breakeven and is not one.
    It is a plug: the state's rule is `real_1y = nominal_1y - breakeven1y`, and the value exists
    to make that identity reproduce the monthly master's 1-year real yield. An earlier draft of
    this function read outputs/curve_latest.csv's 1-year row -- 2.0783, 68 basis points away, on
    a row that file itself flags `reliability 0.0, provenance front-constructed`. That would have
    moved real_1y from 1.10 to 1.78 and the published cost of equity by 4.6bp with every test
    green. `test_breakeven1y_is_never_read_off_the_pipeline_curve` stops it recurring.

    WHY NOT CARRY IT. The first working version carried it forward with an alarm, because its
    apparent source -- history/real_yield_curve_v3_MASTER.csv -- is written by NOTHING in this
    repository and ends 2026-06-01. James, 2026-08-19: "All of this data is carried by FRED. It
    should all be updating every single day." He is right, and the fetcher already existed:
    `asfp.datasources.fetch_expinf()` pulls Cleveland Fed EXPINF1YR..EXPINF30YR from FRED. It
    was simply never pointed at this input.

    WHAT IS HELD AND WHAT MOVES. The decomposition is
        breakeven1y  =  expected inflation (MOVES, from FRED)  +  wedge (HELD)
    rather than holding the whole number. That is a real improvement and not a complete one:
    the wedge is still anchored to a single date, June 2026. It is a slow structural quantity --
    an inflation risk premium plus a TIPS liquidity effect -- so holding it is defensible in a
    way that holding a level that tracks realized inflation was not. It is NOT a measurement,
    and it belongs on the open register until the 1-year point has a fitted construction of its
    own. Recorded here rather than buried, because a held wedge that nobody remembers is holding
    is exactly how this project's nine silent-input failures happened.

    BY CONSTRUCTION THIS REPRODUCES THE JUNE ANCHOR EXACTLY. Feeding June's own EXPINF1YR back
    in returns 2.76 to the last decimal, so adopting it moved no published number on the day it
    landed; from here it tracks."""
    amber = []
    try:
        from asfp import datasources as DS
        key = os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY not set")
        expinf, expinf_asof = DS.fetch_expinf(key)
        e1 = float(expinf[0])
        if math.isnan(e1):
            raise RuntimeError("EXPINF1YR came back NaN")
        val = e1 + BE1Y_WEDGE
        log(f"  breakeven1y: {val:.4f} = EXPINF1YR {e1:.4f} + wedge {BE1Y_WEDGE:+.4f} "
            f"(Cleveland Fed via FRED, as of {expinf_asof})")
        # Cleveland publishes monthly. Older than a quarter means the series itself has stalled.
        if expinf_asof:
            ay, am = (int(x) for x in str(expinf_asof)[:7].split("-"))
            age = (asof.year - ay) * 12 + (asof.month - am)
            if age > 2:
                amber.append(f"EXPINF1YR is {age} months stale (as of {expinf_asof}); "
                             f"breakeven1y derived from it may be lagging.")
        return val, amber
    except Exception as e:
        # FALL BACK TO THE CARRY, LOUDLY. A missing FRED read must not stop the re-anchor -- the
        # other seven inputs are fine and refusing would keep vs(T) out of production over one
        # series. But it must never be quiet.
        prev = prior.get("breakeven1y")
        if prev is None:
            raise ReanchorRefused(f"cannot derive breakeven1y ({e}) and nothing to carry")
        log(f"  breakeven1y: {float(prev):.4f} CARRIED -- FRED derivation failed ({e})")
        amber.append(f"breakeven1y CARRIED at {float(prev):.4f}: the FRED derivation failed "
                     f"({e}). Worth ~4.6bp per 68bp of drift in the 1-year real rate.")
        return float(prev), amber


def _retired_carry_breakeven1y(root: str, prior: dict, asof: dt.date, log=print):
    """RETIRED 2026-08-19, kept only as the record of what the carry looked like.

    THIS FUNCTION USED TO READ outputs/curve_latest.csv AND THAT WAS WRONG. The mistake is
    recorded here rather than deleted, because it is the tenth instance of this project's
    standing failure and it was nearly shipped inside the very module written to prevent it.

    `breakeven1y` LOOKS like an inflation breakeven and is not one. It is a PLUG. The state's
    own rule is `real_1y = nominal_1y - breakeven1y`, and the value exists to make that identity
    reproduce the monthly master's 1-year real yield. Proof, exact to four decimals:

        history/real_yield_curve_v3_MASTER.csv, 2026-06-01:  real1_tips = 1.069601
        June's nominal_1y (FRED DGS1)                     :  3.83
        3.83 - 1.069601 = 2.760399    and the state carries  breakeven1y = 2.76

    outputs/curve_latest.csv's 1-year row reads 2.0783 — a DIFFERENT construction, 68 basis
    points away, and its own row is flagged `reliability 0.0, provenance front-constructed`.
    Substituting it would have moved real_1y from 1.10 to 1.78 and the published eff_coe by
    4.6bp, on a number labelled unreliable by the file that produced it, with every test green.

    SO WHY IS IT CARRIED RATHER THAN DERIVED? Because its real source cannot be refreshed.
    `history/real_yield_curve_v3_MASTER.csv` is written by NOTHING in this repository — no
    script, no workflow references it — and it stops at 2026-06-01. Until something updates the
    master, or the 1-year real yield is given a construction of its own, there is no honest
    automatic derivation. Carrying it is the truthful option; the alarm below is what stops the
    carry from becoming invisible the way it has been for two months.

    This is NOT free the way carrying normalized_X4 is free. 68bp of breakeven is worth 4.6bp of
    cost of equity, so a genuinely moving 1-year real rate does reach the published number."""
    val = prior.get("breakeven1y")
    if val is None or math.isnan(float(val)):
        raise ReanchorRefused("no breakeven1y in the prior state to carry forward")
    val = float(val)
    amber = []
    master = os.path.join(root, "history", "real_yield_curve_v3_MASTER.csv")
    if os.path.exists(master):
        last = str(pd.read_csv(master, usecols=["date"])["date"].iloc[-1])[:7]
        ly, lm = (int(x) for x in last.split("-"))
        age = (asof.year - ly) * 12 + (asof.month - lm)
        log(f"  breakeven1y: {val:.4f} CARRIED (master ends {last}, {age} months back)")
        if age > 1:
            amber.append(
                f"breakeven1y carried at {val:.4f}: real_yield_curve_v3_MASTER.csv ends {last}, "
                f"{age} months stale, and nothing in the repository writes it. Worth ~4.6bp per "
                f"68bp of drift in the 1-year real rate.")
    else:
        log(f"  breakeven1y: {val:.4f} CARRIED (master file absent)")
        amber.append(f"breakeven1y carried at {val:.4f}: the monthly master is missing entirely.")
    return val, amber


def derive_vs(asof: dt.date, prior_state: dict, log=print) -> tuple[list, float, dict]:
    """The vs(T) term structure via the six-tier VIX1Y chain. Tier 5 (stale) and tier 6
    (refusing) are both RED: a held volatility input is exactly what this module exists to stop,
    so accepting a stale one here would reintroduce the defect at the point of its repair."""
    prior_vs = prior_state.get("vs")
    lkg = float(prior_vs[0]) if isinstance(prior_vs, list) else (
        float(prior_vs) if prior_vs is not None else None)
    res = RR.refresh_vol_scale(asof.isoformat(), last_known_good=lkg,
                               last_known_good_date=prior_state.get("anchor_vintage"), log=log)
    if res.get("vs") is None:
        raise ReanchorRefused(
            f"VIX1Y source chain refused to publish (tier {res.get('tier_used')}): "
            f"{res.get('message')}")
    if res.get("tier_used") == 5:
        raise ReanchorRefused(
            f"VIX1Y chain fell through to tier 5 (holding a {res.get('stale_days')}-day-stale "
            f"value). Re-anchoring onto a held volatility number is the defect this module "
            f"exists to prevent; refusing.")
    return [float(x) for x in res["vs"]], float(res["vs_1y"]), res


def last_business_day(year: int, month: int) -> dt.date:
    d = dt.date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def months_to_replay(prior_vintage: str, target: tuple[int, int]) -> list:
    """Every month-end between the prior vintage and the target, inclusive of the prior and
    exclusive of the target.

    ONE STEP PER MONTH, NOT ONE STEP PER RE-ANCHOR. fey_out = 0.7*fey_in + 0.3*eff_coe and
    D_out = 0.6*D_in + 0.4*dur(eff_coe) are MONTHLY exponential updates. When the re-anchor has
    been skipped -- as it was for July and August 2026 -- taking a single step would silently
    compress two months of state evolution into one and put the monthly series out of step with
    the 1877-> history it extends. That is the same family of defect as the inert input this
    module exists to fix, so it is handled rather than assumed away."""
    py, pm = (int(x) for x in str(prior_vintage).split("-"))
    out = []
    y, m = py, pm
    while (y, m) < target:
        out.append(last_business_day(y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def carry_state(prior_state: dict, month_ends: list, log=print) -> tuple[float, float]:
    """Walk the state machine forward one month-end at a time and read fey_out / D_out back as
    the incoming state for the new vintage.

    Month end, not "latest": build_erp_daily's own docstring says the effective number is the
    live current-month observation of the monthly series and FINALIZES at month-end. Carrying
    forward from an arbitrary mid-month day would put the monthly series on a different footing
    from the history it extends.

    The SLOW inputs (vs, cost, corp_prem) are held at the prior state's values across every
    intermediate month. That is not an approximation -- it is what the engine actually did.
    With no re-anchor written for July or August, the daily job really did publish June's vs,
    cost and corp_prem every day through both months, so replaying them reproduces the realized
    path rather than a counterfactual one."""
    fey, dur_ = float(prior_state["fey_in"]), float(prior_state["D_in"])
    for me in month_ends:
        reals, nom, sp = RR.fetch_daily_inputs(me.isoformat())
        real, norm_ey = RR.construct_legs(prior_state, reals, nom, sp)
        r = BED.build_asof(real, norm_ey, prior_state["vs"], fey, dur_,
                           prior_state["cost"], prior_state["corp_prem"])
        log(f"  replay {me}: eff_coe={r['eff_coe']:.4f}  fey {fey:.4f}->{r['fey_out']:.4f}  "
            f"D {dur_:.4f}->{r['D_out']:.4f}")
        fey, dur_ = float(r["fey_out"]), float(r["D_out"])
    return fey, dur_


# ============================================================================================
# (3) THE GUARD
# ============================================================================================

def apply_guards(new: dict, prior: dict, log=print) -> list:
    """RED raises. AMBER is returned as a list of warnings for the caller to exit non-zero on."""
    amber = []
    prior_vs = prior.get("vs")
    prior_cmp = {
        "vs_1y": (float(prior_vs[0]) if isinstance(prior_vs, list)
                  else float(prior_vs) if prior_vs is not None else None),
        "corp_prem": prior.get("corp_prem"),
        "breakeven1y": prior.get("breakeven1y"),
        "cost": prior.get("cost"),
        "fey_in": prior.get("fey_in"),
        "D_in": prior.get("D_in"),
    }
    for key, (lo, hi, max_move) in BANDS.items():
        val = new[key]
        if val is None or math.isnan(val):
            raise ReanchorRefused(f"{key} is NaN/None")
        if not (lo <= val <= hi):
            raise ReanchorRefused(
                f"{key}={val:.4f} is outside its absolute band [{lo}, {hi}]. Refusing to write; "
                f"the prior vintage stands and the daily job keeps publishing off it.")
        was = prior_cmp.get(key)
        if was is not None:
            move = abs(val - float(was))
            if move > max_move:
                amber.append(f"{key} moved {move:.4f} (limit {max_move}): {float(was):.4f} -> {val:.4f}")
            log(f"  guard {key:12s} {float(was):9.4f} -> {val:9.4f}  (move {move:.4f}, limit {max_move})")
        else:
            log(f"  guard {key:12s} {'  (new)':>9s} -> {val:9.4f}")
    return amber


# ============================================================================================
# (4) THE RE-ANCHOR
# ============================================================================================

def reanchor(root: str = ".", asof: str | None = None, dry_run: bool = False,
             force: bool = False, log=print) -> dict:
    asof_d = dt.date.fromisoformat(asof) if asof else dt.datetime.now(dt.timezone.utc).date()
    target = (asof_d.year, asof_d.month)
    target_str = "%04d-%02d" % target

    log(f"re-anchor: target vintage {target_str}, as of {asof_d}")

    existing = dict((("%04d-%02d" % v), p) for v, p in HS.list_held_states(root))
    if target_str in existing and not force:
        log(f"  {target_str} already exists -- nothing to do (idempotent)")
        return dict(status="noop", vintage=target_str, amber=[])
    if target_str in existing:
        # --force REWRITES a vintage. Needed when a derivation is corrected mid-month, as on
        # 2026-08-19 when breakeven1y moved from a carry to a FRED read. The PRIOR vintage for
        # the state-machine walk is the newest one BEFORE the target, not the target itself.
        log(f"  {target_str} exists and --force was given: rewriting it")
        os.remove(os.path.join(root, "ERP_HELD_STATE_%s.json" % target_str))

    prior_path, prior = HS.resolve_held_state(root, asof=asof_d.isoformat(), log=log)
    log(f"  prior vintage: {prior['anchor_vintage']} ({os.path.basename(prior_path)})")

    replays = months_to_replay(prior["anchor_vintage"], target)
    if not replays:
        raise ReanchorRefused(
            f"prior vintage {prior['anchor_vintage']} is not before target {target_str}; "
            f"nothing to step forward.")
    log(f"  stepping the state machine forward {len(replays)} month(s): "
        f"{', '.join(str(d) for d in replays)}")

    vs_curve, vs_1y, vs_diag = derive_vs(asof_d, prior, log=log)
    fey_in, d_in = carry_state(prior, replays, log=log)
    corp_prem = derive_corp_prem(root, asof_d, log=log)
    breakeven1y, be_amber = derive_breakeven1y(root, prior, asof_d, log=log)
    cost = cost_for(asof_d)
    log(f"  cost: {cost:.4f} (raw {BED.cost_of_year(asof_d.year + (asof_d.timetuple().tm_yday-1)/365.0):.4f}, "
        f"floored at {COST_FLOOR})")

    guard_view = dict(vs_1y=vs_1y, corp_prem=corp_prem, breakeven1y=breakeven1y,
                      cost=cost, fey_in=fey_in, D_in=d_in)
    amber = apply_guards(guard_view, prior, log=log) + be_amber

    state = {
        "anchor_vintage": target_str,
        "note": prior.get("note"),
        "vs": vs_curve,
        "fey_in": round(fey_in, 6),
        "D_in": round(d_in, 6),
        "cost": round(cost, 6),
        "corp_prem": round(corp_prem, 6),
        "breakeven1y": round(breakeven1y, 6),
        # CARRIED FORWARD, DELIBERATELY AND IN WRITING. The S&P 500 EPS normalization job does
        # not exist yet. Measured influence on the published eff_coe: moving normalized_X4 from
        # the June 234.1507 to the July 236.8641 changes eff_coe by 0.54 BASIS POINTS, and a
        # 10% error in normalized earnings is worth 4.6bp. The plateau presets landed 2026-08-12
        # blend the normalized-earnings term to zero weight at 30 years, and the effective
        # number is duration-collapsed at ~25 years, so almost all the weight sits where this
        # input has already been blended out. It is carried, not fetched, and that is a stated
        # choice rather than an oversight. Revisit if the presets ever change.
        "normalized_X4": prior["normalized_X4"],
        "cpi_factor": prior["cpi_factor"],
        "sources": prior.get("sources"),
        "derivation": {
            "written_by": "reanchor.py",
            "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "prior_vintage": prior["anchor_vintage"],
            "month_ends_replayed": [d.isoformat() for d in replays],
            "vs": {"vix1y": vs_diag.get("value"), "tier_used": vs_diag.get("tier_used"),
                   "source": vs_diag.get("source"), "alarm": vs_diag.get("alarm"),
                   "median": getattr(__import__("vol_scale_v3"), "VIX1Y_MEDIAN", None),
                   "vs_1y": round(vs_1y, 6)},
            "corp_prem": "asfp.volsurface.floor_from_credit_grid(outputs/market_credit_latest.csv, wedge=0.50)",
            "breakeven1y": (f"EXPINF1YR (Cleveland Fed via FRED) + held wedge {BE1Y_WEDGE:+.6f}, "
                            f"the wedge anchored on {BE1Y_ANCHOR} / {EXPINF1Y_ANCHOR:.6f} at the "
                            f"2026-06 vintage. Expected inflation moves; the inflation-risk-plus-"
                            f"liquidity wedge is held and is NOT a measurement -- open register."),
            "cost": (f"build_erp_daily.cost_of_year(), glide continues at its own rate, floored "
                     f"at {COST_FLOOR} (reached mid-2032). James's ruling 2026-08-19."),
            "fey_in_D_in": "prior state replayed through build_asof at the prior month's last business day",
            "normalized_X4_cpi_factor": "CARRIED FORWARD from the prior vintage; normalization job not built. Worth <1bp -- see the inline note.",
            "prior_values": {k: prior.get(k) for k in
                             ("fey_in", "D_in", "cost", "corp_prem", "breakeven1y")},
            "amber": amber,
        },
    }

    out = os.path.join(root, "ERP_HELD_STATE_%s.json" % target_str)
    if dry_run:
        log(f"  DRY RUN -- would write {os.path.basename(out)}")
    else:
        with open(out, "w") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        log(f"  wrote {os.path.basename(out)}")
        # Prove the thing we just wrote is the thing the daily job will read. The whole class of
        # defect this module addresses is a correct file that nothing consumes.
        rp, rs = HS.resolve_held_state(root, asof=asof_d.isoformat(), log=lambda *_: None)
        if os.path.abspath(rp) != os.path.abspath(out):
            raise ReanchorRefused(
                f"wrote {out} but resolve_held_state() still returns {rp} -- the new vintage "
                f"would have been inert.")
        if not isinstance(rs["vs"], list) or len(rs["vs"]) != 30:
            raise ReanchorRefused("the written state's vs is not a 30-long vs(T) vector")
        log(f"  resolver confirms the daily job will read {os.path.basename(rp)} "
            f"with a {len(rs['vs'])}-element vs(T)")

    return dict(status="written", vintage=target_str, amber=amber, state=state)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=".")
    ap.add_argument("--asof", default=None, help="ISO date; defaults to today (UTC)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rewrite the target vintage if it already exists")
    a = ap.parse_args(argv)
    try:
        res = reanchor(root=a.root, asof=a.asof, dry_run=a.dry_run, force=a.force)
    except ReanchorRefused as e:
        print(f"\nRED -- REFUSED, nothing written: {e}", file=sys.stderr)
        print("The prior vintage stands and the daily job continues publishing off it.",
              file=sys.stderr)
        return 2
    if res["amber"]:
        print("\nAMBER -- the state WAS written and WILL publish. Look at these moves:",
              file=sys.stderr)
        for m in res["amber"]:
            print(f"  * {m}", file=sys.stderr)
        print("Failing the run so this is not silent. No action is required if the moves are "
              "real.", file=sys.stderr)
        return 1
    print(f"\nGREEN -- {res['status']} {res['vintage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
