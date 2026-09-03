#!/usr/bin/env python3
"""apply_erp_overlay.py — re-apply ERP's published basis on top of freshly generated
pipeline outputs (James's Decision B, 2026-07-22; AUTO-LATEST per ERP 1826).

WHY THIS EXISTS
---------------
The asfp pipeline builds its own real curve and its own variance-based ERP. James
decided the VALUATION basis is ERP's term structure instead. That was first applied by
hand-editing the generated outputs — which the weekday `asfp.run` promptly overwrote
(asfp-bot b701da3 reverted the risk-free leg and left the engine discounting at a
HYBRID: pipeline rf + ERP erp). So the overlay runs as a PIPELINE STEP, after
generation and before the commit: regeneration can no longer silently revert it.

AUTO-LATEST (no pinned constants, no per-vintage hand edit)
-----------------------------------------------------------
ERP publishes two STABLE-NAMED files each vintage and we read whatever they contain:
    history/TODAY_forward_curve_latest.csv   tenor,fwd_real_yield,fwd_erp,fwd_coe,
                                             spot_real_yield,spot_erp,spot_coe
    history/ERP_effective_latest.csv         vintage,date,eff_tips_ry,eff_erp,eff_coe,duration
ERP_OVERLAY.json holds ONLY paths + the source->target column mapping. It never carries
a number, so a new vintage needs no edit anywhere: ERP overwrites the two _latest files
and the next pipeline run picks them up.

The earlier design pinned the numbers in JSON. That was a silent-staleness hole: a stale
but internally-consistent vintage re-applies forever and trips no check. Hence auto-latest
plus the staleness guard below.

WHAT IT TOUCHES (only files with a verified consumer)
-----------------------------------------------------
  curve_latest_annual.csv        real <- spot_real_yield, real_fwd1y <- fwd_real_yield
      The aeg engine writes `real` into Market Data row 23 (SPOT) and DERIVES its forward
      in row 24 via f_t=(1+z_t)^t/(1+z_{t-1})^{t-1}-1 — the same bootstrap ERP uses — so
      feeding ERP's spot reproduces ERP's forward EXACTLY. nominal columns are recomputed
      from the unchanged breakeven so the file stays self-consistent.
  coe_v2_<T>_latest_annual.csv   real_rf <- fwd_real_yield, market_erp <- fwd_erp
      TWO COLUMNS, AND THE FILE NOW HAS NO OTHERS (2026-09-02). It used to carry
      `idiosyncratic`, `company_erp` and `real_coe` as well; all three were the retired
      single-name construction, none reached a valuation, and aeg-valuation reads only the
      two above. See asfp/total_risk_erp.py's docstring and register item A6.
  coe_v2_<T>_effective(.csv|_annual.csv)   RETIRED AND DELETED 2026-09-02. See the note where
      overlay_effective() used to be, below.

DELIBERATELY NOT TOUCHED: coe_v2_<T>_latest.csv (percent term structure, no verified
consumer) and curve_latest.csv (raw construction artifact carrying phi/reliability/
provenance that must not be synthesised for an externally supplied curve).

CONVENTIONS (verified against the published files — do not "simplify")
----------------------------------------------------------------------
  *_annual : real_rf is a RATE, market_erp is a PREMIUM, and they convert differently.
  See the units block above RATE_TARGETS — that distinction is load-bearing and was got
  wrong once already (2026-08-22).

FAILURE POLICY
--------------
  HARD FAIL (exit 1)  : decomposition/identity breakage — fwd_coe != rf+erp,
                        eff_coe != eff_tips_ry+eff_erp, malformed grid, missing columns.
                        Never commit a silently wrong rate.
  LOUD WARNING only   : vintage older than max_vintage_age_days. Now a DEAD-FEED alarm:
                        the daily-close job (erp_daily.yml) rewrites the vintage every weekday,
                        so a healthy age is 0-5 days (weekends/holidays). >5 means the daily
                        job stopped or the feed broke. Warns; never breaks the pipeline.
"""
import csv, datetime as dt, glob, json, math, os, sys

CONFIG = os.path.join("history", "ERP_OVERLAY.json")
PROVENANCE = os.path.join("outputs", "erp_overlay_provenance.csv")
TOL_DEC, TOL_PCT, TOL_IDENT = 1e-9, 5e-4, 1e-3
DEFAULT_MAX_AGE_DAYS = 5   # dead-feed alarm for the daily-close job (~3 business days incl. weekends)


class OverlayError(Exception):
    """Identity/structure breakage. Hard fail — never ship a bad rate."""


def _rows(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def _write(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ sources
def load_curve(path):
    """tenor -> {col: float}; contiguous 1..30 and fwd_coe == fwd_real_yield + fwd_erp."""
    need = ["tenor", "fwd_real_yield", "fwd_erp", "fwd_coe", "spot_real_yield"]
    _, rows = _rows(path)
    curve = {}
    for r in rows:
        missing = [c for c in need if c not in r]
        if missing:
            raise OverlayError(f"{path}: missing columns {missing}")
        curve[int(float(r["tenor"]))] = {k: float(v) for k, v in r.items()}
    for t in range(1, 31):
        if t not in curve:
            raise OverlayError(f"{path}: tenor grid must contain 1..30; missing {t}")
        c = curve[t]
        if abs(c["fwd_real_yield"] + c["fwd_erp"] - c["fwd_coe"]) > TOL_IDENT:
            raise OverlayError(
                f"{path} tenor {t}: fwd_coe != fwd_real_yield + fwd_erp "
                f"({c['fwd_coe']} vs {c['fwd_real_yield'] + c['fwd_erp']})")
    return curve


def load_effective(path):
    """Single-row vintage file: vintage,date,eff_tips_ry,eff_erp,eff_coe,duration."""
    need = ["vintage", "date", "eff_tips_ry", "eff_erp", "eff_coe"]
    _, rows = _rows(path)
    if len(rows) != 1:
        raise OverlayError(f"{path}: expected exactly 1 vintage row, got {len(rows)}")
    r = rows[0]
    missing = [c for c in need if c not in r]
    if missing:
        raise OverlayError(f"{path}: missing columns {missing}")
    eff = {"vintage": r["vintage"].strip(), "date": r["date"].strip(),
           "eff_tips_ry": float(r["eff_tips_ry"]), "eff_erp": float(r["eff_erp"]),
           "eff_coe": float(r["eff_coe"])}
    eff["duration"] = float(r["duration"]) if r.get("duration") not in (None, "") else None
    if abs(eff["eff_tips_ry"] + eff["eff_erp"] - eff["eff_coe"]) > TOL_IDENT:
        raise OverlayError(
            f"{path}: eff_coe != eff_tips_ry + eff_erp "
            f"({eff['eff_coe']} vs {eff['eff_tips_ry'] + eff['eff_erp']})")
    return eff


def vintage_age_days(eff, today=None):
    try:
        d = dt.date.fromisoformat(eff["date"])
    except ValueError:
        return None
    return ((today or dt.date.today()) - d).days


# ------------------------------------------------------------------ targets
# ---------------------------------------------------------------------------------------------
# ⛔ UNITS. THE `_annual` FILES ARE ANNUAL-COMPOUNDED. THE OVERLAY WAS WRITING cc INTO THEM.
#
# Found 2026-08-22 by James, who noticed the screen quoting a 30-year real yield of 3.03% when
# FRED's DFII30 read 2.95%. It was neither: `outputs/curve_latest_annual.csv` carried the RAW
# continuously-compounded value from history/TODAY_forward_curve_latest.csv, divided by 100 and
# written into a file whose own contract (aeg-valuation rate_feed.py) is
#     *_latest.csv  -> continuously-compounded PERCENT
#     *_annual.csv  -> ANNUAL-COMPOUNDED DECIMAL,  annual = exp(cc/100) - 1
#
# `asfp/run.py` builds that file CORRECTLY, applying units.annualize_rate to every rate column.
# This overlay then overwrote those values with cc/100 and undid it. Verified across the curve:
# every tenor of the published outputs file matched the raw cc of the history file exactly.
#
# The error is small and one-directional -- it UNDERSTATES every real rate, so it OVERSTATES
# every valuation the engine produces: 1.8bp at tenor 1, 3.2bp at 10, 4.6bp at 30, growing with
# the rate. It survived because both numbers are plausible reals a few basis points apart, which
# is precisely the profile of a number that is silently wrong while every gate stays green.
#
# THE CONVERSION IS PER COLUMN TYPE, not blanket. real-yields' own units.py draws the line:
#   RATE / YIELD  (real, real_fwd1y, real_rf)   ->  annual = exp(cc/100) - 1   [annualize_rate]
#   PREMIUM       (market_erp)                  ->  plain cc/100               [to_decimal]
# A premium is an additive spread on top of a base rate, and the engine's decomposition requires
# the pieces to SUM to the annual total, so premia must NOT be compounded. Applying expm1 to
# everything would fix one error by introducing a smaller one.
RATE_TARGETS = frozenset({"real", "real_fwd1y", "real_rf"})


def _to_annual(target, cc_percent):
    """cc percent -> the units the `_annual` contract requires for THIS column."""
    if target in RATE_TARGETS:
        return math.expm1(cc_percent / 100.0)
    return cc_percent / 100.0          # a premium stays additive; see the note above
# ---------------------------------------------------------------------------------------------


def overlay_curve(path, curve, mapping):
    fn, rows = _rows(path)
    for row in rows:
        t = int(float(row["tenor"]))
        if t not in curve:
            continue
        for target, source in mapping.items():
            if target in row:
                row[target] = f"{_to_annual(target, curve[t][source]):.9f}"
        if "nominal" in row and "breakeven" in row:
            row["nominal"] = f'{(1 + float(row["real"])) * (1 + float(row["breakeven"])) - 1:.9f}'
        if "nominal_fwd1y" in row and "breakeven_fwd1y" in row:
            row["nominal_fwd1y"] = (
                f'{(1 + float(row["real_fwd1y"])) * (1 + float(row["breakeven_fwd1y"])) - 1:.9f}')
    _write(path, fn, rows)
    return f"curve_latest_annual: {', '.join(f'{k}<-{v}' for k, v in mapping.items())}"


def overlay_coe_termstructure(path, curve, mapping):
    """Rewrite real_rf and market_erp from the ERP engine's forward curve.

    TWO COLUMNS, AND SINCE 2026-09-02 THE FILE HAS NO OTHERS. It used to also carry
    `idiosyncratic`, `company_erp` and `real_coe`, and this function recomputed the last two and
    asserted `rf + erp + idio == real_coe`. That assertion is deleted because it has nothing left
    to assert -- and because it was never the check it looked like. It is arithmetic: it holds by
    construction the moment you write both sides, whatever the pieces mean. The pieces did not
    mean the same thing (see the retirement note in asfp/total_risk_erp.py), and the assertion
    passed all the way through.
    """
    fn, rows = _rows(path)
    for row in rows:
        t = int(float(row["tenor"]))
        if t not in curve:
            continue
        for target, source in mapping.items():
            if target in row:
                row[target] = f"{_to_annual(target, curve[t][source]):.9f}"
    _write(path, fn, rows)
    return f"{os.path.basename(path)}: real_rf + market_erp <- ERP forward curve"


# RETIRED 2026-09-02 -- overlay_effective() and METHODOLOGY_NOTE.
#
# This function rewrote `coe_v2_<T>_effective.csv` and `_effective_annual.csv`, and it is the
# single clearest artifact of what register item A6 was about. Its own METHODOLOGY_NOTE said, in
# the file it published: real_rf and market_erp come from the "Decision-B state-machine
# (duration-weighted spot average)"; `idiosyncratic` comes from a "cash-flow-PV YTM collapse of
# the same-day forward curve (different methodology, different curve representation)"; and
# real_coe/company_erp "add the two together". Then it asserted the sum, and the assertion passed,
# because adding two incommensurable numbers is still addition.
#
# For AMCR the file published a real cost of equity of 11.873%. The valuation discounted at
# 6.2169%. Nothing read the file -- not the engine, not the screener, not the Dashboard -- but it
# sat in a public repository looking like the company's cost of equity.
#
# Both files are deleted and asfp/run_company.py no longer writes them. James ruled 2026-09-02
# that there is ONE approved method for a company's idiosyncratic premium, the four-block risk
# score in aeg-valuation/idio/, and that the retired ones must not be referred to anywhere.


def write_provenance(eff, curve_path, age, stale):
    """Stamp the CONSUMED vintage so COCKPIT/James can SEE which curve is live."""
    os.makedirs("outputs", exist_ok=True)
    with open(PROVENANCE, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "value"])
        for k, v in (
            ("erp_vintage", eff["vintage"]), ("erp_vintage_date", eff["date"]),
            ("eff_tips_ry_pct", eff["eff_tips_ry"]), ("eff_erp_pct", eff["eff_erp"]),
            ("eff_coe_pct", eff["eff_coe"]), ("equity_duration_yrs", eff["duration"]),
            ("forward_curve_file", os.path.basename(curve_path)),
            ("vintage_age_days", age), ("vintage_stale", str(bool(stale)).lower()),
            ("overlay_applied_utc", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ):
            w.writerow([k, v])
    return f"provenance: vintage {eff['vintage']} (age {age}d) -> {PROVENANCE}"


def main():
    if not os.path.exists(CONFIG):
        print(f"[erp-overlay] no {CONFIG}; nothing to apply (outputs left as generated)")
        return 0
    cfg = json.load(open(CONFIG))
    maps = cfg.get("mapping") or {}
    curve_path, eff_path = cfg["forward_curve"], cfg["effective"]
    for p in (curve_path, eff_path):
        if not os.path.exists(p):
            raise OverlayError(f"config points at a missing file: {p}")

    curve = load_curve(curve_path)
    eff = load_effective(eff_path)

    max_age = int(cfg.get("max_vintage_age_days", DEFAULT_MAX_AGE_DAYS))
    age = vintage_age_days(eff)
    stale = age is not None and age > max_age

    applied = []
    cpath = os.path.join("outputs", "curve_latest_annual.csv")
    if os.path.exists(cpath):
        applied.append(overlay_curve(cpath, curve, maps.get("curve_latest_annual", {})))
    for p in sorted(glob.glob(os.path.join("outputs", "coe_v2_*_latest_annual.csv"))):
        applied.append(overlay_coe_termstructure(p, curve, maps.get("coe_v2_latest_annual", {})))
    # The `coe_v2_effective` pass ran here until 2026-09-02; see the note above overlay_curve's
    # retired sibling. `eff` is still loaded and still validated -- it is what stamps the
    # provenance file below, and aeg-valuation now REFUSES a valuation when that file is absent,
    # stale, or older than the company job's own stamp (rate_feed.check_overlay_vintage, a2a4c52).
    applied.append(write_provenance(eff, curve_path, age, stale))

    print(f"[erp-overlay] Decision B re-applied — ERP vintage {eff['vintage']} "
          f"({eff['date']}), eff_coe {eff['eff_coe']}%:")
    for line in applied:
        print("  -", line)

    if stale:
        msg = (f"ERP vintage {eff['vintage']} ({eff['date']}) is {age} days old, older than "
               f"max_vintage_age_days={max_age}. The overlay re-applied it, so the published "
               f"basis may be stale. DEAD-FEED ALARM: the daily-close job (erp_daily.yml) should "
               f"rewrite history/TODAY_forward_curve_latest.csv + history/ERP_effective_latest.csv "
               f"every weekday — check that the scheduled run is succeeding (FRED feed / Actions).")
        print(f"::warning title=ERP vintage stale::{msg}")
        print(f"[erp-overlay] WARNING: {msg}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OverlayError as e:
        print(f"[erp-overlay] FAILED: {e}", file=sys.stderr)
        print(f"::error title=ERP overlay identity break::{e}")
        sys.exit(1)
