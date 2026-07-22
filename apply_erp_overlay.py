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
  coe_v2_<T>_effective(.csv|_annual.csv)  real_rf <- eff_tips_ry, market_erp <- eff_erp
  idiosyncratic is the firm's own Martin-Wagner term and is NEVER touched.

DELIBERATELY NOT TOUCHED: coe_v2_<T>_latest.csv (percent term structure, no verified
consumer) and curve_latest.csv (raw construction artifact carrying phi/reliability/
provenance that must not be synthesised for an externally supplied curve).

CONVENTIONS (verified against the published files — do not "simplify")
----------------------------------------------------------------------
  *_annual / value_decimal : ADDITIVE      real_rf + market_erp + idiosyncratic == real_coe
  percent value_pct        : SEQUENTIAL-LOG (marginal compounding), additive in CC space.
  The percent file is authoritative for _effective and stays percent-pinned (ERP 1826);
  the decimal variant is derived from it, so it may wobble in the 5th decimal. Intended.

FAILURE POLICY
--------------
  HARD FAIL (exit 1)  : decomposition/identity breakage — fwd_coe != rf+erp,
                        eff_coe != eff_tips_ry+eff_erp, malformed grid, missing columns.
                        Never commit a silently wrong rate.
  LOUD WARNING only   : vintage older than max_vintage_age_days. A late vintage must not
                        break the weekday pipeline (monthly cadence with slack).
"""
import csv, datetime as dt, glob, json, math, os, sys

CONFIG = os.path.join("history", "ERP_OVERLAY.json")
PROVENANCE = os.path.join("outputs", "erp_overlay_provenance.csv")
TOL_DEC, TOL_PCT, TOL_IDENT = 1e-9, 5e-4, 1e-3
DEFAULT_MAX_AGE_DAYS = 45


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
def overlay_curve(path, curve, mapping):
    fn, rows = _rows(path)
    for row in rows:
        t = int(float(row["tenor"]))
        if t not in curve:
            continue
        for target, source in mapping.items():
            if target in row:
                row[target] = f"{curve[t][source] / 100:.9f}"
        if "nominal" in row and "breakeven" in row:
            row["nominal"] = f'{(1 + float(row["real"])) * (1 + float(row["breakeven"])) - 1:.9f}'
        if "nominal_fwd1y" in row and "breakeven_fwd1y" in row:
            row["nominal_fwd1y"] = (
                f'{(1 + float(row["real_fwd1y"])) * (1 + float(row["breakeven_fwd1y"])) - 1:.9f}')
    _write(path, fn, rows)
    return f"curve_latest_annual: {', '.join(f'{k}<-{v}' for k, v in mapping.items())}"


def overlay_coe_termstructure(path, curve, mapping):
    fn, rows = _rows(path)
    for row in rows:
        t = int(float(row["tenor"]))
        if t not in curve:
            continue
        for target, source in mapping.items():
            row[target] = f"{curve[t][source] / 100:.9f}"
        rf, erp = float(row["real_rf"]), float(row["market_erp"])
        idio = float(row["idiosyncratic"])          # NEVER overwritten
        row["company_erp"] = f"{erp + idio:.9f}"
        row["real_coe"] = f"{rf + erp + idio:.9f}"
        if abs((rf + erp + idio) - float(row["real_coe"])) > TOL_DEC:
            raise OverlayError(f"{path} tenor {t}: decomposition broke")
    _write(path, fn, rows)
    return f"{os.path.basename(path)}: rf+market_erp <- forward curve, idio kept"


def overlay_effective(ticker, eff, mapping):
    """Percent file is authoritative (ERP 1826: keep percent-pinned); decimal derived."""
    out = []
    ppath = os.path.join("outputs", f"coe_v2_{ticker}_effective.csv")
    apath = os.path.join("outputs", f"coe_v2_{ticker}_effective_annual.csv")
    if not os.path.exists(ppath):
        return out

    fn, rows = _rows(ppath)
    d = {r["field"]: r["value_pct"] for r in rows}
    if "idiosyncratic" not in d:
        raise OverlayError(f"{ppath}: no idiosyncratic field")
    idio_pct = float(d["idiosyncratic"])            # NEVER overwritten
    for target, source in mapping.items():
        d[target] = f"{eff[source]}"
    rf_pct, erp_pct = float(d["real_rf"]), float(d["market_erp"])
    d["company_erp"] = f"{erp_pct + idio_pct:.4f}"
    d["real_coe"] = f"{rf_pct + erp_pct + idio_pct:.4f}"
    for r in rows:
        r["value_pct"] = d[r["field"]]
    _write(ppath, fn, rows)
    if abs(rf_pct + erp_pct + idio_pct - float(d["real_coe"])) > TOL_PCT:
        raise OverlayError(f"{ppath}: percent decomposition broke")
    out.append(f"coe_v2_{ticker}_effective: {rf_pct}+{erp_pct}+{idio_pct} = {d['real_coe']}")

    if os.path.exists(apath):
        rf = math.exp(rf_pct / 100) - 1
        erp = (1 + rf) * math.exp(erp_pct / 100) - 1 - rf
        idio = (1 + rf + erp) * math.exp(idio_pct / 100) - 1 - rf - erp
        vals = {"real_rf": rf, "market_erp": erp, "idiosyncratic": idio,
                "company_erp": erp + idio, "real_coe": rf + erp + idio}
        afn, arows = _rows(apath)
        for r in arows:
            if r["field"] in vals:
                r["value_decimal"] = f"{vals[r['field']]:.6f}"
        _write(apath, afn, arows)
        if abs(vals["real_rf"] + vals["market_erp"] + vals["idiosyncratic"] - vals["real_coe"]) > TOL_DEC:
            raise OverlayError(f"{apath}: decimal decomposition broke")
        out.append(f"coe_v2_{ticker}_effective_annual: real_coe={vals['real_coe']:.6f}")
    return out


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
    eff_map = maps.get("coe_v2_effective", {})
    if eff_map:
        for tk in sorted({os.path.basename(p).split("_")[2]
                          for p in glob.glob(os.path.join("outputs", "coe_v2_*_effective.csv"))}):
            applied += overlay_effective(tk, eff, eff_map)
    applied.append(write_provenance(eff, curve_path, age, stale))

    print(f"[erp-overlay] Decision B re-applied — ERP vintage {eff['vintage']} "
          f"({eff['date']}), eff_coe {eff['eff_coe']}%:")
    for line in applied:
        print("  -", line)

    if stale:
        msg = (f"ERP vintage {eff['vintage']} ({eff['date']}) is {age} days old, older than "
               f"max_vintage_age_days={max_age}. The overlay re-applied it, so the published "
               f"basis may be a cycle behind. ERP: publish a new vintage by overwriting "
               f"history/TODAY_forward_curve_latest.csv + history/ERP_effective_latest.csv.")
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
