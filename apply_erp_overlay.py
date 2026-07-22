#!/usr/bin/env python3
"""apply_erp_overlay.py — re-apply ERP's forward-curve basis on top of freshly
generated pipeline outputs (James's Decision B, 2026-07-22).

WHY THIS EXISTS
---------------
The asfp pipeline builds its own real curve and its own variance-based ERP. James
decided (ERP 0700, "Decision B") that the VALUATION basis is ERP's term structure
instead: the engine's risk-free and market ERP both come from ERP's forward curve.
That decision was first applied by hand-editing the generated outputs — which the
weekday `asfp.run` promptly overwrote (asfp-bot b701da3 reverted the risk-free leg
and left the engine discounting at a HYBRID: pipeline rf + ERP erp).

So the overlay runs as a PIPELINE STEP, after generation and before the commit.
Regeneration can no longer silently revert the decision.

WHAT IT TOUCHES (only files with a verified consumer)
-----------------------------------------------------
  curve_latest_annual.csv          real <- spot_real_yield, real_fwd1y <- fwd_real_yield
      The aeg engine writes `real` into Market Data row 23 (SPOT) and DERIVES its
      forward in row 24 via f_t=(1+z_t)^t/(1+z_{t-1})^{t-1}-1 — the same bootstrap ERP
      uses — so feeding ERP's spot reproduces ERP's forward EXACTLY. nominal columns
      are recomputed from the unchanged breakeven so the file stays self-consistent.
  coe_v2_<T>_latest_annual.csv     real_rf <- fwd_real_yield, market_erp <- fwd_erp
      (cockpit's per-tenor COE table + the engine's ERP row). idiosyncratic is the
      firm's own Martin-Wagner term and is NEVER touched.
  coe_v2_<T>_effective(.csv|_annual.csv)   real_rf <- eff_tips_ry, market_erp <- eff_erp
      The duration-collapsed single COE (cockpit effective headline, Normalization g,
      Valuation&AEG cost-of-equity block). These two constants are ERP's EFFECTIVE
      series, not derivable from the forward curve, so they live in the config.

DELIBERATELY NOT TOUCHED: coe_v2_<T>_latest.csv (percent term structure — no verified
consumer) and curve_latest.csv (raw construction artifact carrying phi/reliability/
provenance that must not be synthesised for an externally supplied curve).

CONVENTIONS (verified against the published files, do not "simplify")
---------------------------------------------------------------------
  *_annual / value_decimal : ADDITIVE          real_rf + market_erp + idiosyncratic == real_coe
  percent value_pct        : SEQUENTIAL-LOG (marginal compounding), which is additive in
                             CC space:  rf=ln(1+rf_a)*100
                                        erp=ln((1+rf_a+erp_a)/(1+rf_a))*100
                                        idio=ln((1+rf_a+erp_a+idio_a)/(1+rf_a+erp_a))*100
Idempotent: applying twice is a no-op. Fail-LOUD: any identity break raises and fails
the job rather than committing a silently wrong rate.
"""
import csv, glob, json, math, os, sys

CONFIG = os.path.join("history", "ERP_OVERLAY.json")
TOL_DEC, TOL_PCT = 1e-9, 5e-4


class OverlayError(Exception):
    """Raised on a malformed curve or a broken identity. Never ship a bad rate."""


def _rows(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def _write(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_curve(path):
    """tenor -> {col: float}. Requires a contiguous 1..30 grid and the identity
    fwd_coe == fwd_real_yield + fwd_erp (ERP publishes it as a check column)."""
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
        if abs(c["fwd_real_yield"] + c["fwd_erp"] - c["fwd_coe"]) > 1e-3:
            raise OverlayError(
                f"{path} tenor {t}: fwd_coe != fwd_real_yield + fwd_erp "
                f"({c['fwd_coe']} vs {c['fwd_real_yield'] + c['fwd_erp']})")
    return curve


def overlay_curve(path, curve):
    fn, rows = _rows(path)
    for row in rows:
        t = int(float(row["tenor"]))
        if t not in curve:
            continue
        row["real"] = f'{curve[t]["spot_real_yield"] / 100:.9f}'
        row["real_fwd1y"] = f'{curve[t]["fwd_real_yield"] / 100:.9f}'
        if "nominal" in row and "breakeven" in row:
            row["nominal"] = f'{(1 + float(row["real"])) * (1 + float(row["breakeven"])) - 1:.9f}'
        if "nominal_fwd1y" in row and "breakeven_fwd1y" in row:
            row["nominal_fwd1y"] = (
                f'{(1 + float(row["real_fwd1y"])) * (1 + float(row["breakeven_fwd1y"])) - 1:.9f}')
    _write(path, fn, rows)
    return f"curve: real/real_fwd1y <- ERP forward curve ({len(rows)} rows)"


def overlay_coe_termstructure(path, curve):
    """rf + market_erp from the forward curve; idiosyncratic preserved."""
    fn, rows = _rows(path)
    for row in rows:
        t = int(float(row["tenor"]))
        if t not in curve:
            continue
        rf = curve[t]["fwd_real_yield"] / 100
        erp = curve[t]["fwd_erp"] / 100
        idio = float(row["idiosyncratic"])          # NEVER overwritten
        row["real_rf"] = f"{rf:.9f}"
        row["market_erp"] = f"{erp:.9f}"
        row["company_erp"] = f"{erp + idio:.9f}"
        row["real_coe"] = f"{rf + erp + idio:.9f}"
        if abs((rf + erp + idio) - float(row["real_coe"])) > TOL_DEC:
            raise OverlayError(f"{path} tenor {t}: decomposition broke")
    _write(path, fn, rows)
    return f"{os.path.basename(path)}: rf+market_erp <- forward curve, idio kept"


def overlay_effective(ticker, rf_pct, erp_pct):
    """ERP's duration-collapsed effective COE. Percent file is the cockpit's source
    (_COEEFF); the decimal file is derived from it so the two round-trip."""
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
    d["real_rf"] = f"{rf_pct}"
    d["market_erp"] = f"{erp_pct}"
    d["company_erp"] = f"{erp_pct + idio_pct:.4f}"
    d["real_coe"] = f"{rf_pct + erp_pct + idio_pct:.4f}"
    for r in rows:
        r["value_pct"] = d[r["field"]]
    _write(ppath, fn, rows)
    if abs(float(d["real_rf"]) + float(d["market_erp"]) + idio_pct - float(d["real_coe"])) > TOL_PCT:
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


def main():
    if not os.path.exists(CONFIG):
        print(f"[erp-overlay] no {CONFIG}; nothing to apply (pipeline output left as generated)")
        return 0
    cfg = json.load(open(CONFIG))
    curve_path = cfg["forward_curve"]
    if not os.path.exists(curve_path):
        raise OverlayError(f"config points at a missing forward curve: {curve_path}")
    curve = load_curve(curve_path)

    applied = []
    cpath = os.path.join("outputs", "curve_latest_annual.csv")
    if os.path.exists(cpath):
        applied.append(overlay_curve(cpath, curve))
    for p in sorted(glob.glob(os.path.join("outputs", "coe_v2_*_latest_annual.csv"))):
        applied.append(overlay_coe_termstructure(p, curve))

    eff = cfg.get("effective") or {}
    if "real_rf_pct" in eff and "market_erp_pct" in eff:
        tickers = sorted({os.path.basename(p).split("_")[2]
                          for p in glob.glob(os.path.join("outputs", "coe_v2_*_effective.csv"))})
        for tk in tickers:
            applied += overlay_effective(tk, float(eff["real_rf_pct"]), float(eff["market_erp_pct"]))

    print("[erp-overlay] Decision B re-applied on top of generated outputs:")
    for line in applied:
        print("  -", line)
    if not applied:
        print("  (no target files present)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OverlayError as e:
        print(f"[erp-overlay] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
