"""
run_erp_daily.py  —  daily-close RUNNER/WRITER around build_erp_daily.build_asof.
Fetch daily inputs -> construct the ERP-owned legs (1y short-end, norm_ey) -> build_asof
-> write the two _latest files the auto-latest overlay consumes.

EXEC wires fetch_daily_inputs() to the repo's data step. Everything below fetch is ERP-owned
and deterministic. __main__ is a self-contained SMOKE that reproduces June from committed anchors
and writes the two files, so the writer path is verifiable without a live feed.
"""
import json, csv, numpy as np
from build_erp_daily import build_asof

def construct_legs(state, reals_5_10_20_30, nominal_1y, sp_close):
    """ERP-owned input construction. reals_* is {5,10,20,30:pct}. Returns (real_5pt dict, norm_ey)."""
    real={k:float(reals_5_10_20_30[k]) for k in (5,10,20,30)}
    real[1]=float(nominal_1y)-float(state["breakeven1y"])           # item 1: 1y short-end
    norm_ey=100.0*float(state["normalized_X4"])*float(state["cpi_factor"])/float(sp_close)  # item 2: deflator
    return real, norm_ey

def write_outputs(asof_date, r, outdir="."):
    with open(f"{outdir}/TODAY_forward_curve_latest.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["tenor","fwd_real_yield","fwd_erp","fwd_coe","spot_real_yield","spot_erp","spot_coe"])
        for i in range(30):
            w.writerow([i+1,round(r["fwd_real"][i],4),round(r["fwd_erp"][i],4),round(r["fwd_coe"][i],4),
                        round(r["spot_real"][i],4),round(r["spot_erp"][i],4),round(r["spot_coe"][i],4)])
    with open(f"{outdir}/ERP_effective_latest.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["vintage","date","eff_tips_ry","eff_erp","eff_coe","duration"])
        w.writerow([asof_date, asof_date, round(r["eff_tips"],4), round(r["eff_erp"],4), round(r["eff_coe"],4), round(r["D_out"],2)])

def run(asof_date, reals_5_10_20_30, nominal_1y, sp_close, state, outdir="."):
    real, norm_ey = construct_legs(state, reals_5_10_20_30, nominal_1y, sp_close)
    r=build_asof(real, norm_ey, state["vs"], state["fey_in"], state["D_in"], state["cost"], state["corp_prem"])
    write_outputs(asof_date, r, outdir)
    return r

SP500_SERIES = "SP500"   # FRED S&P 500 index level (daily close). ERP: confirm on the live eyeball.

def fetch_daily_inputs(asof_date, api_key=None):
    """ADAPTER — wired to the repo's FRED data step (asfp.datasources). Returns
       (reals_5_10_20_30: {5,10,20,30 -> real par yield pct}, nominal_1y: pct, sp_close: index level),
       each the latest observation on/before asof_date.
       Sources (per ERP_HELD_STATE_*.json['sources']):
         real par 5/10/20/30 -> FRED DFII5/DFII10/DFII20/DFII30 (Treasury Daily Par REAL Yield Curve),
         nominal 1y          -> FRED DGS1 (Treasury Daily Par Yield Curve, 1Y),
         S&P 500 close        -> FRED SP500.
       Import is function-local so the hermetic gate/tests stay fully offline."""
    import os
    from asfp import datasources as ds
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set (required for the live daily fetch)")
    reals = {k: ds.fetch_fred_asof(key, ds.DFII_MAP[k], asof_date)[0] for k in (5, 10, 20, 30)}
    nominal_1y = ds.fetch_fred_asof(key, ds.DGS_MAP[1], asof_date)[0]
    sp_close = ds.fetch_fred_asof(key, SP500_SERIES, asof_date)[0]
    missing = [n for n, v in [("nominal_1y", nominal_1y), ("sp_close", sp_close)] if v is None]
    missing += [f"real_{k}y" for k, v in reals.items() if v is None]
    if missing:
        raise RuntimeError(f"FRED returned no value for {missing} as of {asof_date}")
    return reals, nominal_1y, sp_close

if __name__=="__main__":
    # SMOKE: reproduce June-2026 from committed anchors + June daily inputs, and write the two files.
    state=json.load(open("ERP_HELD_STATE_2026-06.json"))
    r=run("2026-06-01", {5:1.885,10:2.204,20:2.745,30:2.73}, nominal_1y=3.83, sp_close=7450.03, state=state, outdir=".")
    print("RUNNER SMOKE (June): eff tips=%.3f erp=%.3f coe=%.3f dur=%.2f"%(r["eff_tips"],r["eff_erp"],r["eff_coe"],r["D_out"]))
    assert abs(r["eff_tips"]-2.349)<0.01 and abs(r["eff_erp"]-3.887)<0.01 and abs(r["eff_coe"]-6.236)<0.01, "SMOKE FAILED"
    import pandas as pd
    got=pd.read_csv("TODAY_forward_curve_latest.csv"); ref=pd.read_csv("TODAY_forward_curve_2026-06.csv")
    sp=max(abs(got.spot_coe-ref.spot_coe)); print("  wrote _latest files; spot_coe max|delta| vs committed June = %.4f pp"%sp)
    print("  ERP_effective_latest.csv ->", open("ERP_effective_latest.csv").read().strip().replace(chr(10)," | "))
    assert sp<0.01, "WRITE MISMATCH"
    print("  RUNNER SMOKE PASSED")
