"""
run_coe_history.py — TICKER-ENTRY runner for the historical cost-of-equity model.
Enter a ticker; get its monthly COE history (real + nominal) + an annual rollup. No hand-calibration.

    python -m run_coe_history AAPL
    -> outputs/coe_history_AAPL.csv         (MONTHLY: real_yield, market_erp, idio_erp, coe_real, exp_infl, coe_nominal)
       outputs/coe_history_AAPL_annual.csv  (annual year-end + calendar-mean rollup)

Company-agnostic: the ONLY per-ticker input is that ticker's monthly price history. Everything else
(market real-yield + ERP history, S&P 500 index history, expected inflation) is shared/committed.
A single universal price-of-idiosyncratic-risk (coe_history.K_UNIVERSAL) prices the idiosyncratic leg,
so a never-seen ticker runs with zero manual calibration.

Shared inputs (committed real-yields paths; override via env):
  ERP_MONTHLY   real-yields monthly decomposition of record (cols: date, eff_tips_ry, eff_erp)  [FINAL_decomposition_v4]
  EPI_CSV       yields-repo expected_inflation_termstructure_v2.csv (mkey, Epi_10y)  [NOT Shiller]
  SPX_MONTHLY   optional S&P 500 monthly adj-close fixture (date,adj_close). If unset, GSPC is fetched live.
EXEC wire: fetch_monthly_adj() hits EODHD monthly adjusted close by ticker (same pattern as the daily
engine's fetch adapter + refresh_bonds' EODHD client). Everything below fetch is deterministic/offline.
"""
import os, sys, numpy as np, pandas as pd
import coe_history as C


def _p(name, default):
    return os.environ.get(name, default)


def fetch_monthly_adj(ticker, api_key=None):
    """ADAPTER - EODHD monthly adjusted close by ticker. Returns a pd.Series indexed by pandas
    Period('M'), full history. Symbol maps TICKER -> TICKER.US unless an exchange is given
    (e.g. 'GSPC.INDX'). Import is function-local so offline tests never hit the network."""
    import json, urllib.request, urllib.parse
    key = api_key or os.environ.get("EODHD_API_KEY")
    if not key:
        raise RuntimeError("EODHD_API_KEY not set (required for the live monthly-price fetch)")
    sym = ticker if "." in ticker else f"{ticker.upper()}.US"
    url = (f"https://eodhd.com/api/eod/{urllib.parse.quote(sym)}"
           f"?period=m&fmt=json&api_token={urllib.parse.quote(key)}")
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = json.load(r)
    idx, vals = [], []
    for o in rows:
        ac = o.get("adjusted_close", o.get("close"))
        if ac in (None, "", "."):
            continue
        idx.append(pd.Period(str(o["date"])[:7], freq="M"))
        vals.append(float(ac))
    if not vals:
        raise RuntimeError(f"EODHD returned no monthly prices for {sym}")
    s = pd.Series(vals, index=pd.PeriodIndex(idx, freq="M")).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(ticker.upper())


def _load_shared(need_spx=True):
    """Read the committed shared inputs at call time (env-overridable). SPX is a committed fixture
    if SPX_MONTHLY points at one, else fetched live from EODHD (GSPC.INDX)."""
    v4 = pd.read_csv(_p("ERP_MONTHLY", "history/FINAL_decomposition_v4_1877_2026.csv"))
    v4["m"] = pd.to_datetime(v4["date"]).dt.to_period("M")
    v4 = v4.set_index("m")
    epi = pd.read_csv(_p("EPI_CSV", "history/expected_inflation_termstructure_v2.csv"))
    epi["m"] = pd.PeriodIndex(epi["mkey"], freq="M")
    epi = epi.set_index("m")["Epi_10y"].astype(float)
    spx = None
    if need_spx:
        spx_path = _p("SPX_MONTHLY", "")
        if spx_path and os.path.exists(spx_path):
            spx = pd.read_csv(spx_path)
            spx["m"] = pd.to_datetime(spx["date"]).dt.to_period("M")
            spx = spx.set_index("m")["adj_close"].astype(float)
        else:
            spx = fetch_monthly_adj("GSPC.INDX")
    return v4, epi, spx


def run(ticker, px=None, spx=None, outdir=None):
    outdir = outdir or _p("OUTDIR", "outputs")
    v4, epi, spx_loaded = _load_shared(need_spx=(spx is None))
    if spx is None:
        spx = spx_loaded
    if px is None:
        px = fetch_monthly_adj(ticker)
    D = C.build_coe(ticker.upper(), px, spx, v4, epi)              # monthly real + nominal COE
    os.makedirs(outdir, exist_ok=True)
    D.round(3).to_csv(f"{outdir}/coe_history_{ticker.upper()}.csv")
    D["yr"] = D.index.year
    ann = D.groupby("yr").agg(coe_real_dec=("coe_real", "last"), coe_nom_dec=("coe_nominal", "last"),
                              coe_real_mean=("coe_real", "mean"), coe_nom_mean=("coe_nominal", "mean"),
                              idio=("idio_erp", "last")).round(2)
    ann.to_csv(f"{outdir}/coe_history_{ticker.upper()}_annual.csv")
    return D, ann


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if args and args[0] not in ("--selftest", "-t"):
        D, ann = run(args[0])
        c = D.iloc[-1]
        print(f"{args[0].upper()} latest {D.index[-1]}: COE_real {c.coe_real:.2f}  "
              f"COE_nominal {c.coe_nominal:.2f}  ({len(D)} monthly rows)")
    else:
        # HERMETIC SELF-TEST - offline; committed compact fixtures reproduce the AAPL pilot to <0.05.
        base = os.path.dirname(os.path.abspath(__file__))
        fx = os.path.join(base, "tests", "fixtures", "coe_history")
        a = np.array([float(x) for x in open(os.path.join(fx, "aapl_adj_2015.csv")).read().split(",")])
        px = pd.Series(a, index=pd.period_range("2015-01", periods=len(a), freq="M"))
        spx = pd.read_csv(os.path.join(fx, "spx_monthly_2015.csv"))
        spx["m"] = pd.to_datetime(spx["date"]).dt.to_period("M")
        spx = spx.set_index("m")["adj_close"].astype(float)
        os.environ.setdefault("ERP_MONTHLY", os.path.join(base, "history", "FINAL_decomposition_v4_1877_2026.csv"))
        os.environ.setdefault("EPI_CSV", os.path.join(base, "history", "expected_inflation_termstructure_v2.csv"))
        D, ann = run("AAPL", px=px, spx=spx, outdir=os.environ.get("OUTDIR", "outputs"))
        c = D.loc[pd.Period("2026-06", "M")]
        print("SELF-TEST AAPL 2026-06: COE_real=%.2f (exp 8.15)  COE_nominal=%.2f (exp 11.50)  "
              "idio=%.3f  rows=%d" % (c.coe_real, c.coe_nominal, c.idio_erp, len(D)))
        assert abs(c.coe_real - 8.15) < 0.05 and abs(c.coe_nominal - 11.50) < 0.05, "SELF-TEST FAILED"
        print("SELF-TEST PASSED - monthly series written to outputs/coe_history_AAPL.csv")
