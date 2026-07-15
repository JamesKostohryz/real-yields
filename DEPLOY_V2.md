# Deploying the v2 total-risk COE pipeline (live data)

This deploy makes the system **pull the real market data** for the total-risk single-name
COE we designed: the VIX term structure, the long-dated (futures/LEAPS) index options, and
each name's own option-implied vol term structure — and it emits both the **term structure**
and the **effective** (collapsed, YTM-style) ERP for a ticker.

Everything is **non-breaking**: the existing v1 outputs are untouched. The v2 files are new,
and every network pull is wrapped so a source being down degrades gracefully instead of
failing the run.

---

## 1. What gets pulled live, and how it degrades

**Market ERP (the weekly job, `asfp.run`):**

| Source | What | If it's down |
|---|---|---|
| FRED `VIXCLS` (30d), `VXVCLS` (3m) | the reliable VIX-term-structure base | needs a FRED key; this is the floor of what must work |
| CBOE `VIX9D`, `VIX6M`, `VIX1Y` | fills out the 9-day-to-1-year term structure | dropped; front just has fewer points |
| SPX/SPY LEAPS (yfinance) | **long-dated index options** — extends the *observed* ERP out to ~3y | dropped; ERP hands off to the bond blend sooner |
| CME E-mini settlement | furthest-out futures-options point (hook) | returns `[]` today — see §6 |

The observed front runs out to the **last** vol tenor we can see; past that, the
"split-the-distance" blend glides to the bond floor by year 30, and the Merton elevator owns
the tail. So the more of the long end we can see, the longer the ERP stays purely
options-driven — which is exactly the behavior you asked for.

**Single-name COE (the ticker job, `asfp.run_company`):**

- The name's **option-implied vol term structure** (1m, 3m, 6m, 1y, 1.5y, 2y ATM IV) is
  pulled from its option chain and becomes the risk ratio `R_i(t) = σ_i(t)/σ_mkt(t)` at the
  front. If the long-dated chain is thin, it falls back to the single hardened 1y point
  (realized-vol-backstopped), so the job always has a usable curve.
- The index vol it's divided against is derived from the market ERP via Martin
  (variance = ERP), sampled at the *same* tenors as the stock so the ratio lines up
  tenor-by-tenor.

---

## 2. New output files (all additive; v1 files unchanged)

Weekly job:

- `market_erp_v2_latest.csv` / `_annual.csv` — market ERP term structure, 1–150y.
- `index_vol_ts_latest.csv` — the exact vol points that fed it (`tenor, index_vol`), for
  audit and for charting the observed front.

Ticker job (`<T>` = ticker):

- `coe_v2_<T>_latest.csv` / `_annual.csv` — the single-name real-COE **term structure**:
  `real_rf, market_erp, idiosyncratic, company_erp, real_coe` (1–150y). The annual file is
  exact-additive decimal.
- `coe_v2_<T>_effective.csv` / `_annual.csv` — the **effective** (collapsed) numbers: the
  whole curve summarized as one cash-flow-PV-weighted rate, the equity analogue of a bond's
  YTM. Fields: `real_rf, market_erp, idiosyncratic, company_erp, real_coe` (+ `cf_growth`).
  The effective pieces are collapsed as nested cumulative curves so they still **add up**
  (`real_rf + market_erp + idiosyncratic = real_coe`).

---

## 3. Files to upload to the repo

Safest is to upload the whole refreshed `asfp/` package (several modules changed and the
pieces interlock). The v2-relevant changes are:

- **Modified:** `asfp/company.py` (adds the single-name IV term structure `equity_vol_ts`),
  `asfp/volsurface.py` (adds the long-dated index extension + vol-curve merge/dedup),
  `asfp/run.py` (assembles the full vol curve into the market ERP + writes
  `index_vol_ts_latest.csv`), `asfp/run_company.py` (uses the IV term structure + writes the
  effective ERP).
- **Supporting (already part of the v2 bundle):** `asfp/total_risk_erp.py`,
  `asfp/elevator.py`, `asfp/collapse.py`, `asfp/volsurface.py`.

Also upload the tests (`tests/test_v2_wiring.py` is new) and, if you haven't already, the
`.github/workflows/` files.

---

## 4. Deploy steps

1. Upload the refreshed `asfp/` package (and `tests/`) to the repo.
2. Confirm the repo secret **`FRED_API_KEY`** is set (Settings → Secrets → Actions).
3. Run the **weekly** job (`weekly-real-yields`, or `python -m asfp.run` locally). Check the
   log line: `market ERP v2 written: N vol pts (obs to X.XXy; short=[…], long=K), floor=…`.
   You want `obs to` reaching past 1y (ideally ~2–3y) once the LEAPS pull works on the
   runner, and `market_erp_v2_latest.csv` present.
4. Run the **ticker** job for AAPL (workflow dispatch with `AAPL`, or set the sheet's TICKER
   cell). Check: `coe v2: R(front)=… obs_to=…y coe(1y)=… coe(100y)=…` and
   `coe v2 EFFECTIVE: real_coe=… company_erp=… (mkt … + idio …)`.
5. Confirm the new CSVs committed to `outputs/`.

---

## 5. Knobs (environment variables on the ticker job)

- `OBS_CATEGORY` — obsolescence durability for the elevator tail: `A` durable (ORY 50),
  `B` moderate (ORY 40, default), `C` exposed (ORY 30). **For AAPL, `A` (durable) is the
  natural choice** — confirm before the production run.
- `ORY_OVERRIDE` — set the Obsolescence Risk Year directly (overrides the category's ORY).
- `COE_CF_GROWTH` — real cash-flow growth rate used to weight the effective collapse
  (default 2.0%). This only affects the *effective* single number, not the term structure.

(Phase 4 will read `OBS_CATEGORY` / ORY per company from the Google Sheet instead of the
env; for now it's an env var.)

---

## 6. The one caveat — the CME futures-options hook

`volsurface.fetch_cme_settlement_vols()` is a **hook that returns `[]` today**. CME's public
option-settlement surface isn't a stable, documented free CSV, so rather than scrape an
unverified endpoint that could silently return garbage into a valuation input, it's left as a
clearly-marked stub. The long-dated reach is currently carried by **SPX/SPY LEAPS via
yfinance (~3y)**, which is reliable and does the same job (ATM IV at long tenors). If you
have a confirmed free ES-options settlement feed (the "prior settle to 2030" data you
mentioned), that endpoint drops straight into this one function and pushes the observed front
from ~3y out toward ~5y — no other change needed.

---

## 7. Verification done before this deploy

- Full test suite: **79 passing** (adds `tests/test_v2_wiring.py` — the IV term structure,
  the long-dated extension, the vol-curve merge/dedup, and the effective collapse, all
  exercised offline with synthetic option chains).
- End-to-end AAPL-like smoke (injected vols + a realistic IV term structure, real library
  calls): market ERP glides 3.8%→1.57% floor with the observed front reaching 3y; the
  single-name term structure holds the additive identity to 0, keeps idiosyncratic ≥ 0 and
  company ERP ≥ market everywhere, and the durable-category elevator lifts the tail; the
  effective real COE collapses to ~6.0% (company ERP ~3.3% = ~2.3 market + ~1.0 idio).
- Live endpoints (FRED/CBOE/Yahoo/CME) can't be reached from the build sandbox; they run on
  the GitHub Actions runner. All are non-fatal, so first-run behavior is safe.
