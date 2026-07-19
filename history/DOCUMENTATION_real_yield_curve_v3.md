# U.S. Real Yield Curve, 1877–2026 — Full Documentation (v3, final integrated series)

A monthly reconstruction of U.S. Treasury **real** yields at 1, 5, 10, 20, and 30-year
tenors, September 1877 – June 2026, in two risk concepts (ex-ante and TIPS-like), with
per-tenor uncertainty bands and a source flag on every value. Every segment is anchored on
the most reliable source available for its era and cross-validated against independent series.

Primary file: `real_yield_curve_v3_MASTER.csv`.

---

## 1. What the dataset contains

Two distinct real-yield objects, because they answer different questions:

**Ex-ante real yield** = nominal yield − expected inflation. The real cost of capital as
*perceived at the time*. Column `real{t}_exante`.

**TIPS-like real yield** = nominal yield − breakeven inflation = ex-ante − inflation risk
premium. What an inflation-protected bond would have yielded; the market/observable real
rate. Column `real{t}_tips`. **This is the object for cost-of-equity work** (a TIPS-based
real risk-free rate).

The gap between them, `phi{t}`, is the **inflation risk premium** (IRP).

---

## 2. Source and construction, by leg and era

### Nominal yields (the spine of both concepts)
| Era | Source |
|---|---|
| 1961+ | Gürkaynak-Sack-Wright (GSW) `feds200628`, Svensson params evaluated per tenor |
| pre-1961 | Monthly Nelson-Siegel fit to available government points (GFD tenor files, T-bills, commercial paper); pre-1920 10y blended with Homer-Sylla (Shiller) |

Cross-validated against Homer-Sylla: 0.18pp agreement 1920+, ~0.35pp (blended) before.

### Expected inflation (drives the ex-ante leg)
- **Model:** UCSV-style local-level trend with stochastic volatility + a regime-aware term
  structure (gold-standard endpoint ≈ 0.47%, fiat endpoint = the trend; transition 1933–1960).
- **Inputs:** long CPI splice (David-Solar → Fed cost-of-living → Snyder General Price Index
  1875–1912 → BLS from 1913), YoY.
- **Validation:** vs SPF, 1-year MAE 0.54; 10-year MAE 0.42 / corr 0.91 (disinflation-inclusive).

### Ex-ante real yield
| Era | Construction |
|---|---|
| 1982+ | **Cleveland Fed anchored:** nominal − Cleveland `EXPINF` (per tenor). Reproduces Cleveland's published 10y real rate (corr 0.94, gap 0.08pp) and extends it to all tenors. |
| pre-1982 | Reconstruction: nominal − model expected inflation. Joined to 1982 by a continuity correction (fixes the model's disinflation overshoot, decaying to zero by 1976) → no jump, no blend. |

### TIPS-like real yield
| Era | Source |
|---|---|
| 1999+ | **Actual TIPS** — GSW real curve `feds200805` (TIPSY 5/10/20) + DFII (30y). |
| 1971–1999 | **Groen (NY Fed) synthetic** real rate (10y, 20y direct; 5/30 via curve shape). |
| pre-1971 | Reconstruction: ex-ante − a small regime-based IRP (≈0 under the gold standard, ramping to ~0.4pp by 1971). |

Validated against actual TIPS at every tenor, 1999–2026: corr 0.93–0.99, MAE 0.14–0.46pp.

### Inflation risk premium (`phi`)
Market/synthetic-measured where breakevens exist (Groen 1971–99, actual TIPS 1999+); ≈0 on
average in the modern era, though volatile (−3.6pp at 5y in the 2008 crisis). Regime-based
and wide-banded pre-1971. Note: the earlier `κ·E[π]` model of the IRP was retired — the
market revealed a smaller premium that does not scale with the inflation level as assumed.

---

## 3. Cross-validation summary

| Check | Result |
|---|---|
| Ex-ante vs Cleveland Fed real rate (1982+) | corr **0.94**, gap 0.08pp |
| TIPS-like vs actual TIPS (1999+) | corr **0.93–0.99**, MAE 0.14–0.46pp |
| Nominal vs Homer-Sylla (1920+) | MAE **0.18pp** |
| Ex-ante vs Groen, TIPS-like vs Groen | consistent once sorted by concept |
| Nelson-Siegel vs tenor-by-tenor regression | 0.02–0.23pp |

Four independent constructions (Cleveland, Groen, Homer-Sylla, actual TIPS) corroborate the
series in their respective windows.

---

## 4. Known limitations — read before citing

1. **1970s ex-ante is biased low.** The CPI-driven expected-inflation model overshoots when
   inflation is high and expectations lagged (the Great Inflation); the ex-ante real yield is
   too low there. **Use the TIPS-like (Groen) series for the 1970s.** Post-1982 both concepts
   are solid. The same over-responsiveness appears mildly in the post-COVID period — which is
   exactly why the modern ex-ante is anchored on Cleveland rather than the raw model.
2. **Pre-1920** rests on a composite long-bond nominal and a pre-modern price index (Snyder's
   commodity/wage/rent blend, not a consumer basket). Read as a regime characterization
   (real yields ~2.5–3% under the classical gold standard), not precise monthly points.
3. **WWI (1917–19)** — gold convertibility suspended; the gold-anchor assumption is strained.
4. **The IRP** is the most model-dependent layer where no breakevens exist (pre-1971). Trust
   its sign and regime pattern; treat pre-1971 magnitudes as indicative (hence the wide bands).

Uncertainty bands (`_se` columns) widen with historical depth: ~0.4–0.6pp modern, ~0.8–1.0pp
pre-1920; TIPS-like bands are ~0 where actual TIPS exist.

---

## 5. Column dictionary — `real_yield_curve_v3_MASTER.csv`

For each tenor t ∈ {1, 5, 10, 20, 30}:
- `date` — month-end.
- `nom{t}` — nominal zero-coupon yield (%).
- `real{t}_exante` — ex-ante real yield (%); `real{t}_exante_se` — 1 s.e. band (pp).
- `real{t}_tips` — TIPS-like real yield (%); `real{t}_tips_se` — 1 s.e. band (pp).
- `phi{t}` — inflation risk premium = ex-ante − TIPS-like (pp).
- `tips_source{t}` — provenance: `market-TIPS` / `groen-synthetic` / `breakeven-implied` /
  `regime-extrapolated`.

---

## 6. Source files

GSW `feds200628.csv` (nominal), `feds200805.csv` (real/TIPS); GFD tenor files
(IGUSA1/5/10/20/30D, ITUSA3D), `Interest Rate Data.xlsx` (commercial paper), `GFD_CPI.xlsx`
+ FRED `CPIAUCNS`; Philadelphia Fed SPF (`Inflation.xlsx`, `Additional-CPIE10.xlsx`); NY Fed
Groen `Synthetic_TIPS_Breakeven_Rates.xlsx`; Cleveland Fed `Inflation_expectations.xlsx`
(EXPINF + real rate); FRED `DFII5/7/10/20/30`; Shiller `ie_data.xls` (Homer-Sylla, cross-val).

## 7. Reproducibility status

`build_real_yield_curve.py` regenerates the **ex-ante v2** stage from source and is verified
to 0.002pp. The v3 integration (Cleveland anchoring, Groen/TIPS splice, market phi) was done
interactively and is documented here but **not yet consolidated into that single script** —
that is the remaining step to make the full v3 self-regenerating.
