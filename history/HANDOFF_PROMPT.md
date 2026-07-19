# Handoff prompt — paste into a new chat, attach the two data files

Attach `real_yield_curve_v3_MASTER.csv` and `DOCUMENTATION_real_yield_curve_v3.md`, then paste
the text below.

---

I'm giving you a dataset: `real_yield_curve_v3_MASTER.csv`, a monthly reconstruction of U.S.
Treasury **real** yields at 1, 5, 10, 20, and 30-year tenors, September 1877 – June 2026. A
companion file, `DOCUMENTATION_real_yield_curve_v3.md`, describes the methodology in full — read
it before doing analysis.

**Structure.** For each tenor t ∈ {1,5,10,20,30}, columns are: `date` (month-end); `nom{t}`
(nominal zero-coupon yield, %); `real{t}_exante` and `real{t}_exante_se` (ex-ante real yield and
its 1-standard-error band); `real{t}_tips` and `real{t}_tips_se` (TIPS-like real yield and band);
`phi{t}` (inflation risk premium = ex-ante − TIPS-like); and `tips_source{t}` (provenance flag).

**Two real-yield concepts — use the right one:**
- **Ex-ante** = nominal − expected inflation (the perceived real cost of capital at the time).
- **TIPS-like** = nominal − breakeven = ex-ante − inflation risk premium (the market/observable
  real rate; the correct object for a TIPS-based cost-of-equity or real-return analysis).

**Critical caveats (do not skip):**
1. The **1970s ex-ante is biased low** (the expected-inflation model overshoots during the Great
   Inflation). For the 1970s, use `real{t}_tips`, not `real{t}_exante`. The `tips_source` flags
   and the documentation explain this.
2. **Pre-1920** is a regime characterization (real yields ~2.5–3% under the gold standard), not
   precise monthly points — the price index and long-bond nominal are approximate that far back.
3. **Uncertainty bands widen with historical depth**; respect the `_se` columns. TIPS-like bands
   are ~0 where actual TIPS exist (1999+).
4. From **1982 forward both concepts are solid** (ex-ante anchored on the Cleveland Fed model,
   TIPS-like on Groen/actual TIPS), cross-validated at 0.93–0.99 correlation.

**When you load it:** parse `date` as a datetime, treat each `real*` column as percent, and always
state which concept (ex-ante vs TIPS-like) and which tenor you're using. Don't average across the
two concepts. If a task needs data before 1877 or a real yield the file doesn't contain, say so
rather than extrapolating.
