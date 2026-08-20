# SUPERSEDED 2026-08-20 — do not read these files

Every `coe_history_<TICKER>.csv` here was built on
`history/FINAL_decomposition_v4_1877_2026.csv`, which is on a DIFFERENT equity-risk-premium
level from the live engine. Measured for the same month, June 2026:

    FINAL_decomposition_v4        eff_erp  3.887
    history/ERP_effective_latest  eff_erp  3.370

A 0.52 percentage point break between two published series describing the same object, with no
splice logic anywhere and nothing to notice it. `coe_history_KO.csv` and the live
`coe_v2_KO_latest_annual.csv` were on two different premium levels at the same time.

Three further things, all verified 2026-08-20:

* **Nothing built the decomposition.** It was committed once and never regenerated; the only
  code that mentioned it read it.
* **`coe_history.py` read it from a hardcoded `/tmp/calib/` path.**
* Its pre-1995 `cost` column follows a straight line from 2.50 to 1.50 that exists in no code,
  and the live `cost_of_year()` returns a **complex number** for 1877 — it raises a negative
  base to a fractional power.

**The replacement is `aeg-valuation/outputs/market_coe_history.csv`**, built by
`aeg-valuation/idio/market_coe_history.py` from current components only: the real-rate leg from
`history/real_yield_curve_v3_MASTER.csv` (reused unchanged — it is a rate curve, not a risk
construction, and it carries a provenance flag on every cell), and the premium from the
pre-registered semi-deviation bridge in `aeg-valuation/idio/market_semidev_bridge.py`.

Kept rather than deleted so the old numbers can still be inspected. Nothing may read them.
