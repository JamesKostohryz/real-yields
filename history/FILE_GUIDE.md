# FILE GUIDE — Long Term Inflation Expectations & Real Yields

Complete contents of this package. **Single source of truth: `real_yield_curve_v3_MASTER.csv`.**
Everything else is documentation, scaffolding (intermediate build stages, kept for audit), or charts.

Note: this package contains the OUTPUTS the assistant produced. It does NOT contain the raw
SOURCE data files (GSW feds200628/feds200805, GFD tenor/CPI files, Cleveland/SPF/Groen
spreadsheets, DFII, Shiller) — those live on the GitHub repo or were uploaded per-chat; see
DOCUMENTATION §6 and the HAND-OFF doc §3 for their names and links.

====================================================================
START HERE
====================================================================
- 00_HANDOFF_PACKAGE.txt ....... The full 11-section hand-off (module, sources, status, repo
                                  mechanics, next steps). Read this first for the big picture.
- FILE_GUIDE.md ................ This file.
- DOCUMENTATION_real_yield_curve_v3.md . Full methodology + column dictionary for the master CSV.
- HANDOFF_PROMPT.md ............ Paste-in prompt for a NEW analysis chat (attach the master CSV
                                  + the documentation with it).
- GOING_LIVE_gameplan.md ....... Step-by-step plan to put this on GitHub (Level 1 store now,
                                  Level 2 automate later).

====================================================================
THE DATASET (authoritative)
====================================================================
- real_yield_curve_v3_MASTER.csv  ★ FINAL. Monthly 1877–2026, tenors 1/5/10/20/30.
    Per tenor t: nom{t}, real{t}_exante (+_se), real{t}_tips (+_se), phi{t} (=IRP),
    tips_source{t} (market-TIPS / groen-synthetic / breakeven-implied / regime-extrapolated).
- real_10y_v3_FINAL.png ........ Hero chart: 10y ex-ante vs TIPS-like, 1877–2026, era-shaded.

====================================================================
BUILD / DEPLOY
====================================================================
- build_real_yield_curve.py .... Reproducible build script. Regenerates the EX-ANTE v2 stage
                                  from source (verified 0.002pp). v3 integration NOT yet folded in.
- README_real_yields.md ........ Repo README (run instructions, source manifest).
- real_yields_monthly.yml ...... Draft monthly GitHub Action (refresh GSW+FRED, rerun, commit).
- METHODS_real_yield_curve_v2.md . v2 methods doc (superseded by the v3 documentation; still valid for v2).

====================================================================
INTERMEDIATE DATASETS (scaffolding — keep for audit, not authoritative)
====================================================================
Real-yield curve stages, earliest→latest:
- real_yield_curve_1877_present.csv ....... 5-tenor curve, tenor-by-tenor REGRESSION method (cross-check).
- real_yield_curve_NSS_1877.csv ........... Same, Nelson-Siegel-Svensson (one arbitrage-free curve).
- real_yield_curve_v2.csv / _v2_MASTER.csv  Ex-ante v2 curve + uncertainty bands.
- real_yield_curve_v2_tipslike.csv ........ v2 TIPS-like using MODELED IRP (κ·E[π]) — later retired.
- real_yield_curve_cleveland_anchored.csv . First Cleveland splice (had a 1982–86 blend) — superseded.
- real_yield_curve_cleveland_anchored_v2.csv  No-jump Cleveland-anchored ex-ante (continuity correction).
  (v3_MASTER combines: cleveland_anchored_v2 ex-ante + Groen/actual-TIPS TIPS-like + market phi.)

Single-tenor / era-specific stages:
- real_10y_1877_present.csv ............... 10y real to 1877 (pre-v2).
- real_10y_1914_present_FINAL.csv ......... 10y real, 1914 start (before gold-standard extension).
- real_10y_extended_1914.csv .............. 10y nominal/real extension work.
- real_1y_1877_present.csv ................ 1y real (commercial-paper/T-bill short-end build).
- real_yields_reconstructed_1961_present.csv  5y/10y real from GSW era (1961+ first cut).
- synthetic_real_yields_1971_2013.csv ..... Groen synthetic real (parsed NY Fed file).

Expected-inflation engine:
- expected_inflation_termstructure_v2.csv . ★ The engine: regime-aware E[π] at 1/5/10/20/30 (UCSV + gold anchor).
- expected_inflation_termstructure_1877.csv  Earlier E[π] term structure.
- expected_inflation_10y_1914_present.csv . 10y E[π], model+survey, 1914+.
- expected_inflation_10y_1979_present.csv . Unified survey 10y E[π] (Blue Chip→SPF), 1979+.
- model_expected_inflation_1914_present.csv  UCSV trend backfill (1914+).
- spf_expected_inflation.csv / spf_inflation_expectations_tidy.csv  Parsed Philadelphia Fed SPF.

====================================================================
CHARTS (png)
====================================================================
Final / headline:
- real_10y_v3_FINAL.png ................. Final 10y, both concepts, era-shaded.
- tips_validation_panels.png ........... Our real vs ACTUAL TIPS at 5/10/20/30, 1999–2026 (0.93–0.99 corr).
- three_way_comparison.png ............. Ours vs Cleveland Fed vs Groen (NY Fed), 10y.
- cleveland_vs_ours.png ................ Our ex-ante vs Cleveland real rate (corr 0.94, gap 0.08).
- per_tenor_validation_current.png ..... Current-curve check: anchoring fixes the modern overshoot.
Construction / validation:
- nominal_crossvalidation.png .......... Our 10y nominal vs Homer-Sylla (Shiller).
- method_comparison.png ................ Regression vs Nelson-Siegel (agree to ~0.1–0.2pp).
- real_10y_v2_vs_v1.png ................ v2 (regime-aware) vs v1 — pre-1933 revision.
- real_10y_cleveland_anchored.png ...... Cleveland-anchored splice (first, blended version).
- exante_vs_tipslike_10y.png ........... Ex-ante vs TIPS-like, IRP wedge, 1877–2026.
- implied_10y_inflation_risk_premium.png  IRP measured from Groen − survey, 1979–98.
- expected_inflation_termstructure.png . E[π] term structure 1877–2026 (1y cyclical vs 10y/30y anchor).
- expected_inflation_backfill_1914.png . E[π] model backfill + survey validation.
- spf_vs_groen_breakeven_overlap.png ... SPF vs Groen breakeven overlap.
- synthetic_real_yield_10y.png ......... Groen synthetic 10y real.
Historical single-tenor:
- real_yield_curve_1877.png ............ Full 5-tenor real curve + regime snapshots.
- real_10y_1877_present.png ............ 10y real to 1877 (gold standard → today).
- real_1y_1877_present.png ............. 1y real, colored by nominal source era.
- real_10y_1914_present_FINAL.png ...... 10y real, 1914 milestone.
- real_yields_reconstructed_1961.png ... 5y/10y real from 1961 (first GSW-era cut).

====================================================================
PROVENANCE NOTE
====================================================================
Built across multiple sessions; the integrated v3 work was completed 2026-07-17. The v3
master and its documentation supersede all v1/v2/intermediate files for any downstream use.
