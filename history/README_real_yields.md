# U.S. Real Yield Curve, 1877–2026 — build repo

Monthly nominal & real yields at 1/5/10/20/30 years, in two risk concepts
(ex-ante and TIPS-like), with uncertainty bands. See `METHODS_real_yield_curve_v2.md`.

## Run
```
pip install pandas numpy statsmodels python-calamine openpyxl
SRC=./data OUT=./out python build_real_yield_curve.py
```
`SRC` holds the source files (names in the `F` dict at the top of the script).
Output: `real_yield_curve_v2.csv` — columns per tenor t:
`nom{t}`, `real{t}_exante`, `real{t}_exante_se`, `irp{t}`, `real{t}_tips`, `real{t}_tips_se`.

## Sources (place in ./data)
Static (historical, never change): GFD tenor files 1Y/5Y/10Y/20Y/30Y, 3M T-bill,
Interest_Rate_Data (commercial paper), GFD_CPI, ie_data (Shiller/Homer-Sylla),
Groen synthetic breakevens, SPF Inflation + Additional-CPIE10.
Updating monthly: `feds200628.csv` (GSW) and `CPIAUCNS.csv` (FRED CPI).

## Monthly refresh
Only GSW and FRED CPI change month to month; everything else is fixed history.
The included GitHub Action refreshes those two and reruns the build.

## Validation (baked into the build)
- Nominal vs Homer-Sylla: 0.18pp (1920+); pre-1920 blended, ~0.35pp spread carried as band.
- Expected inflation vs SPF: 1y MAE 0.54; 10y MAE 0.42 / corr 0.91 (disinflation-inclusive).
- Nominal curve: NS fit reproduces observed points to 4.5bp; GSW params reproduce SVENY to 0.000.
- Cross-method (NS vs regression) agreement: 0.02–0.23pp by tenor.
