"""The equity cost premium of record (James, 2026-10-08; aeg-project
docs/RULING-Cost-Premium-Of-Record-2026-10-08.md). cost_of_year() reads one annual line,
cost_premium_of_record.csv, and is the only cost premium the engine, the re-anchor and the
market cost-of-equity history use."""
import csv, datetime as dt, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import build_erp_daily as BED
import reanchor as RA

TABLE = os.path.join(ROOT, "cost_premium_of_record.csv")


def _rows():
    with open(TABLE, newline="") as fh:
        return {int(r["year"]): r for r in csv.DictReader(fh)}


def test_the_table_is_the_ruled_line():
    t = _rows()
    assert min(t) == 1870 and max(t) >= 2025
    assert float(t[1870]["cost_premium"]) == 2.0                       # James's start level
    peak = max(t, key=lambda y: float(t[y]["cost_premium"]))
    assert peak == 1972 and abs(float(t[1972]["cost_premium"]) - 2.5464) < 1e-9
    for y in t:                                                          # measured from 1973
        if y > 1972:
            assert float(t[y]["cost_premium"]) == float(t[y]["measured_central"])
    for y in t:                                                          # the band brackets it
        lo, c, hi = (float(t[y][k]) for k in ("cost_premium_low", "cost_premium", "cost_premium_high"))
        assert lo <= c <= hi


def test_each_value_sits_at_mid_year_and_is_flat_outside():
    t = _rows()
    for y in (1870, 1931, 1972, 1999, 2025):
        assert abs(BED.cost_of_year(y + 0.5) - float(t[y]["cost_premium"])) < 1e-12
    mid = 0.5 * (float(t[1990]["cost_premium"]) + float(t[1991]["cost_premium"]))
    assert abs(BED.cost_of_year(1991.0) - mid) < 1e-12
    last = float(t[max(t)]["cost_premium"])
    assert BED.cost_of_year(2040.0) == last and BED.cost_of_year(1800.0) == 2.0


def test_the_reanchor_uses_the_same_line_floored():
    c = RA.cost_for(dt.date(2026, 10, 9))
    assert c == max(BED.cost_of_year(2025.9), RA.COST_FLOOR)
    assert abs(c - 0.3099) < 1e-9
    lo, hi, _ = RA.BANDS["cost"]
    assert lo == RA.COST_FLOOR and hi >= max(float(r["cost_premium"]) for r in _rows().values())
