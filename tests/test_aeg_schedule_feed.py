"""Tests for asfp/aeg_schedule_feed.py — the real-distribution-profile fetch used to
wire real cash flows into collapse_rate() (Task 2, AEG-ERP-Collapse-Function-AUDIT-
2026-08-12.md section 5). All offline: local_dir fixtures stand in for the HTTPS
raw.githubusercontent GET, exactly like rate_feed.py's own local_dir test seam.
"""
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import aeg_schedule_feed as asf


def _write_schedule(tmp_path, ticker, rows, header=("t", "phase", "dps_real")):
    fn = tmp_path / f"{ticker}_aeg_schedule.csv"
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return tmp_path


def test_fetches_a_clean_positive_profile(tmp_path):
    rows = [(i, "explicit" if i <= 4 else "continuing", round(8.0 - 0.1 * i, 4))
            for i in range(1, 31)]
    _write_schedule(tmp_path, "ZZZ", rows)
    grid, cf = asf.fetch_distribution_profile("ZZZ", local_dir=str(tmp_path))
    assert len(grid) == len(cf) == 30
    assert grid[0] == 1.0 and grid[-1] == 30.0
    assert all(v > 0 for v in cf)


def test_missing_file_raises_scheduleFetchError(tmp_path):
    try:
        asf.fetch_distribution_profile("NOPE", local_dir=str(tmp_path))
        assert False, "expected ScheduleFetchError"
    except asf.ScheduleFetchError:
        pass


def test_missing_dps_real_column_raises(tmp_path):
    fn = tmp_path / "ZZZ_aeg_schedule.csv"
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "phase", "normal_eps"])
        w.writerow([1, "explicit", 5.0])
    try:
        asf.fetch_distribution_profile("ZZZ", local_dir=str(tmp_path))
        assert False, "expected ScheduleFetchError"
    except asf.ScheduleFetchError as e:
        assert "dps_real" in str(e)


def test_empty_schedule_raises(tmp_path):
    _write_schedule(tmp_path, "ZZZ", [])
    try:
        asf.fetch_distribution_profile("ZZZ", local_dir=str(tmp_path))
        assert False, "expected ScheduleFetchError"
    except asf.ScheduleFetchError:
        pass


def test_non_positive_dps_real_is_rejected(tmp_path):
    # Reproduces the AZO/HD/MCD shape found 2026-08-12: an implied-distribution path
    # under the two-of-three closure that goes negative in some years. Must NOT be
    # silently accepted as collapse weights.
    rows = [(1, "explicit", 5.0), (2, "explicit", -1.2), (3, "explicit", 4.0)]
    _write_schedule(tmp_path, "AZO", rows)
    try:
        asf.fetch_distribution_profile("AZO", local_dir=str(tmp_path))
        assert False, "expected ScheduleFetchError"
    except asf.ScheduleFetchError as e:
        assert "non-positive" in str(e)


def test_stops_at_first_gap_without_zero_filling(tmp_path):
    rows = [(1, "explicit", 5.0), (2, "explicit", 5.2), (3, "explicit", ""), (4, "explicit", 5.5)]
    _write_schedule(tmp_path, "ZZZ", rows)
    grid, cf = asf.fetch_distribution_profile("ZZZ", local_dir=str(tmp_path))
    assert grid == [1.0, 2.0]                    # stopped before the gap, year 4 dropped too
    assert cf == [5.0, 5.2]


def test_unreachable_url_raises_scheduleFetchError():
    # No local_dir -> real HTTPS path, pointed at a URL that cannot resolve/serve.
    try:
        asf.fetch_distribution_profile(
            "NOPE", base_url="https://raw.githubusercontent.com/JamesKostohryz/"
                              "aeg-valuation/main/outputs/does-not-exist",
            timeout=5)
        assert False, "expected ScheduleFetchError"
    except asf.ScheduleFetchError:
        pass
