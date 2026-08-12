"""
aeg_schedule_feed.py — fetch a company's own forecast DISTRIBUTION stream from the
AEG valuation engine's published schedule, for use as REAL cash-flow weights when
collapsing a rate term structure to a single number (see asfp/collapse.py, Task 2,
AEG-ERP-Collapse-Function-AUDIT-2026-08-12.md section 5).

Mirrors aeg-valuation's rate_feed.py contract in the OPPOSITE direction: plain HTTPS
GET on raw.githubusercontent, no auth. aeg-valuation already fetches real-yields'
curve CSVs this way (see rate_feed.py's docstring); this is the same "locked CSV
contract" pattern applied the other way, so real-yields can read the distribution
stream aeg-valuation's engine actually produced for a given ticker.

FAILURE POLICY — deliberately different from rate_feed.py's fail-loud stance. There
a missing/bad rate feed is always a bug (the feed is supposed to exist). Here a
missing schedule is an EXPECTED case: a brand-new ticker has no forecast yet, or a
name's implied distribution stream is not usable as collapse weights (see below).
ScheduleFetchError is raised for every one of these cases; the CALLER (asfp/collapse
call sites in run_company.py) decides to catch it and fall back to the synthetic
growth profile, printing a warning that says why.

WHY THE PROFILE CAN BE UNUSABLE, NOT JUST MISSING
---------------------------------------------------
Under the canonical operating closure (aeg-valuation, commit 2c7401a and after),
distributions are IMPLIED — financing absorbs whatever the driven operating plan
doesn't fund. For some names (checked 2026-08-12: AZO, HD, MCD) that produces
per-year real distributions that go negative or swing by two orders of magnitude
year to year. collapse_rate()'s bisection assumes the cash-flow weights are
non-negative (PV must be monotone in the flat rate); feeding it a profile with
negative or wildly unstable entries would not error — it would silently return a
number that reprices nothing real. So this module validates positivity and raises
ScheduleFetchError rather than let the caller collapse on a broken weight vector.
"""
from __future__ import annotations

import csv
import io
import os
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/JamesKostohryz/aeg-valuation/main/outputs"


class ScheduleFetchError(Exception):
    """The company's aeg_schedule.csv is unavailable, malformed, or not usable as a
    collapse weight profile. Expected for a name with no forecast yet, or one whose
    implied-distribution path is not positive throughout. Callers should catch this
    and fall back to collapse_rate's synthetic growth profile — never let it abort
    the run, and never silently zero-fill a gap to make it fit."""


def _fetch_text(ticker, *, base_url=BASE_URL, local_dir=None, timeout=30):
    """Raw text of <ticker>_aeg_schedule.csv. Production: HTTPS GET on
    raw.githubusercontent (no auth). Testing/offline: pass local_dir. Exactly one
    source, same shape as aeg-valuation's rate_feed.py::_fetch_text."""
    fname = f"{ticker}_aeg_schedule.csv"
    if local_dir is not None:
        path = os.path.join(local_dir, fname)
        if not os.path.exists(path):
            raise ScheduleFetchError(f"[fetch] local fixture missing: {path}")
        with open(path, "r", newline="") as fh:
            return fh.read()
    url = f"{base_url}/{fname}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                raise ScheduleFetchError(f"[fetch] {url} -> HTTP {r.status}")
            return r.read().decode("utf-8")
    except ScheduleFetchError:
        raise
    except Exception as e:      # urllib.error.*, socket timeout, DNS, etc.
        raise ScheduleFetchError(f"[fetch] {url} -> {type(e).__name__}: {e}")


def fetch_distribution_profile(ticker, *, base_url=BASE_URL, local_dir=None, timeout=30):
    """Return (grid, cashflows): 1-based forecast years and the company's own real
    (inflation-deflated) distribution per share, dps_real, read straight off its
    published AEG schedule — the same column the engine's own self-verify check
    (write_aeg_schedule) already ties to the intrinsic value before it is written.

    Raises ScheduleFetchError on anything that makes the profile unfit to use as
    collapse weights: unreachable/missing file, missing columns, no usable rows,
    or any non-positive entry (see module docstring). Never returns a partial or
    zero-patched profile silently.
    """
    text = _fetch_text(ticker, base_url=base_url, local_dir=local_dir, timeout=timeout)
    rd = csv.DictReader(io.StringIO(text))
    fieldnames = rd.fieldnames or []
    if "t" not in fieldnames or "dps_real" not in fieldnames:
        raise ScheduleFetchError(
            f"{ticker}_aeg_schedule.csv: missing t/dps_real columns "
            f"(got {fieldnames})")
    rows = list(rd)
    if not rows:
        raise ScheduleFetchError(f"{ticker}_aeg_schedule.csv: empty")

    grid, cf = [], []
    for row in rows:
        t, d = row.get("t"), row.get("dps_real")
        if t in (None, "") or d in (None, ""):
            break                                    # stop at the first gap; don't zero-fill
        grid.append(float(t))
        cf.append(float(d))
    if not grid:
        raise ScheduleFetchError(f"{ticker}_aeg_schedule.csv: no usable dps_real rows")
    if any(v <= 0 for v in cf):
        n_bad = sum(1 for v in cf if v <= 0)
        raise ScheduleFetchError(
            f"{ticker}_aeg_schedule.csv: {n_bad}/{len(cf)} dps_real entries are "
            f"non-positive (implied-distribution profile unusable as collapse "
            f"weights — collapse_rate's bisection assumes a positive cash-flow "
            f"stream)")
    return grid, cf
