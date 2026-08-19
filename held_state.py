"""
held_state.py — resolve WHICH ERP_HELD_STATE_*.json the daily job should read.

WHY THIS EXISTS. Until 2026-08-18 the filename `ERP_HELD_STATE_2026-06.json` was typed
literally into four places: `.github/workflows/erp_daily.yml` twice (the live-preview job
and the publish job), `run_erp_daily.py`'s `__main__` smoke, and
`tests/test_erp_daily_writer.py`. There was no "current vintage" pointer of any kind.

That meant the monthly re-anchor had a silent failure mode with teeth: writing a correct
new `ERP_HELD_STATE_2026-08.json` would change NOTHING. The daily job would go on
publishing off June, the acceptance gate would stay green, every test would pass, and the
only symptom would be a published cost of equity built on a stale volatility input. That
is this project's standing suspicion #1 — a number that is silently wrong or silently
inert while every gate reports success — in its purest form, and it was sitting directly
in the path of the one action the vs(T) landing still needs.

WHAT THIS DOES NOT DO. It does not repoint the two June REPRODUCTION harnesses.
`run_erp_daily.__main__` and `tests/test_erp_daily_writer.py` deliberately rebuild
June-2026 from June's committed anchors and assert 2.349 / 3.400 / 5.748. Those are
regression fixtures, not live reads, and pointing them at "the newest vintage" would break
them the moment a new vintage lands — replacing an inert-input bug with a moving-fixture
bug. They stay pinned, and they now say so in code via `JUNE_REPRO_STATE` instead of
repeating a bare string.

THE TWO CHECKS, AND WHY EACH ONE IS HERE.

1. FRESHNESS. The re-anchor is monthly. A held state materially older than that means the
   volatility, normalized-earnings and credit inputs have stopped tracking. The resolver
   refuses to return a state older than `max_age_months` (default 4 — three months of
   slippage before publishing stops). Fail-closed, deliberately: a warning printed into a
   green CI log every morning is the thing everybody learns to scroll past.

2. THE FILENAME MUST AGREE WITH THE CONTENTS. `ERP_HELD_STATE_2026-08.json` containing
   `anchor_vintage: "2026-06"` is exactly the silent hybrid this project keeps getting
   bitten by — a file that looks re-anchored and is not. The resolver reads
   `anchor_vintage` out of the JSON and refuses if it disagrees with the vintage in the
   filename. Mislabelling is now impossible rather than merely discouraged.

Landed 2026-08-18. Verified a strict no-op on the tree as it stands: the June file is the
only `ERP_HELD_STATE_*.json` in the repository, so `resolve_held_state()` returns exactly
what the four hardcoded reads returned, and the published curve is byte-identical.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re

# ERP_HELD_STATE_<YYYY>-<MM>.json
STATE_GLOB = "ERP_HELD_STATE_*.json"
STATE_RE = re.compile(r"ERP_HELD_STATE_(\d{4})-(\d{2})\.json$")

# The June reproduction fixture. Named, not repeated as a bare string, so that a future
# reader can see at a glance that it is pinned ON PURPOSE and is not an oversight.
JUNE_REPRO_STATE = "ERP_HELD_STATE_2026-06.json"

# Months of slippage tolerated before the daily publish stops. The re-anchor is monthly;
# 4 leaves three months of room and still fails long before a stale input can quietly
# become the norm.
MAX_STATE_AGE_MONTHS = 4


def _months_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (b[0] - a[0]) * 12 + (b[1] - a[1])


def list_held_states(root: str = ".") -> list[tuple[tuple[int, int], str]]:
    """[( (year, month), path ), ...] ascending by vintage. Unparseable names are ignored."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, STATE_GLOB))):
        m = STATE_RE.search(os.path.basename(p))
        if m:
            out.append(((int(m.group(1)), int(m.group(2))), p))
    out.sort()
    return out


def resolve_held_state(root: str = ".", asof: str | None = None,
                       max_age_months: int = MAX_STATE_AGE_MONTHS,
                       log=print):
    """Return (path, state_dict) for the NEWEST held state, or raise.

    `asof` is an ISO date; defaults to today (UTC). Raises RuntimeError if no state file
    exists, if the newest is older than `max_age_months`, or if the vintage in the
    filename disagrees with `anchor_vintage` inside the file.
    """
    states = list_held_states(root)
    if not states:
        raise RuntimeError(
            "no %s found under %r -- the daily publish has no held state to run from"
            % (STATE_GLOB, os.path.abspath(root)))

    vintage, path = states[-1]
    state = json.load(open(path))

    declared = str(state.get("anchor_vintage", "")).strip()
    expected = "%04d-%02d" % vintage
    if declared != expected:
        raise RuntimeError(
            "held state %s declares anchor_vintage=%r but its filename says %r. A file "
            "named for one vintage holding another is the silent-hybrid failure; refusing "
            "to publish. Fix the file, do not relax this check."
            % (os.path.basename(path), declared, expected))

    today = dt.date.fromisoformat(asof) if asof else dt.datetime.now(dt.timezone.utc).date()
    age = _months_between(vintage, (today.year, today.month))
    if age > max_age_months:
        raise RuntimeError(
            "newest held state is %s, %d months old as of %s; the monthly re-anchor is "
            "overdue and the volatility / normalized-earnings / credit inputs have stopped "
            "tracking. Publishing is stopped by design. Write the next "
            "ERP_HELD_STATE_YYYY-MM.json, or raise max_age_months deliberately and say why."
            % (expected, age, today.isoformat()))

    if len(states) > 1:
        prior = "%04d-%02d" % states[-2][0]
        log("  held state: %s (age %d months; previous vintage %s)" % (expected, age, prior))
    else:
        log("  held state: %s (age %d months; only vintage in the tree)" % (expected, age))
    return path, state


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    found = list_held_states(root)
    print("held states found: %d" % len(found))
    for v, p in found:
        print("  %04d-%02d  %s" % (v[0], v[1], p))
    p, s = resolve_held_state(root)
    print("resolved -> %s" % p)
    print("  anchor_vintage=%s  vs=%s" % (
        s.get("anchor_vintage"),
        ("%d-vector" % len(s["vs"])) if isinstance(s.get("vs"), list) else s.get("vs")))
