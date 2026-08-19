"""Tests for held_state.resolve_held_state — the "which vintage does the daily job read"
pointer added 2026-08-18.

These exist because the failure this module prevents is invisible by construction: before
it, writing a correct new ERP_HELD_STATE_YYYY-MM.json changed nothing at all, and every
gate in the repository stayed green while the daily publish went on using June. A test
that only checked "the resolver returns a file" would have the same blind spot, so each
test below asserts a specific way the resolver must REFUSE.

Fully hermetic: every case is built in a temporary directory. Nothing here reads the live
tree except the last test, which asserts the no-op property on the repository as it stands.
"""
import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import held_state as hs  # noqa: E402


def _write(root, vintage, declared=None, vs=0.9348):
    """Create ERP_HELD_STATE_<vintage>.json; `declared` defaults to matching the name."""
    body = {"anchor_vintage": declared if declared is not None else vintage,
            "vs": vs, "fey_in": 6.02, "D_in": 24.72, "cost": 0.503,
            "corp_prem": 1.02, "breakeven1y": 2.76, "cpi_factor": 0.99844,
            "normalized_X4": 234.1507}
    p = os.path.join(root, "ERP_HELD_STATE_%s.json" % vintage)
    with open(p, "w") as f:
        json.dump(body, f)
    return p


def test_newest_vintage_wins_not_alphabetical_luck():
    d = tempfile.mkdtemp()
    _write(d, "2026-06")
    _write(d, "2026-08")
    _write(d, "2025-12")
    path, state = hs.resolve_held_state(d, asof="2026-08-18", log=lambda *_a: None)
    assert os.path.basename(path) == "ERP_HELD_STATE_2026-08.json"
    assert state["anchor_vintage"] == "2026-08"


def test_no_state_file_at_all_raises():
    d = tempfile.mkdtemp()
    with pytest.raises(RuntimeError, match="no ERP_HELD_STATE"):
        hs.resolve_held_state(d, asof="2026-08-18", log=lambda *_a: None)


def test_stale_state_stops_publishing():
    """The whole point: a re-anchor that never happened must stop the publish, not be
    absorbed silently."""
    d = tempfile.mkdtemp()
    _write(d, "2026-01")
    with pytest.raises(RuntimeError, match="months old"):
        hs.resolve_held_state(d, asof="2026-08-18", max_age_months=4, log=lambda *_a: None)
    # ...and is fine when it is inside the window
    path, _ = hs.resolve_held_state(d, asof="2026-03-01", max_age_months=4,
                                    log=lambda *_a: None)
    assert path.endswith("ERP_HELD_STATE_2026-01.json")


def test_filename_disagreeing_with_contents_is_refused():
    """A file NAMED for August holding JUNE's anchor is the silent-hybrid failure mode.
    It must be impossible to publish from, not merely discouraged."""
    d = tempfile.mkdtemp()
    _write(d, "2026-08", declared="2026-06")
    with pytest.raises(RuntimeError, match="anchor_vintage"):
        hs.resolve_held_state(d, asof="2026-08-18", log=lambda *_a: None)


def test_vector_vs_is_accepted():
    """vs(T) landed 2026-08-18 as a 30-element term structure; the resolver must not care
    which form it is, because build_asof accepts both."""
    d = tempfile.mkdtemp()
    _write(d, "2026-08", vs=[0.95] * 30)
    _, state = hs.resolve_held_state(d, asof="2026-08-18", log=lambda *_a: None)
    assert isinstance(state["vs"], list) and len(state["vs"]) == 30


def test_boundary_exactly_at_the_age_limit_is_allowed():
    d = tempfile.mkdtemp()
    _write(d, "2026-04")
    path, _ = hs.resolve_held_state(d, asof="2026-08-18", max_age_months=4,
                                    log=lambda *_a: None)
    assert path.endswith("ERP_HELD_STATE_2026-04.json")
    with pytest.raises(RuntimeError):
        hs.resolve_held_state(d, asof="2026-09-01", max_age_months=4, log=lambda *_a: None)


def test_live_tree_resolves_to_its_newest_vintage():
    """RETIRED AND REPLACED 2026-08-19. This was 'the no-op proof': on the tree as landed the
    resolver had to return exactly the file previously typed into the workflow, so that
    introducing the resolver could not move a number. That was the right test for one day.

    It was an assertion that JUNE IS THE NEWEST VINTAGE, and the resolver's whole purpose is to
    stop being true the moment a newer one lands. The first automated re-anchor wrote
    ERP_HELD_STATE_2026-08.json and this test turned the entire daily ERP publish red -- a test
    that fails when the system starts working is a pin on a date, not a regression test.

    What survives is the property that actually matters and is true forever: the resolver
    returns the NEWEST vintage in the tree, and that file's name agrees with its contents."""
    path, state = hs.resolve_held_state(ROOT, log=lambda *_a: None)
    newest = hs.list_held_states(ROOT)[-1]
    assert os.path.abspath(path) == os.path.abspath(newest[1]), "resolver did not take the newest"
    assert state["anchor_vintage"] == "%04d-%02d" % newest[0]
    assert state == json.load(open(path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
