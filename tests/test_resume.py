"""Resumable-sweep bookkeeping in main.py.

E1 and E2 train an ensemble per sweep point, so an interrupted run must be able
to continue rather than restart. That logic is load-bearing -- a bug in it would
silently drop or duplicate sweep points, and the resulting table would look
perfectly plausible.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def m(tmp_path, monkeypatch):
    """Load main.py with its results directory pointed at a temp dir."""
    spec = importlib.util.spec_from_file_location("_main_under_test", ROOT / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_main_under_test"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "RESULTS", tmp_path)
    return mod


def _write(mod, name, rows, key="beta", **extra):
    payload = {"rows": [{key: v} for v in rows], **extra}
    (mod.RESULTS / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_remaining_is_everything_when_nothing_saved(m):
    assert m._remaining((0.6, 1.0, 1.6), "e1", "beta") == (0.6, 1.0, 1.6)


def test_remaining_skips_completed_points(m):
    _write(m, "e1", [0.6, 1.6])
    assert m._remaining((0.6, 1.0, 1.6, 2.5), "e1", "beta") == (1.0, 2.5)


def test_remaining_empty_when_all_done(m):
    _write(m, "e1", [0.6, 1.0])
    assert m._remaining((0.6, 1.0), "e1", "beta") == ()


def test_merge_combines_and_sorts(m):
    _write(m, "e1", [1.6, 0.6])
    merged = m._merge({"rows": [{"beta": 1.0}], "complete": False}, "e1", "beta")
    assert [r["beta"] for r in merged["rows"]] == [0.6, 1.0, 1.6]


def test_merge_does_not_duplicate_existing_points(m):
    """A retry that recomputes an existing point must not append a second copy."""
    _write(m, "e1", [0.6, 1.0])
    merged = m._merge({"rows": [{"beta": 1.0}, {"beta": 2.5}]}, "e1", "beta")
    assert [r["beta"] for r in merged["rows"]] == [0.6, 1.0, 2.5]


def test_merge_preserves_saved_rows_over_partial(m):
    """Previously saved rows survive even when the new partial lacks them."""
    _write(m, "e2", [250, 500, 1000], key="n_train")
    merged = m._merge({"rows": [{"n_train": 2000}]}, "e2", "n_train")
    assert [r["n_train"] for r in merged["rows"]] == [250, 500, 1000, 2000]


def test_merge_carries_non_row_fields(m):
    _write(m, "e2", [250], key="n_train")
    merged = m._merge(
        {"rows": [{"n_train": 500}], "loglog_slope_epistemic_vs_n": -0.9,
         "complete": True},
        "e2", "n_train",
    )
    assert merged["loglog_slope_epistemic_vs_n"] == -0.9
    assert merged["complete"] is True


def test_round_trip_resume_reaches_the_full_sweep(m):
    """Simulate two interruptions: the union must be the whole sweep, in order."""
    wanted = (0.6, 1.0, 1.6, 2.5, 4.0)
    for _ in range(10):
        todo = m._remaining(wanted, "e1", "beta")
        if not todo:
            break
        # one point completes, then the process "dies"
        part = {"rows": [{"beta": todo[0]}], "complete": False}
        (m.RESULTS / "e1.json").write_text(
            json.dumps(m._merge(part, "e1", "beta")), encoding="utf-8"
        )
    saved = json.loads((m.RESULTS / "e1.json").read_text())
    assert [r["beta"] for r in saved["rows"]] == list(wanted)
