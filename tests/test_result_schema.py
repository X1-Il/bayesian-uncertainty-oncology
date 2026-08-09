"""The `_`-prefix convention for result payloads.

Saved results interleave data with metadata: `save()` stamps `_preset` into every
payload, `exp_ood` adds `_shift` and `_semantic`, `exp_selective` adds
`_curve_msp` and `_semantic_split`. Consumers must therefore filter on the
prefix rather than assume every key is data.

Three separate bugs came from ignoring this -- a crash after 90 minutes of E5,
and two silent failures where `list(res.keys())[0]` picked up the metadata string
and iterated its characters. These tests encode the contract so the next
consumer does not rediscover it the same way.
"""

import numpy as np
import pytest

from cancer_unc.plots import plot_ood


def _fake_ood_result(with_preset: bool) -> dict:
    scores = ("msp", "entropy", "energy", "epistemic", "aleatoric", "mahalanobis")
    shift = {s: {"auroc": 0.6, "aupr_out": 0.6, "fpr@95tpr": 0.5} for s in scores}
    res = {}
    if with_preset:
        res["_preset"] = "full"  # exactly what save() prepends
    for name, semantic in (("noise_3", False), ("decoupled", True)):
        res[name] = {**{k: dict(v) for k, v in shift.items()},
                     "_shift": name, "_semantic": semantic}
    return res


def test_shift_extraction_ignores_metadata_keys():
    res = _fake_ood_result(with_preset=True)
    shifts = [k for k in res if not k.startswith("_")]
    assert shifts == ["noise_3", "decoupled"]
    # the naive version picks the metadata string instead
    assert list(res.keys())[0] == "_preset"


def test_plot_ood_survives_the_preset_stamp(tmp_path):
    """Regression: plot_ood crashed with 'string indices must be integers'
    once every payload gained a `_preset` key."""
    out = plot_ood(_fake_ood_result(with_preset=True), tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_plot_ood_matches_with_and_without_metadata(tmp_path):
    """The stamp must not change what is plotted, only what is filtered out."""
    a = plot_ood(_fake_ood_result(with_preset=True), tmp_path / "a")
    b = plot_ood(_fake_ood_result(with_preset=False), tmp_path / "b")
    assert a.exists() and b.exists()


def test_make_tables_shift_extraction(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_mt_under_test", root / "paper" / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    sys.modules["_mt_under_test"] = mt
    spec.loader.exec_module(mt)

    tex = mt.tab_e4(_fake_ood_result(with_preset=True))
    assert "_preset" not in tex
    assert "noise" in tex and "decoupled" in tex


def test_selective_metadata_is_not_a_metric(tmp_path):
    """`_curve_msp` and `_semantic_split` are not metric dicts."""
    from cancer_unc.eval import compare_confidence_functions

    rng = np.random.default_rng(0)
    correct = rng.random(150) < 0.7
    res = compare_confidence_functions({"msp": rng.random(150)}, correct)
    res["_curve_msp"] = {"coverage": [0.5], "risk": [0.1]}
    res["_semantic_split"] = "decoupled"

    metrics = {k: v for k, v in res.items() if not k.startswith("_")}
    assert set(metrics) == {"msp"}
    assert all("aurc" in v for v in metrics.values())
