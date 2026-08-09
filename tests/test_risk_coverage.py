"""Selective prediction. The invariance test here is the load-bearing one."""

import numpy as np

from cancer_unc.eval import (
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


def test_full_coverage_risk_equals_error_rate():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 1000)
    correct = rng.random(1000) < 0.8
    curve = risk_coverage_curve(conf, correct)
    assert abs(curve.risk[-1] - (~correct).mean()) < 1e-12
    assert abs(curve.coverage[-1] - 1.0) < 1e-12


def test_perfect_confidence_attains_optimal_aurc():
    """A confidence function that ranks every correct case above every
    incorrect one is the oracle, so its excess AURC must be zero."""
    correct = np.array([True] * 700 + [False] * 300)
    conf = correct.astype(float)  # perfectly informative
    curve = risk_coverage_curve(conf, correct)
    assert abs(curve.excess_aurc) < 1e-12
    assert abs(curve.aurc - curve.aurc_optimal) < 1e-12


def test_aurc_invariant_under_monotone_transform():
    """AURC depends only on the *ranking*, so temperature scaling cannot move it.

    This is the formal content of the warning in the module docstring: a
    reported AURC 'improvement' from temperature scaling is necessarily a bug.
    Any strictly increasing map of the confidence must leave the curve fixed.
    """
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.01, 0.99, 2000)
    correct = rng.random(2000) < conf

    base = risk_coverage_curve(conf, correct)
    for transform in (
        lambda c: c / 2.5,                       # temperature-like
        lambda c: np.sqrt(c),
        lambda c: 1 / (1 + np.exp(-4 * (c - 0.5))),
    ):
        moved = risk_coverage_curve(transform(conf), correct)
        np.testing.assert_allclose(moved.risk, base.risk, atol=1e-12)
        assert abs(moved.aurc - base.aurc) < 1e-12


def test_random_confidence_gives_flat_risk():
    """Uninformative confidence => selective risk ~ constant at the error rate."""
    rng = np.random.default_rng(2)
    correct = rng.random(20000) < 0.75
    conf = rng.uniform(0, 1, 20000)  # independent of correctness
    curve = risk_coverage_curve(conf, correct)
    mid = curve.risk[len(curve.risk) // 2 :]
    assert abs(mid.mean() - 0.25) < 0.02


def test_risk_decreases_with_informative_confidence():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 1.0, 5000)
    correct = rng.random(5000) < conf
    curve = risk_coverage_curve(conf, correct)
    assert risk_at_coverage(curve, 0.5) < risk_at_coverage(curve, 1.0)


def test_coverage_at_risk_returns_largest_feasible_coverage():
    """Non-monotone risk curves make the naive bisection wrong; check we take
    the largest feasible coverage, not the first crossing."""
    correct = np.array([True] * 90 + [False] * 10)
    conf = np.linspace(1.0, 0.0, 100)
    curve = risk_coverage_curve(conf, correct)
    got = coverage_at_risk(curve, 0.05)
    assert got["risk"] <= 0.05 + 1e-12
    assert got["coverage"] >= 0.9 - 1e-9


def test_coverage_at_unreachable_risk_is_zero():
    correct = np.zeros(100, dtype=bool)  # everything wrong
    conf = np.linspace(0, 1, 100)
    curve = risk_coverage_curve(conf, correct)
    assert coverage_at_risk(curve, 0.01)["coverage"] == 0.0


def test_metadata_keys_are_distinguishable_from_metrics():
    """Result dicts mix metric entries with `_`-prefixed metadata.

    Every consumer -- printing, plotting, table generation -- must filter on
    that prefix. A loop that assumed all values were metric dicts crashed E5
    after ~90 minutes of compute, so the convention is pinned here.
    """
    from cancer_unc.eval import compare_confidence_functions

    rng = np.random.default_rng(0)
    correct = rng.random(200) < 0.7
    res = compare_confidence_functions({"a": rng.random(200)}, correct)
    res["_meta"] = "decoupled"

    metrics = {k: v for k, v in res.items() if not k.startswith("_")}
    assert set(metrics) == {"a"}
    for v in metrics.values():
        assert isinstance(v, dict) and "aurc" in v
    assert isinstance(res["_meta"], str)


def test_custom_loss_is_respected():
    """Cost-sensitive losses (false negatives costlier) must change the curve."""
    correct = np.array([True, False, True, False] * 50)
    conf = np.linspace(1, 0, 200)
    zero_one = risk_coverage_curve(conf, correct)
    weighted = risk_coverage_curve(conf, correct, loss=(~correct).astype(float) * 5.0)
    assert weighted.aurc > zero_one.aurc
