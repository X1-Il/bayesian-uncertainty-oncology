"""Calibration metrics and post-hoc scaling."""

import numpy as np
import pytest

from cancer_unc.uncertainty import (
    TemperatureScaler,
    VectorScaler,
    adaptive_ece,
    brier_decomposition,
    ece,
    nll,
)


def _calibrated_sample(n=40000, seed=0):
    """Draw labels from the model's own stated probabilities.

    By construction this sample is *perfectly* calibrated, so any ECE it shows
    is pure estimator bias plus sampling noise -- which is exactly the quantity
    worth knowing before interpreting an ECE on real predictions.
    """
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p1).astype(np.int64)
    return np.stack([1 - p1, p1], axis=1), y


def test_perfectly_calibrated_has_near_zero_ece():
    probs, y = _calibrated_sample()
    assert ece(probs, y, n_bins=15) < 0.02
    assert adaptive_ece(probs, y, n_bins=15) < 0.02


def test_overconfident_model_has_large_ece():
    """Sharpen a calibrated model's probabilities: calibration must degrade."""
    probs, y = _calibrated_sample()
    logit = np.log(probs[:, 1] / probs[:, 0])
    sharp = 1 / (1 + np.exp(-3.0 * logit))  # temperature 1/3
    probs_sharp = np.stack([1 - sharp, sharp], axis=1)
    assert ece(probs_sharp, y) > 5 * ece(probs, y)


def test_temperature_scaling_preserves_accuracy_exactly():
    """The argmax-invariance theorem, tested rather than asserted in prose.

    Dividing every logit by the same T > 0 is strictly monotone, so it cannot
    reorder classes. If this ever fails, `exp_calibration`'s assertion would
    fire too -- both exist because a calibration method that quietly changed
    accuracy would invalidate the entire comparison.
    """
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, size=(3000, 2))
    y = (rng.random(3000) < 0.5).astype(np.int64)

    scaler = TemperatureScaler().fit(logits, y)
    before = logits.argmax(1)
    after = scaler.transform(logits).argmax(1)
    assert np.array_equal(before, after)
    assert scaler.temperature > 0


def test_temperature_scaling_fixes_overconfidence():
    """Fit on held-out data, evaluate on a disjoint set."""
    rng = np.random.default_rng(2)
    n = 20000
    p1 = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p1).astype(np.int64)
    logits = np.stack([np.zeros(n), 3.0 * np.log(p1 / (1 - p1))], axis=1)  # overconfident

    half = n // 2
    scaler = TemperatureScaler().fit(logits[:half], y[:half])

    raw = np.exp(logits[half:]) / np.exp(logits[half:]).sum(1, keepdims=True)
    cal = scaler.transform(logits[half:])
    assert ece(cal, y[half:]) < ece(raw, y[half:])
    assert nll(cal, y[half:]) < nll(raw, y[half:])
    # should recover T ~ 3
    assert 2.0 < scaler.temperature < 4.5


def test_brier_decomposition_sums_to_brier():
    """reliability - resolution + uncertainty == Brier score."""
    probs, y = _calibrated_sample(n=20000, seed=3)
    d = brier_decomposition(probs, y, n_bins=20)
    # the identity is exact only up to within-bin discretisation
    assert abs(d["brier_from_decomposition"] - d["brier_direct"]) < 5e-3
    assert d["reliability"] >= 0
    assert d["resolution"] >= 0


def test_adaptive_ece_differs_on_skewed_confidence():
    """Equal-width and equal-mass binning must disagree for a confident model.

    If they agreed, `adaptive_ece` would be redundant. They disagree precisely
    because equal-width bins leave the high-confidence region under-resolved,
    which is the failure mode it exists to fix.
    """
    rng = np.random.default_rng(4)
    n = 20000
    p1 = rng.beta(0.15, 0.15, n)  # mass piled at 0 and 1
    y = (rng.random(n) < np.clip(p1 * 0.85 + 0.075, 0, 1)).astype(np.int64)
    probs = np.stack([1 - p1, p1], axis=1)
    assert abs(ece(probs, y) - adaptive_ece(probs, y)) > 1e-4


def test_bootstrap_with_zero_resamples_returns_point_estimate():
    """The sweeps pass n_boot=0 to skip resampling. Regression test: this used
    to take a quantile of an empty array and crash partway through E1."""
    from cancer_unc.uncertainty import bootstrap_ci, calibration_report

    probs, y = _calibrated_sample(n=1000, seed=6)
    point, lo, hi = bootstrap_ci(ece, probs, y, n_boot=0)
    assert point == lo == hi
    rep = calibration_report(probs, y, n_boot=0)
    assert np.isfinite(rep["ece"]) and np.isfinite(rep["nll"])


def test_vector_scaling_may_change_accuracy():
    """Documented contrast with temperature scaling: more parameters, fewer
    guarantees. This is why temperature scaling is the default."""
    rng = np.random.default_rng(5)
    logits = rng.normal(0, 2, size=(4000, 2))
    y = (logits[:, 1] + rng.normal(0, 3, 4000) > 0).astype(np.int64)
    vs = VectorScaler().fit(logits, y)
    assert vs.transform(logits).shape == (4000, 2)  # runs; argmax not guaranteed
