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


def test_mean_of_softmax_differs_from_softmax_of_mean():
    """The premise behind EnsembleTemperatureScaler.

    If these two collapsed to the same thing, scaling the averaged logit would
    be fine. They do not, which is why the ensemble needs its own scaler.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(10)
    z = torch.as_tensor(rng.normal(0, 3, size=(500, 4, 2)), dtype=torch.float32)
    mean_of_softmax = F.softmax(z, dim=-1).mean(dim=1)
    softmax_of_mean = F.softmax(z.mean(dim=1), dim=-1)
    assert not torch.allclose(mean_of_softmax, softmax_of_mean, atol=1e-3)


def test_ensemble_scaler_calibrates_the_mixture():
    """Fit on held-out data, check ECE improves on a disjoint half."""
    from cancer_unc.uncertainty import EnsembleTemperatureScaler

    rng = np.random.default_rng(11)
    n, m = 8000, 5
    p1 = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p1).astype(np.int64)
    true_logit = np.log(p1 / (1 - p1))
    # overconfident members, each with its own jitter
    member_logits = np.stack(
        [np.stack([np.zeros(n), 3.0 * true_logit + rng.normal(0, 0.3, n)], axis=1)
         for _ in range(m)], axis=1,
    )

    half = n // 2
    sc = EnsembleTemperatureScaler().fit(member_logits[:half], y[:half])

    def mixture(z):
        e = np.exp(z - z.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)).mean(axis=1)

    raw = mixture(member_logits[half:])
    cal = sc.transform(member_logits[half:])
    assert ece(cal, y[half:]) < ece(raw, y[half:])
    assert 1.5 < sc.temperature < 5.0


def test_ensemble_scaling_may_move_accuracy_slightly():
    """The single-model argmax theorem does NOT extend to a mixture.

    This test documents the failure rather than hiding it: the shift should be
    small, but demanding it be exactly zero is what made `exp_calibration`
    abort on a correct result.
    """
    from cancer_unc.uncertainty import EnsembleTemperatureScaler

    rng = np.random.default_rng(12)
    n, m = 4000, 5
    member_logits = rng.normal(0, 2.5, size=(n, m, 2))
    y = (member_logits[:, :, 1].mean(1) + rng.normal(0, 1, n) > 0).astype(np.int64)

    sc = EnsembleTemperatureScaler().fit(member_logits, y)
    shift = sc.accuracy_shift(member_logits, y)
    assert abs(shift) < 0.05, f"shift {shift} implausibly large"


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
