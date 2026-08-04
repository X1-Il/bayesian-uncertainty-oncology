"""The decomposition identity is a theorem. These tests hold it to that."""

import numpy as np
import pytest

from cancer_unc.uncertainty import decompose, entropy


def _probs(p1):
    """(N, T, 2) from an (N, T) array of positive-class probabilities."""
    p1 = np.asarray(p1, dtype=float)
    return np.stack([1 - p1, p1], axis=-1)


def test_identity_holds_exactly():
    """total = epistemic + aleatoric, to floating-point precision."""
    rng = np.random.default_rng(0)
    mp = _probs(rng.uniform(0.01, 0.99, size=(500, 16)))
    d = decompose(mp)
    np.testing.assert_allclose(d["total"], d["epistemic"] + d["aleatoric"], atol=1e-12)


def test_epistemic_is_zero_when_members_agree():
    """No disagreement between posterior samples => no epistemic uncertainty.

    Crucially the members here all agree on a *maximally uncertain* prediction
    (p=0.5). Total uncertainty is at its maximum, yet epistemic must be exactly
    zero -- this is the case that separates a real decomposition from a
    monotone rescaling of total uncertainty.
    """
    mp = _probs(np.full((100, 20), 0.5))
    d = decompose(mp)
    assert np.allclose(d["epistemic"], 0.0, atol=1e-12)
    assert np.allclose(d["aleatoric"], np.log(2), atol=1e-12)
    assert np.allclose(d["total"], np.log(2), atol=1e-12)


def test_epistemic_is_maximal_when_members_maximally_disagree():
    """Half the members certain of class 0, half certain of class 1.

    Each member is individually certain (zero aleatoric), but the posterior
    predictive is uniform -- so all of the uncertainty is epistemic.
    """
    p1 = np.zeros((50, 20))
    p1[:, 10:] = 1.0
    d = decompose(_probs(np.clip(p1, 1e-12, 1 - 1e-12)))
    assert np.allclose(d["aleatoric"], 0.0, atol=1e-9)
    assert np.allclose(d["epistemic"], np.log(2), atol=1e-9)


def test_both_terms_non_negative():
    rng = np.random.default_rng(1)
    for _ in range(20):
        mp = _probs(rng.beta(0.3, 0.3, size=(200, 8)))  # heavy at the extremes
        d = decompose(mp)
        assert d["epistemic"].min() >= 0.0
        assert d["aleatoric"].min() >= 0.0


def test_entropy_bounded_by_log_c():
    rng = np.random.default_rng(2)
    mp = _probs(rng.uniform(0, 1, size=(300, 12)))
    d = decompose(mp)
    assert d["total"].max() <= np.log(2) + 1e-9


def test_variance_decomposition_matches_law_of_total_variance():
    """E[Var(p|w)] + Var[E(p|w)] must equal the total Bernoulli variance."""
    rng = np.random.default_rng(3)
    p1 = rng.uniform(0.05, 0.95, size=(400, 24))
    d = decompose(_probs(p1))

    p_bar = p1.mean(axis=1)
    total_var = p_bar * (1 - p_bar)
    np.testing.assert_allclose(
        d["var_aleatoric"] + d["var_epistemic"], total_var, atol=1e-12
    )


def test_entropy_matches_scipy_convention():
    p = np.array([[0.25, 0.75], [0.5, 0.5]])
    expected = np.array([
        -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)),
        np.log(2),
    ])
    np.testing.assert_allclose(entropy(p), expected, atol=1e-12)


def test_more_members_does_not_decrease_expected_epistemic():
    """The O(1/T) bias is downward: small T under-estimates epistemic.

    Averaging over many random draws, the T=32 estimate should exceed the T=2
    estimate. This is the bias that `mc_convergence` exists to expose, so if it
    ever reverses, the convergence diagnostic is meaningless.
    """
    rng = np.random.default_rng(4)
    # a genuinely dispersed posterior, so there is epistemic mass to under-count
    p1 = rng.beta(0.5, 0.5, size=(2000, 32))
    small = decompose(_probs(p1[:, :2]))["epistemic"].mean()
    large = decompose(_probs(p1))["epistemic"].mean()
    assert large > small
