"""OOD ranking metrics."""

import numpy as np

from cancer_unc.uncertainty import MahalanobisScorer, aupr, auroc, fpr_at_tpr


def test_identical_distributions_give_chance_auroc():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert abs(auroc(a, b) - 0.5) < 0.02


def test_perfect_separation_gives_one():
    assert auroc(np.zeros(100), np.ones(100)) == 1.0


def test_perfect_inversion_gives_zero():
    assert auroc(np.ones(100), np.zeros(100)) == 0.0


def test_all_ties_give_exactly_half():
    """Saturated scores (e.g. msp pinned at 1.0) are all ties. Without the 0.5
    tie correction a naive threshold sweep reports 1.0 here, which would make a
    degenerate score look perfect."""
    assert auroc(np.ones(500), np.ones(500)) == 0.5


def test_auroc_is_rank_based():
    """Invariant to any strictly monotone rescaling of the scores."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 1000), rng.normal(1, 1, 1000)
    base = auroc(a, b)
    for f in (np.exp, lambda x: 3 * x - 7, lambda x: np.tanh(x / 2)):
        assert abs(auroc(f(a), f(b)) - base) < 1e-9


def test_fpr_at_tpr_bounds():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 4000), rng.normal(4, 1, 4000)
    v = fpr_at_tpr(a, b, 0.95)
    assert 0.0 <= v <= 1.0
    assert v < 0.1  # well separated


def test_fpr_at_tpr_is_one_when_inseparable():
    rng = np.random.default_rng(3)
    a = rng.normal(size=3000)
    assert fpr_at_tpr(a, rng.normal(size=3000), 0.95) > 0.85


def test_aupr_beats_chance_when_separated():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)
    assert aupr(a, b) > 0.9


def test_mahalanobis_scores_outliers_higher():
    rng = np.random.default_rng(5)
    d = 16
    f0 = rng.normal(0, 1, size=(500, d))
    f1 = rng.normal(3, 1, size=(500, d))
    feats = np.vstack([f0, f1])
    labels = np.array([0] * 500 + [1] * 500)

    scorer = MahalanobisScorer().fit(feats, labels)
    far = rng.normal(20, 1, size=(200, d))
    assert scorer.score(far).mean() > scorer.score(feats).mean()
    assert auroc(scorer.score(feats), scorer.score(far)) > 0.95


def test_mahalanobis_precision_is_well_conditioned():
    """Shrinkage must keep the precision matrix usable when D > N per class,
    which is the regime that makes untied/unshrunk covariance degenerate."""
    rng = np.random.default_rng(6)
    feats = rng.normal(size=(40, 128))  # far fewer samples than dimensions
    labels = (rng.random(40) < 0.5).astype(int)
    scorer = MahalanobisScorer(shrinkage=0.1).fit(feats, labels)
    s = scorer.score(feats)
    assert np.all(np.isfinite(s))
