"""The phantom generator and its oracle quantities."""

from dataclasses import replace

import numpy as np
import pytest

from cancer_unc.data import (
    PhantomConfig,
    SHIFTS,
    bayes_posterior,
    binary_entropy,
    make_benchmark,
    make_split,
    oracle_stats,
)


def test_images_are_valid():
    s = make_split(64, PhantomConfig(), seed=0)
    assert s.images.shape == (64, 1, 64, 64)
    assert s.images.dtype == np.float32
    assert s.images.min() >= 0.0 and s.images.max() <= 1.0
    assert set(np.unique(s.labels)) <= {0, 1}


def test_labels_follow_the_stated_link_function():
    """The empirical label rate must match sigma(beta z) within sampling error.

    This is what makes the oracle trustworthy: if the sampler and the analytic
    posterior ever disagreed, every E1/E2 target would be silently wrong.
    """
    cfg = PhantomConfig()
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200_000)
    p = bayes_posterior(z, cfg)
    y = rng.random(200_000) < p

    for lo, hi in [(-2, -1), (-1, 0), (0, 1), (1, 2)]:
        m = (z >= lo) & (z < hi)
        assert abs(y[m].mean() - p[m].mean()) < 0.01


def test_oracle_stats_match_closed_form():
    """Bayes error for a logistic link, checked against direct integration."""
    cfg = PhantomConfig()
    o = oracle_stats(cfg, n=400_000, seed=1)

    rng = np.random.default_rng(99)
    z = rng.standard_normal(400_000)
    p = bayes_posterior(z, cfg)
    assert abs(o["bayes_error"] - np.minimum(p, 1 - p).mean()) < 2e-3
    assert abs(o["aleatoric_entropy"] - binary_entropy(p).mean()) < 2e-3
    # for a Bernoulli, expected NLL under the true posterior *is* its entropy
    assert abs(o["bayes_nll"] - o["aleatoric_entropy"]) < 2e-3


def test_beta_controls_irreducible_noise_monotonically():
    """Higher beta => sharper link => less aleatoric entropy and lower Bayes error.

    E1 depends on this being a genuine monotone dial, not an incidental one.
    """
    prev_h, prev_e = np.inf, np.inf
    for beta in (0.5, 1.0, 2.0, 4.0, 8.0):
        o = oracle_stats(replace(PhantomConfig(), beta=beta), n=100_000)
        assert o["aleatoric_entropy"] < prev_h
        assert o["bayes_error"] < prev_e
        prev_h, prev_e = o["aleatoric_entropy"], o["bayes_error"]


def test_epistemic_target_is_zero():
    assert oracle_stats(PhantomConfig(), n=10_000)["epistemic_entropy"] == 0.0


def test_lesion_contrast_tracks_the_latent():
    """Positive z must render brighter than negative z, on average.

    If the sign convention were broken, the label would be uncorrelated with
    the image and the whole benchmark would be noise -- which a model would
    quietly report as very high aleatoric uncertainty rather than as an error.
    """
    cfg = PhantomConfig()
    rng = np.random.default_rng(0)
    from cancer_unc.data.synthetic import render_one

    hi = np.mean([render_one(2.0, cfg, rng).max() for _ in range(40)])
    lo = np.mean([render_one(-2.0, cfg, rng).max() for _ in range(40)])
    assert hi > lo


def test_splits_do_not_share_phantoms():
    """Train/val/test must be disjoint draws, or every held-out number is a lie.

    Compared on whole-image bytes rather than a pixel prefix: the corners are
    air (exactly 0.0) in every phantom, so any prefix-based check passes
    vacuously.
    """
    b = make_benchmark(PhantomConfig(), n_train=200, n_val=200, n_test=200,
                       n_ood=100, shifts=("novel",))
    digests = {}
    for name in ("train", "val", "test"):
        digests[name] = {img.tobytes() for img in b[name].images}
        assert len(digests[name]) == len(b[name].images), f"duplicates within {name}"

    assert not (digests["train"] & digests["test"])
    assert not (digests["train"] & digests["val"])
    assert not (digests["val"] & digests["test"])


def test_shifts_actually_shift_the_distribution():
    cfg = PhantomConfig()
    base = make_split(200, cfg, seed=7, shift=SHIFTS["id"])
    for key in ("noise_3", "blur_3", "modality", "novel"):
        other = make_split(200, cfg, seed=7, shift=SHIFTS[key])
        # same seed, same latents -- so any difference is due to the shift alone
        assert not np.allclose(base.images, other.images)
        np.testing.assert_allclose(base.z, other.z)


def test_noise_free_labels_are_deterministic_given_z():
    s = make_split(500, PhantomConfig(), seed=3, resample_labels=False)
    np.testing.assert_array_equal(s.labels, (s.p_true > 0.5).astype(np.int64))


def test_only_decoupled_is_flagged_semantic():
    """The covariate/semantic distinction drives how E4 is interpreted.

    `novel` is deliberately NOT semantic: it changes the lesion's shape but
    still encodes the latent in its signed amplitude, so the label stays
    recoverable. Only `decoupled` severs the image/label link.
    """
    assert SHIFTS["decoupled"].semantic
    for key in ("noise_1", "noise_3", "blur_2", "modality", "novel"):
        assert not SHIFTS[key].semantic


def test_decoupled_shift_destroys_the_image_label_link():
    """The defining property: lesion contrast must be independent of z.

    Under `novel` the rendered contrast still tracks z (that is exactly why it
    is not a semantic shift). Under `decoupled` it must not, or the split does
    not test what E4 claims it tests.
    """
    cfg = PhantomConfig()
    n = 400
    from cancer_unc.data import matched_filter_statistic

    for key, expect_link in (("novel", True), ("decoupled", False)):
        s = make_split(n, cfg, seed=4242, shift=SHIFTS[key])
        stat = matched_filter_statistic(s.images, cfg)
        r = abs(np.corrcoef(stat, s.z)[0, 1])
        if expect_link:
            assert r > 0.4, f"{key}: expected contrast to track z, got r={r:.3f}"
        else:
            assert r < 0.2, f"{key}: contrast still tracks z (r={r:.3f}); the "
    # under `decoupled` the label is a coin flip given the image
    s = make_split(2000, cfg, seed=99, shift=SHIFTS["decoupled"])
    assert 0.4 < s.labels.mean() < 0.6
