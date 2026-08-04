"""Out-of-distribution detection scores and their evaluation.

Framing
-------
OOD detection is a *ranking* problem, not a classification one: we score every
input by how in-distribution it looks and ask whether ID inputs rank above OOD
inputs. That is why AUROC (a rank statistic, threshold-free) is the headline
metric -- it avoids committing to a threshold that would depend on an assumed
OOD prevalence nobody knows.

The scores implemented here differ in what evidence they use, and the point of
the comparison is that the *Bayesian* score uses evidence the others cannot see:

  msp           max softmax probability (Hendrycks & Gimpel 2017). The baseline.
                Uses only the point prediction, so a confidently-wrong network
                is invisible to it.
  entropy       H of the posterior predictive. Uses the whole distribution, but
                still conflates "ambiguous" with "unfamiliar".
  energy        -logsumexp(z) (Liu et al. 2020). Proportional to the unnormalised
                log-density the classifier implicitly assigns to x, so unlike
                softmax scores it is not destroyed by the normalisation that
                throws away logit magnitude.
  epistemic     the mutual information I(y; w | x) from the decomposition. This
                is the score that should win on *semantic* novelty: an unfamiliar
                input is one the posterior disagrees about, which is exactly what
                MI measures. It is also the only score here that is near-zero for
                an input that is merely ambiguous, since all members agree it is
                ambiguous.
  mahalanobis   distance in penultimate feature space to the nearest
                class-conditional Gaussian under a shared covariance
                (Lee et al. 2018). A density estimate on features, so it can flag
                inputs the classifier head is oblivious to.

The distinction the evaluation preserves is *covariate* shift (noise, blur,
scanner) versus *semantic* shift (novel lesion morphology). On covariate shift
the right behaviour is to stay calibrated and keep predicting; on semantic shift
it is to abstain. Averaging them into one "OOD AUROC" hides the only interesting
result, so `ood_report` keeps them separate.

Sign convention: every score below is oriented so that **higher means more OOD**.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from .decomposition import UncertaintyOutput, entropy

EPS = 1e-12


# --------------------------------------------------------------------------
# scores (higher = more out-of-distribution)
# --------------------------------------------------------------------------
def score_msp(out: UncertaintyOutput) -> np.ndarray:
    return -out.probs.max(axis=-1)


def score_entropy(out: UncertaintyOutput) -> np.ndarray:
    return entropy(out.probs)


def score_energy(out: UncertaintyOutput, temperature: float = 1.0) -> np.ndarray:
    """E(x) = -T logsumexp(z / T); high energy = low implicit density."""
    return -temperature * logsumexp(out.logits / temperature, axis=-1)


def score_epistemic(out: UncertaintyOutput) -> np.ndarray:
    return out.epistemic


def score_aleatoric(out: UncertaintyOutput) -> np.ndarray:
    """Reported as a *negative control*. Aleatoric uncertainty should NOT
    detect semantic novelty -- if it does, the decomposition is leaking."""
    return out.aleatoric


class MahalanobisScorer:
    """Class-conditional Gaussians with a shared (tied) covariance.

    Tied covariance is not a convenience: with D-dimensional features and only a
    few thousand training points, per-class covariances are rank-deficient and
    the score degenerates. Shrinkage toward a scaled identity is applied on top,
    which keeps the precision matrix well conditioned.
    """

    def __init__(self, shrinkage: float = 0.05):
        self.shrinkage = shrinkage
        self.means_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MahalanobisScorer":
        classes = np.unique(labels)
        d = features.shape[1]
        means, centred = [], []
        for c in classes:
            f = features[labels == c]
            mu = f.mean(axis=0)
            means.append(mu)
            centred.append(f - mu)
        self.means_ = np.stack(means)

        x = np.concatenate(centred, axis=0)
        cov = (x.T @ x) / max(len(x) - len(classes), 1)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.trace(cov) / d * np.eye(d)
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        assert self.means_ is not None, "call fit() first"
        dists = []
        for mu in self.means_:
            delta = features - mu
            dists.append(np.einsum("ij,jk,ik->i", delta, self.precision_, delta))
        return np.min(np.stack(dists, axis=1), axis=1)  # nearest class


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U identity, with correct tie handling.

    AUROC = P(score_out > score_in) + 0.5 P(score_out = score_in). Using ranks
    rather than a threshold sweep makes this exact and O(n log n); the 0.5
    tie term matters because several scores here saturate (msp pinned at 1.0)
    and a naive implementation silently rewards that saturation.
    """
    n_out, n_in = len(scores_out), len(scores_in)
    if n_out == 0 or n_in == 0:
        return float("nan")
    all_s = np.concatenate([scores_out, scores_in])
    r = rankdata(all_s)  # average ranks => ties handled
    u = r[:n_out].sum() - n_out * (n_out + 1) / 2
    return float(u / (n_out * n_in))


def aupr(scores_in: np.ndarray, scores_out: np.ndarray, positive: str = "out") -> float:
    """Average precision treating `positive` as the positive class."""
    if positive == "out":
        s = np.concatenate([scores_out, scores_in])
        y = np.concatenate([np.ones(len(scores_out)), np.zeros(len(scores_in))])
    else:
        s = np.concatenate([-scores_out, -scores_in])
        y = np.concatenate([np.zeros(len(scores_out)), np.ones(len(scores_in))])

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return float((precision * y).sum() / n_pos) if n_pos > 0 else float("nan")


def fpr_at_tpr(scores_in: np.ndarray, scores_out: np.ndarray, tpr: float = 0.95) -> float:
    """FPR on ID data at the threshold achieving `tpr` recall on OOD data.

    The operationally honest number: "to catch 95% of the scans my model has no
    business judging, what fraction of normal scans do I needlessly flag?"
    AUROC can look excellent while this is unusable.
    """
    if len(scores_in) == 0 or len(scores_out) == 0:
        return float("nan")
    thresh = np.quantile(scores_out, 1.0 - tpr)  # detect OOD when score >= thresh
    return float((scores_in >= thresh).mean())


def ood_report(
    out_id: UncertaintyOutput,
    out_ood: UncertaintyOutput,
    mahalanobis: MahalanobisScorer | None = None,
) -> dict[str, dict[str, float]]:
    """AUROC / AUPR / FPR@95 for every score, on one ID-vs-OOD pair."""
    scorers = {
        "msp": score_msp,
        "entropy": score_entropy,
        "energy": score_energy,
        "epistemic": score_epistemic,
        "aleatoric": score_aleatoric,  # negative control
    }
    report: dict[str, dict[str, float]] = {}
    for name, fn in scorers.items():
        s_in, s_out = fn(out_id), fn(out_ood)
        report[name] = {
            "auroc": auroc(s_in, s_out),
            "aupr_out": aupr(s_in, s_out),
            "fpr@95tpr": fpr_at_tpr(s_in, s_out),
        }

    if mahalanobis is not None and out_id.features is not None and out_ood.features is not None:
        s_in = mahalanobis.score(out_id.features)
        s_out = mahalanobis.score(out_ood.features)
        report["mahalanobis"] = {
            "auroc": auroc(s_in, s_out),
            "aupr_out": aupr(s_in, s_out),
            "fpr@95tpr": fpr_at_tpr(s_in, s_out),
        }
    return report
