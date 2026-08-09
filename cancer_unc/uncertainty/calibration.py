"""Calibration: measurement, decomposition, and post-hoc correction.

Definition
----------
A probabilistic classifier is *perfectly calibrated* if

    P( y = yhat | conf(x) = c ) = c    for all c in [0, 1].

Among all scans the model calls malignant with 80% confidence, 80% should be
malignant. This is exactly the property a clinician needs in order to use the
number as a risk, and it is orthogonal to accuracy: a model can be highly
accurate and badly calibrated, or useless and perfectly calibrated (predict the
base rate every time).

Estimating ECE
--------------
The definition conditions on a continuous quantity, which is unobservable from
a finite sample, so we bin. With bins B_1..B_M,

    ECE = sum_m (|B_m| / n) | acc(B_m) - conf(B_m) |.

Three things about this estimator that reported ECE numbers usually gloss over,
and which this module makes visible:

1. **It is biased.** Within-bin variation is averaged away, so binned ECE
   *under*-estimates the true calibration error; the bias shrinks as bins
   narrow but the variance grows. There is no unbiased choice of M.
2. **It is bin-scheme dependent.** Equal-width bins leave the high-confidence
   bins nearly empty for a confident model -- which is precisely where the
   clinically relevant errors live. `adaptive_ece` uses equal-*mass* bins so
   every bin carries the same weight of evidence.
3. **It has real sampling variance.** An ECE of 0.02 vs 0.03 on n=2000 is
   usually noise. `bootstrap_ci` reports an interval, and comparisons in this
   project are made on intervals, not point estimates.

Brier decomposition
-------------------
Murphy (1973) splits the Brier score into three interpretable terms:

    BS = reliability - resolution + uncertainty
       = E[(conf - acc)^2]  -  E[(acc - abar)^2]  +  abar(1 - abar)

*reliability* is calibration error (lower better), *resolution* is the ability
to separate classes (higher better), *uncertainty* is the irreducible variance
of the labels -- a property of the dataset that no model changes. Reading the
three separately shows whether a post-hoc fix bought calibration at the price of
sharpness, which a single scalar hides.

Temperature scaling
-------------------
Fit a single scalar T > 0 on held-out data and predict softmax(logits / T),
choosing T by minimising validation NLL.

Two facts make it the right default. First, **accuracy is provably unchanged**:
dividing by T > 0 is strictly monotone and applied identically to every logit,
so argmax_c z_c / T = argmax_c z_c for every input. Calibration is therefore
free -- there is no accuracy/calibration trade-off to negotiate. Second, with
one parameter it cannot meaningfully overfit a validation set of any reasonable
size, unlike vector or matrix scaling (C and C^2 parameters), which are provided
here for comparison and which do move accuracy.

The objective is convex in 1/T for the binary case, so the 1-D optimisation is
well behaved; we still fit on a *held-out* split, never on train or test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
@dataclass
class BinStats:
    """Per-bin reliability-diagram data."""

    edges: np.ndarray
    counts: np.ndarray
    conf: np.ndarray  # mean confidence in bin
    acc: np.ndarray  # empirical accuracy in bin


def _bin_stats(conf: np.ndarray, correct: np.ndarray, edges: np.ndarray) -> BinStats:
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
    m = len(edges) - 1
    counts = np.bincount(idx, minlength=m).astype(float)
    sum_conf = np.bincount(idx, weights=conf, minlength=m)
    sum_acc = np.bincount(idx, weights=correct.astype(float), minlength=m)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / np.maximum(counts, 1), np.nan)
        mean_acc = np.where(counts > 0, sum_acc / np.maximum(counts, 1), np.nan)
    return BinStats(edges, counts, mean_conf, mean_acc)


def reliability(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, adaptive: bool = False
) -> BinStats:
    """Reliability-diagram data for the top-label confidence."""
    conf = probs.max(axis=-1)
    correct = probs.argmax(axis=-1) == labels
    if adaptive:
        # equal-mass bins: quantiles of the confidence distribution
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(conf, qs))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    else:
        edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    return _bin_stats(conf, correct, edges)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins."""
    b = reliability(probs, labels, n_bins, adaptive=False)
    mask = b.counts > 0
    w = b.counts[mask] / b.counts.sum()
    return float((w * np.abs(b.acc[mask] - b.conf[mask])).sum())


def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-mass bins -- the estimator to trust for confident models."""
    b = reliability(probs, labels, n_bins, adaptive=True)
    mask = b.counts > 0
    w = b.counts[mask] / b.counts.sum()
    return float((w * np.abs(b.acc[mask] - b.conf[mask])).sum())


def mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Maximum calibration error -- the worst bin, i.e. the worst case a
    clinician could encounter, not the average one."""
    b = reliability(probs, labels, n_bins, adaptive=False)
    mask = b.counts > 0
    return float(np.abs(b.acc[mask] - b.conf[mask]).max()) if mask.any() else 0.0


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE averaged over every class's predicted probability, not just the top one.

    Top-label ECE ignores what the model says about the class it did *not*
    predict. For a two-class screening problem the malignant probability matters
    even when the prediction is benign, so this is the stricter and more
    relevant measure.
    """
    n, c = probs.shape
    total = 0.0
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    for k in range(c):
        b = _bin_stats(probs[:, k], (labels == k), edges)
        mask = b.counts > 0
        w = b.counts[mask] / b.counts.sum()
        total += float((w * np.abs(b.acc[mask] - b.conf[mask])).sum())
    return total / c


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)
    return float(-np.log(p).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_decomposition(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """Murphy's reliability / resolution / uncertainty split (binary, class 1)."""
    p = probs[:, 1]
    y = (labels == 1).astype(float)
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    n = len(y)
    ybar = y.mean()
    rel = res = 0.0
    for m in range(n_bins):
        sel = idx == m
        nk = sel.sum()
        if nk == 0:
            continue
        pk, yk = p[sel].mean(), y[sel].mean()
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - ybar) ** 2
    rel, res = rel / n, res / n
    unc = ybar * (1 - ybar)
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "brier_from_decomposition": float(rel - res + unc),
        "brier_direct": float(((p - y) ** 2).mean()),
    }


def bootstrap_ci(
    fn, probs: np.ndarray, labels: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap for any metric(probs, labels)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    point = fn(probs, labels)
    if n_boot <= 0:
        # Callers inside the sweeps pass n_boot=0 to skip resampling, which is
        # 500x the cost of the point estimate and is not used there. Return the
        # point estimate with a degenerate interval rather than pretending to
        # have quantified uncertainty we did not compute.
        return float(point), float(point), float(point)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = fn(probs[i], labels[i])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def calibration_report(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, n_boot: int = 500
) -> dict[str, float]:
    acc = float((probs.argmax(1) == labels).mean())
    e, e_lo, e_hi = bootstrap_ci(lambda p, y: ece(p, y, n_bins), probs, labels, n_boot)
    a, a_lo, a_hi = bootstrap_ci(
        lambda p, y: adaptive_ece(p, y, n_bins), probs, labels, n_boot
    )
    out = {
        "accuracy": acc,
        "ece": e,
        "ece_lo": e_lo,
        "ece_hi": e_hi,
        "adaptive_ece": a,
        "adaptive_ece_lo": a_lo,
        "adaptive_ece_hi": a_hi,
        "mce": mce(probs, labels, n_bins),
        "classwise_ece": classwise_ece(probs, labels, n_bins),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
    }
    out.update(brier_decomposition(probs, labels, n_bins))
    return out


# --------------------------------------------------------------------------
# post-hoc calibrators
# --------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Single-parameter post-hoc calibrator: p = softmax(z / T).

    We parameterise by log T so the optimiser is unconstrained while T stays
    strictly positive -- which is what guarantees the argmax (and hence the
    accuracy) is preserved.
    """

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        z = torch.as_tensor(logits, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.as_tensor(logits, dtype=torch.float32)
            return F.softmax(self.forward(z), dim=-1).numpy()


class EnsembleTemperatureScaler(nn.Module):
    r"""Temperature scaling for a *posterior predictive*, not a single softmax.

    Why this class exists
    ---------------------
    The predictive distribution of an ensemble (or of MC dropout) is a mixture,

        p(y|x) = (1/M) \sum_m softmax(z_m).

    Applying `TemperatureScaler` to the *averaged* logits computes
    softmax(\bar z / T), which is a different estimator: softmax is non-linear,
    so mean-of-softmax != softmax-of-mean. Calibrating one and reporting the
    other silently compares two different models.

    The correct operation scales inside the average,

        p_T(y|x) = (1/M) \sum_m softmax(z_m / T),

    which is what this class fits and applies.

    The invariance caveat
    ---------------------
    For a single model, dividing logits by T > 0 provably preserves the argmax,
    so accuracy is untouched. **That guarantee does not extend to the mixture.**
    The argmax of a mixture of softmaxes is not temperature-invariant: raising T
    flattens each member toward uniform, which re-weights how much each member
    contributes to the mixture, and the winning class can change. So accuracy can
    move here -- typically by a fraction of a point, but not by exactly zero.

    This is worth stating because the single-model theorem is quoted routinely
    and applied to ensembles, where it is false. We measure the shift rather than
    assume it away (`accuracy_shift`).
    """

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def _mixture_log_probs(self, member_logits: torch.Tensor) -> torch.Tensor:
        """log (1/M) sum_m softmax(z_m / T), computed in log-space. (N, C)"""
        z = member_logits / self.log_t.exp()  # (N, M, C)
        log_p = z - torch.logsumexp(z, dim=-1, keepdim=True)
        m = member_logits.shape[1]
        return torch.logsumexp(log_p, dim=1) - torch.log(
            torch.tensor(float(m), device=z.device)
        )

    def fit(self, member_logits: np.ndarray, labels: np.ndarray,
            max_iter: int = 200) -> "EnsembleTemperatureScaler":
        z = torch.as_tensor(member_logits, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.nll_loss(self._mixture_log_probs(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def transform(self, member_logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.as_tensor(member_logits, dtype=torch.float32)
            return self._mixture_log_probs(z).exp().numpy()

    def accuracy_shift(self, member_logits: np.ndarray, labels: np.ndarray) -> float:
        """Signed change in accuracy caused by scaling. Expected to be small but
        not exactly zero -- see the caveat in the class docstring."""
        base = torch.as_tensor(member_logits, dtype=torch.float32)
        with torch.no_grad():
            raw = F.softmax(base, dim=-1).mean(dim=1).argmax(-1).numpy()
        cal = self.transform(member_logits).argmax(-1)
        return float((cal == labels).mean() - (raw == labels).mean())


class VectorScaler(nn.Module):
    """p = softmax(diag(w) z + b): 2C parameters.

    Included as the natural next rung on the ladder. It is strictly more
    expressive than temperature scaling and *does* change the argmax, so unlike
    temperature scaling it can trade accuracy for calibration -- which is why it
    needs a genuinely held-out split and why it is not the default here.
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.w + self.b

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 300) -> "VectorScaler":
        z = torch.as_tensor(logits, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        opt = torch.optim.LBFGS([self.w, self.b], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.as_tensor(logits, dtype=torch.float32)
            return F.softmax(self.forward(z), dim=-1).numpy()
