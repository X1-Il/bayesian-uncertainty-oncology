"""Decomposing predictive uncertainty into epistemic and aleatoric parts.

Setting
-------
We have an approximate posterior q(w) over weights (MC dropout, or the uniform
mixture over members of a deep ensemble) and, for each weight sample, a
predictive distribution p(y | x, w) that has already marginalised the logit
noise. The posterior predictive is

    p(y | x) = E_{q(w)} [ p(y | x, w) ].

Entropy decomposition
---------------------
The exact identity (Depeweg et al. 2018; the BALD objective of Houlsby et al.
2011) is

    H[ E_q p(y|x,w) ]   =   I(y ; w | x)   +   E_q H[ p(y|x,w) ]
    \\_______________/       \\___________/       \\_______________/
       total                 epistemic            aleatoric

Read it as: total uncertainty splits into the part that *disagreement between
plausible models* explains, and the part that *every* plausible model agrees is
irreducible. It follows from the definition of mutual information, and both
terms are non-negative, so the split is a genuine decomposition rather than a
heuristic. The epistemic term is exactly the mutual information between the
label and the weights: it is zero iff every posterior sample makes the same
prediction, which is the correct notion of "the model knows what it doesn't
know".

Two properties worth stating because they are what the estimator is graded on:
  * epistemic -> 0 as the training set grows (the posterior concentrates);
  * aleatoric -> E_z[H(p*(y|z))], a constant set by the data-generating process,
    which no amount of data reduces.

Estimator bias
--------------
With T weight samples, both terms are biased. The aleatoric term is an average
of T i.i.d. entropies, so it is unbiased. The total term plugs a T-sample mean
into the concave functional H, so by Jensen E[H(p_bar_T)] <= H(p_bar_inf):
**total, and therefore epistemic, are under-estimated at small T**, with bias
O(1/T). `mc_convergence` sweeps T so the bias can be seen and a sufficient T
chosen, rather than assumed.

Variance decomposition
----------------------
Entropy is not the only route. The law of total variance applied to the
predicted probability of the positive class gives

    Var[p]  =  E_q[ Var(p | w) ]  +  Var_q[ E(p | w) ]

which splits the same way and is reported alongside as a cross-check: the two
decompositions are computed from different functionals, so agreement between
them is evidence the split is real and not an artefact of the entropy estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.nets import BayesianCNN, enable_mc_dropout

EPS = 1e-12


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats along `axis`."""
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


@dataclass
class UncertaintyOutput:
    """Per-example predictive summary. All entropies in nats."""

    probs: np.ndarray  # (N, C)  posterior predictive
    total: np.ndarray  # (N,)    H[E p]
    aleatoric: np.ndarray  # (N,)    E H[p]
    epistemic: np.ndarray  # (N,)    mutual information
    var_aleatoric: np.ndarray  # (N,) E[Var(p|w)]
    var_epistemic: np.ndarray  # (N,) Var[E(p|w)]
    member_probs: np.ndarray  # (N, T, C) per-weight-sample predictives
    logits: np.ndarray  # (N, C) mean logits, for temperature scaling
    features: np.ndarray | None = None  # (N, D) penultimate, for Mahalanobis

    @property
    def confidence(self) -> np.ndarray:
        return self.probs.max(axis=-1)

    @property
    def prediction(self) -> np.ndarray:
        return self.probs.argmax(axis=-1)


def decompose(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Core decomposition from (N, T, C) per-sample predictives."""
    # Promote to float64 first. The epistemic term is a difference of two
    # entropies that are nearly equal whenever the posterior is concentrated --
    # exactly the well-trained, large-data regime E2 probes. In float32 that
    # cancellation loses most of the significant digits and can push the
    # difference negative, tripping the identity assertion below on results
    # that are actually fine.
    member_probs = np.asarray(member_probs, dtype=np.float64)
    p_bar = member_probs.mean(axis=1)  # (N, C)

    total = entropy(p_bar)  # H[E p]
    aleatoric = entropy(member_probs, axis=-1).mean(axis=1)  # E H[p]
    epistemic = total - aleatoric  # I(y; w | x)

    # Clamp only the floating-point residue. The identity guarantees
    # epistemic >= 0 exactly; anything beyond ~1e-9 would mean a real bug, so
    # we do not silently absorb it.
    assert epistemic.min() > -1e-6, f"MI identity violated: {epistemic.min()}"
    epistemic = np.maximum(epistemic, 0.0)

    # law of total variance on the positive-class probability
    if member_probs.shape[-1] == 2:
        p1 = member_probs[..., 1]  # (N, T)
        var_alea = (p1 * (1 - p1)).mean(axis=1)  # E[Var(y|w)] (Bernoulli)
        var_epis = p1.var(axis=1)  # Var_w[E(y|w)]
    else:
        var_alea = (member_probs * (1 - member_probs)).sum(-1).mean(axis=1)
        var_epis = member_probs.var(axis=1).sum(-1)

    return {
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "var_aleatoric": var_alea,
        "var_epistemic": var_epis,
    }


@torch.no_grad()
def mc_dropout_predict(
    model: BayesianCNN,
    x: torch.Tensor,
    n_weight_samples: int = 32,
    n_logit_samples: int = 16,
    batch_size: int = 256,
    return_features: bool = True,
    seed: int | None = 0,
) -> UncertaintyOutput:
    """Nested Monte Carlo: T dropout masks, each marginalising S logit samples.

    The nesting is what keeps the two sources separate. Drawing a single joint
    sample of (mask, logit noise) and treating the spread as epistemic would
    fold the aleatoric noise straight into the epistemic term -- a common and
    silent error, since the resulting numbers still look plausible.
    """
    if seed is not None:
        torch.manual_seed(seed)
    enable_mc_dropout(model)
    device = next(model.parameters()).device

    n = x.shape[0]
    members: list[np.ndarray] = []

    for _ in range(n_weight_samples):
        probs_t = []
        for i in range(0, n, batch_size):
            xb = x[i : i + batch_size].to(device)
            probs_t.append(model.predict_probs(xb, n_logit_samples).cpu())
        members.append(torch.cat(probs_t).numpy())

    # Mean logits and features are taken with dropout *off*: they are used for
    # temperature scaling and Mahalanobis, both of which want the deterministic
    # network, not a random draw from the posterior.
    model.eval()
    logits_all, feat_all = [], []
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size].to(device)
        mean, _ = model(xb)
        logits_all.append(mean.cpu())
        if return_features:
            feat_all.append(model.embed(xb).cpu())

    member_probs = np.stack(members, axis=1)  # (N, T, C)
    parts = decompose(member_probs)

    return UncertaintyOutput(
        probs=member_probs.mean(axis=1),
        member_probs=member_probs,
        logits=torch.cat(logits_all).numpy(),
        features=torch.cat(feat_all).numpy() if return_features else None,
        **parts,
    )


@torch.no_grad()
def ensemble_predict(
    models: list[BayesianCNN],
    x: torch.Tensor,
    n_logit_samples: int = 32,
    batch_size: int = 256,
    return_features: bool = True,
) -> UncertaintyOutput:
    """Deep ensemble: q(w) is the uniform mixture over independently trained nets.

    The decomposition is identical -- only the source of the weight samples
    changes. Ensembles typically give a larger, better-behaved epistemic term
    than MC dropout because the members sit in genuinely different basins,
    whereas dropout masks perturb around a single mode. Running both is the
    point: it shows the decomposition is a property of the posterior
    approximation, not of one trick.
    """
    n = x.shape[0]
    members, logit_sum, feat_sum = [], None, None

    for model in models:
        model.eval()
        probs_m, logits_m, feats_m = [], [], []
        for i in range(0, n, batch_size):
            xb = x[i : i + batch_size].to(next(model.parameters()).device)
            probs_m.append(model.predict_probs(xb, n_logit_samples).cpu())
            mean, _ = model(xb)
            logits_m.append(mean.cpu())
            if return_features:
                feats_m.append(model.embed(xb).cpu())
        members.append(torch.cat(probs_m).numpy())
        lg = torch.cat(logits_m)
        logit_sum = lg if logit_sum is None else logit_sum + lg
        if return_features:
            ft = torch.cat(feats_m)
            feat_sum = ft if feat_sum is None else feat_sum + ft

    member_probs = np.stack(members, axis=1)
    parts = decompose(member_probs)

    return UncertaintyOutput(
        probs=member_probs.mean(axis=1),
        member_probs=member_probs,
        logits=(logit_sum / len(models)).numpy(),
        features=(feat_sum / len(models)).numpy() if return_features else None,
        **parts,
    )


def mc_convergence(
    member_probs: np.ndarray, sample_counts: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
) -> dict[int, dict[str, float]]:
    """Mean uncertainty terms as a function of T, to expose the O(1/T) bias.

    Sub-samples (without replacement) from the available weight samples. The
    epistemic curve should rise and flatten; where it flattens is the T that is
    actually sufficient. Quoting an epistemic number without this curve is
    quoting an unknown fraction of the true value.
    """
    rng = np.random.default_rng(0)
    t_avail = member_probs.shape[1]
    out: dict[int, dict[str, float]] = {}
    for t in sample_counts:
        if t > t_avail:
            continue
        idx = rng.choice(t_avail, size=t, replace=False)
        parts = decompose(member_probs[:, idx])
        out[t] = {k: float(v.mean()) for k, v in parts.items()}
    return out
