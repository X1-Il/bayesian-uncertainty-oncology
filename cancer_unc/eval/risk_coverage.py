"""Selective prediction: the risk-coverage trade-off.

This is where uncertainty stops being a diagnostic and becomes a *decision*. A
screening model does not have to answer every scan. Given a confidence function
kappa(x), it can answer the fraction it is surest about and refer the rest to a
radiologist. The question is how much accuracy that buys.

Definitions (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017)
-------------------------------------------------------------
For a selection rule g(x) = 1[kappa(x) >= tau],

    coverage(tau)        = E[ g(x) ]
    selective risk R(tau)= E[ loss(x) g(x) ] / E[ g(x) ]

R is the error rate *among the cases the model chose to answer*. The curve
tau -> (coverage, risk) is the risk-coverage curve, and its area

    AURC = integral of R over coverage

summarises the confidence function with a single number. Lower is better.

Why AURC alone is not enough
----------------------------
AURC is bounded below by the model's own error rate: even a *perfect* confidence
ranking cannot make a wrong prediction right, it can only defer it last. That
optimal curve is achievable by an oracle that ranks every correct prediction
above every incorrect one. The *excess* AURC

    E-AURC = AURC - AURC_optimal

therefore isolates the quality of the ranking from the quality of the classifier.
This matters for the comparison being run here: temperature scaling is a strictly
monotone transform of the logits, so it **cannot change the ranking, the
risk-coverage curve, or AURC at all**. Any paper reporting an AURC improvement
from temperature scaling has a bug. What temperature scaling changes is the
*threshold semantics* -- with calibrated probabilities, "refer everything below
90% confidence" means what it says. That is the claim this module is built to
support, and `coverage_at_risk` is where it pays off: choosing tau to hit a
target risk requires the probabilities to be calibrated, otherwise the achieved
risk misses the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskCoverageCurve:
    coverage: np.ndarray
    risk: np.ndarray
    threshold: np.ndarray
    aurc: float
    aurc_optimal: float
    excess_aurc: float


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray, loss: np.ndarray | None = None
) -> RiskCoverageCurve:
    """Full curve, evaluated at every achievable coverage 1/n .. 1.

    `loss` defaults to 0/1 error. Passing a cost-sensitive loss (e.g. weighting
    false negatives more heavily, which is the realistic choice in screening)
    changes the curve but nothing else in the computation.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    err = (~np.asarray(correct, dtype=bool)).astype(np.float64) if loss is None else np.asarray(loss, float)

    order = np.argsort(-conf, kind="mergesort")  # most confident first
    err_sorted = err[order]
    n = len(err)

    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k
    aurc = float(risk.mean())

    # oracle: same losses, perfectly ranked (all correct answered first)
    err_opt = np.sort(err_sorted, kind="mergesort")
    risk_opt = np.cumsum(err_opt) / k
    aurc_opt = float(risk_opt.mean())

    return RiskCoverageCurve(
        coverage=coverage,
        risk=risk,
        threshold=conf[order],
        aurc=aurc,
        aurc_optimal=aurc_opt,
        excess_aurc=aurc - aurc_opt,
    )


def risk_at_coverage(curve: RiskCoverageCurve, target_coverage: float) -> float:
    """Selective risk when answering `target_coverage` of cases."""
    i = int(np.clip(np.searchsorted(curve.coverage, target_coverage), 0, len(curve.risk) - 1))
    return float(curve.risk[i])


def coverage_at_risk(curve: RiskCoverageCurve, target_risk: float) -> dict[str, float]:
    """Largest coverage whose selective risk stays at or below `target_risk`.

    The clinically phrased question: "if I will tolerate at most a 2% error rate,
    what fraction of the workload can this model take off the radiologist?"

    Scanning from full coverage downward and taking the first feasible point is
    deliberate: the risk curve is not monotone in coverage (adding one more
    correct answer lowers risk), so a bisection would find an arbitrary
    crossing. We want the *largest* feasible coverage.
    """
    feasible = np.where(curve.risk <= target_risk)[0]
    if len(feasible) == 0:
        return {"coverage": 0.0, "risk": float("nan"), "threshold": float("nan")}
    i = int(feasible.max())
    return {
        "coverage": float(curve.coverage[i]),
        "risk": float(curve.risk[i]),
        "threshold": float(curve.threshold[i]),
    }


def selective_report(
    confidence: np.ndarray,
    correct: np.ndarray,
    coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 1.0),
    risks: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> dict[str, float]:
    curve = risk_coverage_curve(confidence, correct)
    out = {
        "aurc": curve.aurc,
        "aurc_optimal": curve.aurc_optimal,
        "excess_aurc": curve.excess_aurc,
        "full_coverage_risk": float(curve.risk[-1]),
    }
    for c in coverages:
        out[f"risk@cov{c:g}"] = risk_at_coverage(curve, c)
    for r in risks:
        out[f"cov@risk{r:g}"] = coverage_at_risk(curve, r)["coverage"]
    return out


def compare_confidence_functions(
    confidences: dict[str, np.ndarray], correct: np.ndarray
) -> dict[str, dict[str, float]]:
    """Rank candidate confidence functions by how well they support deferral.

    Expected result on this benchmark: the *negated total/aleatoric* uncertainty
    should beat raw MSP slightly, and epistemic alone should be a poor selective
    predictor on in-distribution data -- because ID errors are driven by label
    noise, which is aleatoric. That epistemic wins at OOD detection and loses at
    ID deferral is the cleanest evidence that the decomposition separates two
    genuinely different things.
    """
    return {name: selective_report(c, correct) for name, c in confidences.items()}
