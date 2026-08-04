"""The experiments the project actually claims results for.

Each function answers one question and returns plain dicts/arrays so results can
be serialised, re-plotted, or dropped into a report without re-running anything.

  E1  Does the aleatoric estimate recover the analytic label noise?
  E2  Does the epistemic estimate vanish as data grows?
  E3  What does temperature scaling fix, and does it survive distribution shift?
  E4  Which uncertainty score detects novelty, and is the split meaningful?
  E5  What does uncertainty buy in a deferral workflow?
  E6  How many MC samples are actually needed?

E1 and E2 are the ones that make this more than a benchmark run: they are
falsifiable checks against ground truth that a real dataset cannot support.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .data import (
    PhantomConfig,
    SHIFTS,
    as_tensors,
    binary_entropy,
    estimate_latent,
    matched_filter_statistic,
    make_benchmark,
    make_loaders,
    make_split,
    oracle_stats,
)
from .eval import compare_confidence_functions, risk_coverage_curve, selective_report
from .models import BayesianCNN
from .train import TrainConfig, train_ensemble
from .uncertainty import (
    MahalanobisScorer,
    TemperatureScaler,
    calibration_report,
    ensemble_predict,
    mc_convergence,
    mc_dropout_predict,
    ood_report,
    reliability,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def predict_split(models, split, n_logit_samples: int = 32):
    x, _ = as_tensors(split)
    if len(models) == 1:
        return mc_dropout_predict(models[0], x, n_weight_samples=32,
                                  n_logit_samples=n_logit_samples)
    return ensemble_predict(models, x, n_logit_samples=n_logit_samples)


# --------------------------------------------------------------------------
# E1 -- aleatoric recovery
# --------------------------------------------------------------------------
def exp_aleatoric_recovery(
    betas: tuple[float, ...] = (0.6, 1.0, 1.6, 2.5, 4.0),
    base_phantom: PhantomConfig | None = None,
    cfg: TrainConfig | None = None,
    n_train: int = 4000,
    verbose: bool = True,
) -> dict:
    """Sweep the label-noise dial and compare estimate against closed form.

    beta controls the logistic link's steepness and therefore the irreducible
    entropy E_z[H(sigma(beta z))], which we know exactly. A correct aleatoric
    estimator must track that curve downward as beta grows. A model that simply
    reports "more uncertainty when less accurate" would produce a curve of the
    right shape but the wrong values -- so we report the estimate against the
    target on the same axis, plus per-example correlation with H(p*(y|x)),
    which the shape-only explanation cannot fake.
    """
    base = base_phantom or PhantomConfig()
    cfg = cfg or TrainConfig()
    rows = []

    for beta in betas:
        phantom = replace(base, beta=float(beta))
        oracle = oracle_stats(phantom)
        if verbose:
            print(f"[E1] beta={beta}  target aleatoric={oracle['aleatoric_entropy']:.4f}", flush=True)

        bench = make_benchmark(phantom, n_train=n_train, n_val=1200, n_test=2500,
                               seed=cfg.seed, shifts=())
        loaders = make_loaders(bench, batch_size=cfg.batch_size)
        models, _ = train_ensemble(cfg, loaders, verbose=False)

        out = predict_split(models, bench["test"])
        test = bench["test"]

        # per-example oracle entropy
        h_true = binary_entropy(test.p_true)
        corr = float(np.corrcoef(out.aleatoric, h_true)[0, 1])
        corr_epis = float(np.corrcoef(out.epistemic, h_true)[0, 1])

        rows.append({
            "beta": float(beta),
            "target_aleatoric": oracle["aleatoric_entropy"],
            "est_aleatoric": float(out.aleatoric.mean()),
            "est_epistemic": float(out.epistemic.mean()),
            "est_total": float(out.total.mean()),
            "target_bayes_error": oracle["bayes_error"],
            "test_error": float((out.prediction != test.labels).mean()),
            "target_bayes_nll": oracle["bayes_nll"],
            "test_nll": calibration_report(out.probs, test.labels, n_boot=0)["nll"],
            "corr_aleatoric_vs_true_H": corr,
            "corr_epistemic_vs_true_H": corr_epis,
        })
        if verbose:
            r = rows[-1]
            print(f"     est={r['est_aleatoric']:.4f}  epis={r['est_epistemic']:.4f}  "
                  f"corr={corr:.3f}", flush=True)
    return {"rows": rows}


# --------------------------------------------------------------------------
# E2 -- epistemic vanishes with data
# --------------------------------------------------------------------------
def exp_epistemic_vs_data(
    sizes: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000),
    phantom: PhantomConfig | None = None,
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Epistemic uncertainty must decay toward 0; aleatoric must not move.

    This is the sharpest test of whether the split is real. Both terms come from
    the same forward passes, so any procedure that merely rescales "uncertainty"
    would move them together. A decaying epistemic curve beside a flat aleatoric
    curve pinned at the analytic value cannot be produced that way.

    The expected decay is roughly O(1/n) in the posterior variance for a
    well-specified model; we report the fitted slope on a log-log axis rather
    than claiming a rate, since dropout is a crude posterior approximation.
    """
    phantom = phantom or PhantomConfig()
    cfg = cfg or TrainConfig()
    oracle = oracle_stats(phantom)
    rows = []

    # one fixed test set for all sizes, so the curves are comparable
    test = make_split(2500, phantom, seed=cfg.seed + 3)

    for n in sizes:
        if verbose:
            print(f"[E2] n_train={n}", flush=True)
        bench = {
            "train": make_split(n, phantom, seed=cfg.seed + 1),
            "val": make_split(max(400, n // 4), phantom, seed=cfg.seed + 2),
        }
        loaders = make_loaders(bench, batch_size=min(cfg.batch_size, max(16, n // 8)))
        models, _ = train_ensemble(cfg, loaders, verbose=False)

        out = predict_split(models, test)
        rows.append({
            "n_train": int(n),
            "est_epistemic": float(out.epistemic.mean()),
            "est_aleatoric": float(out.aleatoric.mean()),
            "est_total": float(out.total.mean()),
            "var_epistemic": float(out.var_epistemic.mean()),
            "var_aleatoric": float(out.var_aleatoric.mean()),
            "target_aleatoric": oracle["aleatoric_entropy"],
            "test_error": float((out.prediction != test.labels).mean()),
            "target_bayes_error": oracle["bayes_error"],
        })
        if verbose:
            r = rows[-1]
            print(f"     epis={r['est_epistemic']:.4f}  alea={r['est_aleatoric']:.4f}  "
                  f"err={r['test_error']:.4f}", flush=True)

    # log-log slope of epistemic vs n
    n_arr = np.array([r["n_train"] for r in rows], dtype=float)
    e_arr = np.array([r["est_epistemic"] for r in rows], dtype=float)
    ok = e_arr > 1e-8
    slope = float(np.polyfit(np.log(n_arr[ok]), np.log(e_arr[ok]), 1)[0]) if ok.sum() > 1 else float("nan")
    return {"rows": rows, "loglog_slope_epistemic_vs_n": slope}


# --------------------------------------------------------------------------
# E3 -- calibration, before/after, in and out of distribution
# --------------------------------------------------------------------------
def exp_calibration(models, benchmark, n_bins: int = 15, verbose: bool = True) -> dict:
    """Fit T on validation, then report calibration everywhere.

    Reporting under shift is the part that matters. Temperature scaling is
    fitted on in-distribution validation data, so there is no reason it should
    transfer to a corrupted test set -- and it largely does not. Demonstrating
    that failure is more useful than the in-distribution improvement, because it
    is the failure mode that would actually harm a deployed screening system.
    """
    out_val = predict_split(models, benchmark["val"])
    scaler = TemperatureScaler().fit(out_val.logits, benchmark["val"].labels)
    if verbose:
        print(f"[E3] fitted T = {scaler.temperature:.4f}", flush=True)

    results = {"temperature": scaler.temperature, "splits": {}}
    out_test = None
    for name, split in benchmark.items():
        if name in ("train",):
            continue
        out = predict_split(models, split)
        if name == "test":
            out_test = out  # reused for the reliability diagrams below
        raw = calibration_report(out.probs, split.labels, n_bins)
        cal = calibration_report(scaler.transform(out.logits), split.labels, n_bins)

        # accuracy invariance is a theorem, not a hope -- assert it
        assert abs(raw["accuracy"] - cal["accuracy"]) < 1e-9, (
            "temperature scaling changed accuracy; the argmax-invariance "
            "argument is violated, which means a bug"
        )

        results["splits"][name] = {
            "shift": split.shift,
            "semantic": SHIFTS.get(name, SHIFTS["id"]).semantic,
            "raw": raw,
            "calibrated": cal,
            "mean_epistemic": float(out.epistemic.mean()),
            "mean_aleatoric": float(out.aleatoric.mean()),
        }
        if verbose:
            print(f"     {name:<10} ece {raw['ece']:.4f} -> {cal['ece']:.4f}  "
                  f"nll {raw['nll']:.4f} -> {cal['nll']:.4f}", flush=True)

    # Both diagrams come from the same forward pass. Recomputing would resample
    # the MC noise and make the "before" and "after" panels describe slightly
    # different predictions, which is exactly the comparison being made.
    assert out_test is not None, "benchmark must contain a 'test' split"
    results["reliability_test_raw"] = _bins_to_dict(
        reliability(out_test.probs, benchmark["test"].labels, n_bins)
    )
    results["reliability_test_cal"] = _bins_to_dict(
        reliability(scaler.transform(out_test.logits), benchmark["test"].labels, n_bins)
    )
    return results


def _bins_to_dict(b) -> dict:
    return {
        "edges": b.edges.tolist(),
        "counts": b.counts.tolist(),
        "conf": np.nan_to_num(b.conf, nan=0.0).tolist(),
        "acc": np.nan_to_num(b.acc, nan=0.0).tolist(),
    }


# --------------------------------------------------------------------------
# E4 -- OOD detection
# --------------------------------------------------------------------------
def exp_ood(models, benchmark, verbose: bool = True) -> dict:
    """Compare scores on covariate vs semantic shift.

    The hypothesis under test: epistemic uncertainty should be the best score on
    the semantic shift (novel lesion morphology) and should *not* dominate on
    pure noise corruption, where the failure is degraded evidence rather than
    unfamiliarity. The aleatoric score is carried through as a negative control
    -- if it detects novelty as well as the epistemic score does, the two terms
    are not actually separated and every other result is suspect.
    """
    out_train = predict_split(models, benchmark["train"])
    maha = MahalanobisScorer().fit(out_train.features, benchmark["train"].labels)

    out_id = predict_split(models, benchmark["test"])
    results = {}
    for name, split in benchmark.items():
        if name in ("train", "val", "test"):
            continue
        out_ood = predict_split(models, split)
        rep = ood_report(out_id, out_ood, maha)
        rep["_semantic"] = SHIFTS[name].semantic
        rep["_shift"] = split.shift
        results[name] = rep
        if verbose:
            best = max((k for k in rep if not k.startswith("_")),
                       key=lambda k: rep[k]["auroc"])
            print(f"[E4] {name:<10} best={best} auroc={rep[best]['auroc']:.4f} "
                  f"(epistemic {rep['epistemic']['auroc']:.4f}, msp {rep['msp']['auroc']:.4f})",
                  flush=True)
    return results


# --------------------------------------------------------------------------
# E5 -- selective prediction
# --------------------------------------------------------------------------
def exp_selective(models, benchmark, verbose: bool = True) -> dict:
    """Risk-coverage on the ID test set, and on a mixed ID+OOD stream.

    The mixed stream is the realistic deployment picture: a screening queue
    contains occasional scans the model has no business judging. A confidence
    function that is excellent in-distribution can fall apart there, and that is
    where the epistemic term earns its place -- it is the only score that is
    high for unfamiliar inputs *and* low for familiar-but-ambiguous ones.
    """
    out = predict_split(models, benchmark["test"])
    correct = out.prediction == benchmark["test"].labels

    confidences = {
        "msp": out.confidence,
        "neg_total_entropy": -out.total,
        "neg_aleatoric": -out.aleatoric,
        "neg_epistemic": -out.epistemic,
    }
    id_results = compare_confidence_functions(confidences, correct)

    curve = risk_coverage_curve(out.confidence, correct)
    id_results["_curve_msp"] = {
        "coverage": curve.coverage[::10].tolist(),
        "risk": curve.risk[::10].tolist(),
    }

    # mixed stream: ID test + the semantic-shift split, where OOD counts as error
    mixed = {}
    if "novel" in benchmark:
        out_ood = predict_split(models, benchmark["novel"])
        conf_mix = {
            "msp": np.concatenate([out.confidence, out_ood.confidence]),
            "neg_total_entropy": -np.concatenate([out.total, out_ood.total]),
            "neg_epistemic": -np.concatenate([out.epistemic, out_ood.epistemic]),
        }
        # every OOD case is scored as an error: the model should have deferred
        correct_mix = np.concatenate([correct, np.zeros(len(out_ood.probs), dtype=bool)])
        mixed = compare_confidence_functions(conf_mix, correct_mix)
        if verbose:
            for k, v in mixed.items():
                print(f"[E5] mixed {k:<20} AURC={v['aurc']:.4f} "
                      f"cov@risk0.05={v['cov@risk0.05']:.3f}", flush=True)

    return {"in_distribution": id_results, "mixed_stream": mixed}


# --------------------------------------------------------------------------
# E6 -- MC sample budget
# --------------------------------------------------------------------------
def exp_mc_budget(models, benchmark, verbose: bool = True) -> dict:
    """Expose the O(1/T) bias in the epistemic estimate."""
    x, _ = as_tensors(benchmark["test"])
    if len(models) == 1:
        out = mc_dropout_predict(models[0], x, n_weight_samples=64, n_logit_samples=16)
    else:
        # for an ensemble, T is capped at the number of members, so we study the
        # dropout budget of a single member instead -- the bias is a property of
        # the MC average, not of which posterior we sampled from
        out = mc_dropout_predict(models[0], x, n_weight_samples=64, n_logit_samples=16)
    conv = mc_convergence(out.member_probs)
    if verbose:
        for t, v in conv.items():
            print(f"[E6] T={t:<3} epistemic={v['epistemic']:.5f} "
                  f"aleatoric={v['aleatoric']:.5f}", flush=True)
    return {"convergence": {str(k): v for k, v in conv.items()}}


# --------------------------------------------------------------------------
# identifiability audit
# --------------------------------------------------------------------------
def exp_identifiability(phantom: PhantomConfig | None = None, n: int = 2000) -> dict:
    """Bracket the validity of the oracle, rather than assuming it.

    `bayes_posterior` claims p*(y|x) = sigma(beta z). That holds exactly only if
    z is recoverable from x. It is not exactly recoverable: lesion position and
    size are unknown, tissue texture is a nuisance field, and the intensity
    clipping at 1.0 compresses the tails. So the honest question is not "is the
    oracle valid" (it is not, exactly) but "how far off can it be".

    The bracket
    -----------
    Let A* = E_z[H(sigma(beta z))] be the oracle aleatoric target, and let
    A_hat be the same quantity computed from a matched-filter estimate z_hat.
    Because a *suboptimal* estimator discards information, it can only inflate
    the apparent noise:

        A* <= A_attainable <= A_hat,

    where A_attainable is what a Bayes-optimal estimator of z from x would see.
    The matched filter is decidedly suboptimal -- it is a fixed linear template
    bank, while the network is free to learn a better statistic -- so A_hat is a
    genuine upper bound and A* a genuine lower bound. Any measured aleatoric
    estimate landing inside [A*, A_hat] is consistent with a correct estimator;
    landing outside it is not.

    Reporting the bracket is the difference between a claim that can be checked
    and a claim that merely sounds rigorous. The width of the bracket is a
    property of the phantom design, so it is quoted alongside every E1/E2 result.
    """
    phantom = phantom or PhantomConfig()
    # two independent draws: one to fit the affine gain, one to evaluate on
    cal = make_split(n, phantom, seed=12345)
    split = make_split(n, phantom, seed=54321)

    cal_stat = matched_filter_statistic(cal.images, phantom)
    z_hat = estimate_latent(split.images, phantom, calibration=(cal_stat, cal.z))

    p_hat = 1.0 / (1.0 + np.exp(-phantom.beta * z_hat))
    h_true = binary_entropy(split.p_true)
    h_hat = binary_entropy(p_hat)

    lower = float(h_true.mean())
    upper = float(h_hat.mean())
    return {
        "aleatoric_lower_bound": lower,
        "aleatoric_upper_bound": upper,
        "bracket_width": upper - lower,
        "bracket_width_relative": (upper - lower) / lower if lower > 0 else float("nan"),
        "corr_z_hat_vs_z": float(np.corrcoef(z_hat, split.z)[0, 1]),
        "rmse_z": float(np.sqrt(np.mean((z_hat - split.z) ** 2))),
        "mean_abs_posterior_error": float(np.mean(np.abs(p_hat - split.p_true))),
        "note": (
            "Lower bound = analytic oracle (z perfectly identified). "
            "Upper bound = matched-filter estimator (suboptimal, so it over-states "
            "noise). A correct aleatoric estimate should fall inside this interval."
        ),
    }
