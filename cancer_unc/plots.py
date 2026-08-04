"""Figures. One per claim, no decoration that does not carry information."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_phantoms(benchmark, out_dir: Path, n: int = 6) -> Path:
    """Sample scans across the shift suite -- the qualitative sanity check."""
    keys = [k for k in ("test", "noise_3", "blur_3", "modality", "novel") if k in benchmark]
    fig, axes = plt.subplots(len(keys), n, figsize=(1.35 * n, 1.45 * len(keys)))
    axes = np.atleast_2d(axes)
    for r, key in enumerate(keys):
        split = benchmark[key]
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(split.images[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c == 0:
                ax.set_ylabel(split.shift, fontsize=7.5, rotation=0,
                              ha="right", va="center")
            if r == 0:
                ax.set_title(f"y={split.labels[c]}", fontsize=7)
    fig.suptitle("Phantoms across the shift suite", fontsize=10)
    return _save(fig, out_dir, "01_phantoms.png")


def plot_aleatoric_recovery(res: dict, bracket: dict | None, out_dir: Path) -> Path:
    """E1: estimate vs analytic target across the label-noise sweep."""
    rows = res["rows"]
    beta = [r["beta"] for r in rows]
    target = [r["target_aleatoric"] for r in rows]
    est = [r["est_aleatoric"] for r in rows]
    epis = [r["est_epistemic"] for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax.plot(beta, target, "k--", marker="o", ms=4, label="analytic $E_z[H(p^*)]$")
    ax.plot(beta, est, marker="s", ms=4, color="#c0392b", label="estimated aleatoric")
    ax.plot(beta, epis, marker="^", ms=4, color="#2980b9", label="estimated epistemic")
    if bracket is not None:
        ax.axhspan(bracket["aleatoric_lower_bound"], bracket["aleatoric_upper_bound"],
                   color="k", alpha=0.07,
                   label=r"identifiability bracket ($\beta$=1.6)")
    ax.set_xlabel(r"$\beta$  (label separability)")
    ax.set_ylabel("entropy (nats)")
    ax.set_title("Aleatoric estimate tracks the analytic target")
    ax.legend(fontsize=7)

    ax2.plot(beta, [r["test_error"] for r in rows], marker="s", ms=4,
             color="#c0392b", label="model error")
    ax2.plot(beta, [r["target_bayes_error"] for r in rows], "k--", marker="o", ms=4,
             label="Bayes error")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("error rate")
    ax2.set_title("Model vs Bayes-optimal error")
    ax2.legend(fontsize=7)
    return _save(fig, out_dir, "02_aleatoric_recovery.png")


def plot_epistemic_vs_data(res: dict, out_dir: Path) -> Path:
    """E2: the decay that separates the two terms."""
    rows = res["rows"]
    n = [r["n_train"] for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax.plot(n, [r["est_epistemic"] for r in rows], marker="^", ms=4,
            color="#2980b9", label="epistemic")
    ax.plot(n, [r["est_aleatoric"] for r in rows], marker="s", ms=4,
            color="#c0392b", label="aleatoric")
    ax.axhline(rows[0]["target_aleatoric"], color="k", ls="--", lw=1,
               label="analytic aleatoric")
    ax.set_xscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel("entropy (nats)")
    ax.set_title("Epistemic decays with data; aleatoric does not")
    ax.legend(fontsize=7)

    ax2.plot(n, [r["est_epistemic"] for r in rows], marker="^", ms=4, color="#2980b9")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("training set size")
    ax2.set_ylabel("epistemic (nats)")
    slope = res.get("loglog_slope_epistemic_vs_n", float("nan"))
    ax2.set_title(f"log-log slope = {slope:.2f}")
    return _save(fig, out_dir, "03_epistemic_vs_data.png")


def plot_reliability(res: dict, out_dir: Path) -> Path:
    """E3: reliability diagrams before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, key, title in [
        (axes[0], "reliability_test_raw", "before"),
        (axes[1], "reliability_test_cal", f"after (T = {res['temperature']:.2f})"),
    ]:
        b = res[key]
        edges = np.array(b["edges"])
        centres = (edges[:-1] + edges[1:]) / 2
        counts = np.array(b["counts"])
        acc = np.array(b["acc"])
        conf = np.array(b["conf"])
        m = counts > 0

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.bar(centres[m], acc[m], width=np.diff(edges)[m] * 0.9,
               color="#2980b9", alpha=0.75, label="accuracy")
        ax.plot(conf[m], acc[m], color="#c0392b", marker="o", ms=3.5, lw=1.2,
                label="acc vs conf")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability, {title}")
        ax.legend(fontsize=7, loc="upper left")

    t = res["splits"]["test"]
    fig.suptitle(
        f"ECE {t['raw']['ece']:.4f} -> {t['calibrated']['ece']:.4f}   |   "
        f"NLL {t['raw']['nll']:.4f} -> {t['calibrated']['nll']:.4f}",
        fontsize=9,
    )
    return _save(fig, out_dir, "04_reliability.png")


def plot_calibration_under_shift(res: dict, out_dir: Path) -> Path:
    """E3: does a temperature fitted in-distribution survive shift?"""
    order = [k for k in ("test", "noise_1", "noise_2", "noise_3",
                         "blur_1", "blur_2", "blur_3", "modality", "novel")
             if k in res["splits"]]
    raw = [res["splits"][k]["raw"]["ece"] for k in order]
    cal = [res["splits"][k]["calibrated"]["ece"] for k in order]
    lo = [res["splits"][k]["raw"]["ece"] - res["splits"][k]["raw"]["ece_lo"] for k in order]
    hi = [res["splits"][k]["raw"]["ece_hi"] - res["splits"][k]["raw"]["ece"] for k in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - 0.2, raw, 0.4, yerr=[lo, hi], capsize=2.5,
           color="#c0392b", alpha=0.85, label="raw")
    ax.bar(x + 0.2, cal, 0.4, color="#2980b9", alpha=0.85, label="temperature-scaled")
    for i, k in enumerate(order):
        if res["splits"][k]["semantic"]:
            ax.axvspan(i - 0.5, i + 0.5, color="k", alpha=0.06)
    ax.set_xticks(x)
    ax.set_xticklabels([res["splits"][k]["shift"] for k in order],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("ECE")
    ax.set_title("Calibration under distribution shift (shaded = semantic shift; "
                 "error bars = 95% bootstrap CI)", fontsize=9)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "05_calibration_under_shift.png")


def plot_ood(res: dict, out_dir: Path) -> Path:
    """E4: AUROC per score per shift, covariate vs semantic."""
    shifts = list(res.keys())
    scores = [k for k in res[shifts[0]] if not k.startswith("_")]
    mat = np.array([[res[s][sc]["auroc"] for s in shifts] for sc in scores])

    fig, ax = plt.subplots(figsize=(1.05 * len(shifts) + 3, 0.52 * len(scores) + 2))
    im = ax.imshow(mat, cmap="RdYlBu", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(
        [f"{res[s]['_shift']}{'*' if res[s]['_semantic'] else ''}" for s in shifts],
        rotation=30, ha="right", fontsize=7.5,
    )
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(scores, fontsize=8)
    ax.grid(False)
    for i in range(len(scores)):
        for j in range(len(shifts)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="AUROC", fraction=0.03)
    ax.set_title("OOD detection AUROC  (* = semantic shift)", fontsize=9)
    return _save(fig, out_dir, "06_ood_auroc.png")


def plot_risk_coverage(res: dict, out_dir: Path) -> Path:
    """E5: what deferral buys, in and out of distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    idr = res["in_distribution"]
    curve = idr.get("_curve_msp")
    if curve:
        axes[0].plot(curve["coverage"], curve["risk"], color="#2980b9", lw=1.6)
        axes[0].set_xlabel("coverage"); axes[0].set_ylabel("selective risk")
        axes[0].set_title("Risk-coverage (in-distribution, MSP)")

    names = [k for k in idr if not k.startswith("_")]
    aurc_id = [idr[k]["aurc"] for k in names]
    mixed = res.get("mixed_stream", {})
    mixed_names = [k for k in mixed if not k.startswith("_")]

    x = np.arange(len(names))
    axes[1].bar(x - 0.2, aurc_id, 0.4, color="#2980b9", alpha=0.85, label="ID only")
    if mixed:
        aurc_mix = [mixed[k]["aurc"] if k in mixed else np.nan for k in names]
        axes[1].bar(x + 0.2, aurc_mix, 0.4, color="#c0392b", alpha=0.85,
                    label="ID + novel stream")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
    axes[1].set_ylabel("AURC (lower is better)")
    axes[1].set_title("Confidence functions compared")
    axes[1].legend(fontsize=7.5)
    return _save(fig, out_dir, "07_risk_coverage.png")


def plot_mc_budget(res: dict, out_dir: Path) -> Path:
    """E6: the O(1/T) bias made visible."""
    conv = res["convergence"]
    ts = sorted(int(k) for k in conv)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(ts, [conv[str(t)]["epistemic"] for t in ts], marker="^", ms=4,
            color="#2980b9", label="epistemic")
    ax.plot(ts, [conv[str(t)]["aleatoric"] for t in ts], marker="s", ms=4,
            color="#c0392b", label="aleatoric")
    ax.plot(ts, [conv[str(t)]["total"] for t in ts], marker="o", ms=4,
            color="k", label="total")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MC weight samples $T$")
    ax.set_ylabel("mean entropy (nats)")
    ax.set_title("Small $T$ under-estimates epistemic")
    ax.legend(fontsize=7.5)
    return _save(fig, out_dir, "08_mc_budget.png")


def plot_uncertainty_vs_oracle(out, split, out_dir: Path) -> Path:
    """Per-example estimate against per-example ground truth.

    The aggregate curves in E1 could in principle be matched by a model that
    gets the average right and the ordering wrong. This is the check that it
    does not: estimated aleatoric should rise with the oracle's H(p*), while
    epistemic should stay flat against it.
    """
    if not np.isfinite(split.p_true).all():
        return out_dir / "_skipped_no_oracle"
    from .data import binary_entropy

    h_true = binary_entropy(split.p_true)
    order = np.argsort(h_true)
    bins = np.array_split(order, 20)
    xs = [h_true[b].mean() for b in bins]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(xs, [out.aleatoric[b].mean() for b in bins], marker="s", ms=4,
            color="#c0392b", label="estimated aleatoric")
    ax.plot(xs, [out.epistemic[b].mean() for b in bins], marker="^", ms=4,
            color="#2980b9", label="estimated epistemic")
    ax.plot([0, max(xs)], [0, max(xs)], "k--", lw=1, label="identity")
    ax.set_xlabel(r"oracle $H(p^*(y|x))$ (nats)")
    ax.set_ylabel("estimated (nats)")
    ax.set_title("Per-example recovery, binned by oracle entropy")
    ax.legend(fontsize=7.5)
    return _save(fig, out_dir, "09_per_example_recovery.png")
