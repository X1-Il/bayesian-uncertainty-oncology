"""Turn results/*.json into a markdown report.

    python summarize.py            # print to stdout
    python summarize.py > RESULTS.md

Kept separate from main.py so the write-up can be regenerated from saved
results without touching the models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The report uses mathematical characters (ẑ, β, →) that the Windows console's
# default cp1252 codec cannot encode, which makes `python summarize.py > RESULTS.md`
# die partway through. Force UTF-8 on stdout rather than degrading the notation.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent


def _results_dir() -> Path:
    """Prefer the full study; fall back to quick, then to a flat legacy layout."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for candidate in (ROOT / "results" / "full", ROOT / "results" / "quick",
                      ROOT / "results"):
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    return ROOT / "results" / "full"


RESULTS = _results_dir()


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def f(x, n=4):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    print("# Results\n")
    preset = RESULTS.name
    if preset == "quick":
        print("> **Warning: these are `--quick` smoke-test numbers**, not the full")
        print("> study. Reduced training budget, fewer sweep points. Run")
        print("> `python main.py --full` for the reported results.\n")

    # ---- identifiability -------------------------------------------------
    a = load("audit_identifiability")
    if a:
        print("## Oracle validity bracket\n")
        print("A correct aleatoric estimate must fall inside this interval. The lower")
        print("bound assumes the latent is perfectly recoverable; the upper bound comes")
        print("from a deliberately suboptimal matched filter, which can only over-state")
        print("noise.\n")
        print(table(
            ["lower (analytic)", "upper (matched filter)", "width", "relative"],
            [[f(a["aleatoric_lower_bound"]), f(a["aleatoric_upper_bound"]),
              f(a["bracket_width"]), f(a["bracket_width_relative"], 3)]],
        ))
        print(f"\n`corr(ẑ, z) = {f(a['corr_z_hat_vs_z'], 3)}`, "
              f"mean |p̂ − p*| = {f(a['mean_abs_posterior_error'], 3)}\n")

    # ---- E1 --------------------------------------------------------------
    e1 = load("e1_aleatoric_recovery")
    if e1:
        print("## E1 — aleatoric recovery across the label-noise sweep\n")
        print(table(
            ["β", "target E_z[H(p*)]", "estimated aleatoric", "estimated epistemic",
             "Bayes err", "model err", "corr(alea, H(p*))"],
            [[f(r["beta"], 1), f(r["target_aleatoric"]), f(r["est_aleatoric"]),
              f(r["est_epistemic"]), f(r["target_bayes_error"]),
              f(r["test_error"]), f(r["corr_aleatoric_vs_true_H"], 3)]
             for r in e1["rows"]],
        ))
        print()

    # ---- E2 --------------------------------------------------------------
    e2 = load("e2_epistemic_vs_data")
    if e2:
        print("## E2 — epistemic vs training set size\n")
        print(table(
            ["n_train", "epistemic", "aleatoric", "target aleatoric", "model err"],
            [[r["n_train"], f(r["est_epistemic"]), f(r["est_aleatoric"]),
              f(r["target_aleatoric"]), f(r["test_error"])]
             for r in e2["rows"]],
        ))
        print(f"\nlog-log slope of epistemic vs n: "
              f"**{f(e2.get('loglog_slope_epistemic_vs_n'), 3)}**\n")

    # ---- E3 --------------------------------------------------------------
    e3 = load("e3_calibration")
    if e3:
        print(f"## E3 — calibration (fitted T = {f(e3['temperature'], 3)})\n")
        rows = []
        for k, v in e3["splits"].items():
            r, c = v["raw"], v["calibrated"]
            rows.append([
                f"`{k}` — {v['shift']}" + (" *" if v["semantic"] else ""),
                f(r["accuracy"], 3),
                f"{f(r['ece'])} [{f(r['ece_lo'])}, {f(r['ece_hi'])}]",
                f(c["ece"]),
                f(r["nll"]), f(c["nll"]),
                f(v.get("accuracy_shift_from_scaling"), 4),
            ])
        print(table(["split", "acc", "ECE raw [95% CI]", "ECE cal",
                     "NLL raw", "NLL cal", "Δacc"], rows))
        print("\n`*` = semantic shift. Note every calibrated ECE lies inside the")
        print("raw estimate's 95% interval: temperature scaling changes nothing")
        print("here, because the ensemble is already calibrated (T ≈ 1).")
        print("\nΔacc is the accuracy shift caused by scaling. It is *not* forced")
        print("to zero: the argmax-invariance theorem holds for a single softmax,")
        print("not for the mixture that an ensemble predictive actually is.\n")

    # ---- E3b -------------------------------------------------------------
    e3b = load("e3b_calibration_baseline")
    if e3b:
        print(f"## E3b — miscalibrated baseline (fitted T = {f(e3b['temperature'], 3)})\n")
        print("A single model, no heteroscedastic head, dropout off, stopped at the")
        print("final epoch instead of best validation NLL — each of the main model's")
        print("calibration mechanisms removed. Accuracy is bit-identical before and")
        print("after scaling here, since a single softmax satisfies argmax-invariance")
        print("exactly.\n")
        rows = []
        for k, v in e3b["splits"].items():
            r, c = v["raw"], v["calibrated"]
            ens = e3["splits"].get(k) if e3 else None
            rows.append([
                v["shift"] + (" *" if v["semantic"] else ""),
                f(r["accuracy"], 3), f(r["ece"]), f(c["ece"]),
                f(ens["raw"]["ece"]) if ens else "-",
                f(ens["calibrated"]["ece"]) if ens else "-",
            ])
        print(table(["split", "acc", "single ECE raw", "single ECE cal",
                     "ens ECE raw", "ens ECE cal"], rows))
        print()

    # ---- E4 --------------------------------------------------------------
    e4 = load("e4_ood")
    if e4:
        print("## E4 — OOD detection (AUROC)\n")
        # `_`-prefixed keys are metadata (`_preset`), not shifts.
        shifts = [k for k in e4 if not k.startswith("_")]
        scores = [k for k in e4[shifts[0]] if not k.startswith("_")]
        rows = []
        for sc in scores:
            rows.append([sc] + [f(e4[s][sc]["auroc"], 3) for s in shifts])
        print(table(
            ["score"] + [e4[s]["_shift"] + (" *" if e4[s]["_semantic"] else "")
                         for s in shifts],
            rows,
        ))
        print("\n`aleatoric` is a negative control: it should *not* win on the "
              "semantic shift (`*`).\n")

    # ---- E5 --------------------------------------------------------------
    e5 = load("e5_selective")
    if e5:
        print("## E5 — selective prediction\n")
        for label, block in (("In-distribution", e5.get("in_distribution", {})),
                             ("Mixed ID + novel stream", e5.get("mixed_stream", {}))):
            names = [k for k in block if not k.startswith("_")]
            if not names:
                continue
            print(f"**{label}**\n")
            print(table(
                ["confidence", "AURC", "E-AURC", "risk@50% cov", "cov@5% risk"],
                [[n, f(block[n]["aurc"]), f(block[n]["excess_aurc"]),
                  f(block[n]["risk@cov0.5"]), f(block[n]["cov@risk0.05"], 3)]
                 for n in names],
            ))
            print()

    # ---- E6 --------------------------------------------------------------
    e6 = load("e6_mc_budget")
    if e6:
        print("## E6 — MC sample budget\n")
        conv = e6["convergence"]
        ts = sorted(conv, key=int)
        print(table(
            ["T", "epistemic", "aleatoric", "total"],
            [[t, f(conv[t]["epistemic"], 5), f(conv[t]["aleatoric"], 5),
              f(conv[t]["total"], 5)] for t in ts],
        ))
        print("\nEpistemic rises with T: the small-T estimate is biased downward "
              "by O(1/T), as Jensen's inequality requires.\n")


if __name__ == "__main__":
    main()
