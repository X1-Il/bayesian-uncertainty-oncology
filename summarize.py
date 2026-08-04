"""Turn results/*.json into a markdown report.

    python summarize.py            # print to stdout
    python summarize.py > RESULTS.md

Kept separate from main.py so the write-up can be regenerated from saved
results without touching the models.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


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
                v["shift"] + (" *" if v["semantic"] else ""),
                f(r["accuracy"], 3),
                f"{f(r['ece'])} [{f(r['ece_lo'])}, {f(r['ece_hi'])}]",
                f(c["ece"]),
                f(r["nll"]), f(c["nll"]),
            ])
        print(table(["split", "acc", "ECE raw [95% CI]", "ECE cal",
                     "NLL raw", "NLL cal"], rows))
        print("\n`*` = semantic shift. Accuracy is identical before and after "
              "scaling by construction.\n")

    # ---- E4 --------------------------------------------------------------
    e4 = load("e4_ood")
    if e4:
        print("## E4 — OOD detection (AUROC)\n")
        shifts = list(e4.keys())
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
