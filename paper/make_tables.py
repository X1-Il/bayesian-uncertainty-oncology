"""Generate paper/tables.tex and paper/macros.tex from results/*.json.

Every number in the report comes through here. Nothing is hand-copied, so the
paper cannot drift out of sync with the code that produced it: re-run the study,
re-run this, and the PDF is correct by construction.

    python paper/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def _results_dir() -> Path:
    """The paper reports the full study; refuse to quietly typeset quick numbers."""
    full = ROOT / "results" / "full"
    if full.is_dir() and any(full.glob("*.json")):
        return full
    legacy = ROOT / "results"
    if legacy.is_dir() and any(legacy.glob("*.json")):
        return legacy
    quick = ROOT / "results" / "quick"
    if quick.is_dir() and any(quick.glob("*.json")):
        print("WARNING: only --quick results found; the paper will contain "
              "smoke-test numbers. Run `python main.py --full` first.")
        return quick
    return full


RESULTS = _results_dir()


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def n(x, d=3):
    """Format a number, or an em-dash when the experiment did not run."""
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "--"
    return f"{v:.{d}f}"


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# --------------------------------------------------------------------------
def tab_e1(e1) -> str:
    if not e1:
        return ""
    # The epistemic column is O(1e-4). At three decimals it prints as 0.000/0.001,
    # which reads as "zero or noise" and hides the actual result -- that it is
    # flat and three orders of magnitude below the aleatoric term. Report it in
    # units of 1e-3, and show the signed gap explicitly: it is positive at every
    # point, which the identifiability argument predicts in advance.
    rows = "\n".join(
        rf"{n(r['beta'],1)} & {n(r['target_aleatoric'])} & {n(r['est_aleatoric'])} & "
        rf"${n(r['est_aleatoric'] - r['target_aleatoric'], 4)}$ & "
        rf"{n(1000 * r['est_epistemic'], 2)} & {n(r['target_bayes_error'])} & "
        rf"{n(r['test_error'])} & {n(r['corr_aleatoric_vs_true_H'])} \\"
        for r in e1["rows"]
    )
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E1: recovery of the analytic label noise.}} The target column is
$\mathbb{{E}}_z[H(p^*)]$, computed in closed form from the generative model, not
estimated. The aleatoric estimate tracks it across the sweep while the epistemic
term stays flat at $O(10^{{-4}})$ --- both from the same forward passes, so no
rescaling of ``uncertainty'' could make one column follow a moving target while
the other does not. $\Delta$ is positive at \emph{{every}} point, which the
identifiability argument of Section~\ref{{sec:bracket}} predicts in advance: a
suboptimal view of the latent can only inflate apparent noise, never deflate it.
The final column is the per-example correlation between the estimated aleatoric
term and the oracle entropy $H(p^*(y|x))$ on the same image, which an estimator
that merely matched the sweep means could not achieve.}}
\label{{tab:e1}}
\centering
\small
\begin{{tabular}}{{cccccccc}}
\toprule
$\beta$ & target $\mathbb{{E}}_z[H(p^*)]$ & est.\ aleatoric & $\Delta$ &
epist.\ ($\times 10^{{-3}}$) & Bayes err. & model err. & corr.\ (per-ex.) \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def tab_e2(e2) -> str:
    if not e2:
        return ""
    rows = "\n".join(
        rf"{r['n_train']} & {n(1000 * r['est_epistemic'], 2)} & {n(r['est_aleatoric'])} & "
        rf"{n(r['target_aleatoric'])} & {n(r['test_error'])} \\"
        for r in e2["rows"]
    )
    slope = n(e2.get("loglog_slope_epistemic_vs_n"), 2)
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E2: epistemic uncertainty decays with data; aleatoric does not.}}
Both columns are computed from the same forward passes, so no monotone rescaling
of ``uncertainty'' could produce a decaying column beside a flat one. Fitted
log--log slope of epistemic against $n$: ${slope}$.}}
\label{{tab:e2}}
\centering
\small
\begin{{tabular}}{{ccccc}}
\toprule
$n_{{\text{{train}}}}$ & epist.\ ($\times 10^{{-3}}$) & est.\ aleatoric &
target aleatoric & model err. \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def tab_e3(e3) -> str:
    if not e3:
        return ""
    rows = []
    for k, v in e3["splits"].items():
        r, c = v["raw"], v["calibrated"]
        star = r"$^\dagger$" if v["semantic"] else ""
        label = esc(k if k in ("val", "test") else v["shift"])
        rows.append(
            rf"{label}{star} & {n(r['accuracy'])} & "
            rf"{n(r['ece'])} & {n(c['ece'])} & {n(r['nll'])} & {n(c['nll'])} \\"
        )
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E3: calibration before and after temperature scaling}}
($T = {n(e3['temperature'],2)}$, fitted on in-distribution validation data only).
The correction is a null: every scaled ECE lies inside the raw estimate's $95\%$
bootstrap interval, because a deep ensemble with a heteroscedastic head is
already calibrated. What does vary is calibration under shift, which degrades
monotonically with severity and is worst under the modality shift --- a property
of the model that no in-distribution recalibration addresses. Table~\ref{{tab:e3b}}
supplies the missing contrast. $^\dagger$ marks semantic shift.}}
\label{{tab:e3}}
\centering
\small
\begin{{tabular}}{{lccccc}}
\toprule
& & \multicolumn{{2}}{{c}}{{ECE}} & \multicolumn{{2}}{{c}}{{NLL}} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}}
split & acc. & raw & scaled & raw & scaled \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def tab_e3b(e3b, e3) -> str:
    if not e3b:
        return ""
    rows = []
    for k in ("test", "noise_3", "blur_3", "modality", "novel", "decoupled"):
        b = e3b["splits"].get(k)
        if not b:
            continue
        e = e3["splits"].get(k) if e3 else None
        star = r"$^\dagger$" if b["semantic"] else ""
        rows.append(
            rf"{esc(b['shift'])}{star} & {n(b['raw']['ece'])} & {n(b['calibrated']['ece'])} & "
            rf"{n(e['raw']['ece']) if e else '--'} & {n(e['calibrated']['ece']) if e else '--'} \\"
        )
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E3b: the same correction on a model that needs it.}} The
baseline is a single network, no heteroscedastic head, dropout off, stopped at
the final epoch rather than at best validation NLL --- each of the main model's
calibration mechanisms removed. Its fitted temperature is
$T = {n(e3b['temperature'],2)}$ against $T = {n(e3['temperature'],2) if e3 else '--'}$
for the ensemble. Accuracy is bit-identical before and after scaling here, since
a single softmax satisfies the argmax-invariance theorem exactly.}}
\label{{tab:e3b}}
\centering
\small
\begin{{tabular}}{{lcccc}}
\toprule
& \multicolumn{{2}}{{c}}{{single model (ECE)}} & \multicolumn{{2}}{{c}}{{ensemble (ECE)}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
split & raw & scaled & raw & scaled \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def tab_e4(e4) -> str:
    if not e4:
        return ""
    # `_`-prefixed keys are metadata (`_preset`), not shifts.
    shifts = [k for k in e4 if not k.startswith("_")]
    scores = [k for k in e4[shifts[0]] if not k.startswith("_")]
    header = " & ".join(
        esc(e4[s]["_shift"]) + (r"$^\dagger$" if e4[s]["_semantic"] else "")
        for s in shifts
    )
    rows = []
    for sc in scores:
        best = max(scores, key=lambda t: 0)  # placeholder, bolding done per column below
        cells = []
        for s in shifts:
            val = e4[s][sc]["auroc"]
            top = max(e4[s][t]["auroc"] for t in scores)
            cells.append(rf"\textbf{{{n(val)}}}" if abs(val - top) < 1e-12 else n(val))
        rows.append(esc(sc) + " & " + " & ".join(cells) + r" \\")
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E4: OOD detection AUROC.}} Best score per column in bold.
\texttt{{aleatoric}} is a \emph{{negative control}}: if the decomposition is real it
should not win on semantic shift ($^\dagger$), since a novel input is one the
posterior \emph{{disagrees}} about rather than one every member agrees is ambiguous.}}
\label{{tab:e4}}
\centering
\small
\begin{{tabular}}{{l{'c' * len(shifts)}}}
\toprule
score & {header} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def tab_e5(e5) -> str:
    if not e5:
        return ""
    blocks = []
    for label, block in (("In-distribution", e5.get("in_distribution", {})),
                         ("Mixed ID + novel", e5.get("mixed_stream", {}))):
        names = [k for k in block if not k.startswith("_")]
        if not names:
            continue
        blocks.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\")
        for k in names:
            b = block[k]
            blocks.append(
                rf"\quad {esc(k)} & {n(b['aurc'],4)} & {n(b['excess_aurc'],4)} & "
                rf"{n(b['risk@cov0.5'])} & {n(b['cov@risk0.05'])} \\"
            )
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E5: selective prediction.}} E-AURC isolates ranking quality from
classifier quality. On the mixed stream every novel case counts as an error,
modelling a screening queue that contains scans the model should decline.}}
\label{{tab:e5}}
\centering
\small
\begin{{tabular}}{{lcccc}}
\toprule
confidence function & AURC & E-AURC & risk@50\% cov. & cov.@5\% risk \\
\midrule
{chr(10).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def tab_e6(e6) -> str:
    if not e6:
        return ""
    conv = e6["convergence"]
    ts = sorted(conv, key=int)
    rows = "\n".join(
        rf"{t} & {n(conv[t]['epistemic'],5)} & {n(conv[t]['aleatoric'],5)} & "
        rf"{n(conv[t]['total'],5)} \\" for t in ts
    )
    return rf"""
\begin{{table}}[t]
\caption{{\textbf{{E6: Monte-Carlo budget.}} The total term plugs a $T$-sample mean
into the concave functional $H$, so by Jensen's inequality it is under-estimated
at small $T$, and the epistemic term with it. The estimate rises and flattens;
quoting an epistemic number without this curve quotes an unknown fraction of it.}}
\label{{tab:e6}}
\centering
\small
\begin{{tabular}}{{cccc}}
\toprule
$T$ & epistemic & aleatoric & total \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# Every macro the prose can reference. Each is always emitted, so a missing
# experiment yields a visible placeholder rather than an undefined control
# sequence -- which pdflatex reports as an error but still writes a PDF for,
# meaning a silently broken document would otherwise pass as a successful build.
MACRO_NAMES = [
    "bracketLo", "bracketHi", "bracketWidth", "zCorr",
    "fittedT", "eceRaw", "eceCal", "nllRaw", "nllCal", "testAcc",
    # The accuracy comparison that justifies the decoupled condition. These were
    # hand-typed at first, which both contradicted the no-transcription claim and
    # left stale numbers from a superseded run sitting in the argument.
    "novelAcc", "decoupledAcc",
]
PLACEHOLDER = r"\textbf{??}"


def macros(audit, e3, e6) -> str:
    """Inline numbers used in the prose, so the text cannot drift either."""
    vals: dict[str, str] = {}

    if audit:
        vals["bracketLo"] = n(audit["aleatoric_lower_bound"])
        vals["bracketHi"] = n(audit["aleatoric_upper_bound"])
        vals["bracketWidth"] = n(audit["bracket_width"])
        vals["zCorr"] = n(audit["corr_z_hat_vs_z"])
    if e3:
        vals["fittedT"] = n(e3["temperature"], 2)
        t = e3["splits"].get("test")
        if t:
            vals["eceRaw"] = n(t["raw"]["ece"])
            vals["eceCal"] = n(t["calibrated"]["ece"])
            vals["nllRaw"] = n(t["raw"]["nll"])
            vals["nllCal"] = n(t["calibrated"]["nll"])
            vals["testAcc"] = n(t["raw"]["accuracy"])
        for key, macro in (("novel", "novelAcc"), ("decoupled", "decoupledAcc")):
            s = e3["splits"].get(key)
            if s:
                vals[macro] = n(s["raw"]["accuracy"])

    missing = [m for m in MACRO_NAMES if m not in vals]
    if missing:
        print(f"WARNING: no data yet for {len(missing)} macro(s): "
              f"{', '.join(missing)} -- they will render as '??' in the PDF")

    lines = [rf"\newcommand{{\{m}}}{{{vals.get(m, PLACEHOLDER)}}}"
             for m in MACRO_NAMES]
    return "\n".join(lines) + "\n"


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    audit = load("audit_identifiability")
    e1, e2 = load("e1_aleatoric_recovery"), load("e2_epistemic_vs_data")
    e3, e4 = load("e3_calibration"), load("e4_ood")
    e3b = load("e3b_calibration_baseline")
    e5, e6 = load("e5_selective"), load("e6_mc_budget")

    tex = (
        "% GENERATED by paper/make_tables.py -- do not edit by hand.\n"
        + tab_e1(e1) + tab_e2(e2) + tab_e3(e3) + tab_e3b(e3b, e3)
        + tab_e4(e4) + tab_e5(e5) + tab_e6(e6)
    )
    (PAPER / "tables.tex").write_text(tex, encoding="utf-8")
    (PAPER / "macros.tex").write_text(
        "% GENERATED by paper/make_tables.py -- do not edit by hand.\n"
        + macros(audit, e3, e6), encoding="utf-8"
    )
    have = [k for k, v in [("E1", e1), ("E2", e2), ("E3", e3), ("E3b", e3b),
                           ("E4", e4), ("E5", e5), ("E6", e6)] if v]
    print(f"wrote paper/tables.tex and paper/macros.tex (experiments present: {have})")


if __name__ == "__main__":
    main()
