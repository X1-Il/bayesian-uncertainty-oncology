# Bayesian Uncertainty Calibration & OOD Detection for Oncological Scans

In screening, a model that is wrong is survivable; a model that is wrong **and
confident** is not. This project treats the predictive distribution — not the
prediction — as the object of study: where the uncertainty comes from, whether
the stated probabilities mean what they say, and whether they support a deferral
decision.

The central methodological choice is that the data are **synthetic by design**.
On a real mammogram there is no ground truth for "how ambiguous is this scan",
so an uncertainty estimate can only be judged by proxies. Here the generative
process is written down, so the Bayes-optimal posterior is available in closed
form and every uncertainty quantity has a **target value it can be checked
against**. That turns the usual benchmark run into a falsifiable experiment.

The pipeline is dataset-agnostic: `cancer_unc.data.load_image_folder` drops a
real Kaggle brain-MRI set into the same interface, and every experiment except
the two oracle checks runs unchanged.



---

## The mathematics

### 1. Decomposing predictive uncertainty

With an approximate posterior `q(w)` over weights, the predictive distribution is
`p(y|x) = E_q[p(y|x,w)]`, and the exact identity (Depeweg et al. 2018; BALD,
Houlsby et al. 2011) is

```
H[ E_q p(y|x,w) ]   =   I(y ; w | x)   +   E_q H[ p(y|x,w) ]
   total                 epistemic          aleatoric
```

Total uncertainty splits into the part that *disagreement between plausible
models* explains, and the part that *every* plausible model agrees is
irreducible. Both terms are non-negative, so this is a genuine decomposition
rather than a heuristic split. Two properties make it testable:

- **epistemic → 0** as the training set grows (the posterior concentrates);
- **aleatoric → `E_z[H(p*)]`**, a constant fixed by the data-generating process.

A second, independent route via the law of total variance,
`Var[p] = E_q[Var(p|w)] + Var_q[E(p|w)]`, is computed alongside as a cross-check.

Posterior samples come from two sources — **MC dropout** and a **deep ensemble** —
because the decomposition is a property of the posterior approximation, not of
one trick.

### 2. Aleatoric uncertainty as learned logit noise

The heteroscedastic head (Kendall & Gal 2017) places a Gaussian on the logits and
integrates the softmax over it:

```
u_t = f(x) + σ(x)·ε_t ,  ε_t ~ N(0,I)
L   = −log (1/S) Σ_t softmax(u_t)_y
```

computed entirely in log-space for stability. Note `σ(x)` is learned **without
supervision**: inflating it on an ambiguous input raises the log-mean-exp of the
correct-class log-probability even as it lowers the mean logit. That is what
makes it an aleatoric estimate rather than a regulariser.

### 3. Calibration

A classifier is calibrated if `P(y = ŷ | conf = c) = c`. Estimating this
requires binning, and the module is explicit about what binning costs:

- binned ECE is **biased downward** (within-bin variation is averaged away), and
  no choice of bin count is unbiased;
- equal-width bins under-resolve exactly the high-confidence region where the
  clinically relevant errors live, so **`adaptive_ece`** (equal-mass bins) is the
  estimator to trust;
- ECE has real sampling variance, so every ECE here is reported with a
  **95% bootstrap CI**, and comparisons are made on intervals, not point
  estimates.

Murphy's decomposition `BS = reliability − resolution + uncertainty` separates
calibration error from discriminative power, showing whether a post-hoc fix
bought calibration at the cost of sharpness.

**Temperature scaling** (`p = softmax(z/T)`, one parameter, fitted on held-out
NLL) is the default correction for a reason that is provable rather than
empirical: dividing every logit by the same `T > 0` is strictly monotone, so
`argmax` — and therefore accuracy — is **exactly preserved**.

That theorem is about *one* softmax, and this is where it is routinely
over-applied. The predictive distribution of an ensemble is a **mixture**,
`p = (1/M) Σ_m softmax(z_m)`. Since softmax is non-linear,
`mean-of-softmax ≠ softmax-of-mean`, so scaling the *averaged* logit calibrates a
different estimator than the one being reported. `EnsembleTemperatureScaler`
scales inside the average, `p_T = (1/M) Σ_m softmax(z_m / T)` — and for that
mixture **the argmax guarantee does not hold**: raising `T` flattens each member
toward uniform, re-weighting how much each contributes, so the winning class can
change. We therefore *measure* the accuracy shift
(`accuracy_shift_from_scaling`) instead of assuming it is zero, while still
asserting the exact theorem on the single-model object it applies to.

This distinction is not cosmetic: an assertion that demanded exact invariance for
the mixture aborted a perfectly correct run during development.

### 4. Selective prediction

For a selection rule `g(x) = 1[κ(x) ≥ τ]`, selective risk is
`R(τ) = E[loss·g] / E[g]` — the error rate *among cases the model chose to
answer*. Its area over coverage is AURC.

AURC alone is misleading: it is bounded below by the model's own error rate. The
**excess AURC** `E-AURC = AURC − AURC_optimal` isolates ranking quality from
classifier quality. This matters directly here — since temperature scaling is
monotone, it **cannot change the ranking, the risk-coverage curve, or AURC at
all**. A reported AURC improvement from temperature scaling is necessarily a bug.
What calibration buys is *threshold semantics*: `coverage_at_risk` can only hit a
target risk if the probabilities mean what they say.

### 5. OOD detection

Five scores are compared — MSP, predictive entropy, energy `−logsumexp(z)`,
**epistemic MI**, and Mahalanobis distance in feature space — with AUROC computed
via the Mann–Whitney identity **with tie correction** (several scores saturate,
and a naive threshold sweep rewards that saturation).

The evaluation keeps **covariate shift** (noise, blur, scanner) separate from
**semantic shift** (a lesion morphology never seen in training). The right
behaviour differs: stay calibrated on the former, abstain on the latter.
Averaging them into one "OOD AUROC" hides the only interesting result. The
aleatoric score is carried through as a **negative control** — if it detects
novelty as well as the epistemic score does, the two terms are not actually
separated and every other result is suspect.

---

## The benchmark

`z ~ N(0,1)` is a latent biomarker; the label is `y ~ Bernoulli(σ(βz))`; the
image renders `z` as lesion contrast inside an elliptical head phantom, with
label-independent nuisance (shape, position, texture, bias field, pixel noise).
Because nuisance is conditionally independent of `y`, the chain `y — z — x` gives
`p*(y|x) = σ(βz)` **provided `z` is identifiable from `x`**.

That proviso is not assumed. `exp_identifiability` **brackets** it: the analytic
oracle is a lower bound on attainable aleatoric entropy, and a matched-filter
estimator of `z` — deliberately suboptimal, so it can only over-state noise —
gives an upper bound. A correct aleatoric estimate must land inside
`[A*, Â]`. Reporting a bracket rather than a pass/fail verdict is the difference
between a claim that can be checked and one that merely sounds rigorous.

`β` is the dial controlling irreducible label noise, and it is what makes E1
possible: sweeping it moves the analytic target along a known curve.

**Shift suite:** noise ×3 severities, blur ×3, a "modality shift" (inverted
tissue contrast, finer texture, stronger bias field), an unseen annular lesion
morphology, and a `decoupled` condition.

### An unseen appearance is not an unseen task

The annular morphology was the original candidate for "semantic shift". Measured,
it isn't one: accuracy there is **0.728 vs 0.733 in-distribution**. The renderer
still sets lesion amplitude to `lesion_amp * z`, so the signed contrast still
carries the latent and the network reads it off a ring about as well as off a
blob. It's a covariate shift wearing a semantic costume — and reporting it as
semantic would have made E4's central comparison vacuous.

`decoupled` is the corrected version: same unseen shape, but amplitude drawn from
an *independent* normal. The image then carries no label information,
`p*(y|x) = 1/2` exactly, and no model can beat chance — so abstention is the only
correct response. Marginal appearance statistics still match training, which
isolates the severed image↔label *link* from any change in appearance.
`tests/test_synthetic.py` asserts the difference directly: under `novel` the
recovered contrast correlates with `z` (r > 0.4), under `decoupled` it does not
(r < 0.2).

---

## Experiments

| | Question | Why it is not just a benchmark run |
|---|---|---|
| **E1** | Does the aleatoric estimate recover the analytic label noise? | Compared against a closed form, per-`β` and per-example |
| **E2** | Does epistemic vanish as data grows, while aleatoric stays put? | Both terms come from the same forward passes — no rescaling can fake a decaying curve beside a flat one |
| **E3** | What does temperature scaling fix, and does it survive shift? | Fitted in-distribution; the interesting result is where it *fails* |
| **E4** | Which score detects novelty? | Covariate vs semantic shift kept separate; aleatoric as negative control |
| **E5** | What does uncertainty buy in a deferral workflow? | Evaluated on a mixed ID+novel stream, not just clean ID data |
| **E6** | How many MC samples are actually needed? | Exposes the O(1/T) downward bias in the epistemic estimate |

---

## Running it

```bash
pip install -r requirements.txt

python main.py --quick        # ~6 min, exercises the whole pipeline
python main.py --full         # the full study
python main.py --stage e3     # one experiment
python main.py --plots-only   # regenerate figures from results/*.json

pytest tests/ -q
```

Outputs are **scoped by preset**: `results/full/` and `figures/full/` for
`--full`, `results/quick/` and `figures/quick/` for `--quick`, and likewise for
`checkpoints/`. Each JSON also carries a `_preset` stamp.

This is a correctness guard, not tidiness. `--quick` is the default when neither
flag is given, and the two presets produce structurally identical output that
differs by an order of magnitude in training budget. Sharing one directory means
a single stray invocation silently replaces a two-hour study with smoke-test
numbers, and nothing in the file says which is which. That happened during
development, and cost a full re-run. `summarize.py` now prints a warning banner
when reading quick results, and `make_tables.py` refuses to quietly typeset them
into the paper.

Everything is deterministic given `--seed` — the sweeps reproduce bit-for-bit
across separate processes.

### Resuming

The full study is a couple of hours on CPU, and E1/E2 train a fresh ensemble per
sweep point. Those sweeps write partial results after **every** point and skip
work already on disk, so an interrupted run resumes instead of restarting:

```
[E1] aleatoric recovery across the label-noise sweep
  resuming: 3 point(s) already done, 2 to go
```

Each sweep result carries a `complete` flag, so a partial sweep is never
mistaken for a finished one. `--load-checkpoints` additionally reuses a trained
ensemble for E3–E6 rather than retraining it.

### Using a real dataset

```python
from cancer_unc.data import load_image_folder, stratified_split

split  = load_image_folder("data/raw/brain_mri", class_dirs=("no", "yes"))
splits = stratified_split(split)   # class-stratified train/val/test
```

`z` and `p_true` come back as NaN — on real data the latent and the true
posterior do not exist. Every oracle comparison is guarded on those being
finite and is skipped automatically; calibration, OOD, and risk-coverage run in
full.

---

## Layout

```
cancer_unc/
  data/synthetic.py        phantom generator, oracle quantities, shift suite
  data/loaders.py          torch wrappers + real-data drop-in
  models/nets.py           MC-dropout CNN, heteroscedastic head
  uncertainty/
    decomposition.py       entropy + variance decompositions, nested MC
    calibration.py         ECE family, Brier decomposition, temperature/vector scaling
    ood.py                 five scores, AUROC/AUPR/FPR@95 with tie handling
  eval/risk_coverage.py    selective risk, AURC, E-AURC, coverage@risk
  experiments.py           E1-E6 + identifiability audit
  train.py, plots.py
  experiments.py           E1-E6 + identifiability audit
  train.py, plots.py
paper/
  report.tex               NeurIPS-format write-up
  make_tables.py           results/*.json -> tables.tex + macros.tex
tests/                     44 tests, mostly on mathematical identities
main.py                    CLI
summarize.py               results/*.json -> RESULTS.md
```

## The report

`paper/report.tex` is a NeurIPS-format write-up. Every number in it — tables and
inline figures in the prose alike — is generated by `paper/make_tables.py` from
`results/*.json`; nothing is hand-copied, so the paper cannot drift out of sync
with the code that produced it.

```bash
python main.py --full          # produce results/
python paper/make_tables.py    # regenerate tables.tex + macros.tex
cd paper && pdflatex report && pdflatex report
```

On Overleaf, start from the official NeurIPS 2024 template and drop `report.tex`,
`tables.tex`, `macros.tex` and `figures/` in. Without the style file a fallback
preamble reproduces the layout closely enough to read and page-count.

## References

- Murphy (1973), *A New Vector Partition of the Probability Score*
