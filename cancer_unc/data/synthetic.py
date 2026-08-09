"""Synthetic MRI-like phantoms with an analytically known posterior.

Why bother with synthetic data in a medical imaging project?

Because on real data you can never check whether your uncertainty estimates are
*correct*. You can only check whether they are *useful* (calibration, OOD AUROC,
risk-coverage). Those are downstream proxies. If a model reports "aleatoric
entropy 0.31 nats" on a real mammogram, there is no ground truth to compare it
to -- the true label noise of that image is unobservable.

Here the generative process is written down, so the Bayes-optimal posterior is
available in closed form and every uncertainty quantity has a target value.

Generative model
----------------
For each scan we draw a scalar latent biomarker

    z ~ N(0, 1),

and the label is drawn from a logistic link

    p*(y=1 | z) = sigma(beta * z),          sigma(t) = 1 / (1 + e^{-t}).

The image x is rendered from z plus label-independent nuisance variables
n = (brain shape, lesion position, texture, pixel noise):

    x = render(z, n),        n independent of y given z.

Because n is conditionally independent of y, the conditional independence
y -- z -- x gives

    p*(y | x) = p*(y | z(x)) = sigma(beta * z),

*provided z is identifiable from x*, i.e. z = z(x) is recoverable. That
proviso is not free, so `estimate_latent` implements a matched filter that
recovers z from x, and `tests/test_identifiability.py` checks the recovery
error is small relative to the scale on which sigma(beta*z) varies. Where the
assumption is tight, the oracle below is the true posterior; where it is loose,
the oracle is a slight *under*-estimate of aleatoric uncertainty (the network
sees a noisier z than the oracle does). Reported numbers use it as a reference,
not as gospel.

Consequences we can then compute exactly:

    Bayes error          E_z[ min(p*, 1 - p*) ]
    aleatoric entropy    E_z[ H(p*) ]              <- target for the aleatoric term
    epistemic            0 in the infinite-data limit

`beta` is the dial that controls how much *irreducible* uncertainty exists.
beta -> infinity gives a separable, noise-free problem; beta -> 0 gives pure
coin-flipping. Sweeping it produces the headline experiment: the estimated
aleatoric term should track E_z[H(p*)] across the sweep, while the epistemic
term should stay flat and instead respond to *training set size*.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PhantomConfig:
    """Parameters of the forward rendering model."""

    image_size: int = 64

    # --- label model -------------------------------------------------
    beta: float = 1.6
    """Separability of the logistic link. Controls irreducible label noise."""

    # --- lesion ------------------------------------------------------
    lesion_amp: float = 0.42
    """Peak contrast per unit z. The lesion is hyper-intense for z > 0 and
    hypo-intense for z < 0, so the *sign* of the contrast carries the signal."""

    lesion_sigma_frac: tuple[float, float] = (0.09, 0.15)
    """Lesion radius as a fraction of image size, sampled uniformly."""

    lesion_radius_frac: float = 0.55
    """Lesion centres are confined to this fraction of the brain radius, so
    lesions never straddle the skull boundary."""

    # --- anatomy -----------------------------------------------------
    brain_axes_frac: tuple[float, float] = (0.34, 0.42)
    brain_axes_jitter: float = 0.05
    brain_center_jitter: float = 0.03
    brain_base: float = 0.55
    gradient_amp: float = 0.12
    """Smooth intensity ramp across the brain (bias-field-like)."""

    # --- nuisance ----------------------------------------------------
    texture_std: float = 0.055
    texture_scale: float = 2.2
    """Gaussian-blur sigma (px) applied to white noise to make tissue texture."""

    noise_std: float = 0.05
    """i.i.d. pixel noise added after rendering."""

    edge_softness: float = 1.4
    """Blur applied to the brain mask, in pixels."""


@dataclass(frozen=True)
class ShiftConfig:
    """A named distribution shift, used to build OOD / corruption test sets."""

    name: str
    kind: str  # "none" | "noise" | "blur" | "modality" | "novel_morphology"
               # | "decoupled"
    severity: float = 0.0
    semantic: bool = False
    """True if the shift destroys the link between image and label, as opposed
    to a covariate shift where the label is still recoverable.

    The distinction matters for interpreting results: under covariate shift we
    want the model to stay *calibrated*; under semantic shift we want it to
    *abstain*. Reporting both under one 'OOD' heading hides that.

    Getting this flag right turned out to be subtle. The obvious candidate --
    swapping the lesion for an unseen annular morphology -- is *not* a semantic
    shift, because the rendering still sets the lesion amplitude to
    `lesion_amp * z`. The signed contrast therefore still carries the latent,
    the network reads it off a ring about as well as off a blob, and accuracy
    is unchanged. Measured on this benchmark: novel-morphology accuracy 0.728
    against 0.733 in-distribution, a difference well inside noise. It is a
    covariate shift wearing a semantic costume.

    `decoupled` is the honest version: same unseen morphology, but the
    amplitude is drawn independently of z, so the image carries *no*
    information about the label and p(y|x) = 1/2 exactly. No model can beat
    chance, so the correct behaviour is to abstain -- which is what makes
    epistemic uncertainty, rather than aleatoric, the right detector."""


SHIFTS: dict[str, ShiftConfig] = {
    "id": ShiftConfig("in-distribution", "none"),
    # covariate shift: label still meaningful, image degraded
    "noise_1": ShiftConfig("noise (mild)", "noise", 0.10),
    "noise_2": ShiftConfig("noise (moderate)", "noise", 0.20),
    "noise_3": ShiftConfig("noise (severe)", "noise", 0.35),
    "blur_1": ShiftConfig("blur (mild)", "blur", 1.0),
    "blur_2": ShiftConfig("blur (moderate)", "blur", 2.0),
    "blur_3": ShiftConfig("blur (severe)", "blur", 3.5),
    # acquisition shift: different "scanner"/sequence
    "modality": ShiftConfig("modality shift", "modality", 1.0),
    # unseen morphology, but the latent is still encoded -> covariate, not semantic
    "novel": ShiftConfig("novel morphology", "novel_morphology", 1.0),
    # unseen morphology AND the latent is not encoded -> genuinely semantic
    "decoupled": ShiftConfig("decoupled lesion", "decoupled", 1.0, semantic=True),
}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    ax = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    return np.meshgrid(ax, ax, indexing="ij")


def _brain_mask(cfg: PhantomConfig, rng: np.random.Generator) -> np.ndarray:
    """Soft elliptical head mask with jittered shape, centre and rotation."""
    yy, xx = _coords(cfg.image_size)

    a = cfg.brain_axes_frac[1] * 2 + rng.uniform(-1, 1) * cfg.brain_axes_jitter
    b = cfg.brain_axes_frac[0] * 2 + rng.uniform(-1, 1) * cfg.brain_axes_jitter
    cy = rng.uniform(-1, 1) * cfg.brain_center_jitter
    cx = rng.uniform(-1, 1) * cfg.brain_center_jitter
    theta = rng.uniform(0, np.pi)

    ys, xs = yy - cy, xx - cx
    yr = ys * np.cos(theta) - xs * np.sin(theta)
    xr = ys * np.sin(theta) + xs * np.cos(theta)

    r2 = (yr / b) ** 2 + (xr / a) ** 2
    hard = (r2 <= 1.0).astype(np.float64)
    return gaussian_filter(hard, cfg.edge_softness), (cy, cx, a, b, theta)


def _lesion_kernel(
    cfg: PhantomConfig,
    rng: np.random.Generator,
    geom: tuple[float, float, float, float, float],
    morphology: str,
) -> np.ndarray:
    """Unit-peak spatial profile of the lesion (amplitude applied by caller)."""
    yy, xx = _coords(cfg.image_size)
    cy, cx, a, b, _ = geom

    # place the centre inside the brain, away from the boundary
    rad = rng.uniform(0.0, cfg.lesion_radius_frac)
    ang = rng.uniform(0.0, 2 * np.pi)
    ly = cy + rad * b * np.sin(ang)
    lx = cx + rad * a * np.cos(ang)

    sigma = rng.uniform(*cfg.lesion_sigma_frac) * 2.0
    d2 = (yy - ly) ** 2 + (xx - lx) ** 2

    if morphology == "blob":
        return np.exp(-d2 / (2 * sigma**2))

    if morphology == "ring":
        # annular / rim-enhancing lesion: same total energy scale, different shape.
        # Never present at training time -> genuine semantic novelty.
        r = np.sqrt(d2)
        thickness = sigma * 0.40
        return np.exp(-((r - sigma) ** 2) / (2 * thickness**2))

    raise ValueError(f"unknown morphology: {morphology!r}")


def _texture(cfg: PhantomConfig, rng: np.random.Generator, scale: float) -> np.ndarray:
    raw = rng.standard_normal((cfg.image_size, cfg.image_size))
    sm = gaussian_filter(raw, scale)
    sd = sm.std()
    return sm / sd if sd > 1e-8 else sm


def render_one(
    z: float,
    cfg: PhantomConfig,
    rng: np.random.Generator,
    shift: ShiftConfig = SHIFTS["id"],
) -> np.ndarray:
    """Render a single scan from latent z under an optional distribution shift."""
    mask, geom = _brain_mask(cfg, rng)

    tex_scale = cfg.texture_scale
    base, grad_amp = cfg.brain_base, cfg.gradient_amp
    morphology = "blob"

    if shift.kind == "modality":
        # different "scanner": inverted tissue contrast, finer texture, and a
        # stronger bias field. Lesion contrast direction is preserved, so the
        # label is still well defined -- this is acquisition shift, not novelty.
        tex_scale = cfg.texture_scale * 0.45
        base, grad_amp = 0.38, cfg.gradient_amp * 2.0
    elif shift.kind in ("novel_morphology", "decoupled"):
        morphology = "ring"

    yy, xx = _coords(cfg.image_size)
    img = np.full_like(yy, base)
    img += grad_amp * (yy * np.cos(geom[4]) + xx * np.sin(geom[4]))
    img += cfg.texture_std * _texture(cfg, rng, tex_scale)

    if shift.kind == "modality":
        img = 1.0 - img  # invert tissue contrast

    kernel = _lesion_kernel(cfg, rng, geom, morphology)
    if shift.kind == "decoupled":
        # Amplitude drawn independently of z: the image is rendered from a fresh
        # standard normal, so it carries no information about the label that was
        # sampled from sigma(beta * z). The marginal appearance statistics match
        # the training distribution -- only the image/label *link* is severed,
        # which is what isolates semantic novelty from covariate shift.
        amp_latent = rng.standard_normal()
    else:
        amp_latent = z
    img = img + (cfg.lesion_amp * amp_latent) * kernel

    img = img * mask  # background is air

    if shift.kind == "blur":
        img = gaussian_filter(img, shift.severity)
    if shift.kind == "noise":
        img = img + rng.standard_normal(img.shape) * shift.severity

    img = img + rng.standard_normal(img.shape) * cfg.noise_std
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# oracle quantities
# --------------------------------------------------------------------------
def bayes_posterior(z: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    """p*(y=1 | z) -- the Bayes-optimal posterior."""
    return 1.0 / (1.0 + np.exp(-cfg.beta * np.asarray(z, dtype=np.float64)))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """H(p) in nats, safe at the endpoints."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def oracle_stats(cfg: PhantomConfig, n: int = 200_000, seed: int = 0) -> dict[str, float]:
    """Monte-Carlo the exact quantities implied by the generative model.

    These are the targets the model's uncertainty estimates are graded against.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    p = bayes_posterior(z, cfg)
    return {
        "bayes_error": float(np.minimum(p, 1 - p).mean()),
        "aleatoric_entropy": float(binary_entropy(p).mean()),
        "epistemic_entropy": 0.0,  # by definition, at infinite data
        "bayes_nll": float(-(p * np.log(np.clip(p, 1e-12, 1))
                             + (1 - p) * np.log(np.clip(1 - p, 1e-12, 1))).mean()),
        "bayes_brier": float((p * (1 - p) ** 2 + (1 - p) * p**2).mean()),
    }


def matched_filter_statistic(images: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    """Raw matched-filter response: a statistic that is affine in z.

    The lesion is the only structure whose amplitude scales with z, so
    correlating the image against a bank of normalised lesion templates and
    taking the signed extremal response gives a quantity linear in z plus
    estimation noise. This returns the statistic *uncalibrated* -- converting
    it to z's units requires knowing the gain, which `estimate_latent` fits.

    Two details that matter for the estimator to be any good:

    * templates are energy-normalised (divided by ||h||), otherwise the wider
      templates win purely by integrating more pixels and the sigma search
      degenerates to always picking the largest scale;
    * the search is restricted to the interior of the brain, because the soft
      skull edge is a high-contrast structure that produces larger filter
      responses than any lesion and would otherwise dominate the extremum.
    """
    imgs = np.asarray(images, dtype=np.float64)
    if imgs.ndim == 4:
        imgs = imgs[:, 0]
    n = imgs.shape[-1]

    # interior mask: exclude the skull boundary and the air background
    yy, xx = _coords(n)
    a = cfg.brain_axes_frac[1] * 2 - cfg.brain_axes_jitter
    b = cfg.brain_axes_frac[0] * 2 - cfg.brain_axes_jitter
    margin = 0.80  # stay inside the smallest plausible brain
    interior = ((yy / (b * margin)) ** 2 + (xx / (a * margin)) ** 2) <= 1.0

    sigmas = np.linspace(cfg.lesion_sigma_frac[0], cfg.lesion_sigma_frac[1], 5) * 2.0
    best = np.zeros(imgs.shape[0])

    for s in sigmas:
        px = s * (n / 2.0)
        # Gaussian blur is correlation with a Gaussian template; dividing by the
        # template's L2 norm (1 / (2 sqrt(pi) sigma) in 2-D) makes responses
        # comparable across scales.
        norm = 1.0 / (2.0 * np.sqrt(np.pi) * px)
        for i, im in enumerate(imgs):
            resp = gaussian_filter(im - im[interior].mean(), px) / norm
            vals = resp[interior]
            pos, neg = vals.max(), vals.min()
            signed = pos if abs(pos) >= abs(neg) else neg
            if abs(signed) > abs(best[i]):
                best[i] = signed
    return best


def estimate_latent(
    images: np.ndarray,
    cfg: PhantomConfig,
    calibration: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Estimate z from rendered images, in z's own units.

    The matched-filter statistic is affine in z with an unknown gain and offset
    set by the rendering constants. `calibration=(stat, z)` supplies a separate
    labelled draw from which the affine map is fitted by least squares; the
    fitted map is then applied to `images`. Fitting on a *held-out* draw is what
    keeps the audit honest -- calibrating on the same data being evaluated would
    absorb part of the estimation error into the fit and make z look more
    identifiable than it is.

    This function exists only to audit the assumption behind `bayes_posterior`.
    It is never shown to the model.
    """
    stat = matched_filter_statistic(images, cfg)
    if calibration is None:
        sd = stat.std()
        return (stat - stat.mean()) / sd if sd > 1e-8 else stat
    cal_stat, cal_z = calibration
    gain, offset = np.polyfit(cal_stat, cal_z, 1)
    return gain * stat + offset


# --------------------------------------------------------------------------
# dataset assembly
# --------------------------------------------------------------------------
@dataclass
class PhantomSplit:
    images: np.ndarray  # (N, 1, H, W) float32
    labels: np.ndarray  # (N,) int64
    z: np.ndarray  # (N,) float64   -- latent, for oracle comparisons only
    p_true: np.ndarray  # (N,) float64 -- Bayes posterior, oracle only
    shift: str = "id"

    def __len__(self) -> int:
        return len(self.labels)


def make_split(
    n: int,
    cfg: PhantomConfig,
    seed: int,
    shift: ShiftConfig = SHIFTS["id"],
    resample_labels: bool = True,
) -> PhantomSplit:
    """Draw n scans.

    `resample_labels=False` gives labels at their conditional mode instead of
    sampling them. That produces a *noise-free* version of the same images and
    is how we isolate whether an uncertainty signal is genuinely aleatoric.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    p = bayes_posterior(z, cfg)
    y = (rng.random(n) < p).astype(np.int64) if resample_labels else (p > 0.5).astype(np.int64)

    imgs = np.empty((n, 1, cfg.image_size, cfg.image_size), dtype=np.float32)
    for i in range(n):
        imgs[i, 0] = render_one(float(z[i]), cfg, rng, shift)

    return PhantomSplit(imgs, y, z, p, shift.name)


def make_benchmark(
    cfg: PhantomConfig,
    n_train: int = 6000,
    n_val: int = 1500,
    n_test: int = 3000,
    n_ood: int = 1500,
    seed: int = 0,
    shifts: tuple[str, ...] = ("noise_1", "noise_2", "noise_3",
                               "blur_1", "blur_2", "blur_3",
                               "modality", "novel", "decoupled"),
) -> dict[str, PhantomSplit]:
    """Full benchmark: train / val / test in-distribution, plus shifted sets.

    Separate seeds per split so no phantom is shared across them.
    """
    out = {
        "train": make_split(n_train, cfg, seed + 1),
        "val": make_split(n_val, cfg, seed + 2),
        "test": make_split(n_test, cfg, seed + 3),
    }
    for i, key in enumerate(shifts):
        out[key] = make_split(n_ood, cfg, seed + 10 + i, SHIFTS[key])
    return out
