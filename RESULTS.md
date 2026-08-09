# Results

## Oracle validity bracket

A correct aleatoric estimate must fall inside this interval. The lower
bound assumes the latent is perfectly recoverable; the upper bound comes
from a deliberately suboptimal matched filter, which can only over-state
noise.

| lower (analytic) | upper (matched filter) | width | relative |
|---|---|---|---|
| 0.5138 | 0.5474 | 0.0335 | 0.065 |

`corr(ẑ, z) = 0.844`, mean |p̂ − p*| = 0.101

## E1 — aleatoric recovery across the label-noise sweep

| β | target E_z[H(p*)] | estimated aleatoric | estimated epistemic | Bayes err | model err | corr(alea, H(p*)) |
|---|---|---|---|---|---|---|
| 0.6 | 0.6532 | 0.6619 | 0.0003 | 0.3865 | 0.3960 | 0.742 |
| 1.0 | 0.5993 | 0.6095 | 0.0004 | 0.3250 | 0.3416 | 0.809 |
| 1.6 | 0.5134 | 0.5402 | 0.0005 | 0.2558 | 0.2680 | 0.863 |
| 2.5 | 0.4062 | 0.4243 | 0.0006 | 0.1889 | 0.2020 | 0.902 |
| 4.0 | 0.2904 | 0.3071 | 0.0006 | 0.1285 | 0.1436 | 0.928 |

## E2 — epistemic vs training set size

| n_train | epistemic | aleatoric | target aleatoric | model err |
|---|---|---|---|---|
| 250 | 0.0053 | 0.5144 | 0.5134 | 0.2724 |
| 500 | 0.0027 | 0.4961 | 0.5134 | 0.2672 |
| 1000 | 0.0005 | 0.5285 | 0.5134 | 0.2720 |
| 2000 | 0.0023 | 0.5036 | 0.5134 | 0.2748 |
| 4000 | 0.0003 | 0.5275 | 0.5134 | 0.2652 |

log-log slope of epistemic vs n: **-0.845**

## E3 — calibration (fitted T = 0.998)

| split | acc | ECE raw [95% CI] | ECE cal | NLL raw | NLL cal | Δacc |
|---|---|---|---|---|---|---|
| `val` — in-distribution | 0.730 | 0.0255 [0.0188, 0.0530] | 0.0237 | 0.5303 | 0.5303 | -0.0025 |
| `test` — in-distribution | 0.733 | 0.0166 [0.0119, 0.0359] | 0.0194 | 0.5304 | 0.5306 | 0.0000 |
| `noise_1` — noise (mild) | 0.738 | 0.0218 [0.0165, 0.0511] | 0.0230 | 0.5091 | 0.5089 | 0.0033 |
| `noise_2` — noise (moderate) | 0.744 | 0.0228 [0.0211, 0.0548] | 0.0235 | 0.5384 | 0.5389 | -0.0008 |
| `noise_3` — noise (severe) | 0.662 | 0.0717 [0.0510, 0.1004] | 0.0727 | 0.6104 | 0.6120 | 0.0008 |
| `blur_1` — blur (mild) | 0.718 | 0.0250 [0.0178, 0.0544] | 0.0255 | 0.5529 | 0.5526 | 0.0008 |
| `blur_2` — blur (moderate) | 0.733 | 0.0391 [0.0323, 0.0711] | 0.0400 | 0.5484 | 0.5485 | 0.0008 |
| `blur_3` — blur (severe) | 0.713 | 0.0536 [0.0410, 0.0820] | 0.0555 | 0.5651 | 0.5657 | -0.0008 |
| `modality` — modality shift | 0.669 | 0.0770 [0.0535, 0.1006] | 0.0783 | 0.6063 | 0.6082 | 0.0008 |
| `novel` — novel morphology | 0.728 | 0.0270 [0.0220, 0.0595] | 0.0249 | 0.5337 | 0.5338 | 0.0000 |
| `decoupled` — decoupled lesion * | 0.507 | 0.2346 [0.2123, 0.2629] | 0.2364 | 0.9041 | 0.9089 | 0.0000 |

`*` = semantic shift. Note every calibrated ECE lies inside the
raw estimate's 95% interval: temperature scaling changes nothing
here, because the ensemble is already calibrated (T ≈ 1).

Δacc is the accuracy shift caused by scaling. It is *not* forced
to zero: the argmax-invariance theorem holds for a single softmax,
not for the mixture that an ensemble predictive actually is.

## E3b — miscalibrated baseline (fitted T = 1.026)

A single model, no heteroscedastic head, dropout off, stopped at the
final epoch instead of best validation NLL — each of the main model's
calibration mechanisms removed. Accuracy is bit-identical before and
after scaling here, since a single softmax satisfies argmax-invariance
exactly.

| split | acc | single ECE raw | single ECE cal | ens ECE raw | ens ECE cal |
|---|---|---|---|---|---|
| in-distribution | 0.728 | 0.0201 | 0.0179 | 0.0255 | 0.0237 |
| in-distribution | 0.731 | 0.0155 | 0.0124 | 0.0166 | 0.0194 |
| noise (mild) | 0.744 | 0.0207 | 0.0217 | 0.0218 | 0.0230 |
| noise (moderate) | 0.733 | 0.0745 | 0.0717 | 0.0228 | 0.0235 |
| noise (severe) | 0.677 | 0.1531 | 0.1488 | 0.0717 | 0.0727 |
| blur (mild) | 0.713 | 0.0221 | 0.0233 | 0.0250 | 0.0255 |
| blur (moderate) | 0.738 | 0.0242 | 0.0247 | 0.0391 | 0.0400 |
| blur (severe) | 0.724 | 0.0128 | 0.0145 | 0.0536 | 0.0555 |
| modality shift | 0.690 | 0.0634 | 0.0594 | 0.0770 | 0.0783 |
| novel morphology | 0.729 | 0.0362 | 0.0317 | 0.0270 | 0.0249 |
| decoupled lesion * | 0.513 | 0.2413 | 0.2373 | 0.2346 | 0.2364 |

## E4 — OOD detection (AUROC)

| score | noise (mild) | noise (moderate) | noise (severe) | blur (mild) | blur (moderate) | blur (severe) | modality shift | novel morphology | decoupled lesion * |
|---|---|---|---|---|---|---|---|---|---|
| msp | 0.472 | 0.464 | 0.509 | 0.507 | 0.472 | 0.435 | 0.480 | 0.483 | 0.491 |
| entropy | 0.472 | 0.464 | 0.509 | 0.507 | 0.472 | 0.435 | 0.480 | 0.483 | 0.491 |
| energy | 0.492 | 0.507 | 0.520 | 0.510 | 0.483 | 0.460 | 0.494 | 0.483 | 0.491 |
| epistemic | 0.558 | 0.870 | 0.937 | 0.493 | 0.439 | 0.681 | 0.584 | 0.533 | 0.522 |
| aleatoric | 0.471 | 0.455 | 0.435 | 0.507 | 0.473 | 0.435 | 0.479 | 0.483 | 0.490 |
| mahalanobis | 0.784 | 0.986 | 1.000 | 0.476 | 0.607 | 0.917 | 0.859 | 0.562 | 0.548 |

`aleatoric` is a negative control: it should *not* win on the semantic shift (`*`).

## E5 — selective prediction

**In-distribution**

| confidence | AURC | E-AURC | risk@50% cov | cov@5% risk |
|---|---|---|---|---|
| msp | 0.1489 | 0.1093 | 0.1472 | 0.138 |
| neg_total_entropy | 0.1489 | 0.1093 | 0.1472 | 0.138 |
| neg_aleatoric | 0.1489 | 0.1093 | 0.1472 | 0.138 |
| neg_epistemic | 0.2461 | 0.2065 | 0.2400 | 0.000 |

**Mixed ID + novel stream**

| confidence | AURC | E-AURC | risk@50% cov | cov@5% risk |
|---|---|---|---|---|
| msp | 0.4244 | 0.2674 | 0.4249 | 0.001 |
| neg_total_entropy | 0.4244 | 0.2674 | 0.4249 | 0.001 |
| neg_epistemic | 0.4955 | 0.3384 | 0.4843 | 0.000 |

## E6 — MC sample budget

| T | epistemic | aleatoric | total |
|---|---|---|---|
| 2 | 0.00355 | 0.53275 | 0.53631 |
| 4 | 0.00544 | 0.53434 | 0.53978 |
| 8 | 0.00644 | 0.53421 | 0.54065 |
| 16 | 0.00684 | 0.53372 | 0.54056 |
| 32 | 0.00709 | 0.53379 | 0.54089 |
| 64 | 0.00722 | 0.53371 | 0.54092 |

Epistemic rises with T: the small-T estimate is biased downward by O(1/T), as Jensen's inequality requires.

