"""Backbone and heads.

The network is deliberately small: at 64x64 on CPU, a wide ResNet buys nothing
except wall-clock time, and the object of study here is the *uncertainty*, not
the last half-point of accuracy. What matters is that the architecture supports
the two mechanisms we need:

  1. dropout at test time  -> a variational posterior over weights (epistemic)
  2. a predicted logit variance -> input-dependent label noise (aleatoric)

Heteroscedastic classification (Kendall & Gal, NeurIPS 2017)
------------------------------------------------------------
The head emits a mean logit vector f(x) and a log-variance s(x) = log sigma^2(x).
We place a Gaussian on the *logits* and integrate the softmax over it:

    u_t = f(x) + sigma(x) * eps_t,      eps_t ~ N(0, I),  t = 1..S
    p(y = c | x) ~= (1/S) sum_t softmax(u_t)_c

The loss is the negative log of that MC estimate:

    L = -log (1/S) sum_t exp( u_{t,c} - logsumexp_c' u_{t,c'} )
      = -[ logsumexp_t ( u_{t,c} - logsumexp_c' u_{t,c'} ) - log S ]

Written that way it is computed entirely in log-space, so it stays stable when
sigma is large. Note the loss is *not* simply "cross-entropy plus a penalty":
the sigma term is learned without supervision, because inflating sigma on an
ambiguous input raises the log-mean-exp of the correct-class log-probability
even as it lowers the mean logit. That is what makes it an aleatoric estimate
rather than a regulariser.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv-BN-ReLU x2 + pool, with 2-D dropout kept active at inference."""

    def __init__(self, c_in: int, c_out: int, p_drop: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        # Dropout2d (channel-wise) rather than element-wise: with convolutional
        # features, independent per-pixel masking is largely averaged away by
        # the spatial pooling, which collapses the induced weight posterior.
        self.drop = nn.Dropout2d(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.body(x)))


class BayesianCNN(nn.Module):
    """Small CNN with MC-dropout and an optional heteroscedastic head.

    Parameters
    ----------
    heteroscedastic:
        If True the head also predicts log sigma^2(x) per class and the model is
        trained with the integrated-softmax loss above. If False the model is a
        plain classifier and all of its uncertainty is epistemic-by-construction
        (the aleatoric term then only picks up the softmax's own entropy).
    """

    def __init__(
        self,
        in_ch: int = 1,
        n_classes: int = 2,
        width: int = 32,
        p_drop: float = 0.20,
        heteroscedastic: bool = True,
        n_logit_samples: int = 16,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.heteroscedastic = heteroscedastic
        self.n_logit_samples = n_logit_samples

        w = width
        # Stride-2 stem, as in the ResNet family. Running the first block at
        # full 64x64 resolution costs ~4x the rest of the network combined and
        # buys nothing here: the lesion is a smooth blob 6-10 px across, so
        # nothing the label depends on lives at the finest scale. On a CPU this
        # is the difference between a 150 s epoch and a 35 s one.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ConvBlock(w, w * 2, p_drop * 0.75),
            ConvBlock(w * 2, w * 4, p_drop),
            ConvBlock(w * 4, w * 4, p_drop),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc_mean = nn.Linear(w * 4, n_classes)
        self.fc_logvar = nn.Linear(w * 4, n_classes) if heteroscedastic else None

        if self.fc_logvar is not None:
            # start near sigma^2 = e^-3 ~ 0.05 so early training is dominated by
            # the mean path; the variance grows only where it actually helps.
            nn.init.zeros_(self.fc_logvar.weight)
            nn.init.constant_(self.fc_logvar.bias, -3.0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate features -- consumed by the Mahalanobis OOD score."""
        h = self.features(self.stem(x))
        return torch.flatten(self.pool(h), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.drop(self.embed(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h) if self.fc_logvar is not None else None
        if logvar is not None:
            # keep sigma^2 in [e^-8, e^4]; unbounded logvar lets the model buy
            # loss reduction by declaring everything maximally noisy
            logvar = logvar.clamp(-8.0, 4.0)
        return mean, logvar

    # -- losses -----------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.forward(x)
        if logvar is None:
            return F.cross_entropy(mean, y)
        return heteroscedastic_ce(mean, logvar, y, self.n_logit_samples)

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def probs_from(
        self, mean: torch.Tensor, logvar: torch.Tensor | None,
        n_logit_samples: int | None = None, temperature: float = 1.0,
    ) -> torch.Tensor:
        """Marginalise logit noise into class probabilities, at temperature T.

        `temperature` divides the logits *before* the softmax and before the
        noise is added, which is what makes per-member temperature scaling of an
        ensemble well defined (see `EnsembleTemperatureScaler`).
        """
        m = mean / temperature
        if logvar is None:
            return F.softmax(m, dim=-1)
        s = n_logit_samples or self.n_logit_samples
        sigma = torch.exp(0.5 * logvar) / temperature
        eps = torch.randn(s, *m.shape, device=m.device, dtype=m.dtype)
        u = m.unsqueeze(0) + sigma.unsqueeze(0) * eps
        return F.softmax(u, dim=-1).mean(dim=0)

    @torch.no_grad()
    def predict_probs(
        self, x: torch.Tensor, n_logit_samples: int | None = None
    ) -> torch.Tensor:
        """p(y|x, w) for the *current* dropout mask, marginalising logit noise.

        Returns (N, C). The caller is responsible for looping over dropout
        masks; keeping the two levels separate is what allows the nested
        decomposition in `uncertainty.decomposition`.
        """
        mean, logvar = self.forward(x)
        return self.probs_from(mean, logvar, n_logit_samples)

    @torch.no_grad()
    def predict_probs_and_logits(
        self, x: torch.Tensor, n_logit_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both the marginalised probabilities and the mean logits.

        The logits are kept per posterior sample so that temperature scaling can
        be applied *inside* the posterior average rather than to a collapsed
        point estimate.
        """
        mean, logvar = self.forward(x)
        return self.probs_from(mean, logvar, n_logit_samples), mean


def heteroscedastic_ce(
    mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, n_samples: int = 16
) -> torch.Tensor:
    """-log (1/S) sum_t softmax(f + sigma*eps_t)_y, computed in log-space."""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, *mean.shape, device=mean.device, dtype=mean.dtype)
    u = mean.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, N, C)

    log_p = u - torch.logsumexp(u, dim=-1, keepdim=True)  # (S, N, C)
    idx = y.view(1, -1, 1).expand(n_samples, -1, 1)
    log_p_y = torch.gather(log_p, -1, idx).squeeze(-1)  # (S, N)

    # log-mean-exp over samples
    ll = torch.logsumexp(log_p_y, dim=0) - torch.log(
        torch.tensor(float(n_samples), device=mean.device)
    )
    return -ll.mean()


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-activate dropout.

    BatchNorm must stay in eval mode -- letting it use batch statistics at test
    time would inject a spurious, batch-composition-dependent source of variance
    that would show up in the epistemic term and has nothing to do with weight
    uncertainty. This is the single easiest way to get flattering-looking but
    meaningless MC-dropout numbers, so it is worth being explicit about.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
