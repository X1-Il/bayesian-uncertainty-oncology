"""Training loop and deep-ensemble construction.

A deep ensemble is built by training M networks that differ only in their random
initialisation and data order. That is enough: the loss surface has many basins,
and independent runs land in different ones, so the members disagree in a way
that MC-dropout masks around a single optimum cannot reproduce. No bagging --
resampling the data would shrink each member's effective training set and
inflate the epistemic term for the wrong reason (less data), confounding it with
genuine posterior spread.

Model selection uses validation NLL rather than accuracy. Accuracy is blind to
the probabilities, and the probabilities are the object of this project; a
checkpoint chosen on accuracy is routinely worse calibrated than the one two
epochs earlier.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PhantomConfig, make_benchmark, make_loaders
from .models import BayesianCNN


@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    width: int = 32
    p_drop: float = 0.20
    heteroscedastic: bool = True
    n_logit_samples: int = 16
    ensemble_size: int = 5
    label_smoothing: float = 0.0
    """Left at 0 deliberately. Label smoothing improves ECE by blunting
    confidence, which would confound the calibration study: we want to measure
    what temperature scaling fixes, not pre-fix it in the loss."""
    seed: int = 0
    device: str = "cpu"


@torch.no_grad()
def evaluate_nll(model: BayesianCNN, loader: DataLoader, device: str) -> tuple[float, float]:
    """Deterministic (dropout off) NLL and accuracy -- used for model selection."""
    model.eval()
    tot_nll, tot_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, _ = model(x)
        tot_nll += F.cross_entropy(mean, y, reduction="sum").item()
        tot_correct += (mean.argmax(-1) == y).sum().item()
        n += len(y)
    return tot_nll / n, tot_correct / n


def train_one(
    cfg: TrainConfig,
    loaders: dict[str, DataLoader],
    seed: int,
    verbose: bool = True,
) -> tuple[BayesianCNN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BayesianCNN(
        width=cfg.width,
        p_drop=cfg.p_drop,
        heteroscedastic=cfg.heteroscedastic,
        n_logit_samples=cfg.n_logit_samples,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loaders["train"]), pct_start=0.3
    )

    history: list[dict] = []
    best = {"nll": float("inf"), "state": None, "epoch": -1}

    for epoch in range(cfg.epochs):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            # The heteroscedastic loss can spike early, while logvar is still
            # unconstrained by data; clipping keeps a single bad batch from
            # destroying the run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)

        val_nll, val_acc = evaluate_nll(model, loaders["val"], cfg.device)
        history.append(
            {"epoch": epoch, "train_loss": run_loss / n, "val_nll": val_nll,
             "val_acc": val_acc, "secs": time.time() - t0}
        )
        if val_nll < best["nll"]:
            best = {
                "nll": val_nll,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
        if verbose:
            print(
                f"    epoch {epoch:>3} | train {run_loss / n:.4f} | "
                f"val nll {val_nll:.4f} | val acc {val_acc:.4f} | {time.time() - t0:.1f}s",
                flush=True,
            )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"], "best_val_nll": best["nll"]}


def train_ensemble(
    cfg: TrainConfig,
    loaders: dict[str, DataLoader],
    out_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[list[BayesianCNN], list[dict]]:
    models, logs = [], []
    for m in range(cfg.ensemble_size):
        if verbose:
            print(f"  [member {m + 1}/{cfg.ensemble_size}]", flush=True)
        model, log = train_one(cfg, loaders, seed=cfg.seed + 1000 * m, verbose=verbose)
        models.append(model)
        logs.append(log)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(cfg)},
                out_dir / f"member_{m}.pt",
            )
    return models, logs


def load_ensemble(out_dir: Path, cfg: TrainConfig) -> list[BayesianCNN]:
    models = []
    for p in sorted(out_dir.glob("member_*.pt")):
        ckpt = torch.load(p, map_location=cfg.device, weights_only=True)
        model = BayesianCNN(
            width=cfg.width,
            p_drop=cfg.p_drop,
            heteroscedastic=cfg.heteroscedastic,
            n_logit_samples=cfg.n_logit_samples,
        ).to(cfg.device)
        model.load_state_dict(ckpt["state_dict"])
        models.append(model)
    if not models:
        raise FileNotFoundError(f"no checkpoints in {out_dir}")
    return models


def build_and_train(
    phantom: PhantomConfig,
    cfg: TrainConfig,
    n_train: int = 6000,
    out_dir: Path | None = None,
    verbose: bool = True,
):
    benchmark = make_benchmark(phantom, n_train=n_train, seed=cfg.seed)
    loaders = make_loaders(benchmark, batch_size=cfg.batch_size)
    models, logs = train_ensemble(cfg, loaders, out_dir, verbose)
    return benchmark, models, logs
