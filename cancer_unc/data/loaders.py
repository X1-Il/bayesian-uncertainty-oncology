"""Torch wrappers around the phantom benchmark, plus the real-data drop-in.

The whole pipeline downstream of here consumes tensors of shape (N, 1, H, W)
and int64 labels. Nothing else in the project knows or cares whether those came
from the phantom generator or from a folder of DICOM-derived PNGs, which is the
point: swapping in a real dataset is a change to this file only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .synthetic import PhantomSplit


class PhantomDataset(Dataset):
    """In-memory dataset. The phantoms are small enough that keeping the whole
    benchmark resident costs ~100 MB and removes the dataloader from the
    profile entirely -- which matters when everything runs on CPU."""

    def __init__(self, split: PhantomSplit, augment: bool = False):
        self.x = torch.from_numpy(split.images)
        self.y = torch.from_numpy(split.labels)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.augment:
            # Flips only. Anything that changes *intensity* would corrupt the
            # experiment: the label depends on lesion contrast, so brightness or
            # contrast jitter would inject label noise the oracle does not know
            # about and break the comparison against the analytic aleatoric term.
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-2])
        return x, self.y[i]


def make_loaders(
    benchmark: dict[str, PhantomSplit],
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders = {}
    for name, split in benchmark.items():
        train = name == "train"
        loaders[name] = DataLoader(
            PhantomDataset(split, augment=augment and train),
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders


def as_tensors(split: PhantomSplit) -> tuple[torch.Tensor, np.ndarray]:
    """Whole split as (images, labels) for the batched inference helpers."""
    return torch.from_numpy(split.images), split.labels


# --------------------------------------------------------------------------
# real-data drop-in
# --------------------------------------------------------------------------
def load_image_folder(
    root: str | Path,
    class_dirs: tuple[str, str] = ("no", "yes"),
    image_size: int = 64,
) -> PhantomSplit:
    """Load a real dataset laid out as root/<class>/*.png|jpg|jpeg.

    Matches the layout of the common Kaggle brain-MRI tumour sets. Returns the
    same `PhantomSplit` container so the rest of the pipeline is unchanged --
    with `z` and `p_true` filled with NaN, because on real data the latent and
    the true posterior do not exist. Every oracle comparison in the evaluation
    is guarded on those being finite, so they are skipped automatically and the
    calibration / OOD / risk-coverage results still run in full.

    Grayscale conversion uses luminance weights rather than a plain channel mean
    because these datasets are RGB-encoded grayscale of varying provenance.
    """
    from PIL import Image  # imported lazily: only real data needs pillow

    root = Path(root)
    images, labels = [], []
    for label, sub in enumerate(class_dirs):
        d = root / sub
        if not d.is_dir():
            raise FileNotFoundError(f"expected class directory {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            im = Image.open(p).convert("L").resize((image_size, image_size), Image.BILINEAR)
            images.append(np.asarray(im, dtype=np.float32) / 255.0)
            labels.append(label)

    if not images:
        raise RuntimeError(f"no images found under {root}")

    x = np.stack(images)[:, None]
    y = np.asarray(labels, dtype=np.int64)
    nan = np.full(len(y), np.nan)
    return PhantomSplit(x, y, nan, nan, shift=f"real:{root.name}")


def stratified_split(
    split: PhantomSplit, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, PhantomSplit]:
    """Class-stratified train/val/test partition of a single split.

    Stratified rather than random because the calibration set must contain
    enough of the positive class for temperature scaling to be identifiable,
    and real tumour datasets are often imbalanced.
    """
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(split.labels == c)[0] for c in np.unique(split.labels)]
    parts: list[list[np.ndarray]] = [[], [], []]
    for idx in idx_by_class:
        rng.shuffle(idx)
        n = len(idx)
        a = int(round(fractions[0] * n))
        b = a + int(round(fractions[1] * n))
        for j, chunk in enumerate((idx[:a], idx[a:b], idx[b:])):
            parts[j].append(chunk)

    out = {}
    for name, chunks in zip(("train", "val", "test"), parts):
        i = np.concatenate(chunks)
        rng.shuffle(i)
        out[name] = PhantomSplit(
            split.images[i], split.labels[i], split.z[i], split.p_true[i], split.shift
        )
    return out
