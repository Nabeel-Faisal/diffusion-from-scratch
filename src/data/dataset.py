"""
Dataset loaders for Phase 1 (MNIST / Fashion-MNIST).

Images are normalized to [-1, 1] so that x_0 lives in the same range as
the diffusion samples (which start from standard Gaussian noise and are
gradually denoised toward [-1, 1]).  Rescale to [0, 1] only when saving
or visualizing.

Supported dataset names (matches config `dataset.name`):
  "mnist"         — handwritten digits, 28×28 grayscale
  "fashion_mnist" — clothing items,     28×28 grayscale
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Root directory where torchvision will download / cache data
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}

# torchvision's ToTensor() maps uint8 [0, 255] → float [0, 1].
# Normalize((0.5,), (0.5,)) then maps [0, 1] → [-1, 1]:
#   x_norm = (x - 0.5) / 0.5
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_dataset(name: str, train: bool = True) -> Dataset:
    """
    Return a torchvision dataset with pixels normalised to [-1, 1].

    Args:
        name:  "mnist" or "fashion_mnist"
        train: if True, returns the training split; False returns test split.

    Returns:
        A torch Dataset whose items are (image, label) where
        image ∈ ℝ^{1×28×28} with values in [-1, 1].
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_DATASETS)}")

    cls = _DATASETS[name]
    return cls(root=_DATA_ROOT, train=train, download=True, transform=_TRANSFORM)


def get_dataloader(
    name: str,
    batch_size: int,
    train: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, int]:
    """
    Build and return a DataLoader for the requested dataset.

    Args:
        name:        "mnist" or "fashion_mnist"
        batch_size:  number of images per batch
        train:       training vs test split
        num_workers: parallel data-loading workers
        pin_memory:  pin host memory for faster GPU transfer

    Returns:
        (dataloader, num_classes) — num_classes is always 10 for both datasets
    """
    dataset = get_dataset(name, train=train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,          # shuffle only during training
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # keeps all batches the same size
    )
    return loader, 10


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Verifies:
      1. Both dataset names load without error.
      2. Batch shape is (B, 1, 28, 28).
      3. Pixel values are in [-1, 1].
      4. DataLoader iterates without error.
    """
    import sys

    for name in ("mnist", "fashion_mnist"):
        print(f"\n--- {name} ---")
        loader, num_classes = get_dataloader(name, batch_size=8, train=True, num_workers=0)

        images, labels = next(iter(loader))

        # Shape
        assert images.shape == torch.Size([8, 1, 28, 28]), \
            f"Unexpected shape: {images.shape}"
        print(f"[PASS] Batch shape: {images.shape}")

        # Pixel range
        lo, hi = images.min().item(), images.max().item()
        assert lo >= -1.0 and hi <= 1.0, f"Pixel range out of [-1, 1]: [{lo:.3f}, {hi:.3f}]"
        print(f"[PASS] Pixel range: [{lo:.3f}, {hi:.3f}]  (expected within [-1, 1])")

        # Label range
        assert labels.min() >= 0 and labels.max() < num_classes, \
            f"Label out of range: {labels.min()}..{labels.max()}"
        print(f"[PASS] Labels: {labels.tolist()}  ({num_classes} classes)")

        # Dataset length
        ds = get_dataset(name, train=True)
        print(f"[INFO] Training set size: {len(ds):,}")

    print("\nAll sanity checks passed.")
