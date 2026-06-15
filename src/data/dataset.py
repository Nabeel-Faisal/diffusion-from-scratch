"""
Dataset loaders for Phase 1–3.

Phase 1–2 (MNIST / Fashion-MNIST):
  Images normalised to [-1, 1].  Items: (image: Tensor[1,28,28], label: int).

Phase 3 (CIFAR-10):
  Images normalised to [-1, 1].  Items: (image: Tensor[3,32,32], caption: str).
  Each integer label is mapped to a templated caption: "a photo of a {class_name}".
  The 10 class names (in label order) are exposed as CaptionedCIFAR10.CLASSES.

Normalisation formula (same for all datasets):
  x_norm = (x / 255.0 - 0.5) / 0.5
i.e. torchvision ToTensor() maps uint8 → [0, 1], then Normalize(0.5, 0.5) maps
[0, 1] → [-1, 1].  Rescale to [0, 1] only when saving or visualising.

Supported dataset names (matches config `dataset.name`):
  "mnist"         — handwritten digits, 28×28 grayscale
  "fashion_mnist" — clothing items,     28×28 grayscale
  "cifar10"       — natural images,     32×32 RGB, 10 classes with captions
"""

from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Root directory where torchvision will download / cache data.
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Grayscale normalisation (MNIST / Fashion-MNIST): 1 channel.
_GRAY_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# RGB normalisation (CIFAR-10): 3 channels.
_RGB_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Grayscale dataset registry (Phase 1–2).
_GRAY_DATASETS = {
    "mnist":         datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}


# ---------------------------------------------------------------------------
# CIFAR-10 with templated captions (Phase 3)
# ---------------------------------------------------------------------------

class CaptionedCIFAR10(Dataset):
    """
    CIFAR-10 wrapper that replaces integer labels with caption strings.

    Each of the 10 class labels is mapped to:
      "a photo of a {class_name}"

    This template follows the zero-shot CLIP paper (Radford et al. 2021,
    Appendix B) and gives the text encoder meaningful context rather than a
    bare class name.

    Items: (image: Tensor[3, 32, 32] in [-1, 1],  caption: str)

    CIFAR-10 label order (0–9):
      airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """

    CLASSES: List[str] = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    CAPTION_TEMPLATE: str = "a photo of a {}"

    def __init__(self, root: Path, train: bool = True, download: bool = True):
        self._base = datasets.CIFAR10(
            root=root, train=train, download=download,
            transform=_RGB_TRANSFORM,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image, label = self._base[idx]
        caption = self.CAPTION_TEMPLATE.format(self.CLASSES[label])
        return image, caption


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def get_dataset(name: str, train: bool = True) -> Dataset:
    """
    Return a dataset with pixels normalised to [-1, 1].

    For "mnist" / "fashion_mnist": items are (image[1,28,28], label: int).
    For "cifar10":                 items are (image[3,32,32], caption: str).

    Args:
        name:  "mnist", "fashion_mnist", or "cifar10".
        train: Training split if True; test split otherwise.
    """
    if name in _GRAY_DATASETS:
        cls = _GRAY_DATASETS[name]
        return cls(root=_DATA_ROOT, train=train, download=True,
                   transform=_GRAY_TRANSFORM)
    if name == "cifar10":
        return CaptionedCIFAR10(root=_DATA_ROOT, train=train, download=True)
    raise ValueError(
        f"Unknown dataset '{name}'. Choose from: {list(_GRAY_DATASETS) + ['cifar10']}"
    )


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
        name:        "mnist", "fashion_mnist", or "cifar10".
        batch_size:  Number of images per batch.
        train:       Training vs test split.
        num_workers: Parallel data-loading workers.
        pin_memory:  Pin host memory for faster GPU transfer.

    Returns:
        (dataloader, num_classes)
        - num_classes is always 10 for all supported datasets.
        - For "cifar10", each batch yields (image, caption_list) where
          caption_list is a list of B strings.
        - For grayscale datasets, each batch yields (image, label_tensor).
    """
    dataset = get_dataset(name, train=train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader, 10


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Checks all three datasets.

    MNIST / Fashion-MNIST:
      1. Batch shape (B, 1, 28, 28).
      2. Pixel values in [-1, 1].
      3. Integer labels in [0, 9].

    CIFAR-10:
      4. Batch shape (B, 3, 32, 32).
      5. Pixel values in [-1, 1].
      6. Each item in a batch is a caption string.
      7. Captions follow the "a photo of a {class}" template.
      8. All 10 CIFAR-10 class names appear in a full epoch.
    """
    print("=== Grayscale datasets (Phase 1–2) ===")
    for name in ("mnist", "fashion_mnist"):
        print(f"\n--- {name} ---")
        loader, num_classes = get_dataloader(name, batch_size=8, train=True,
                                             num_workers=0, pin_memory=False)
        images, labels = next(iter(loader))

        assert images.shape == torch.Size([8, 1, 28, 28]), \
            f"Unexpected shape: {images.shape}"
        print(f"[PASS] Batch shape: {images.shape}")

        lo, hi = images.min().item(), images.max().item()
        assert lo >= -1.0 and hi <= 1.0, f"Pixel range out of [-1, 1]: [{lo:.3f}, {hi:.3f}]"
        print(f"[PASS] Pixel range: [{lo:.3f}, {hi:.3f}]")

        assert isinstance(labels[0].item(), int) or labels.dtype in (torch.int64, torch.int32)
        assert labels.min() >= 0 and labels.max() < 10
        print(f"[PASS] Labels (int): {labels.tolist()}")

    print("\n=== CIFAR-10 (Phase 3) ===")
    loader, num_classes = get_dataloader("cifar10", batch_size=8, train=True,
                                         num_workers=0, pin_memory=False)
    images, captions = next(iter(loader))

    # 4. Shape
    assert images.shape == torch.Size([8, 3, 32, 32]), \
        f"Unexpected shape: {images.shape}"
    print(f"[PASS] Batch shape: {images.shape}")

    # 5. Pixel range
    lo, hi = images.min().item(), images.max().item()
    assert lo >= -1.0 and hi <= 1.0, f"Pixel range: [{lo:.3f}, {hi:.3f}]"
    print(f"[PASS] Pixel range: [{lo:.3f}, {hi:.3f}]")

    # 6. Captions are strings
    assert isinstance(captions, (list, tuple)), f"Expected list of strings, got {type(captions)}"
    assert all(isinstance(c, str) for c in captions), "Not all captions are strings"
    assert len(captions) == 8, f"Expected 8 captions, got {len(captions)}"
    print(f"[PASS] Captions are strings ({len(captions)} per batch)")

    # 7. Template check
    assert all(c.startswith("a photo of a ") for c in captions), \
        f"Caption doesn't match template: {captions[0]}"
    print(f"[PASS] Template: '{captions[0]}'")

    # 8. All 10 class names appear across a full dataset pass
    seen = set()
    ds_check = get_dataset("cifar10", train=True)
    for i in range(len(ds_check)):
        _, cap = ds_check[i]
        seen.add(cap.replace("a photo of a ", ""))
        if seen == set(CaptionedCIFAR10.CLASSES):
            break
    assert seen == set(CaptionedCIFAR10.CLASSES), \
        f"Missing classes: {set(CaptionedCIFAR10.CLASSES) - seen}"
    print(f"[PASS] All 10 class captions present: {sorted(seen)}")

    # invalid name raises
    try:
        get_dataset("flowers102")
        assert False
    except ValueError:
        print("[PASS] Unknown dataset name raises ValueError")

    print("\nAll sanity checks passed.")
