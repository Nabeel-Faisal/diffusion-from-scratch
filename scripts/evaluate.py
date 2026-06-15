"""
FID evaluation for trained DDPM/DDIM checkpoints.

Compares DDPM (T steps) and DDIM (configurable step counts, η=0) under the
beta schedule stored in the checkpoint.  Results are printed as a table and
saved as a CSV for inclusion in the write-up.

Fréchet Inception Distance (Heusel et al. 2017):

  FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g − 2·(Σ_r Σ_g)^{1/2})

  where μ, Σ are the mean / covariance of 2048-d InceptionV3 pool-layer
  features extracted from the real (r) and generated (g) image sets.

  Lower is better.  Well-trained DDPM on Fashion-MNIST typically lands
  in the range 10–40; values above ~100 indicate poor sample quality.

  Note: FID is only reliable for N ≥ 2048.  For N < 2048 a warning is
  printed but the number is still reported so the script is useful during
  development (e.g. --n-samples 64 for a quick debugging pass).

Usage:
    # Standard evaluation (1000 samples — DDPM + DDIM at several step counts)
    python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_0030.pt

    # Custom options
    python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_0030.pt \\
        --n-samples 2048 --ddim-steps 10 50 200 --device cuda

    # Self-test (no checkpoint, no InceptionV3 download, runs in seconds)
    python scripts/evaluate.py --sanity-check
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_dataloader
from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler
from src.utils.sampling import SamplingCoeffs, ddim_sample, ddpm_sample


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor
# ---------------------------------------------------------------------------

_INCEPTION_MEAN = [0.485, 0.456, 0.406]
_INCEPTION_STD  = [0.229, 0.224, 0.225]


class _InceptionFeatureExtractor:
    """
    Wraps a pretrained InceptionV3 to extract 2048-d pool features.

    Registers a forward hook on `avgpool` (the global average pool before the
    classifier) to capture the penultimate representation.  This is the same
    layer used by the original FID paper and pytorch-fid.

    Images must be in [0, 1] RGB at any spatial resolution; they are resized
    to 299×299 and normalised with ImageNet statistics before inference.
    """

    def __init__(self, device: torch.device):
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        model.eval()
        model.to(device)

        self._device  = device
        self._model   = model
        self._feats: Optional[torch.Tensor] = None

        # Hook avgpool to capture (B, 2048, 1, 1) before the FC head.
        self._hook = model.avgpool.register_forward_hook(self._save_hook)

        self._preprocess = transforms.Compose([
            transforms.Resize(299, antialias=True),
            transforms.Normalize(mean=_INCEPTION_MEAN, std=_INCEPTION_STD),
        ])

    def _save_hook(self, _module, _inp, output):
        self._feats = output.flatten(1)          # (B, 2048)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """
        Args:
            images: (B, C, H, W) tensor in [-1, 1], C ∈ {1, 3}.

        Returns:
            features: (B, 2048) float32 numpy array.
        """
        # Rescale from diffusion [-1,1] to [0,1].
        imgs = (images.clamp(-1.0, 1.0) + 1.0) / 2.0     # (B, C, H, W)

        # InceptionV3 expects 3 channels.
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        imgs = self._preprocess(imgs.to(self._device))
        self._model(imgs)                                  # triggers hook
        return self._feats.cpu().numpy()

    def remove_hook(self):
        self._hook.remove()


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def _fid_from_stats(
    mu_r: np.ndarray, sigma_r: np.ndarray,
    mu_g: np.ndarray, sigma_g: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g − 2·sqrtm(Σ_r @ Σ_g))

    Uses scipy.linalg.sqrtm.  When the product Σ_r @ Σ_g is near-singular
    (which happens when N < D, i.e. fewer samples than feature dimensions),
    we regularise by adding ε·I to each covariance matrix before multiplying:

      sqrtm((Σ_r + ε·I) @ (Σ_g + ε·I))

    This follows the pytorch-fid convention.  The real part of the result is
    taken (imaginary components arise only from floating-point noise).
    The final value is clamped to 0 — negative FID is a numerical artefact.
    """
    diff = mu_r - mu_g
    cov_product = sigma_r @ sigma_g

    sqrt_cov, _ = sqrtm(cov_product, disp=False)

    if not np.isfinite(sqrt_cov).all():
        # Near-singular product: regularise each matrix individually.
        offset   = np.eye(sigma_r.shape[0]) * eps
        sqrt_cov = sqrtm((sigma_r + offset) @ (sigma_g + offset))

    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * sqrt_cov))
    return max(0.0, fid)


def _stats_from_features(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) for a feature matrix of shape (N, D)."""
    mu    = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# ---------------------------------------------------------------------------
# Feature collection helpers
# ---------------------------------------------------------------------------

def _collect_real_features(
    dataset_name: str,
    n: int,
    extractor: _InceptionFeatureExtractor,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Load n images from the test set and extract InceptionV3 features.

    Images come from the dataset loader in [-1, 1]; they are passed directly
    to the extractor which re-scales to [0, 1] internally.

    Returns:
        features: (n, 2048) float32 array.
    """
    loader, _ = get_dataloader(
        name=dataset_name, batch_size=batch_size,
        train=False, num_workers=0, pin_memory=False,
    )

    all_feats: List[np.ndarray] = []
    collected = 0

    for images, _ in loader:
        remaining = n - collected
        if remaining <= 0:
            break
        batch = images[:remaining]
        all_feats.append(extractor.extract(batch))
        collected += batch.shape[0]

    return np.concatenate(all_feats, axis=0)[:n]


def _collect_fake_features(
    model: UNet,
    scheduler: NoiseScheduler,
    img_shape: Tuple[int, int, int],
    n: int,
    device: torch.device,
    extractor: _InceptionFeatureExtractor,
    sampler_type: str,
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    gen_batch_size: int = 64,
    sampling_coeffs: Optional[SamplingCoeffs] = None,
) -> Tuple[np.ndarray, float]:
    """
    Generate n fake images, extract features, return (features, elapsed_sec).

    Images are generated in batches of gen_batch_size to fit in memory.
    """
    all_feats: List[np.ndarray] = []
    remaining  = n
    t0 = time.perf_counter()

    while remaining > 0:
        b = min(gen_batch_size, remaining)

        if sampler_type == "ddim":
            batch = ddim_sample(
                model, scheduler, b, img_shape, device,
                n_steps=ddim_steps, eta=ddim_eta, show_progress=False,
            )
        else:
            batch = ddpm_sample(
                model, scheduler, b, img_shape, device,
                show_progress=False, coeffs=sampling_coeffs,
            )

        all_feats.append(extractor.extract(batch))
        remaining -= b

    elapsed = time.perf_counter() - t0
    return np.concatenate(all_feats, axis=0)[:n], elapsed


# ---------------------------------------------------------------------------
# Resolve device
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[WARN] CUDA not available — falling back to MPS.")
            return torch.device("mps")
        print("[WARN] CUDA not available — falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[WARN] MPS not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: Path,
    n_samples: int = 1000,
    device_str: Optional[str] = None,
    ddim_steps_list: Optional[List[int]] = None,
    ddim_eta: float = 0.0,
    output_path: Optional[Path] = None,
    gen_batch_size: int = 64,
) -> List[dict]:
    """
    Evaluate a checkpoint with DDPM and DDIM sampling; compute FID for each.

    Runs the following configurations (one row each in the output table):
      - DDPM, all T steps
      - DDIM, n_steps ∈ ddim_steps_list, η=ddim_eta

    Args:
        checkpoint_path: .pt file saved by scripts/train.py.
        n_samples:        Number of generated images to evaluate.
        device_str:       Device override ('cpu'/'cuda'/'mps').
        ddim_steps_list:  DDIM step counts to evaluate.  Default: [10, 50, 200].
        ddim_eta:         DDIM stochasticity (0 = deterministic).
        output_path:      Where to write results CSV.  Defaults to
                          outputs/logs/fid_<timestamp>.csv.
        gen_batch_size:   Batch size for image generation.

    Returns:
        List of result dicts (one per configuration).
    """
    if ddim_steps_list is None:
        ddim_steps_list = [10, 50, 200]

    if n_samples < 2048:
        print(
            f"[WARN] n_samples={n_samples} < 2048.  FID estimates will be "
            "noisy and should not be compared across runs."
        )

    # --- Load checkpoint ---
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    epoch = ckpt.get("epoch", 0)

    device = _resolve_device(device_str or cfg["training"]["device"])

    dataset_name  = cfg["dataset"]["name"]
    img_channels  = cfg["dataset"]["channels"]
    img_size      = cfg["dataset"]["image_size"]
    img_shape     = (img_channels, img_size, img_size)
    schedule_name = cfg["diffusion"].get("beta_schedule", "linear")

    print(f"\nCheckpoint : {checkpoint_path.name}  (epoch {epoch})")
    print(f"Device     : {device}")
    print(f"Schedule   : {schedule_name}")
    print(f"Dataset    : {dataset_name}  (test set)")
    print(f"N samples  : {n_samples}")

    # --- Build scheduler ---
    scheduler = NoiseScheduler(
        timesteps  = cfg["diffusion"]["timesteps"],
        beta_start = cfg["diffusion"]["beta_start"],
        beta_end   = cfg["diffusion"]["beta_end"],
        schedule   = schedule_name,
        cosine_s   = cfg["diffusion"].get("cosine_s", 0.008),
    ).to(device)

    # --- Build model ---
    model = UNet(
        img_channels  = img_channels,
        base_channels = cfg["model"]["base_channels"],
        channel_mults = tuple(cfg["model"]["channel_mults"]),
        num_res_blocks= cfg["model"]["num_res_blocks"],
        time_emb_dim  = cfg["model"]["time_emb_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Precompute DDPM coefficients (reused across runs).
    ddpm_coeffs = SamplingCoeffs(scheduler)

    # --- Inception extractor ---
    print("\nLoading InceptionV3 weights (first run downloads ~100 MB)...")
    extractor = _InceptionFeatureExtractor(device)

    # --- Real features (collected once) ---
    print(f"Extracting features from {n_samples} real {dataset_name} images...")
    t0 = time.perf_counter()
    real_feats  = _collect_real_features(dataset_name, n_samples, extractor, gen_batch_size)
    real_time   = time.perf_counter() - t0
    mu_r, sig_r = _stats_from_features(real_feats)
    print(f"  → real features: {real_feats.shape}  ({real_time:.1f}s)")

    # --- Evaluate each configuration ---
    configs = [("ddpm", scheduler.T, ddim_eta)]
    for s in ddim_steps_list:
        configs.append(("ddim", s, ddim_eta))

    results = []

    for sampler_type, steps, eta in configs:
        label = f"DDPM ({steps} steps)" if sampler_type == "ddpm" else f"DDIM ({steps} steps, η={eta})"
        print(f"\nGenerating {n_samples} samples — {label}...")

        fake_feats, elapsed = _collect_fake_features(
            model, scheduler, img_shape, n_samples, device, extractor,
            sampler_type=sampler_type,
            ddim_steps=steps, ddim_eta=eta,
            gen_batch_size=gen_batch_size,
            sampling_coeffs=ddpm_coeffs if sampler_type == "ddpm" else None,
        )
        mu_g, sig_g = _stats_from_features(fake_feats)
        fid = _fid_from_stats(mu_r, sig_r, mu_g, sig_g)

        print(f"  → FID = {fid:.2f}   time = {elapsed:.1f}s")
        results.append({
            "checkpoint": checkpoint_path.name,
            "epoch":      epoch,
            "schedule":   schedule_name,
            "sampler":    sampler_type,
            "steps":      steps,
            "eta":        eta,
            "n_samples":  n_samples,
            "fid":        round(fid, 4),
            "time_s":     round(elapsed, 2),
        })

    extractor.remove_hook()

    # --- Print table ---
    header = f"\n{'Schedule':<10}  {'Sampler':<25}  {'Steps':>6}  {'FID':>8}  {'Time (s)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        sampler_str = f"{r['sampler'].upper()} (η={r['eta']})"
        print(f"{r['schedule']:<10}  {sampler_str:<25}  {r['steps']:>6}  {r['fid']:>8.2f}  {r['time_s']:>10.1f}")

    # --- Save CSV ---
    if output_path is None:
        log_dir = Path(cfg["paths"].get("log_dir", "outputs/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = log_dir / f"fid_{ts}.csv"

    fields = ["checkpoint", "epoch", "schedule", "sampler", "steps", "eta",
              "n_samples", "fid", "time_s"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {output_path}")
    return results


# ---------------------------------------------------------------------------
# Sanity check — no checkpoint, no InceptionV3 download
# ---------------------------------------------------------------------------

def _sanity_check() -> None:
    """
    Verifies end-to-end correctness without downloading InceptionV3.

    Uses random gaussian vectors as stand-in for inception features to test:
      1. _fid_from_stats gives FID ≈ 0 when real == generated distribution.
      2. _fid_from_stats gives FID > 0 when distributions differ.
      3. Fake image generation (DDPM + DDIM) runs to completion and produces
         correct shapes.
      4. _collect_fake_features returns (n, 2048)-shaped features (mocked).
    """
    import tempfile

    print("=== evaluate.py sanity check ===\n")
    rng = np.random.default_rng(42)

    # 1. FID ≈ 0 for identical distributions.
    # Use D=64 so N (256) >> D and the sample covariance is full-rank.
    # (In production D=2048 with N≥2048; here we verify the formula, not scale.)
    D, N = 64, 256
    feats = rng.standard_normal((N, D)).astype(np.float32)
    mu, sig = _stats_from_features(feats)
    fid_same = _fid_from_stats(mu, sig, mu, sig)
    assert abs(fid_same) < 0.5, f"FID(same) = {fid_same:.4f}, expected ≈ 0"
    print(f"[PASS] FID(same distribution)  = {fid_same:.6f}  (≈ 0)")

    # 2. FID > 0 for shifted distributions.
    feats2 = feats + 10.0           # shift mean by 10
    mu2, sig2 = _stats_from_features(feats2)
    fid_diff = _fid_from_stats(mu, sig, mu2, sig2)
    assert fid_diff > 1.0, f"FID(diff) = {fid_diff:.2f}, expected > 1.0"
    print(f"[PASS] FID(shifted by 10)      = {fid_diff:.2f}  (> 1.0)")

    # 3. Generation pipeline runs.
    T, C, H, W = 20, 1, 28, 28
    device = torch.device("cpu")

    scheduler = NoiseScheduler(timesteps=T, beta_start=1e-4, beta_end=0.02).to(device)
    model = UNet(
        img_channels=C, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64,
    ).to(device)

    N_gen = 4
    img_shape = (C, H, W)

    ddpm_out = ddpm_sample(model, scheduler, N_gen, img_shape, device,
                           show_progress=False)
    assert ddpm_out.shape == (N_gen, C, H, W)
    assert torch.isfinite(ddpm_out).all()
    print(f"[PASS] DDPM output shape       = {tuple(ddpm_out.shape)}")

    ddim_out = ddim_sample(model, scheduler, N_gen, img_shape, device,
                           n_steps=5, eta=0.0, show_progress=False)
    assert ddim_out.shape == (N_gen, C, H, W)
    assert torch.isfinite(ddim_out).all()
    print(f"[PASS] DDIM output shape       = {tuple(ddim_out.shape)}")

    # 4. Mocked feature collection → FID roundtrip.
    #    Replace the extractor with a lambda that returns random features.
    _MOCK_D = 64   # smaller than N_gen for full-rank covariance

    class _MockExtractor:
        def extract(self, imgs):
            return rng.standard_normal((imgs.shape[0], _MOCK_D)).astype(np.float32)

    mock_ext = _MockExtractor()
    fake_feats, elapsed = _collect_fake_features(
        model, scheduler, img_shape, N_gen, device, mock_ext,
        sampler_type="ddim", ddim_steps=5, ddim_eta=0.0,
        gen_batch_size=2,
    )
    assert fake_feats.shape == (N_gen, _MOCK_D), f"Unexpected shape: {fake_feats.shape}"
    print(f"[PASS] Mocked feature shape    = {fake_feats.shape}")

    # FID on tiny N — just verify it's a finite number.
    real_feats_mock = rng.standard_normal((N_gen, _MOCK_D)).astype(np.float32)
    mu_r, sig_r = _stats_from_features(real_feats_mock)
    mu_g, sig_g = _stats_from_features(fake_feats)
    fid_mock = _fid_from_stats(mu_r, sig_r, mu_g, sig_g)
    assert np.isfinite(fid_mock), f"FID is not finite: {fid_mock}"
    print(f"[PASS] FID on mocked features  = {fid_mock:.2f}  (finite)")

    print("\nAll sanity checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute FID for a trained DDPM checkpoint"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint .pt file (saved by scripts/train.py)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of images to generate (default: 1000; use ≥2048 for reliable FID)",
    )
    parser.add_argument(
        "--ddim-steps", type=int, nargs="+", default=[10, 50, 200],
        metavar="S",
        help="DDIM step counts to evaluate (default: 10 50 200)",
    )
    parser.add_argument(
        "--ddim-eta", type=float, default=0.0,
        help="DDIM stochasticity η (default: 0.0 = deterministic)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override: cpu / cuda / mps",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for generation and feature extraction (default: 64)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: outputs/logs/fid_<timestamp>.csv)",
    )
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Run self-test without a checkpoint or InceptionV3 download and exit",
    )
    args = parser.parse_args()

    if args.sanity_check:
        _sanity_check()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required (or use --sanity-check)")
        evaluate(
            checkpoint_path = Path(args.checkpoint),
            n_samples       = args.n_samples,
            device_str      = args.device,
            ddim_steps_list = args.ddim_steps,
            ddim_eta        = args.ddim_eta,
            output_path     = Path(args.output) if args.output else None,
            gen_batch_size  = args.batch_size,
        )
