"""
Generate images from a trained DDPM checkpoint.

Loads a checkpoint saved by scripts/train.py, reconstructs the model and
noise scheduler from the embedded config, runs DDPM ancestral sampling, and
saves a PNG image grid.

Usage:
    # Generate 16 samples (uses config embedded in the .pt file)
    python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt

    # Override number of samples, layout, output path, and device
    python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt \\
        --n-samples 64 --nrow 8 --output my_grid.png --device cpu

    # Self-test (no checkpoint needed)
    python scripts/sample.py --sanity-check
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler
from src.utils.sampling import SamplingCoeffs, ddpm_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> torch.device:
    """Return the best available device, falling back gracefully."""
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


def _save_grid(samples: torch.Tensor, path: Path, nrow: int) -> None:
    """Rescale samples from [-1, 1] to [0, 1] and write a PNG grid."""
    grid = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(grid, path, nrow=nrow)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate(
    checkpoint_path: Path,
    n_samples=None,
    device_str=None,
    output_path=None,
    nrow=None,
) -> Path:
    """
    Load a checkpoint and generate a grid of images via DDPM sampling.

    Args:
        checkpoint_path: Path to a .pt file saved by scripts/train.py.
                         Must contain keys: model_state_dict, config, epoch.
        n_samples:       Number of images to generate.  Defaults to
                         config['training']['num_samples'].
        device_str:      Device string ('cpu', 'cuda', 'mps').  Defaults to
                         config['training']['device'].
        output_path:     Where to write the PNG.  Defaults to
                         <sample_dir>/sampled_epoch_<N>.png.
        nrow:            Images per row in the grid.  Defaults to
                         floor(sqrt(n_samples)).

    Returns:
        The path of the saved PNG.
    """
    # --- Load checkpoint ---
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    epoch = ckpt.get("epoch", 0)

    # --- Resolve runtime parameters ---
    device    = _resolve_device(device_str or cfg["training"]["device"])
    n         = n_samples if n_samples is not None else cfg["training"]["num_samples"]
    grid_nrow = nrow if nrow is not None else max(1, int(math.floor(n ** 0.5)))

    if output_path is None:
        sample_dir = Path(cfg["paths"]["sample_dir"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f"sampled_epoch_{epoch:04d}.png"

    # --- Build scheduler ---
    scheduler = NoiseScheduler(
        timesteps  = cfg["diffusion"]["timesteps"],
        beta_start = cfg["diffusion"]["beta_start"],
        beta_end   = cfg["diffusion"]["beta_end"],
    ).to(device)

    # --- Build model and load weights ---
    model = UNet(
        img_channels  = cfg["dataset"]["channels"],
        base_channels = cfg["model"]["base_channels"],
        channel_mults = tuple(cfg["model"]["channel_mults"]),
        num_res_blocks= cfg["model"]["num_res_blocks"],
        time_emb_dim  = cfg["model"]["time_emb_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- Sample ---
    img_shape = (
        cfg["dataset"]["channels"],
        cfg["dataset"]["image_size"],
        cfg["dataset"]["image_size"],
    )
    print(f"Checkpoint : {checkpoint_path.name}  (epoch {epoch})")
    print(f"Device     : {device}")
    print(f"Generating : {n} images  |  grid {grid_nrow} per row  |  shape {img_shape}")

    coeffs  = SamplingCoeffs(scheduler)
    samples = ddpm_sample(
        model, scheduler,
        n_samples=n, img_shape=img_shape,
        device=device, show_progress=True, coeffs=coeffs,
    )

    # --- Save ---
    _save_grid(samples, output_path, nrow=grid_nrow)
    print(f"Saved      : {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check() -> None:
    """
    End-to-end test using a tiny model and T=20 steps — runs in seconds on CPU.

    Verifies:
      1. generate() creates a non-empty PNG at the requested output path.
      2. The auto-derived output filename follows the epoch naming convention.
      3. nrow default = floor(sqrt(n_samples)) for several values of n.
    """
    import tempfile

    T = 20
    cfg_tiny = {
        "diffusion": {
            "timesteps": T, "beta_start": 1e-4,
            "beta_end": 0.02, "beta_schedule": "linear",
        },
        "model": {
            "base_channels": 32, "channel_mults": [1, 2],
            "num_res_blocks": 1, "time_emb_dim": 64,
        },
        "dataset": {
            "name": "fashion_mnist", "image_size": 28,
            "channels": 1, "batch_size": 4,
        },
        "training": {
            "epochs": 1, "learning_rate": 2e-4, "device": "cpu",
            "save_every": 1, "sample_every": 1, "num_samples": 4,
        },
        "paths": {
            "checkpoint_dir": "outputs/checkpoints",
            "sample_dir":     "outputs/samples",
            "log_dir":        "outputs/logs",
        },
    }

    # Tiny untrained model saved as a fake checkpoint
    model = UNet(
        img_channels=1, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64,
    )
    ckpt_path = Path(tempfile.mktemp(suffix=".pt"))
    torch.save(
        {"epoch": 7, "model_state_dict": model.state_dict(),
         "optimizer_state_dict": {}, "config": cfg_tiny},
        ckpt_path,
    )

    # 1. Explicit output path
    out_path = Path(tempfile.mktemp(suffix=".png"))
    result = generate(ckpt_path, n_samples=4, device_str="cpu",
                      output_path=out_path, nrow=2)
    assert result == out_path,          f"Wrong return path: {result}"
    assert out_path.exists(),           "Output PNG not created"
    assert out_path.stat().st_size > 0, "Output PNG is empty"
    print(f"[PASS] generate() wrote {out_path.stat().st_size} bytes → {out_path.name}")

    # 2. Auto-derived filename uses epoch from checkpoint
    auto_result = generate(ckpt_path, n_samples=4, device_str="cpu",
                           output_path=None, nrow=2)
    assert auto_result.name == "sampled_epoch_0007.png", \
        f"Unexpected filename: {auto_result.name}"
    print(f"[PASS] Auto filename: {auto_result.name}")
    auto_result.unlink(missing_ok=True)

    # 3. nrow default = floor(sqrt(n_samples))
    for n, expected in [(1, 1), (4, 2), (7, 2), (9, 3), (16, 4)]:
        got = max(1, int(math.floor(n ** 0.5)))
        assert got == expected, f"nrow({n}) = {got}, expected {expected}"
    print("[PASS] Default nrow = floor(sqrt(n_samples))")

    # Cleanup
    ckpt_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    print("\nAll sanity checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample images from a trained DDPM checkpoint"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint .pt file (saved by scripts/train.py)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of images to generate (default: from checkpoint config)",
    )
    parser.add_argument(
        "--nrow", type=int, default=None,
        help="Images per row in the output grid (default: floor(sqrt(n_samples)))",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: outputs/samples/sampled_epoch_NNNN.png)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device: cpu / cuda / mps",
    )
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Run self-test without a real checkpoint and exit",
    )
    args = parser.parse_args()

    if args.sanity_check:
        _sanity_check()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required (or use --sanity-check)")
        generate(
            checkpoint_path = Path(args.checkpoint),
            n_samples       = args.n_samples,
            device_str      = args.device,
            output_path     = Path(args.output) if args.output else None,
            nrow            = args.nrow,
        )
