"""
Training script for Phase 1: Unconditional DDPM on Fashion-MNIST.

Algorithm (Ho et al. 2020, Algorithm 1):
  1. Sample x_0 from the dataset.
  2. Sample t ~ Uniform{0, ..., T-1}.
  3. Sample ε ~ N(0, I).
  4. Compute x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε   (closed-form forward diffusion).
  5. Predict ε̂ = UNet(x_t, t).
  6. Minimise  L = ||ε̂ - ε||²   (simplified ELBO, Eq. 14).

Usage:
    python scripts/train.py --config configs/phase1_unconditional.yaml
    python scripts/train.py --config configs/phase1_unconditional.yaml --device cpu
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

# Allow `from src.xxx import` when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_dataloader
from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> torch.device:
    """Return the best available device, falling back gracefully from the request."""
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


def save_checkpoint(path: Path, epoch: int, model: UNet,
                    optimizer: torch.optim.Optimizer, cfg: dict) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }, path)


@torch.no_grad()
def sample_ddpm(
    model: UNet,
    scheduler: NoiseScheduler,
    n_samples: int,
    img_channels: int,
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    DDPM ancestral sampling (Algorithm 2, Ho et al. 2020).

    Starting from x_T ~ N(0, I), iterates the learned reverse process:

      x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + √β_t · z

    where z ~ N(0, I) for t > 0 and z = 0 at the final step t = 0.

    The variance σ²_t = β_t follows the "fixed small variance" choice from the paper.

    Returns:
        x_0 estimates, shape (n_samples, img_channels, img_size, img_size), in [-1, 1].
    """
    model.eval()
    x = torch.randn(n_samples, img_channels, img_size, img_size, device=device)

    for t_val in tqdm(reversed(range(scheduler.T)), total=scheduler.T,
                      desc="Sampling", leave=False):
        t_batch = torch.full((n_samples,), t_val, dtype=torch.long, device=device)

        eps_pred = model(x, t_batch)

        alpha_t          = scheduler.alphas[t_val]            # α_t
        beta_t           = scheduler.betas[t_val]             # β_t
        sqrt_1m_ab       = scheduler.sqrt_one_minus_alpha_bars[t_val]  # √(1-ᾱ_t)

        # Posterior mean (DDPM Eq. 11)
        mean = (1.0 / alpha_t.sqrt()) * (x - (beta_t / sqrt_1m_ab) * eps_pred)

        if t_val > 0:
            x = mean + beta_t.sqrt() * torch.randn_like(x)
        else:
            x = mean  # t = 0: no noise added

    model.train()
    return x


def save_sample_grid(samples: torch.Tensor, path: Path, nrow: int = 4) -> None:
    """Clamp to [-1,1], rescale to [0,1], and write a PNG grid."""
    grid = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(grid, path, nrow=nrow)


def save_loss_plot(step_losses: list, path: Path) -> None:
    """Save a loss-vs-step plot with a 200-step moving-average overlay."""
    steps  = [s for s, _ in step_losses]
    losses = [l for _, l in step_losses]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses, alpha=0.25, color="steelblue", linewidth=0.8)

    window = min(200, len(losses))
    if window > 1:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(steps[window - 1:], ma, color="steelblue",
                 linewidth=1.5, label=f"{window}-step avg")
        plt.legend()

    plt.xlabel("Global step")
    plt.ylabel("MSE loss")
    plt.title("DDPM training loss")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, device_override=None) -> None:
    device = resolve_device(device_override or cfg["training"]["device"])
    use_cuda = device.type == "cuda"

    # --- Build components ---
    scheduler = NoiseScheduler(
        timesteps  = cfg["diffusion"]["timesteps"],
        beta_start = cfg["diffusion"]["beta_start"],
        beta_end   = cfg["diffusion"]["beta_end"],
    ).to(device)

    model = UNet(
        img_channels  = cfg["dataset"]["channels"],
        base_channels = cfg["model"]["base_channels"],
        channel_mults = tuple(cfg["model"]["channel_mults"]),
        num_res_blocks= cfg["model"]["num_res_blocks"],
        time_emb_dim  = cfg["model"]["time_emb_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )

    loader, _ = get_dataloader(
        name        = cfg["dataset"]["name"],
        batch_size  = cfg["dataset"]["batch_size"],
        train       = True,
        num_workers = 4 if use_cuda else 0,
        pin_memory  = use_cuda,
    )

    # --- Output directories ---
    ckpt_dir   = Path(cfg["paths"]["checkpoint_dir"])
    sample_dir = Path(cfg["paths"]["sample_dir"])
    log_dir    = Path(cfg["paths"]["log_dir"])
    for d in (ckpt_dir, sample_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Loss log ---
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"loss_{run_id}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "loss"])

    # --- Config ---
    n_epochs     = cfg["training"]["epochs"]
    save_every   = cfg["training"]["save_every"]
    sample_every = cfg["training"]["sample_every"]
    num_samples  = cfg["training"]["num_samples"]
    img_channels = cfg["dataset"]["channels"]
    img_size     = cfg["dataset"]["image_size"]
    nrow         = int(num_samples ** 0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device     : {device}")
    print(f"Model      : {n_params:,} parameters")
    print(f"Dataset    : {cfg['dataset']['name']}  (batch {cfg['dataset']['batch_size']})")
    print(f"Epochs     : {n_epochs}  |  save every {save_every}  |  sample every {sample_every}")

    step_losses: list = []
    global_step = 0

    # --- Epoch loop ---
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", leave=True)
        for x0, _ in pbar:
            x0 = x0.to(device)
            B  = x0.shape[0]

            # Step 2: sample t ~ Uniform{0, T-1}
            t = torch.randint(0, scheduler.T, (B,), device=device, dtype=torch.long)

            # Step 3: sample ε ~ N(0, I)
            noise = torch.randn_like(x0)

            # Step 4: forward diffusion — x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
            xt = scheduler.q_sample(x0, t, noise)

            # Step 5: predict noise
            pred_noise = model(xt, t)

            # Step 6: simplified noise-prediction loss (Ho et al., Eq. 14)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            global_step += 1
            epoch_loss_sum += loss_val
            step_losses.append((global_step, loss_val))

            log_writer.writerow([global_step, epoch, f"{loss_val:.6f}"])
            log_file.flush()

            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = epoch_loss_sum / len(loader)
        print(f"  → epoch {epoch:3d}  avg loss {avg_loss:.4f}")

        # --- Checkpoint ---
        if epoch % save_every == 0 or epoch == n_epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, cfg)
            print(f"     checkpoint → {ckpt_path}")

        # --- Sample grid ---
        if epoch % sample_every == 0:
            samples = sample_ddpm(
                model, scheduler, num_samples, img_channels, img_size, device
            )
            sp = sample_dir / f"epoch_{epoch:04d}.png"
            save_sample_grid(samples, sp, nrow=nrow)
            print(f"     samples    → {sp}")

            plot_path = log_dir / f"loss_{run_id}.png"
            save_loss_plot(step_losses, plot_path)

    # Final loss plot
    save_loss_plot(step_losses, log_dir / f"loss_{run_id}.png")

    log_file.close()
    print(f"\nDone. Loss log → {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on MNIST / Fashion-MNIST")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None,
                        help="Override device from config (e.g. cpu, cuda, mps)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, device_override=args.device)
