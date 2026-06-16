"""
Training script for DDPM — Phase 1/2 (unconditional) and Phase 3 (text-conditioned).

Phase 1/2 algorithm (Ho et al. 2020, Algorithm 1):
  1. Sample x_0 from the dataset.
  2. Sample t ~ Uniform{0, ..., T-1}.
  3. Sample ε ~ N(0, I).
  4. Compute x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε   (closed-form forward diffusion).
  5. Predict ε̂ = UNet(x_t, t).
  6. Minimise  L = ||ε̂ - ε||²   (simplified ELBO, Ho et al. 2020, Eq. 14).

Phase 3 additions (CIFAR-10, dataset.name == "cifar10"):
  - Frozen CLIP text encoder converts captions → text_emb (B, 512).
  - Classifier-free guidance (CFG) dropout (Ho et al. 2022):
      per sample, with probability p_uncond, replace its caption with "".
      This trains the UNet for both conditional and unconditional generation
      from shared weights, enabling CFG at inference.
  - text_emb passed to UNet.forward(); cross-attention conditions on it.
  - EMA of UNet weights (ema_decay=0.9999) for cleaner sample grids.
  - Linear LR warmup + gradient norm clipping (standard for larger models).
  - Sample grids show one image per CIFAR-10 class via CFG-DDIM so
    per-class quality can be tracked visually across epochs.

Usage:
    # Phase 1/2 (unconditional):
    python scripts/train.py --config configs/phase1_unconditional.yaml
    python scripts/train.py --config configs/phase2_improved.yaml --device cpu

    # Phase 3 (text-conditioned CIFAR-10):
    python scripts/train.py --config configs/phase3_cifar10.yaml
    python scripts/train.py --config configs/phase3_cifar10.yaml --device cpu

    # CPU smoke test — 2 training steps, verifies the full Phase 3 pipeline:
    python scripts/train.py --config configs/phase3_cifar10.yaml \\
        --device cpu --smoke-test
"""

import argparse
import copy
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CaptionedCIFAR10, get_dataloader
from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler
from src.utils.sampling import SamplingCoeffs, ddim_cfg_sample, ddim_sample, ddpm_sample


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


def save_checkpoint(
    path: Path,
    epoch: int,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    ema_model: Optional[UNet] = None,
) -> None:
    """Save model weights, optional EMA weights, optimizer state, and config."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }, path)


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

def train(
    cfg: dict,
    device_override: Optional[str] = None,
    smoke_test: bool = False,
) -> None:
    device  = resolve_device(device_override or cfg["training"]["device"])
    use_cuda = device.type == "cuda"

    # ------------------------------------------------------------------
    # Smoke-test overrides — tiny model, 2 steps, fast sampling.
    # Exists purely for rapid pipeline validation on CPU; not for Kaggle.
    # ------------------------------------------------------------------
    if smoke_test:
        cfg = copy.deepcopy(cfg)
        cfg["dataset"]["batch_size"] = 4
        cfg["training"]["epochs"] = 1
        cfg["training"]["save_every"] = 1
        cfg["training"]["sample_every"] = 1
        cfg["training"]["ddim_steps"] = 5
        # Shrink model to make forward/backward fast on CPU.
        cfg["model"]["base_channels"] = max(cfg["model"]["base_channels"] // 4, 16)

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------
    is_phase3 = cfg["dataset"]["name"] == "cifar10"

    # ------------------------------------------------------------------
    # Noise scheduler
    # ------------------------------------------------------------------
    scheduler = NoiseScheduler(
        timesteps  = cfg["diffusion"]["timesteps"],
        beta_start = cfg["diffusion"].get("beta_start", 1e-4),
        beta_end   = cfg["diffusion"].get("beta_end", 0.02),
        schedule   = cfg["diffusion"].get("beta_schedule", "linear"),
        cosine_s   = cfg["diffusion"].get("cosine_s", 0.008),
    ).to(device)

    # ------------------------------------------------------------------
    # U-Net (with optional cross-attention for Phase 3)
    # ------------------------------------------------------------------
    model = UNet(
        img_channels  = cfg["dataset"]["channels"],
        base_channels = cfg["model"]["base_channels"],
        channel_mults = tuple(cfg["model"]["channel_mults"]),
        num_res_blocks= cfg["model"]["num_res_blocks"],
        time_emb_dim  = cfg["model"]["time_emb_dim"],
        text_emb_dim  = cfg["model"].get("text_emb_dim"),    # None → unconditional
        attn_heads    = cfg["model"].get("attn_heads", 8),
    ).to(device)

    # ------------------------------------------------------------------
    # EMA shadow model (Phase 3 default: ema_decay=0.9999).
    # EMA weights are used for sample grids and saved in checkpoints.
    # The main model (not EMA) is what the optimizer updates.
    # ------------------------------------------------------------------
    ema_decay: Optional[float] = cfg["training"].get("ema_decay")
    ema_model: Optional[UNet]  = None
    if ema_decay is not None:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Frozen CLIP text encoder (Phase 3 only)
    # ------------------------------------------------------------------
    clip_enc = None
    if is_phase3:
        from src.models.clip_encoder import CLIPTextEncoder
        clip_model_name = cfg.get("clip", {}).get("model_name", "ViT-B/32")
        print(f"Loading CLIP ({clip_model_name}) …")
        clip_enc = CLIPTextEncoder(model_name=clip_model_name, device=device).to(device)
        print(f"  CLIP ready  (embedding_dim={clip_enc.embedding_dim}, all params frozen)")

    # ------------------------------------------------------------------
    # Optimizer + linear LR warmup
    # ------------------------------------------------------------------
    lr = cfg["training"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    warmup_steps = cfg["training"].get("lr_warmup_steps", 0)
    lr_sched = None
    if warmup_steps > 0:
        lr_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )

    grad_clip: Optional[float] = cfg["training"].get("grad_clip")

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    num_workers = cfg["dataset"].get("num_workers", 4 if use_cuda else 0)
    loader, _ = get_dataloader(
        name        = cfg["dataset"]["name"],
        batch_size  = cfg["dataset"]["batch_size"],
        train       = True,
        num_workers = 0 if smoke_test else (num_workers if use_cuda else 0),
        pin_memory  = use_cuda,
    )

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    paths      = cfg.get("paths", {})
    ckpt_dir   = Path(paths.get("checkpoint_dir", "outputs/checkpoints"))
    sample_dir = Path(paths.get("sample_dir",    "outputs/samples"))
    log_dir    = Path(paths.get("log_dir",        "outputs/logs"))
    for d in (ckpt_dir, sample_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path   = log_dir / f"loss_{run_id}.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "loss"])

    # ------------------------------------------------------------------
    # Sampling config
    # ------------------------------------------------------------------
    n_epochs     = cfg["training"]["epochs"]
    save_every   = cfg["training"]["save_every"]
    sample_every = cfg["training"]["sample_every"]
    img_channels = cfg["dataset"]["channels"]
    img_size     = cfg["dataset"]["image_size"]
    img_shape    = (img_channels, img_size, img_size)

    sampler_type   = cfg["training"].get("sampler", "ddpm")
    ddim_steps     = cfg["training"].get("ddim_steps", 50)
    ddim_eta       = cfg["training"].get("ddim_eta", 0.0)
    guidance_scale = cfg.get("sampling", {}).get("guidance_scale", 0.0)
    p_uncond       = cfg["training"].get("p_uncond", 0.0)

    sampling_coeffs = SamplingCoeffs(scheduler) if sampler_type == "ddpm" else None

    # Fixed sample prompts for the periodic grid.
    # Phase 3: one per CIFAR-10 class — tracks per-class quality over epochs.
    # Smoke test: use just 2 prompts to keep sampling fast.
    if is_phase3:
        sample_prompts: Optional[List[str]] = [
            CaptionedCIFAR10.CAPTION_TEMPLATE.format(c)
            for c in CaptionedCIFAR10.CLASSES
        ]
        if smoke_test:
            sample_prompts = sample_prompts[:2]
        n_samples = len(sample_prompts)
        nrow = min(5, n_samples)
    else:
        sample_prompts = None
        n_samples = cfg["training"].get("num_samples", 16)
        nrow = int(n_samples ** 0.5)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters())
    n_attn   = sum(p.numel() for n, p in model.named_parameters() if "attn" in n)
    print(f"Device     : {device}")
    print(f"Model      : {n_params:,} parameters"
          + (f"  ({n_attn:,} in cross-attn)" if is_phase3 and n_attn else ""))
    print(f"Dataset    : {cfg['dataset']['name']}  (batch {cfg['dataset']['batch_size']})")
    print(f"Schedule   : {cfg['diffusion'].get('beta_schedule', 'linear')}")
    if sampler_type == "ddim":
        print(f"Sampler    : ddim  ({ddim_steps} steps, η={ddim_eta}"
              + (f", guidance_scale={guidance_scale}" if is_phase3 else "") + ")")
    else:
        print(f"Sampler    : ddpm")
    if is_phase3:
        print(f"CFG        : p_uncond={p_uncond}  guidance_scale={guidance_scale}")
        if ema_model is not None:
            print(f"EMA        : decay={ema_decay}")
        if warmup_steps > 0:
            print(f"LR warmup  : {warmup_steps} steps")
    print(f"Epochs     : {n_epochs}  |  save every {save_every}  |  sample every {sample_every}")
    if smoke_test:
        print("[SMOKE TEST] 2 steps / epoch, tiny model, 5 DDIM steps")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    step_losses: list = []
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        steps_this_epoch = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", leave=True)
        for step_in_epoch, batch in enumerate(pbar):
            # Unpack batch:
            #   Phase 3 (CIFAR-10): (image[B,3,32,32], captions: List[str])
            #   Phase 1/2:          (image[B,1,28,28], labels: Tensor[B])
            x0 = batch[0].to(device)
            captions: Optional[List[str]] = batch[1] if is_phase3 else None
            B = x0.shape[0]

            # Step 2: sample t ~ Uniform{0, T-1}
            t = torch.randint(0, scheduler.T, (B,), device=device, dtype=torch.long)

            # Step 3: sample ε ~ N(0, I)
            noise = torch.randn_like(x0)

            # Step 4: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
            xt = scheduler.q_sample(x0, t, noise)

            # ---- CFG dropout + text embedding (Phase 3) ----
            # Per-sample dropout: replace caption with "" with prob p_uncond.
            # All captions (including dropped ones) are encoded in one CLIP
            # forward pass — "" encodes to the null embedding used at inference.
            text_emb: Optional[torch.Tensor] = None
            if clip_enc is not None and captions is not None:
                effective_captions = [
                    "" if (torch.rand(1).item() < p_uncond) else c
                    for c in captions
                ]
                text_emb = clip_enc(effective_captions)   # (B, 512), no_grad

            # Step 5: ε̂ = UNet(x_t, t, [text_emb])
            pred_noise = model(xt, t, text_emb=text_emb)

            # Step 6: simplified noise-prediction loss
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if lr_sched is not None:
                lr_sched.step()

            # EMA update:  ema_p ← ema_decay · ema_p + (1-ema_decay) · p
            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.lerp_(p, 1.0 - ema_decay)

            loss_val = loss.item()
            global_step += 1
            epoch_loss_sum += loss_val
            steps_this_epoch += 1
            step_losses.append((global_step, loss_val))

            log_writer.writerow([global_step, epoch, f"{loss_val:.6f}"])
            log_file.flush()

            pbar.set_postfix(loss=f"{loss_val:.4f}")

            # Smoke test: stop after 2 gradient steps per epoch.
            if smoke_test and step_in_epoch >= 1:
                break

        avg_loss = epoch_loss_sum / max(1, steps_this_epoch)
        print(f"  → epoch {epoch:3d}  avg loss {avg_loss:.4f}")

        # ---- Checkpoint ----
        if epoch % save_every == 0 or epoch == n_epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, cfg, ema_model)
            print(f"     checkpoint → {ckpt_path}")

        # ---- Sample grid ----
        if epoch % sample_every == 0:
            # Use EMA weights if available — they give cleaner samples.
            sample_model = ema_model if ema_model is not None else model

            if is_phase3 and clip_enc is not None and sample_prompts is not None:
                # One image per CIFAR-10 class, generated with CFG.
                samples = ddim_cfg_sample(
                    model          = sample_model,
                    clip_enc       = clip_enc,
                    prompts        = sample_prompts,
                    scheduler      = scheduler,
                    device         = device,
                    img_shape      = img_shape,
                    n_steps        = ddim_steps,
                    eta            = ddim_eta,
                    guidance_scale = guidance_scale,
                )
            elif sampler_type == "ddim":
                samples = ddim_sample(
                    sample_model, scheduler, n_samples, img_shape, device,
                    n_steps=ddim_steps, eta=ddim_eta,
                )
            else:
                samples = ddpm_sample(
                    sample_model, scheduler, n_samples, img_shape, device,
                    coeffs=sampling_coeffs,
                )

            sp = sample_dir / f"epoch_{epoch:04d}.png"
            save_sample_grid(samples, sp, nrow=nrow)
            print(f"     samples    → {sp}")

            save_loss_plot(step_losses, log_dir / f"loss_{run_id}.png")

    save_loss_plot(step_losses, log_dir / f"loss_{run_id}.png")
    log_file.close()
    print(f"\nDone. Loss log → {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DDPM — Phase 1/2 unconditional or Phase 3 text-conditioned"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None,
                        help="Override device from config (cpu / cuda / mps)")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Run 2 gradient steps on a tiny model to verify the full pipeline "
            "(CLIP loading, CFG dropout, EMA, CFG-DDIM sampling) in under 60s on CPU. "
            "Not a training run — use this before submitting to Kaggle."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, device_override=args.device, smoke_test=args.smoke_test)
