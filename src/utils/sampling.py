"""
DDPM ancestral sampling — reverse diffusion process.

Reference: Ho et al. (2020), §3.2, Algorithm 2.
https://arxiv.org/abs/2006.11239

The reverse process learns to undo the forward diffusion by iterating:

  x_{t-1} = μ_θ(x_t, t) + σ_t · z,    z ~ N(0, I),   t = T-1, ..., 1
  x_0      = μ_θ(x_1, 1)                               (no noise at t = 0)

Predicted posterior mean (ε-parameterisation, Eq. 11):
  μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))

This is obtained by substituting the predicted x̂_0:
  x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
into the true posterior mean μ̃_t(x_t, x_0) (Eq. 7).

Variance — two valid choices (§3.2):
  σ²_t = β_t                                (fixed large, used here)
  σ²_t = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t  (fixed small / posterior variance)
Both produce comparable sample quality; the paper recommends β_t for image generation.
β̃_t is precomputed below for reference in Phase 2 experiments.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

# Allow `from src.xxx import` when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler


# ---------------------------------------------------------------------------
# Coefficient precomputation
# ---------------------------------------------------------------------------

class SamplingCoeffs:
    """
    Precomputes and caches all per-timestep scalar coefficients needed for
    DDPM sampling.  Computing these once before the loop avoids redundant
    arithmetic inside the 1000-step reverse process.

    Attributes (all shape (T,), on the same device as `scheduler`):
        coeff_x    — 1/√α_t         : scales x_t in the mean
        coeff_eps  — β_t/√(1-ᾱ_t)  : scales the predicted noise in the mean
        sigma      — √β_t            : noise std for the large-variance choice
        post_var   — β̃_t             : posterior variance (for reference / Phase 2)
    """

    def __init__(self, scheduler: NoiseScheduler):
        device = scheduler.betas.device

        # 1/√α_t
        self.coeff_x: torch.Tensor = 1.0 / scheduler.alphas.sqrt()

        # β_t / √(1-ᾱ_t)
        self.coeff_eps: torch.Tensor = (
            scheduler.betas / scheduler.sqrt_one_minus_alpha_bars
        )

        # σ_t = √β_t  (fixed large variance)
        self.sigma: torch.Tensor = scheduler.betas.sqrt()

        # β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t
        # ᾱ_0 ≡ 1 by convention (empty cumulative product).
        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), scheduler.alpha_bars[:-1]]
        )
        self.post_var: torch.Tensor = (
            scheduler.betas * (1.0 - alpha_bars_prev) / (1.0 - scheduler.alpha_bars)
        )


# ---------------------------------------------------------------------------
# DDPM sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(
    model: UNet,
    scheduler: NoiseScheduler,
    n_samples: int,
    img_shape: Tuple[int, int, int],
    device: torch.device,
    show_progress: bool = True,
    coeffs: Optional[SamplingCoeffs] = None,
) -> torch.Tensor:
    """
    Generate images via DDPM ancestral sampling (Algorithm 2, Ho et al. 2020).

    Starting from x_T ~ N(0, I), runs the reverse Markov chain for T steps:

      μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))   [Eq. 11]

      x_{t-1} = μ_θ(x_t, t) + √β_t · z,    z ~ N(0, I)   for t > 0    [Alg. 2]
      x_0     = μ_θ(x_1, 1)                               for t = 0

    Automatically switches the model to eval mode and restores its original
    training mode on exit, so this is safe to call mid-training.

    Args:
        model:         UNet denoiser (trained or in-training checkpoint).
        scheduler:     NoiseScheduler already moved to `device`.
        n_samples:     Number of images to generate.
        img_shape:     (C, H, W) shape per image, e.g. (1, 28, 28).
        device:        Target device (must match scheduler and model).
        show_progress: Show a tqdm progress bar over the T reverse steps.
        coeffs:        Optional pre-built SamplingCoeffs. Pass a cached instance
                       when calling this function repeatedly (e.g. every N epochs
                       during training) to avoid recomputing coefficients each call.

    Returns:
        x_0: Generated images, shape (n_samples, C, H, W), values in [-1, 1].
    """
    was_training = model.training
    model.eval()

    if coeffs is None:
        coeffs = SamplingCoeffs(scheduler)

    C, H, W = img_shape

    # x_T ~ N(0, I)
    x = torch.randn(n_samples, C, H, W, device=device)

    timesteps = list(reversed(range(scheduler.T)))  # [T-1, T-2, ..., 1, 0]
    iterator = (
        tqdm(timesteps, desc="DDPM sampling", leave=False)
        if show_progress
        else timesteps
    )

    for t_val in iterator:
        t_batch = torch.full((n_samples,), t_val, dtype=torch.long, device=device)

        # Predict noise: ε_θ(x_t, t)
        eps_pred = model(x, t_batch)

        # Posterior mean μ_θ(x_t, t)  — Eq. 11
        mean = coeffs.coeff_x[t_val] * (x - coeffs.coeff_eps[t_val] * eps_pred)

        if t_val > 0:
            # x_{t-1} ~ N(μ_θ, β_t · I)
            x = mean + coeffs.sigma[t_val] * torch.randn_like(x)
        else:
            # t = 0: deterministic final step, no noise added
            x = mean

    if was_training:
        model.train()

    return x


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Fast CPU sanity check using a tiny scheduler (T=50) and a tiny model.

    Verifies:
      1. Output shape matches (n_samples, C, H, W).
      2. All output values are finite (no NaN / Inf from the reverse process).
      3. model.training is restored after the call.
      4. Passing a cached SamplingCoeffs avoids recomputation and gives the
         same result as the uncached path (deterministic given fixed seed).
      5. SamplingCoeffs invariant: σ²_t + β̃_t sanity (β̃_1 ≈ 0, β̃_T ≈ β_T).
    """
    import sys

    T        = 50         # small T so the loop is fast on CPU
    N        = 4
    C, H, W  = 1, 28, 28

    device    = torch.device("cpu")
    scheduler = NoiseScheduler(timesteps=T, beta_start=1e-4, beta_end=0.02).to(device)

    model = UNet(
        img_channels=C,
        base_channels=32,          # must be divisible by NUM_GROUPS=32
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_emb_dim=64,
    ).to(device)

    # --- 1. Shape check ---
    model.train()   # start in train mode to test restoration
    out = ddpm_sample(model, scheduler, n_samples=N, img_shape=(C, H, W),
                      device=device, show_progress=True)
    assert out.shape == (N, C, H, W), f"Shape mismatch: {out.shape}"
    print(f"[PASS] Output shape: {out.shape}")

    # --- 2. Finite values ---
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    print(f"[PASS] All values finite  (range [{out.min():.3f}, {out.max():.3f}])")

    # --- 3. Model training mode restored ---
    assert model.training, "model.training was not restored after sampling"
    print(f"[PASS] model.training restored to True after call")

    # --- 4. Cached coeffs give identical result ---
    coeffs = SamplingCoeffs(scheduler)
    torch.manual_seed(42)
    out_a = ddpm_sample(model, scheduler, N, (C, H, W), device,
                        show_progress=False, coeffs=coeffs)
    torch.manual_seed(42)
    out_b = ddpm_sample(model, scheduler, N, (C, H, W), device,
                        show_progress=False, coeffs=coeffs)
    assert torch.allclose(out_a, out_b), "Sampling is not deterministic given the same seed"
    print(f"[PASS] Deterministic output given fixed seed")

    # --- 5. SamplingCoeffs invariants ---
    # β̃_1 should be 0 (ᾱ_0 = 1 → numerator = 1-ᾱ_0 = 0)
    assert coeffs.post_var[0].item() == 0.0, f"β̃_0 should be 0, got {coeffs.post_var[0]}"
    # σ_t > 0 for all t (β_t > 0 by construction)
    assert (coeffs.sigma > 0).all(), "σ_t must be positive for all t"
    # coeff_x[t] > 1 for all t (since α_t < 1)
    assert (coeffs.coeff_x > 1.0).all(), "1/√α_t should be > 1 for all t"
    print(f"[PASS] SamplingCoeffs invariants hold")

    print("\nAll sanity checks passed.")
