"""
Reverse diffusion samplers: DDPM (Ho et al. 2020) and DDIM (Song et al. 2020).

DDPM — ancestral sampling (Algorithm 2, Ho et al. 2020)
  Runs T stochastic steps.  Reliable but slow.

  x_{t-1} = (1/√α_t) · (x_t - β_t/√(1-α̅_t) · ε_θ(x_t, t)) + √β_t · z

DDIM — implicit model (Eq. 12, Song et al. 2020)
  Uses a user-chosen subsequence τ ⊂ {0,...,T-1} of S ≤ T steps.
  Much faster at S=50 steps with η=0 (deterministic) and near-identical
  quality to DDPM at S=T, η=1.

  x̂_0     = (x - √(1-α̅_τ) · ε_θ(x, τ)) / √α̅_τ

  x_{τ_prev} = √α̅_{τ_prev} · x̂_0
             + √(1 - α̅_{τ_prev} - σ²) · ε_θ           [direction to x_τ]
             + σ · z

  σ = η · √((1-α̅_{τ_prev}) / (1-α̅_τ)) · √(1 - α̅_τ/α̅_{τ_prev})

  η = 0 → deterministic (DDIM); η = 1 → recovers DDPM posterior variance β̃_t.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.unet import UNet
from src.utils.diffusion import NoiseScheduler


# ---------------------------------------------------------------------------
# DDPM coefficient precomputation
# ---------------------------------------------------------------------------

class SamplingCoeffs:
    """
    Precomputes per-timestep scalar coefficients for DDPM ancestral sampling.

    All tensors have shape (T,) and live on the same device as `scheduler`.

        coeff_x   — 1/√α_t          : scales x_t in the posterior mean
        coeff_eps — β_t/√(1-α̅_t)   : scales predicted noise in the mean
        sigma     — √β_t             : noise std (fixed large variance)
        post_var  — β̃_t              : posterior (fixed small) variance,
                                       documented here for Phase 2 ablations
    """

    def __init__(self, scheduler: NoiseScheduler):
        device = scheduler.betas.device

        self.coeff_x:   torch.Tensor = 1.0 / scheduler.alphas.sqrt()
        self.coeff_eps: torch.Tensor = scheduler.betas / scheduler.sqrt_one_minus_alpha_bars
        self.sigma:     torch.Tensor = scheduler.betas.sqrt()

        # β̃_t = (1-α̅_{t-1}) / (1-α̅_t) · β_t,  with α̅_0 ≡ 1
        ab_prev = torch.cat([torch.ones(1, device=device), scheduler.alpha_bars[:-1]])
        self.post_var: torch.Tensor = (
            scheduler.betas * (1.0 - ab_prev) / (1.0 - scheduler.alpha_bars)
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
    DDPM ancestral sampling (Algorithm 2, Ho et al. 2020).

    Starting from x_T ~ N(0, I), iterates the reverse Markov chain for T steps:

      μ_θ(x_t, t) = (1/√α_t) · (x_t - β_t/√(1-α̅_t) · ε_θ(x_t, t))   [Eq. 11]

      x_{t-1} = μ_θ(x_t, t) + √β_t · z,    z ~ N(0, I)   for t > 0    [Alg. 2]
      x_0     = μ_θ(x_1, 1)                               for t = 0

    Variance σ²_t = β_t (fixed large; §3.2). The tighter posterior variance
    β̃_t = (1-α̅_{t-1})/(1-α̅_t) · β_t is precomputed in SamplingCoeffs.post_var
    for ablations but is not used in the default sampling path here.

    Switches model to eval and restores training mode on exit.

    Args:
        model:         UNet denoiser.
        scheduler:     NoiseScheduler on `device`.
        n_samples:     Number of images to generate.
        img_shape:     (C, H, W) per image.
        device:        Target device.
        show_progress: Show tqdm bar.
        coeffs:        Pre-built SamplingCoeffs; built here if None.

    Returns:
        x_0: shape (n_samples, C, H, W), values in [-1, 1].
    """
    was_training = model.training
    model.eval()

    if coeffs is None:
        coeffs = SamplingCoeffs(scheduler)

    C, H, W = img_shape
    x = torch.randn(n_samples, C, H, W, device=device)

    timesteps = list(reversed(range(scheduler.T)))
    iterator  = (
        tqdm(timesteps, desc="DDPM sampling", leave=False)
        if show_progress else timesteps
    )

    for t_val in iterator:
        t_batch  = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
        eps_pred = model(x, t_batch)

        mean = coeffs.coeff_x[t_val] * (x - coeffs.coeff_eps[t_val] * eps_pred)

        if t_val > 0:
            x = mean + coeffs.sigma[t_val] * torch.randn_like(x)
        else:
            x = mean

    if was_training:
        model.train()

    return x


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    model: UNet,
    scheduler: NoiseScheduler,
    n_samples: int,
    img_shape: Tuple[int, int, int],
    device: torch.device,
    n_steps: int = 50,
    eta: float = 0.0,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    DDIM sampling (Song et al. 2020, Eq. 12).

    Runs S = `n_steps` denoising passes over a uniformly-spaced subsequence
    τ ⊂ {0,...,T-1}, enabling high-quality samples with far fewer NFE than DDPM.

    At each step (τ_i → τ_{i-1}):

      x̂_0 = (x - √(1-α̅_{τ_i}) · ε_θ(x, τ_i)) / √α̅_{τ_i}       [predicted x_0]
      x̂_0 = clamp(x̂_0, -1, 1)                                    [stabilise]

      σ = η · √((1-α̅_{τ_{i-1}}) / (1-α̅_{τ_i})) · √(1 - α̅_{τ_i}/α̅_{τ_{i-1}})

      x_{τ_{i-1}} = √α̅_{τ_{i-1}} · x̂_0
                  + √(1 - α̅_{τ_{i-1}} - σ²) · ε_θ               [direction]
                  + σ · z,   z ~ N(0, I)                          [noise term]

    At the final step τ_{S-1} → "τ_{-1}", α̅_{τ_{-1}} ≡ 1, so σ = 0,
    the direction vanishes, and x = x̂_0 (clean predicted image).

    η = 0  → fully deterministic (no z); reproducible from seed.
    η = 1  → stochastic; approximates DDPM posterior variance β̃_t.

    Switches model to eval and restores training mode on exit.

    Args:
        model:         UNet denoiser.
        scheduler:     NoiseScheduler on `device`.
        n_samples:     Number of images to generate.
        img_shape:     (C, H, W) per image.
        device:        Target device.
        n_steps:       Number of denoising steps S ≤ T.
        eta:           Stochasticity level: 0 = deterministic, 1 ≈ DDPM.
        show_progress: Show tqdm bar.

    Returns:
        x_0: shape (n_samples, C, H, W), values in [-1, 1].
    """
    was_training = model.training
    model.eval()

    C, H, W = img_shape

    # Subsequence τ: S timesteps, evenly spaced from T-1 down to 0.
    # τ[0] = T-1 (most noisy), τ[-1] = 0 (least noisy).
    tau = torch.linspace(scheduler.T - 1, 0, n_steps).round().long().tolist()

    x = torch.randn(n_samples, C, H, W, device=device)

    iterator = (
        tqdm(range(n_steps), total=n_steps,
             desc=f"DDIM sampling ({n_steps} steps, η={eta})", leave=False)
        if show_progress else range(n_steps)
    )

    for i in iterator:
        t_val = tau[i]

        ab_cur  = scheduler.alpha_bars[t_val]   # α̅_{τ_i}

        # α̅_{τ_{i-1}}: use 1.0 past the clean boundary (final step)
        t_prev  = tau[i + 1] if i + 1 < n_steps else -1
        ab_prev = (
            scheduler.alpha_bars[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=device)
        )

        # Predict noise ε_θ(x_{τ_i}, τ_i)
        t_batch  = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
        eps_pred = model(x, t_batch)

        # Predicted x̂_0 (Eq. 9 / 12)
        x0_pred = (
            (x - scheduler.sqrt_one_minus_alpha_bars[t_val] * eps_pred)
            / scheduler.sqrt_alpha_bars[t_val]
        )
        x0_pred = x0_pred.clamp(-1.0, 1.0)   # numerical stability

        # σ_{τ_i}: DDIM stochasticity parameter (0 when η=0 or at final step)
        sigma = (
            eta
            * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_cur).clamp(min=1e-8))
            * torch.sqrt(torch.clamp(1.0 - ab_cur / ab_prev.clamp(min=1e-8), min=0.0))
        )

        # "Direction pointing to x_{τ_i}" (Eq. 12, second term coefficient)
        dir_coeff = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0))

        # DDIM update (Eq. 12)
        x = ab_prev.sqrt() * x0_pred + dir_coeff * eps_pred
        if eta > 0.0:
            x = x + sigma * torch.randn_like(x)

    if was_training:
        model.train()

    return x


# ---------------------------------------------------------------------------
# CFG + DDIM sampler (Phase 3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_cfg_sample(
    model: UNet,
    clip_enc,
    prompts: List[str],
    scheduler: NoiseScheduler,
    device: torch.device,
    img_shape: Tuple[int, int, int],
    n_steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 7.5,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    DDIM sampling with classifier-free guidance (CFG, Ho et al. 2022).

    Generates one image per prompt using DDIM and combines conditional and
    unconditional noise predictions at each step:

        ε_cfg = (1 + w) · ε_cond − w · ε_uncond        (Ho et al. 2022, Eq. 6)

    where w = guidance_scale.  w=0 → pure conditional (ε_cond); larger w
    amplifies the text signal at the cost of sample diversity.

    Both conditional and unconditional passes are batched into a single UNet
    forward by doubling the batch:

        x_doubled   = [x | x]          shape (2B, C, H, W)
        text_doubled = [cond | null]    shape (2B, D)

        ε_cond, ε_uncond = UNet(x_doubled, t, text_doubled).chunk(2)

    This halves wall time per step compared to two separate model calls.
    The null embedding encodes "" (empty string) via CLIP — same space
    as real captions, consistent with CFG training convention.

    The DDIM update formula follows ddim_sample exactly; only the noise
    prediction is replaced with the CFG-combined estimate.

    Args:
        model:          Conditional UNet with text_emb support.
        clip_enc:       Frozen CLIPTextEncoder; callable as clip_enc(List[str]).
        prompts:        B caption strings; generates one image per prompt.
        scheduler:      NoiseScheduler on `device`.
        device:         Target device.
        img_shape:      (C, H, W) per image.
        n_steps:        Number of DDIM denoising steps S ≤ T.
        eta:            Stochasticity: 0 = deterministic, 1 ≈ DDPM.
        guidance_scale: w in the CFG formula.
        show_progress:  Show tqdm bar.

    Returns:
        x_0: shape (B, C, H, W), values in [-1, 1].
    """
    was_training = model.training
    model.eval()

    B = len(prompts)
    C, H, W = img_shape

    # Encode embeddings once before the denoising loop.
    cond_emb = clip_enc(prompts)           # (B, D)
    null_emb = clip_enc([""] * B)         # (B, D) — null embedding for CFG
    text_doubled = torch.cat([cond_emb, null_emb], dim=0)  # (2B, D)

    tau = torch.linspace(scheduler.T - 1, 0, n_steps).round().long().tolist()

    x = torch.randn(B, C, H, W, device=device)

    iterator = (
        tqdm(range(n_steps), total=n_steps,
             desc=f"CFG-DDIM ({n_steps} steps, w={guidance_scale})", leave=False)
        if show_progress else range(n_steps)
    )

    for i in iterator:
        t_val = tau[i]

        ab_cur  = scheduler.alpha_bars[t_val]
        t_prev  = tau[i + 1] if i + 1 < n_steps else -1
        ab_prev = (
            scheduler.alpha_bars[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=device)
        )

        # Single batched forward for both conditional and unconditional paths.
        x_doubled = torch.cat([x, x], dim=0)                           # (2B, C, H, W)
        t_doubled  = torch.full((2 * B,), t_val, dtype=torch.long, device=device)
        eps_both   = model(x_doubled, t_doubled, text_emb=text_doubled) # (2B, C, H, W)
        eps_cond, eps_uncond = eps_both[:B], eps_both[B:]

        # CFG combination (Ho et al. 2022, Eq. 6):
        #   ε_cfg = (1+w)·ε_cond − w·ε_uncond
        eps = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

        # DDIM update (identical to ddim_sample after this point).
        x0_pred = (
            (x - scheduler.sqrt_one_minus_alpha_bars[t_val] * eps)
            / scheduler.sqrt_alpha_bars[t_val]
        ).clamp(-1.0, 1.0)

        sigma = (
            eta
            * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_cur).clamp(min=1e-8))
            * torch.sqrt(torch.clamp(1.0 - ab_cur / ab_prev.clamp(min=1e-8), min=0.0))
        )
        dir_coeff = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0))

        x = ab_prev.sqrt() * x0_pred + dir_coeff * eps
        if eta > 0.0:
            x = x + sigma * torch.randn_like(x)

    if was_training:
        model.train()

    return x


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Fast CPU checks for both DDPM and DDIM samplers using a tiny model (T=50).

    DDPM checks (existing):
      1. Output shape matches.
      2. Values finite.
      3. model.training restored.
      4. Deterministic given fixed seed.
      5. SamplingCoeffs invariants.

    DDIM checks (new):
      6. Output shape matches.
      7. Values finite.
      8. model.training restored.
      9. η=0 is deterministic (same seed → same output).
      10. η=0 and η=1 produce different outputs (stochasticity is live).
      11. n_steps << T completes without error.
    """
    T       = 50
    N       = 4
    C, H, W = 1, 28, 28
    device  = torch.device("cpu")

    scheduler = NoiseScheduler(timesteps=T, beta_start=1e-4, beta_end=0.02).to(device)

    model = UNet(
        img_channels=C, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64,
    ).to(device)

    # ------------------------------------------------------------------ DDPM
    print("=== DDPM ===")

    model.train()
    out = ddpm_sample(model, scheduler, N, (C, H, W), device, show_progress=True)
    assert out.shape == (N, C, H, W)
    print(f"[PASS] Shape: {out.shape}")

    assert torch.isfinite(out).all()
    print(f"[PASS] Values finite  [{out.min():.3f}, {out.max():.3f}]")

    assert model.training
    print("[PASS] model.training restored")

    coeffs = SamplingCoeffs(scheduler)
    torch.manual_seed(0)
    a = ddpm_sample(model, scheduler, N, (C, H, W), device, False, coeffs)
    torch.manual_seed(0)
    b = ddpm_sample(model, scheduler, N, (C, H, W), device, False, coeffs)
    assert torch.allclose(a, b)
    print("[PASS] Deterministic given same seed")

    assert coeffs.post_var[0].item() == 0.0
    assert (coeffs.sigma > 0).all()
    assert (coeffs.coeff_x > 1.0).all()
    print("[PASS] SamplingCoeffs invariants")

    # ------------------------------------------------------------------ DDIM
    print("\n=== DDIM ===")

    # 6. Shape
    model.train()
    out = ddim_sample(model, scheduler, N, (C, H, W), device,
                      n_steps=10, eta=0.0, show_progress=True)
    assert out.shape == (N, C, H, W)
    print(f"[PASS] Shape: {out.shape}  (n_steps=10, T={T})")

    # 7. Finite values
    assert torch.isfinite(out).all()
    print(f"[PASS] Values finite  [{out.min():.3f}, {out.max():.3f}]")

    # 8. model.training restored
    assert model.training
    print("[PASS] model.training restored")

    # 9. η=0 is deterministic
    torch.manual_seed(7)
    d1 = ddim_sample(model, scheduler, N, (C, H, W), device, n_steps=10, eta=0.0, show_progress=False)
    torch.manual_seed(7)
    d2 = ddim_sample(model, scheduler, N, (C, H, W), device, n_steps=10, eta=0.0, show_progress=False)
    assert torch.allclose(d1, d2)
    print("[PASS] η=0 is deterministic (same seed → same output)")

    # 10. η=0 vs η=1 differ (stochasticity is live)
    torch.manual_seed(7)
    det = ddim_sample(model, scheduler, N, (C, H, W), device, n_steps=10, eta=0.0, show_progress=False)
    torch.manual_seed(7)
    sto = ddim_sample(model, scheduler, N, (C, H, W), device, n_steps=10, eta=1.0, show_progress=False)
    assert not torch.allclose(det, sto)
    print("[PASS] η=0 and η=1 produce different outputs")

    # 11. n_steps=1 (extreme case) completes without error
    out1 = ddim_sample(model, scheduler, N, (C, H, W), device, n_steps=1, eta=0.0, show_progress=False)
    assert out1.shape == (N, C, H, W) and torch.isfinite(out1).all()
    print("[PASS] n_steps=1 completes successfully")

    # ------------------------------------------------------------------ CFG-DDIM
    print("\n=== CFG-DDIM (mock CLIP, Phase 3) ===")

    # Mock CLIP encoder: returns random unit-normalised vectors.
    # Real CLIPTextEncoder is not used here to keep the check dependency-free.
    class _MockCLIP:
        """Minimal stand-in for CLIPTextEncoder.forward()."""
        def __call__(self, captions: List[str]) -> torch.Tensor:
            vecs = torch.randn(len(captions), 512)
            return vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Conditional UNet with tiny text_emb_dim for the mock.
    cond_model = UNet(
        img_channels=C, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64, text_emb_dim=512, attn_heads=8,
    ).to(device)

    mock_enc = _MockCLIP()
    prompts  = ["a photo of a cat", "a photo of a dog", "a photo of a frog", "a photo of a ship"]

    # 12. Output shape
    model.train()
    out_cfg = ddim_cfg_sample(
        cond_model, mock_enc, prompts, scheduler, device,
        img_shape=(C, H, W), n_steps=5, eta=0.0, guidance_scale=7.5,
    )
    assert out_cfg.shape == (len(prompts), C, H, W), f"Shape: {out_cfg.shape}"
    print(f"[PASS] 12. CFG-DDIM output shape: {tuple(out_cfg.shape)}")

    # 13. Values finite
    assert torch.isfinite(out_cfg).all()
    print(f"[PASS] 13. Values finite  [{out_cfg.min():.3f}, {out_cfg.max():.3f}]")

    # 14. model.training restored (cond_model was in train mode before call)
    assert cond_model.training
    print("[PASS] 14. model.training restored after CFG sampling")

    # 15. Guidance scale affects output (w=0 vs w=7.5 differ)
    torch.manual_seed(3)
    out_w0  = ddim_cfg_sample(cond_model, mock_enc, prompts, scheduler, device,
                               img_shape=(C, H, W), n_steps=5, guidance_scale=0.0,
                               show_progress=False)
    torch.manual_seed(3)
    out_w75 = ddim_cfg_sample(cond_model, mock_enc, prompts, scheduler, device,
                               img_shape=(C, H, W), n_steps=5, guidance_scale=7.5,
                               show_progress=False)
    assert not torch.allclose(out_w0, out_w75), "guidance_scale has no effect"
    print("[PASS] 15. w=0 and w=7.5 produce different outputs (guidance is live)")

    print("\nAll sanity checks passed.")
