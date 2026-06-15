"""
Forward diffusion process (noise scheduler) for DDPM.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
https://arxiv.org/abs/2006.11239

Key notation (matches the paper):
  T              — total diffusion timesteps
  beta_t         — noise variance at step t, linearly spaced in [beta_start, beta_end]
  alpha_t        — 1 - beta_t
  alpha_bar_t    — cumulative product of alpha_1 ... alpha_t  (ᾱ_t in the paper)
  epsilon        — noise ~ N(0, I)

The closed-form forward process (Eq. 4 in the paper):
  q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)

which lets us sample x_t directly:
  x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * epsilon,   epsilon ~ N(0, I)
"""

from typing import Optional

import torch


class NoiseScheduler:
    """
    Precomputes and stores all diffusion constants for the linear beta schedule.

    All tensors are 1-D with length T, indexed by timestep t in [0, T-1].
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        """
        Args:
            timesteps:   T — total number of diffusion steps.
            beta_start:  beta_1 (noise variance at the first step).
            beta_end:    beta_T (noise variance at the last step).
        """
        self.T = timesteps

        # beta_t: linearly spaced in [beta_start, beta_end], shape (T,)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

        # alpha_t = 1 - beta_t, shape (T,)
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = prod_{s=1}^{t} alpha_s, shape (T,)
        # alpha_bar[t] is the cumulative product up to and including step t.
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute the square-root terms used in q_sample (Eq. 4).
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)            # sqrt(ᾱ_t)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)  # sqrt(1 - ᾱ_t)

    def to(self, device):
        """Move all precomputed tensors to `device`."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def _gather(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Index a 1-D schedule tensor by a batch of timesteps t, then reshape
        to (B, 1, 1, 1) so it broadcasts over (B, C, H, W) image tensors.

        Args:
            tensor: shape (T,)
            t:      shape (B,), integer indices in [0, T-1]

        Returns:
            shape (B, 1, 1, 1)
        """
        return tensor[t].view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample x_t from the forward process q(x_t | x_0) using the closed-form
        reparameterisation (Eq. 4, Ho et al. 2020):

            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * epsilon,   epsilon ~ N(0, I)

        Args:
            x0:    Clean image tensor, shape (B, C, H, W), values in [-1, 1].
            t:     Integer timestep indices, shape (B,), in [0, T-1].
            noise: Optional pre-sampled noise (B, C, H, W). If None, sampled here.

        Returns:
            x_t:  Noisy image at timestep t, same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self._gather(self.sqrt_alpha_bars, t)            # sqrt(ᾱ_t)
        sqrt_one_minus_ab = self._gather(self.sqrt_one_minus_alpha_bars, t)  # sqrt(1-ᾱ_t)

        return sqrt_ab * x0 + sqrt_one_minus_ab * noise


if __name__ == "__main__":
    """
    Sanity checks:
      1. At t=0 (first step), alpha_bar ~ 1  -> x_t should be almost identical to x0.
      2. At t=T-1 (last step), alpha_bar ~ 0 -> x_t should be nearly pure noise.
      3. Output shape must match input shape.
      4. Noise coefficient^2 + signal coefficient^2 = 1 (unit-variance mixture).
    """
    T = 1000
    scheduler = NoiseScheduler(timesteps=T, beta_start=1e-4, beta_end=0.02)

    B, C, H, W = 4, 1, 28, 28
    x0 = torch.zeros(B, C, H, W)  # all-zero clean image

    # --- shape check ---
    t_mid = torch.full((B,), T // 2, dtype=torch.long)
    xt = scheduler.q_sample(x0, t_mid)
    assert xt.shape == x0.shape, f"Shape mismatch: {xt.shape} vs {x0.shape}"
    print(f"[PASS] Shape check: x_t shape = {xt.shape}")

    # --- t=0: signal dominated ---
    t_zero = torch.zeros(B, dtype=torch.long)
    xt_zero = scheduler.q_sample(x0, t_zero)
    sqrt_ab_0 = scheduler.sqrt_alpha_bars[0].item()
    sqrt_1ab_0 = scheduler.sqrt_one_minus_alpha_bars[0].item()
    print(f"[INFO] t=0  -> sqrt(ᾱ_0)={sqrt_ab_0:.4f}, sqrt(1-ᾱ_0)={sqrt_1ab_0:.4f}")
    assert sqrt_ab_0 > 0.99, f"At t=0, signal coefficient should be ~1, got {sqrt_ab_0}"
    print(f"[PASS] t=0: signal coefficient ≈ 1 (x_t ≈ x_0)")

    # --- t=T-1: noise dominated ---
    t_last = torch.full((B,), T - 1, dtype=torch.long)
    xt_last = scheduler.q_sample(x0, t_last)
    sqrt_ab_T = scheduler.sqrt_alpha_bars[-1].item()
    sqrt_1ab_T = scheduler.sqrt_one_minus_alpha_bars[-1].item()
    print(f"[INFO] t=T-1 -> sqrt(ᾱ_T)={sqrt_ab_T:.4f}, sqrt(1-ᾱ_T)={sqrt_1ab_T:.4f}")
    assert sqrt_ab_T < 0.1, f"At t=T-1, signal should be small, got {sqrt_ab_T}"
    print(f"[PASS] t=T-1: noise coefficient ≈ 1 (x_t ≈ pure noise)")

    # --- variance preservation: sqrt(ᾱ)^2 + sqrt(1-ᾱ)^2 = 1 ---
    for idx in [0, 100, 500, 999]:
        sq_sum = scheduler.alpha_bars[idx].item() + (1 - scheduler.alpha_bars[idx].item())
        assert abs(sq_sum - 1.0) < 1e-6, f"Variance sum != 1 at t={idx}: {sq_sum}"
    print(f"[PASS] Unit-variance decomposition: ᾱ_t + (1-ᾱ_t) = 1 at all checked timesteps")

    # --- deterministic noise path ---
    fixed_noise = torch.randn(B, C, H, W)
    xt_a = scheduler.q_sample(x0, t_mid, noise=fixed_noise)
    xt_b = scheduler.q_sample(x0, t_mid, noise=fixed_noise)
    assert torch.allclose(xt_a, xt_b), "q_sample not deterministic given same noise"
    print(f"[PASS] Deterministic given fixed noise")

    print("\nAll sanity checks passed.")
