"""
Forward diffusion process (noise scheduler) for DDPM.

Two beta schedules are supported:

  linear  — Ho et al. (2020): β_t linearly spaced in [β_start, β_end].
             Simple, but can over-destroy information early; α̅_t drops
             sharply and the model sees very little signal by t ≈ T/2.

  cosine  — Nichol & Dhariwal (2021), §3.2: α̅_t defined via a cosine
             function of t, giving a smoother SNR trajectory.
             β_t is derived as 1 - α̅_t/α̅_{t-1}, clipped to [0, 0.999].

Key notation (matches both papers):
  T          — total diffusion timesteps
  β_t        — noise variance at step t
  α_t        — 1 - β_t
  α̅_t        — cumulative product α_1 · ... · α_t
  ε          — noise ~ N(0, I)

Closed-form forward process (Ho et al., Eq. 4) — same for both schedules:
  q(x_t | x_0) = N(x_t;  √α̅_t · x_0,  (1 - α̅_t) · I)
  x_t = √α̅_t · x_0 + √(1 - α̅_t) · ε
"""

import math
from typing import Optional

import torch


class NoiseScheduler:
    """
    Precomputes and caches all diffusion constants for a chosen beta schedule.

    Supports 'linear' (Ho et al. 2020) and 'cosine' (Nichol & Dhariwal 2021).
    All tensors are 1-D with length T, indexed by timestep t in [0, T-1].
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        cosine_s: float = 0.008,
    ):
        """
        Args:
            timesteps:  T — total number of diffusion steps.
            beta_start: β_1 for the linear schedule (ignored for cosine).
            beta_end:   β_T for the linear schedule (ignored for cosine).
            schedule:   'linear' or 'cosine'.
            cosine_s:   Offset s for the cosine schedule (paper default 0.008).
        """
        self.T = timesteps

        if schedule == "linear":
            # β_t linearly spaced in [β_start, β_end], shape (T,)
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            # β_t derived from the cosine α̅_t schedule, shape (T,)
            self.betas = self._cosine_betas(timesteps, cosine_s)
        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose 'linear' or 'cosine'."
            )

        # α_t = 1 - β_t, shape (T,)
        self.alphas = 1.0 - self.betas

        # α̅_t = ∏_{s=1}^{t} α_s, shape (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precomputed square-root terms used in q_sample and the reverse process.
        self.sqrt_alpha_bars           = torch.sqrt(self.alpha_bars)           # √α̅_t
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)     # √(1-α̅_t)

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_betas(timesteps: int, s: float) -> torch.Tensor:
        """
        Cosine schedule (Nichol & Dhariwal 2021, §3.2, Eq. 17).

        Defines signal retention α̅_t as a cosine function of t:

          f(t) = cos²( (t/T + s) / (1 + s) · π/2 )
          α̅_t  = f(t) / f(0)                           (normalised so α̅_0 = 1)

        The small offset s = 0.008 prevents β_t from being too small near
        t = 0, which would make the first few steps nearly noise-free and
        waste model capacity.

        β_t is then derived from consecutive α̅ ratios and clipped:
          β_t = clip(1 - α̅_t / α̅_{t-1},  0,  0.999)

        The 0.999 clip avoids numerical instability when α̅_t ≈ 0 near t = T.

        Args:
            timesteps: T
            s:         offset (default 0.008 from the paper)

        Returns:
            betas: shape (T,), float32
        """
        # Float64 for precision, then cast back to float32.
        t = torch.arange(timesteps + 1, dtype=torch.float64)   # 0, 1, ..., T
        f = torch.cos(((t / timesteps + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        alpha_bars = f / f[0]                              # shape (T+1,), α̅_0 = 1
        betas      = 1.0 - alpha_bars[1:] / alpha_bars[:-1]   # shape (T,)
        return betas.clamp(0.0, 0.999).to(torch.float32)

    # ------------------------------------------------------------------
    def to(self, device):
        """Move all precomputed tensors to `device`."""
        self.betas                    = self.betas.to(device)
        self.alphas                   = self.alphas.to(device)
        self.alpha_bars               = self.alpha_bars.to(device)
        self.sqrt_alpha_bars          = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def _gather(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Index a (T,) schedule tensor by a batch of timesteps, reshape to
        (B, 1, 1, 1) so it broadcasts over (B, C, H, W) image tensors.
        """
        return tensor[t].view(-1, 1, 1, 1)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) via the closed-form reparameterisation
        (Ho et al. 2020, Eq. 4) — identical for both beta schedules:

          x_t = √α̅_t · x_0 + √(1 - α̅_t) · ε,   ε ~ N(0, I)

        Args:
            x0:    Clean image tensor, shape (B, C, H, W), values in [-1, 1].
            t:     Integer timestep indices, shape (B,), in [0, T-1].
            noise: Optional pre-sampled ε; sampled internally if None.

        Returns:
            x_t: Noisy image at timestep t, same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab    = self._gather(self.sqrt_alpha_bars, t)
        sqrt_1m_ab = self._gather(self.sqrt_one_minus_alpha_bars, t)
        return sqrt_ab * x0 + sqrt_1m_ab * noise


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Checks both the linear and cosine schedules share the correct boundary
    properties and that the cosine schedule is more gradual (higher α̅ at T/2).
    """
    T = 1000
    B, C, H, W = 4, 1, 28, 28
    x0 = torch.zeros(B, C, H, W)

    for name, sched in [
        ("linear", NoiseScheduler(T, beta_start=1e-4, beta_end=0.02, schedule="linear")),
        ("cosine", NoiseScheduler(T, schedule="cosine")),
    ]:
        print(f"\n=== {name} schedule ===")

        # q_sample shape
        t_mid = torch.full((B,), T // 2, dtype=torch.long)
        xt = sched.q_sample(x0, t_mid)
        assert xt.shape == x0.shape, f"Shape mismatch: {xt.shape}"
        print(f"[PASS] q_sample shape: {xt.shape}")

        # t=0: signal dominates (√α̅_0 ≈ 1)
        assert sched.sqrt_alpha_bars[0].item() > 0.99, \
            f"√α̅_0 = {sched.sqrt_alpha_bars[0]:.4f}, expected > 0.99"
        print(f"[PASS] t=0  √α̅_t = {sched.sqrt_alpha_bars[0]:.4f}  (signal dominated)")

        # t=T-1: noise dominates (√α̅_{T-1} < 0.1)
        assert sched.sqrt_alpha_bars[-1].item() < 0.1, \
            f"√α̅_{{T-1}} = {sched.sqrt_alpha_bars[-1]:.4f}, expected < 0.1"
        print(f"[PASS] t=T-1 √α̅_t = {sched.sqrt_alpha_bars[-1]:.4f}  (noise dominated)")

        # Unit-variance decomposition: α̅_t + (1-α̅_t) = 1
        for idx in [0, 100, 500, 999]:
            assert abs(sched.alpha_bars[idx].item() + (1 - sched.alpha_bars[idx].item()) - 1.0) < 1e-6
        print("[PASS] α̅_t + (1-α̅_t) = 1 at all checked timesteps")

        # Deterministic given fixed noise
        noise = torch.randn(B, C, H, W)
        assert torch.allclose(sched.q_sample(x0, t_mid, noise), sched.q_sample(x0, t_mid, noise))
        print("[PASS] q_sample deterministic given fixed noise")

        # Betas in (0, 0.999]
        assert (sched.betas > 0).all() and (sched.betas <= 0.999).all(), \
            f"β_t out of range: [{sched.betas.min():.6f}, {sched.betas.max():.4f}]"
        print(f"[PASS] β_t ∈ (0, 0.999]:  min={sched.betas.min():.6f}  max={sched.betas.max():.4f}")

    # Cosine should retain significantly more signal at T/2 than linear
    lin = NoiseScheduler(T, schedule="linear")
    cos = NoiseScheduler(T, schedule="cosine")
    ab_lin = lin.alpha_bars[T // 2].item()
    ab_cos = cos.alpha_bars[T // 2].item()
    assert ab_cos > ab_lin, f"Cosine α̅_{{T/2}} ({ab_cos:.4f}) should > linear ({ab_lin:.4f})"
    assert ab_cos > 0.3,    f"Cosine α̅_{{T/2}} = {ab_cos:.4f}, expected > 0.3"
    print(f"\n[PASS] Cosine α̅_{{T/2}} ({ab_cos:.4f}) >> Linear α̅_{{T/2}} ({ab_lin:.4f})  — more gradual")

    # Invalid schedule name raises ValueError
    try:
        NoiseScheduler(schedule="quadratic")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[PASS] Invalid schedule name raises ValueError")

    print("\nAll sanity checks passed.")
