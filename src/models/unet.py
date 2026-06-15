"""
U-Net denoiser for DDPM (Ho et al. 2020), Phase 1 — unconditional, grayscale 28×28.

Architecture overview
---------------------
  x, t  →  init_conv  →  [down_0 → down_1 → down_2]  →  bottleneck  →  [up_2 → up_1 → up_0]  →  final_conv  →  ε̂

Time conditioning
  t (integer) → SinusoidalTimeEmbedding → MLP  →  t_emb ∈ ℝ^{time_emb_dim}
  Each ResBlock projects t_emb and adds it channel-wise after the first conv.

Down path (channel_mults=[1,2,4], base_channels=64):
  Level 0:  64ch @ 28×28  →  Downsample → 14×14   (skip_0)
  Level 1: 128ch @ 14×14  →  Downsample → 7×7     (skip_1)
  Level 2: 256ch @  7×7   (no downsample)          (skip_2)

Bottleneck:
  2 × ResBlock(256, 256) @ 7×7

Up path (skips consumed in reverse order):
  Concat(256, skip_2=256)=512 → ResBlocks → 256ch  →  Upsample → 14×14
  Concat(256, skip_1=128)=384 → ResBlocks → 128ch  →  Upsample → 28×28
  Concat(128, skip_0= 64)=192 → ResBlocks →  64ch

Output: GroupNorm + SiLU + Conv(64 → img_channels) → ε̂ same shape as x
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GROUPS = 32  # for GroupNorm; divides all channel counts we use (64,128,256,192,384,512)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps integer timestep t to a continuous vector via sinusoidal encoding + MLP.

    Encoding (identical to Transformer positional encoding, Vaswani et al. 2017):
      emb[2i]   = sin(t / 10000^(2i / dim))
      emb[2i+1] = cos(t / 10000^(2i / dim))

    Then projected through an MLP:  dim → dim*4 → SiLU → dim*4 → dim
    (following the improved DDPM design, Dhariwal & Nichol 2021)

    Args:
        dim: output dimension (= time_emb_dim from config)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: integer timestep indices, shape (B,)
        Returns:
            t_emb: shape (B, dim)
        """
        half = self.dim // 2
        # Frequency bands: 10000^(2i/dim) for i in [0, half)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )  # (half,)
        args = t.float()[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """
    Residual block with time-step conditioning (additive injection after first conv).

    Pre-activation layout (norm before activation before conv):
      h = conv1(silu(norm1(x)))
      h = h + time_proj(silu(t_emb))[:, :, None, None]   # broadcast over H×W
      h = conv2(silu(norm2(h)))
      return h + residual_conv(x)                          # skip connection

    Args:
        in_ch:        input channel count
        out_ch:       output channel count
        time_emb_dim: dimensionality of the time embedding vector
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(NUM_GROUPS, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(NUM_GROUPS, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # 1×1 conv to match channels for the residual path when in_ch ≠ out_ch
        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, in_ch, H, W)
            t_emb: (B, time_emb_dim)
        Returns:
            (B, out_ch, H, W)
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    """Halves spatial resolution with a stride-2 convolution."""

    def __init__(self, ch: int):
        super().__init__()
        # kernel 4, stride 2, padding 1 → exact halving for even spatial sizes
        self.conv = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial resolution via nearest-neighbour interpolation + conv."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Small U-Net denoiser for unconditional DDPM.

    Takes a noisy image x_t and timestep t and predicts the noise ε̂ that was
    added to the original image x_0:

        ε̂ = UNet(x_t, t)

    during training the loss is:
        L = || ε̂ - ε ||²   where ε ~ N(0, I) was the true noise

    Args:
        img_channels:  C (1 for grayscale, 3 for RGB)
        base_channels: width of the first feature map (64 for Phase 1)
        channel_mults: per-level channel multipliers relative to base_channels
        num_res_blocks: number of ResBlocks per encoder/decoder level
        time_emb_dim:  dimensionality of the sinusoidal time embedding
    """

    def __init__(
        self,
        img_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        # --- Initial projection ---
        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder (down path) ---
        self.down_blocks: nn.ModuleList = nn.ModuleList()  # one ModuleList per level
        self.downsamples: nn.ModuleList = nn.ModuleList()  # len = len(channel_mults) - 1

        in_ch = base_channels
        skip_channels: List[int] = []   # tracks out_ch at each level for skip-cat in decoder

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level: List[nn.Module] = []
            for _ in range(num_res_blocks):
                level.append(ResBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            self.down_blocks.append(nn.ModuleList(level))
            skip_channels.append(in_ch)  # channel count after this level's ResBlocks

            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(in_ch))

        # --- Bottleneck ---
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim)

        # --- Decoder (up path) ---
        self.up_blocks: nn.ModuleList = nn.ModuleList()   # one ModuleList per level
        self.upsamples: nn.ModuleList = nn.ModuleList()   # len = len(channel_mults) - 1

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = skip_channels.pop()  # skip from the matching encoder level

            level = []
            for j in range(num_res_blocks):
                # First block concatenates the skip; subsequent blocks run at out_ch
                block_in = (in_ch + skip_ch) if j == 0 else out_ch
                level.append(ResBlock(block_in, out_ch, time_emb_dim))
            in_ch = out_ch
            self.up_blocks.append(nn.ModuleList(level))

            if i < len(channel_mults) - 1:
                self.upsamples.append(Upsample(in_ch))

        # --- Output ---
        self.final_norm = nn.GroupNorm(NUM_GROUPS, in_ch)
        self.final_conv = nn.Conv2d(in_ch, img_channels, kernel_size=3, padding=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise ε̂ added to x_0 at timestep t.

        Args:
            x: noisy image x_t, shape (B, C, H, W), values in [-1, 1]
            t: integer timestep indices, shape (B,), in [0, T-1]
        Returns:
            ε̂: predicted noise, shape (B, C, H, W)
        """
        # 1. Time embedding: (B,) → (B, time_emb_dim)
        t_emb = self.time_mlp(t)

        # 2. Initial conv
        h = self.init_conv(x)

        # 3. Encoder — run each level's ResBlocks, save skip, downsample
        skips: List[torch.Tensor] = []
        for i, level_blocks in enumerate(self.down_blocks):
            for block in level_blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.down_blocks) - 1:
                h = self.downsamples[i](h)

        # 4. Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # 5. Decoder — upsample, concat skip, run ResBlocks
        for i, level_blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            for j, block in enumerate(level_blocks):
                if j == 0:
                    h = torch.cat([h, skip], dim=1)  # concat along channel axis
                h = block(h, t_emb)
            if i < len(self.up_blocks) - 1:
                h = self.upsamples[i](h)

        # 6. Output projection
        return self.final_conv(F.silu(self.final_norm(h)))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Verifies:
      1. Output shape matches input shape.
      2. Forward pass runs without error.
      3. Different timesteps produce different outputs (time conditioning is live).
      4. Gradients flow back to all parameters.
      5. Parameter count is printed.
    """
    torch.manual_seed(0)

    model = UNet(
        img_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
    )
    model.eval()

    B, C, H, W = 4, 1, 28, 28
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))

    # --- 1. Shape check ---
    out = model(x, t)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"[PASS] Output shape: {out.shape}")

    # --- 2. Different t → different output ---
    t_zeros = torch.zeros(B, dtype=torch.long)
    t_max   = torch.full((B,), 999, dtype=torch.long)
    out_0 = model(x, t_zeros)
    out_T = model(x, t_max)
    assert not torch.allclose(out_0, out_T), "Time conditioning has no effect"
    print(f"[PASS] Time conditioning: outputs differ for t=0 vs t=T-1")

    # --- 3. Gradient flow ---
    model.train()
    out = model(x, t)
    loss = out.mean()
    loss.backward()
    has_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grads, "Some parameters have no gradient"
    print(f"[PASS] Gradients flow to all parameters")

    # --- 4. Parameter count ---
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {n_params:,}")

    print("\nAll sanity checks passed.")
