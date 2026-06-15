"""
U-Net denoiser for DDPM — supports unconditional (Phase 1/2) and
text-conditioned generation (Phase 3, CIFAR-10).

Architecture overview
---------------------
  x, t, [text_emb]  →  init_conv
      →  [down_0 → down_1 → … → down_L]
      →  mid_block1  →  [mid_attn]  →  mid_block2
      →  [up_L … → up_0]  →  final_conv  →  ε̂

Time conditioning (all phases):
  t (int) → SinusoidalTimeEmbedding → t_emb ∈ ℝ^{time_emb_dim}
  Each ResBlock adds a projected t_emb after its first conv (channel-wise).

Text conditioning (Phase 3, text_emb_dim ≠ None):
  text_emb ∈ ℝ^{B×D} from a frozen CLIP encoder.
  CrossAttentionBlock inserts cross-attention at two points in the network
  (between the two bottleneck ResBlocks, and after the first up-level) so
  the denoiser can condition on the semantics of the caption at the features
  with the richest representation and the smallest spatial resolution.

  Cross-attention (Vaswani et al. 2017):
    Q from spatial features  (B, H×W, C)
    K, V from text embedding  (B, 1, D)   — single global context token
    Attn(Q, K, V) = softmax(Q K^T / √d_k) V
    output shape: (B, H×W, C), reshaped to (B, C, H, W)

  When text_emb=None the cross-attention path is skipped entirely, so the
  Phase 1/2 unconditional checkpoints remain fully forward-compatible.

Down path example (base_channels=64, channel_mults=[1,2,4]):
  Level 0:  64ch @ 28×28  →  Downsample → 14×14   (skip_0)
  Level 1: 128ch @ 14×14  →  Downsample →  7×7    (skip_1)
  Level 2: 256ch @  7×7   (no downsample)          (skip_2)
  Bottleneck: 2 × ResBlock(256) + CrossAttentionBlock(256)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GROUPS = 32  # for GroupNorm; all channel counts used must be divisible by this


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps integer timestep t to a continuous vector via sinusoidal encoding + MLP.

    Encoding (identical to Transformer positional encoding, Vaswani et al. 2017):
      emb[2i]   = sin(t / 10000^(2i / dim))
      emb[2i+1] = cos(t / 10000^(2i / dim))

    Then projected through an MLP:  dim → dim*4 → SiLU → dim
    (following improved DDPM design, Dhariwal & Nichol 2021)

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
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )  # (half,)
        args = t.float()[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """
    Residual block with time-step conditioning (additive injection after first conv).

    Pre-activation layout (norm → activation → conv):
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


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for text-to-image conditioning (Phase 3).

    The CLIP embedding (B, D) is projected into n_ctx virtual context tokens
    before computing multi-head cross-attention:

        Q = W_q · flatten(norm(x))                    shape (B, H×W, C)
        K = reshape(W_k · text_emb, (n_ctx, C))       shape (B, n_ctx, C)
        V = reshape(W_v · text_emb, (n_ctx, C))       shape (B, n_ctx, C)

        Attn = softmax(Q K^T / √d_k)                  shape (B, H×W, n_ctx)
        out  = Attn V                                  shape (B, H×W, C)
        output = x + reshape(W_o · out, (C, H, W))    residual

    Why n_ctx > 1 (default: 4):
        With a single context token (n_ctx=1), softmax(Q K^T) collapses to a
        constant [1.0] regardless of Q and K — the Jacobian of softmax([x])
        with respect to x is zero, so neither W_q nor W_k receive any gradient.
        With n_ctx=4, each spatial position learns to attend selectively to
        four different "aspects" of the CLIP embedding, making Q and K both
        meaningful and fully trained.

    Pre-LayerNorm (GroupNorm) style — identical to the ResBlock convention.

    Args:
        query_dim:   C — spatial feature channels (must be divisible by n_heads)
        context_dim: D — text embedding dimension (512 for CLIP ViT-B/32)
        n_heads:     number of attention heads
        n_ctx:       number of virtual context tokens split from the text embedding
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int = 8,
        n_ctx: int = 4,
    ):
        if query_dim % n_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by n_heads ({n_heads})"
            )
        super().__init__()

        self.n_heads  = n_heads
        self.n_ctx    = n_ctx
        self.dim_head = query_dim // n_heads   # d_k in the attention formula
        self.scale    = self.dim_head ** -0.5  # 1 / √d_k

        self.norm = nn.GroupNorm(NUM_GROUPS, query_dim)

        # Q: spatial features (B, N, C) → (B, N, C)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)

        # K/V: text embedding (B, D) → (B, n_ctx * C), reshaped to (B, n_ctx, C)
        self.to_k = nn.Linear(context_dim, n_ctx * query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, n_ctx * query_dim, bias=False)

        self.to_out = nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, C, H, W) spatial features
            context: (B, D) text embedding

        Returns:
            (B, C, H, W) text-conditioned features (residual output x + attn)
        """
        B, C, H, W = x.shape
        N = H * W     # number of spatial positions

        # Pre-norm + flatten spatial: (B, C, H, W) → (B, N, C)
        h = self.norm(x).reshape(B, C, N).permute(0, 2, 1)

        # Q from spatial: (B, N, C)
        q = self.to_q(h)

        # K, V from text: (B, D) → (B, n_ctx * C) → (B, n_ctx, C)
        k = self.to_k(context).reshape(B, self.n_ctx, C)
        v = self.to_v(context).reshape(B, self.n_ctx, C)

        # Split into multi-head format: (B, n_heads, seq, dim_head)
        q = q.reshape(B, N,         self.n_heads, self.dim_head).transpose(1, 2)
        k = k.reshape(B, self.n_ctx, self.n_heads, self.dim_head).transpose(1, 2)
        v = v.reshape(B, self.n_ctx, self.n_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention over n_ctx context tokens: (B, nh, N, n_ctx)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum: (B, nh, N, dim_head)
        out = attn @ v

        # Merge heads: (B, N, C) and output projection
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.to_out(out)

        # Reshape back to spatial and add residual: (B, C, H, W)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Small U-Net denoiser supporting unconditional (Phase 1/2) and
    text-conditioned (Phase 3) generation.

    Predicts the noise ε̂ added to x_0 at timestep t:

        ε̂ = UNet(x_t, t)                    # unconditional
        ε̂ = UNet(x_t, t, text_emb)          # text-conditioned

    during training the loss is:
        L = || ε̂ - ε ||²   where ε ~ N(0, I)

    Text conditioning uses CrossAttentionBlock at two locations:
      1. Between the two bottleneck ResBlocks (deepest features, global context)
      2. After the first decoder level (before first upsample)

    When text_emb_dim=None (default), no cross-attention blocks are built and
    forward() operates in pure unconditional mode regardless of the text_emb
    argument — Phase 1/2 checkpoints load and run unchanged.

    Args:
        img_channels:  C — 1 for grayscale, 3 for RGB
        base_channels: width of the first feature map
        channel_mults: per-level channel multipliers relative to base_channels
        num_res_blocks: ResBlocks per encoder/decoder level
        time_emb_dim:  dimensionality of the sinusoidal time embedding
        text_emb_dim:  D — text embedding dimension from CLIP (None = unconditional)
        attn_heads:    number of cross-attention heads (must divide bottleneck channels)
    """

    def __init__(
        self,
        img_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        text_emb_dim: Optional[int] = None,
        attn_heads: int = 8,
    ):
        super().__init__()

        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        # --- Initial projection ---
        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder (down path) ---
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()

        in_ch = base_channels
        skip_channels: List[int] = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level: List[nn.Module] = []
            for _ in range(num_res_blocks):
                level.append(ResBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            self.down_blocks.append(nn.ModuleList(level))
            skip_channels.append(in_ch)

            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(in_ch))

        # --- Bottleneck ---
        bottleneck_ch = in_ch   # = base_channels * channel_mults[-1]
        self.mid_block1 = ResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)
        self.mid_block2 = ResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # Cross-attention inserted between the two bottleneck blocks and
        # after the first decoder level (both at the bottleneck channel width).
        if text_emb_dim is not None:
            self.mid_attn = CrossAttentionBlock(bottleneck_ch, text_emb_dim, attn_heads)
            self.up_attn  = CrossAttentionBlock(bottleneck_ch, text_emb_dim, attn_heads)
        else:  # unconditional — no cross-attention parameters created
            self.mid_attn = None
            self.up_attn  = None

        # --- Decoder (up path) ---
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = skip_channels.pop()

            level = []
            for j in range(num_res_blocks):
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
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict the noise ε̂ added to x_0 at timestep t.

        Args:
            x:        noisy image x_t, shape (B, C, H, W), values in [-1, 1]
            t:        integer timestep indices, shape (B,), in [0, T-1]
            text_emb: CLIP text embedding, shape (B, text_emb_dim), or None for
                      unconditional generation / CFG unconditional pass

        Returns:
            ε̂: predicted noise, shape (B, C, H, W)
        """
        # 1. Time embedding: (B,) → (B, time_emb_dim)
        t_emb = self.time_mlp(t)

        # 2. Initial conv
        h = self.init_conv(x)

        # 3. Encoder
        skips: List[torch.Tensor] = []
        for i, level_blocks in enumerate(self.down_blocks):
            for block in level_blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.down_blocks) - 1:
                h = self.downsamples[i](h)

        # 4. Bottleneck — cross-attention injected between the two ResBlocks
        h = self.mid_block1(h, t_emb)
        if self.mid_attn is not None and text_emb is not None:
            h = self.mid_attn(h, text_emb)
        h = self.mid_block2(h, t_emb)

        # 5. Decoder — cross-attention after the first (deepest) up level
        for i, level_blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            for j, block in enumerate(level_blocks):
                if j == 0:
                    h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)

            if i == 0 and self.up_attn is not None and text_emb is not None:
                h = self.up_attn(h, text_emb)

            if i < len(self.up_blocks) - 1:
                h = self.upsamples[i](h)

        # 6. Output projection
        return self.final_conv(F.silu(self.final_norm(h)))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Verifies the unconditional path (Phase 1/2 backward compatibility) and
    the conditional path (Phase 3 cross-attention) separately.

    Unconditional checks (text_emb_dim=None):
      1. Output shape matches input shape.
      2. Different timesteps → different outputs (time conditioning live).
      3. Gradients reach all parameters.

    Conditional checks (text_emb_dim=512):
      4. Output shape matches input shape with text_emb.
      5. Output shape matches input shape without text_emb (None path).
      6. Same image + different text → different outputs (text conditioning live).
      7. Gradients flow through cross-attention Q, K, V, out projections.
      8. Unconditional and conditional outputs differ for the same seed/input.

    Parameter counts for both configs are printed.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    torch.manual_seed(0)
    device = torch.device("cpu")

    # ---- Phase 1/2 unconditional config (tiny) --------------------------
    print("=== Phase 1/2 — unconditional (text_emb_dim=None) ===")
    model_uncond = UNet(
        img_channels=1, base_channels=32, channel_mults=(1, 2),
        num_res_blocks=1, time_emb_dim=64, text_emb_dim=None,
    ).to(device)

    B, C, H, W = 2, 1, 28, 28
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 50, (B,))

    # 1. Shape
    out = model_uncond(x, t)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"[PASS] 1. Output shape:      {tuple(out.shape)}")

    # 2. Time conditioning
    out_0 = model_uncond(x, torch.zeros(B, dtype=torch.long))
    out_T = model_uncond(x, torch.full((B,), 49, dtype=torch.long))
    assert not torch.allclose(out_0, out_T)
    print(f"[PASS] 2. Time conditioning: t=0 ≠ t=T-1")

    # 3. Gradients
    model_uncond.train()
    out = model_uncond(x, t)
    out.mean().backward()
    no_grad = [n for n, p in model_uncond.named_parameters()
               if p.grad is None or p.grad.abs().sum() == 0]
    assert not no_grad, f"Params without grad: {no_grad}"
    print(f"[PASS] 3. Gradients flow to all parameters")

    n_uncond = sum(p.numel() for p in model_uncond.parameters())
    print(f"[INFO] Unconditional params: {n_uncond:,}\n")

    # ---- Phase 3 conditional config (tiny) ------------------------------
    print("=== Phase 3 — conditional (text_emb_dim=512) ===")
    CLIP_DIM = 512
    model_cond = UNet(
        img_channels=3, base_channels=32, channel_mults=(1, 2, 4),
        num_res_blocks=1, time_emb_dim=64, text_emb_dim=CLIP_DIM, attn_heads=8,
    ).to(device)

    B, C, H, W = 2, 3, 32, 32
    x     = torch.randn(B, C, H, W)
    t     = torch.randint(0, 50, (B,))
    text  = torch.randn(B, CLIP_DIM)   # mock CLIP embedding (unit-norm not required here)
    text2 = torch.randn(B, CLIP_DIM)   # different text embedding

    # 4. Shape with text_emb
    out = model_cond(x, t, text_emb=text)
    assert out.shape == (B, C, H, W), f"Shape: {out.shape}"
    print(f"[PASS] 4. Output shape (with text_emb):    {tuple(out.shape)}")

    # 5. Shape without text_emb (None path — unconditional forward)
    out_null = model_cond(x, t, text_emb=None)
    assert out_null.shape == (B, C, H, W)
    print(f"[PASS] 5. Output shape (text_emb=None):    {tuple(out_null.shape)}")

    # 6. Different text → different output (text conditioning live)
    out2 = model_cond(x, t, text_emb=text2)
    assert not torch.allclose(out, out2), "Text conditioning has no effect"
    max_diff = (out - out2).abs().max().item()
    print(f"[PASS] 6. text1 ≠ text2 outputs (max Δ = {max_diff:.4f})")

    # 7. Gradients through cross-attention layers (all 4 projections + norm)
    # With n_ctx=4, softmax is over 4 values so Q and K are both meaningful
    # and receive non-zero gradients (unlike n_ctx=1 where softmax → constant).
    model_cond.train()
    out = model_cond(x, t, text_emb=text)
    out.mean().backward()
    attn_prefixes = ("mid_attn.", "up_attn.")
    for name, p in model_cond.named_parameters():
        if any(name.startswith(pfx) for pfx in attn_prefixes):
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No gradient for cross-attention param: {name}"
    print(f"[PASS] 7. Gradients flow through all cross-attention projections (Q, K, V, out)")

    # 8. Unconditional (None) ≠ conditional output
    model_cond.eval()
    torch.manual_seed(1)
    out_cond   = model_cond(x, t, text_emb=text)
    torch.manual_seed(1)
    out_uncond = model_cond(x, t, text_emb=None)
    assert not torch.allclose(out_cond, out_uncond), \
        "Conditional and unconditional outputs are identical"
    print(f"[PASS] 8. Conditional ≠ unconditional output (cross-attn changes features)")

    n_cond = sum(p.numel() for p in model_cond.parameters())
    n_attn = sum(p.numel() for n, p in model_cond.named_parameters()
                 if "attn" in n)
    print(f"[INFO] Conditional params:   {n_cond:,}  ({n_attn:,} in cross-attn blocks)\n")

    # ---- Full-size Phase 3 config parameter count ----------------------
    print("=== Full Phase 3 config (base_channels=128, channel_mults=[1,2,4]) ===")
    model_phase3 = UNet(
        img_channels=3, base_channels=128, channel_mults=(1, 2, 4),
        num_res_blocks=2, time_emb_dim=256, text_emb_dim=512, attn_heads=8,
    )
    n_phase3 = sum(p.numel() for p in model_phase3.parameters())
    n_attn3  = sum(p.numel() for n, p in model_phase3.named_parameters() if "attn" in n)
    print(f"[INFO] Total parameters:     {n_phase3:,}")
    print(f"[INFO] Cross-attn params:    {n_attn3:,}  ({100*n_attn3/n_phase3:.1f}% of total)")

    print("\nAll sanity checks passed.")
