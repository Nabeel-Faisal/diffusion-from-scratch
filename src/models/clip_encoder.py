"""
Frozen CLIP text encoder for Phase 3 text conditioning.

CLIP (Radford et al. 2021) is used here exclusively as a text embedding
backbone.  It is always frozen — its weights are never updated during
diffusion training.  Only the UNet and its cross-attention layers are trained.

Why frozen CLIP (not trained from scratch):
  Training a text encoder jointly with a diffusion model requires orders of
  magnitude more data and compute than we have available.  Freezing CLIP
  gives us high-quality semantic embeddings for free and is standard practice
  in text-conditioned diffusion (DALL-E 2, DeepFloyd IF, etc.).

Embedding dimension by model:
  "ViT-B/32"  → 512-d   ← default; smallest and fastest
  "ViT-B/16"  → 512-d
  "ViT-L/14"  → 768-d

Null embedding (for classifier-free guidance):
  During training, text conditioning is dropped with probability p_uncond
  (typically 0.1) by replacing captions with an empty string "".  At
  inference, two forward passes are run — one conditional, one unconditional —
  and the noise predictions are combined:

    ε_cfg = ε_uncond + w · (ε_cond − ε_uncond)    (Ho et al. 2022, Eq. 6)

  where w is the guidance scale.  The unconditional pass uses the null
  embedding returned by `null_embedding()`, which encodes "".
"""

from typing import List

import torch
import torch.nn as nn

try:
    import clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False


class CLIPTextEncoder(nn.Module):
    """
    Wraps a pretrained, frozen CLIP text encoder.

    Takes a list of B caption strings and returns L2-normalised float32
    embeddings of shape (B, D) where D = self.embedding_dim.

    CLIP is loaded via clip.load() and registered as a submodule so that
    .to(device) propagates automatically.  All CLIP parameters are frozen
    (requires_grad=False) and the encoder is always kept in eval mode, even
    when the surrounding diffusion model is set to train().

    Args:
        model_name:  CLIP checkpoint identifier, e.g. "ViT-B/32".
        device:      Device on which to load the model.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: torch.device = torch.device("cpu"),
    ):
        if not _CLIP_AVAILABLE:
            raise ImportError(
                "CLIP is not installed.  Run: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        super().__init__()

        clip_model, _ = clip.load(model_name, device=device)

        # Register as a submodule so .to(device) / .state_dict() include it.
        self.clip_model: nn.Module = clip_model

        # Freeze every CLIP parameter — the optimizer must never touch these.
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # text_projection maps from transformer width to the joint embedding space.
        # Its output dimension is the embedding dimension D.
        self.embedding_dim: int = int(clip_model.text_projection.shape[1])
        self._model_name = model_name

    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "CLIPTextEncoder":
        """Override train() to keep CLIP permanently in eval mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, captions: List[str]) -> torch.Tensor:
        """
        Encode a batch of caption strings to L2-normalised embeddings.

        Tokenises with CLIP's BPE tokeniser (context length 77 tokens;
        longer captions are silently truncated).  The result is cast to
        float32 because CLIP uses float16 on GPU by default, but the
        diffusion U-Net operates in float32.

        Args:
            captions: List of B strings.

        Returns:
            embeddings: (B, D) float32 tensor on the same device as CLIP.
        """
        device = next(self.clip_model.parameters()).device
        tokens = clip.tokenize(captions, truncate=True).to(device)

        # encode_text returns (B, D) in the model's native dtype (fp16 on GPU).
        embeddings = self.clip_model.encode_text(tokens).float()

        # L2-normalise: unit-sphere embeddings are more stable to condition on.
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return embeddings

    # ------------------------------------------------------------------
    def null_embedding(self, batch_size: int) -> torch.Tensor:
        """
        Return the CLIP embedding of an empty string, repeated B times.

        Used for classifier-free guidance:
          - During training: replaces real captions with probability p_uncond.
          - During inference: the unconditional forward pass of the UNet.

        Encoding "" rather than returning zeros keeps the embedding in the
        CLIP text embedding space, which is where the UNet's cross-attention
        is trained to operate.

        Args:
            batch_size: B — number of copies to return.

        Returns:
            null_emb: (B, D) float32 tensor.
        """
        null_emb = self.forward([""] * batch_size)   # (B, D)
        return null_emb

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"CLIPTextEncoder(model='{self._model_name}', "
            f"embedding_dim={self.embedding_dim}, frozen=True)"
        )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Fast CPU checks (no GPU required).  Downloads ViT-B/32 weights on first run
    (~350 MB to ~/.cache/clip/).

    Checks:
      1. Module loads without error.
      2. forward() returns correct shape and dtype.
      3. Embeddings are L2-normalised (unit norm).
      4. Different captions produce different embeddings.
      5. Same caption produces identical embeddings (deterministic).
      6. null_embedding() returns correct shape and is not all-zeros
         (it is the CLIP encoding of "").
      7. train(True) does not put CLIP into training mode.
      8. No CLIP parameters have requires_grad=True.
      9. CLIPTextEncoder.to(device) moves the model (smoke test on CPU).
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    device = torch.device("cpu")
    print("Loading CLIPTextEncoder (ViT-B/32)...")
    enc = CLIPTextEncoder(model_name="ViT-B/32", device=device)
    print(f"  {enc}\n")

    # 1. Load
    print("[PASS] 1. Module loads")

    # 2. Shape and dtype
    captions = ["a photo of a cat", "a photo of a dog"]
    emb = enc(captions)
    assert emb.shape == (2, enc.embedding_dim), \
        f"Shape mismatch: {emb.shape} vs expected (2, {enc.embedding_dim})"
    assert emb.dtype == torch.float32, f"Wrong dtype: {emb.dtype}"
    print(f"[PASS] 2. Output shape {emb.shape}, dtype {emb.dtype}")

    # 3. Unit norm
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4), \
        f"Not unit-norm: {norms}"
    print(f"[PASS] 3. Embeddings are unit-norm: {norms.tolist()}")

    # 4. Different captions → different embeddings
    assert not torch.allclose(emb[0], emb[1]), "Cat and dog embeddings are identical"
    cos_sim = (emb[0] @ emb[1]).item()
    print(f"[PASS] 4. Different captions differ  (cosine sim cat↔dog = {cos_sim:.4f})")

    # 5. Deterministic
    emb2 = enc(captions)
    assert torch.allclose(emb, emb2), "Encoder is not deterministic"
    print("[PASS] 5. Deterministic (same input → same output)")

    # 6. null_embedding shape and non-zero
    null = enc.null_embedding(batch_size=4)
    assert null.shape == (4, enc.embedding_dim), f"Wrong null shape: {null.shape}"
    assert not torch.all(null == 0.0), "null_embedding is all-zeros (expected CLIP(\"\")"
    null_norms = null.norm(dim=-1)
    assert torch.allclose(null_norms, torch.ones(4), atol=1e-4), \
        f"null_embedding not unit-norm: {null_norms}"
    print(f"[PASS] 6. null_embedding shape {null.shape}, unit-norm, non-zero")

    # 7. CLIP stays in eval after train()
    enc.train(True)
    assert not enc.clip_model.training, "CLIP was put in training mode"
    print("[PASS] 7. clip_model.training = False after enc.train(True)")

    # 8. No frozen parameters have requires_grad
    grad_params = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert len(grad_params) == 0, f"Params with grad: {grad_params}"
    total_params = sum(p.numel() for p in enc.parameters())
    print(f"[PASS] 8. All {total_params:,} parameters frozen (requires_grad=False)")

    # 9. .to() smoke test (stays on CPU, just checks no crash)
    enc2 = CLIPTextEncoder("ViT-B/32", device=device).to(device)
    emb3 = enc2(["a photo of a ship"])
    assert emb3.shape == (1, enc2.embedding_dim)
    print(f"[PASS] 9. .to(cpu) + forward() works")

    print("\nAll sanity checks passed.")
