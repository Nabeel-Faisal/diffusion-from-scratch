# Project Context for Claude Code

## What this project is
A from-scratch DDPM (Denoising Diffusion Probabilistic Model) in PyTorch, built incrementally
in phases (see README.md for full roadmap). This is a CV/portfolio project — code quality,
clear structure, and correctness of the math matter more than speed of delivery.

## Current Phase
Phase 3: Text-conditioned DDPM on CIFAR-10 (32×32 RGB images, 10 classes).

## Build order for Phase 1 (work through these one at a time, test each before moving on)

1. `src/utils/diffusion.py`
   - Implement the noise scheduler: linear beta schedule, precompute alphas, alpha_bars,
     sqrt_alpha_bars, sqrt_one_minus_alpha_bars.
   - Function `q_sample(x0, t, noise)` -> returns noisy image x_t using closed-form
     q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

2. `src/models/unet.py`
   - Small U-Net: sinusoidal time embedding -> MLP, down blocks (Conv + GroupNorm + SiLU +
     residual), bottleneck, up blocks with skip connections.
   - Input: (image, timestep) -> Output: predicted noise (same shape as image).
   - Keep it small for Phase 1 (base_channels=64, channel_mults=[1,2,4]) so it trains fast
     on Fashion-MNIST.

3. `src/data/dataset.py`
   - Dataloader for MNIST/Fashion-MNIST via torchvision, normalize to [-1, 1].

4. `scripts/train.py`
   - Training loop: sample random t, sample noise, compute x_t via q_sample, predict noise
     with U-Net, MSE loss between predicted and true noise.
   - Log loss per step (save to outputs/logs/ as CSV or via matplotlib at end).
   - Save checkpoints every N epochs to outputs/checkpoints/.
   - Every N epochs, run sampling and save an image grid to outputs/samples/.

5. `src/utils/sampling.py`
   - DDPM ancestral sampling: start from pure noise, iterate t=T..1, predict noise, compute
     mean/variance per the DDPM reverse formula, sample x_{t-1}.

6. `scripts/sample.py`
   - Load a checkpoint, run sampling, save output image grid.

## Conventions
- All configs read from YAML files in `configs/`.
- Use `argparse` in scripts to pass config path.
- Keep tensors in [-1, 1] range for images; rescale to [0,1] only for saving/visualization.
- Write docstrings explaining the math (reference variable names to the DDPM paper notation:
  beta_t, alpha_t, alpha_bar_t, epsilon_theta).
- Prefer small, testable functions over large monolithic ones — this code will be referenced
  in a write-up explaining the math, so clarity > cleverness.

## Testing approach
- For each module, write a quick `if __name__ == "__main__":` block that runs a small sanity
  check (e.g., shapes match, q_sample produces expected noise levels at t=0 vs t=T).
- Local dev uses CPU + tiny batch (batch_size=4, 1 epoch) just to confirm no shape/logic
  errors before running real training on Kaggle GPU.

## Build order for Phase 3 (text-conditioned DDPM on CIFAR-10)

Dataset: CIFAR-10 — 32×32 RGB, 10 classes. Changed from Flowers102 (too large
for Kaggle free tier); CIFAR-10 trains fast and the 10-class structure makes
text conditioning easy to verify visually.

1. `src/data/dataset.py` (extend existing)
   - Add CIFAR-10 support with templated captions: "a photo of a {class_name}".
   - Return (image, caption_str) pairs instead of (image, label_int) for CIFAR-10.
   - CIFAR-10 is RGB 32×32; normalise all 3 channels to [-1, 1].

2. `src/models/clip_encoder.py` (new)
   - Frozen CLIP text encoder (ViT-B/32 → 512-d embeddings).
   - `CLIPTextEncoder(model_name, device)` wraps clip.load(); freezes all params.
   - `forward(captions: List[str]) -> Tensor (B, 512)` — L2-normalised float32.
   - `null_embedding(batch_size) -> Tensor (B, 512)` — encodes "" for CFG dropout.

3. `src/models/unet.py` (extend existing)
   - Add `CrossAttention(query_dim, context_dim, heads, dim_head)` block:
     Q from image features, K/V from text embedding.
   - Add `text_emb_dim: Optional[int]` parameter to UNet; if None, behaves exactly
     like the Phase 1/2 unconditional UNet (zero code-path change for inference).
   - Inject cross-attention in the bottleneck and optionally in up-blocks.
   - Text embedding is projected to match query dim before attention.

4. `configs/phase3_cifar10.yaml` (new)
   - CIFAR-10 dataset, 32×32 RGB, cosine schedule, DDIM sampling.
   - CLIP model name, CFG dropout probability (p_uncond=0.1).
   - Larger U-Net to handle RGB: base_channels=128, channel_mults=[1,2,4].

5. `scripts/train.py` (extend existing)
   - For CIFAR-10: unpack (image, caption) from dataloader.
   - CFG dropout: replace caption with "" with probability p_uncond.
   - Pass text embeddings from CLIPTextEncoder to UNet forward.
   - Conditional vs unconditional sampling path in the training sample grid.

6. `src/utils/sampling.py` (extend existing)
   - `ddim_cfg_sample(model, clip_enc, prompts, scheduler, ...)`:
     runs DDIM with classifier-free guidance weight w.
     Each step: eps = (1+w)*eps_cond - w*eps_uncond  (Ho et al. CFG, Eq. 6).

7. `scripts/sample.py` (extend existing)
   - Accept `--prompt` argument for text-conditioned generation.
   - Accept `--guidance-scale` (CFG weight w, default 7.5).

## DO NOT
- Do not use `diffusers`, pretrained diffusion checkpoints, or pretrained U-Nets for Phase 1-3.
- Do not skip the math docstrings — they matter for the final write-up in docs/.
- Do not move to Phase 2/3 until Phase 1 trains successfully and produces visibly improving
  samples over epochs.
- Do not modify the Phase 1/2 unconditional code path — the text_emb_dim=None branch in the
  U-Net must keep Phase 1/2 checkpoints working without any changes.
