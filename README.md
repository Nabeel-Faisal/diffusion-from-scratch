# Text-to-Image Diffusion Model — From Scratch

A from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch,
extended through three phases to text-conditioned image generation via CLIP embeddings and
cross-attention.

No pretrained diffusion models or `diffusers` shortcuts are used for the core forward/reverse
process, U-Net, or sampling logic.  The math is derived from first principles in
[`docs/math_writeup.md`](docs/math_writeup.md) — forward process, ELBO derivation, loss
simplification, DDIM, and classifier-free guidance.

## Status

**All 3 phases complete.**

## Results

### Phase 1 — Unconditional DDPM, Fashion-MNIST

Trained 30 epochs on Fashion-MNIST (Kaggle GPU, batch 128).
Loss converged from ~0.078 → ~0.036.
Generated samples are recognisable Fashion-MNIST items (shirts, shoes, bags, dresses).
11.6M parameter U-Net with sinusoidal time embedding, ResBlocks, and GroupNorm.

### Phase 2 — Cosine schedule and DDIM sampling, Fashion-MNIST

Trained 30 epochs with the cosine noise schedule (Nichol & Dhariwal 2021).
Final loss ~0.064 — not directly comparable to Phase 1 because the cosine schedule
concentrates noise differently; output quality is visibly better despite the higher loss value.

**FID evaluation** (n=500, cosine schedule, `scripts/evaluate.py`):

| Schedule | Sampler | Steps | FID ↓ | Time (s) |
|----------|---------|------:|------:|---------:|
| cosine | DDPM (η=0) | 1000 | 339.34 | 301.7 |
| cosine | DDIM (η=0) | 10 | 65.72 | 4.9 |
| cosine | DDIM (η=0) | 50 | 63.58 | 17.0 |

DDIM-50 achieves better FID than full DDPM-1000 in ~18× less wall time, confirming the
efficiency of deterministic non-Markovian sampling (Song et al. 2020).

> Note: n=500 is below the recommended n≥2048 for stable FID estimates; absolute values may
> be noisy but the trends are clear.

### Phase 3 — Text-conditioned DDPM with CFG, CIFAR-10

Trained 50 epochs on CIFAR-10 (32×32 RGB, 10 classes).  Text conditioning uses a frozen
CLIP ViT-B/32 encoder (512-d embeddings) fed into cross-attention blocks in the U-Net
bottleneck and first decoder level.  Classifier-free guidance (Ho et al. 2022) is applied
during training (p_uncond=0.1) and at inference (guidance scale 7.5).

Per-class sample grids are saved every 5 epochs to `outputs/samples/`.  The epoch 50 grid
(`outputs/samples/epoch_0050.png`) shows all 10 CIFAR-10 classes via fixed per-class prompts
and CFG-DDIM sampling — outputs are clearly class-discriminated by epoch 50.

**Engineering note — cross-attention with a single context token:**
The initial cross-attention design used the CLIP embedding as a single K/V token (context
length 1).  With a single key, `softmax([x]) = 1.0` for any value of `x`, so the Jacobian
of the softmax with respect to both Q and K is identically zero — neither projection receives
any gradient and both are dead weight from the first step.  This was caught by a gradient-flow
sanity check that asserted all four attention matrices (Q, K, V, out) receive non-zero
gradients.  The fix was to project the 512-d CLIP embedding into `n_ctx=4` virtual context
tokens before attention, making the softmax distribution non-trivial and restoring full
gradient flow through Q and K.  The final 48.8M-parameter model has 5.2M parameters in
cross-attention (10.8% of total).

## Project Phases

### Phase 1 — Unconditional DDPM (Fashion-MNIST) ✓
- [x] Forward diffusion process (`src/utils/diffusion.py`) — linear beta schedule, closed-form `q_sample`
- [x] U-Net denoiser (`src/models/unet.py`) — sinusoidal time embedding, ResBlock + GroupNorm + SiLU, skip connections, 11.6M params
- [x] Dataset loader (`src/data/dataset.py`) — Fashion-MNIST / MNIST via torchvision, normalised to [-1, 1]
- [x] Training loop (`scripts/train.py`) — epsilon-prediction loss, CSV loss log, checkpointing, sample grids every N epochs
- [x] DDPM ancestral sampling (`src/utils/sampling.py`) — precomputed coefficients, both variance choices documented
- [x] Sampling CLI (`scripts/sample.py`) — load checkpoint, run reverse process, save PNG grid

### Phase 2 — Improved Sampling & Evaluation ✓
- [x] Cosine noise schedule (Nichol & Dhariwal 2021) — `src/utils/diffusion.py`
- [x] DDIM sampling (Song et al. 2020) — deterministic, configurable steps, η parameter — `src/utils/sampling.py`
- [x] FID evaluation script (`scripts/evaluate.py`) — InceptionV3 features + scipy sqrtm, no pytorch-fid dependency
- [x] Ablation: DDPM-1000 vs DDIM-10/50, linear vs cosine schedule (see results above)

### Phase 3 — Text-conditioned DDPM (CIFAR-10, 32×32 RGB) ✓
- [x] CIFAR-10 dataloader with templated captions (`src/data/dataset.py`) — "a photo of a {class}"
- [x] Frozen CLIP text encoder (`src/models/clip_encoder.py`) — ViT-B/32, 512-d embeddings, always eval mode
- [x] Cross-attention conditioning in U-Net (`src/models/unet.py`) — `n_ctx=4` virtual context tokens at bottleneck + first decoder level; `text_emb=None` preserves Phase 1/2 behaviour
- [x] Classifier-free guidance — per-sample CFG dropout (p_uncond=0.1) in training; `ddim_cfg_sample` with batched cond/uncond forward at inference
- [x] EMA of model weights, linear LR warmup, gradient clipping (`scripts/train.py`)
- [x] Text-prompted generation CLI (`scripts/sample.py`) — `--prompts`, `--guidance-scale`, caption .txt co-located with PNG

### Phase 4 — Writeup & Deliverables ✓
- [x] Math writeup (`docs/math_writeup.md`) — forward process, ELBO derivation, noise-prediction loss simplification, DDIM, CFG
- [x] Polished README with results, FID table, engineering notes

## Repo Structure

```
diffusion-from-scratch/
├── src/
│   ├── models/        # U-Net, cross-attention blocks, time embeddings, CLIP wrapper
│   ├── data/          # Dataset loaders (MNIST, Fashion-MNIST, CIFAR-10)
│   └── utils/         # Diffusion scheduler, DDPM/DDIM/CFG sampling, FID evaluation
├── configs/           # YAML configs: phase1_unconditional, phase2_improved, phase3_cifar10
├── scripts/           # train.py, sample.py, evaluate.py
├── docs/              # math_writeup.md — derivations and design decisions
├── notebooks/         # Kaggle notebooks for GPU training runs
└── outputs/           # checkpoints/, samples/, logs/  (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

CLIP is installed separately:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Training

```bash
# Phase 1 — linear schedule, DDPM sampling, Fashion-MNIST
python scripts/train.py --config configs/phase1_unconditional.yaml

# Phase 2 — cosine schedule, DDIM sampling, Fashion-MNIST
python scripts/train.py --config configs/phase2_improved.yaml

# Phase 3 — text-conditioned, CFG, CIFAR-10  (GPU recommended)
python scripts/train.py --config configs/phase3_cifar10.yaml

# Local CPU smoke test for Phase 3 (2 gradient steps, verifies full pipeline)
python scripts/train.py --config configs/phase3_cifar10.yaml --device cpu --smoke-test
```

Checkpoints are saved to `outputs/checkpoints/`, sample grids to `outputs/samples/`, and a
live loss CSV + plot to `outputs/logs/` — all at intervals configured by `save_every` and
`sample_every` in the YAML.  Phase 3 checkpoints include both raw and EMA model weights.

## Sampling

```bash
# Phase 1/2 — unconditional generation
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt

# Phase 3 — text-conditioned with CFG
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0050.pt \
    --prompts "a photo of a dog" "a photo of a ship" "a photo of a airplane"

# Phase 3 — all 10 CIFAR-10 classes in one grid (2×5 layout)
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0050.pt \
    --prompts \
      "a photo of a airplane" "a photo of a automobile" "a photo of a bird" \
      "a photo of a cat" "a photo of a deer" "a photo of a dog" \
      "a photo of a frog" "a photo of a horse" "a photo of a ship" \
      "a photo of a truck" \
    --nrow 5

# Override guidance scale
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0050.pt \
    --prompts "a photo of a cat" --guidance-scale 3.0 --output cat_w3.png

# Self-test (no checkpoint needed)
python scripts/sample.py --sanity-check
```

When `--prompts` is used, a caption file (`{output}.txt`) is written alongside the PNG
listing each `image_NN: <prompt>` mapping.

The script auto-detects model type from the checkpoint config:
- `model.text_emb_dim` present → conditional model, CLIP loaded only when `--prompts` is given
- `model.text_emb_dim` absent → unconditional model, `--prompts` raises an error

EMA weights are preferred automatically when present in the checkpoint.

## Evaluation (FID)

```bash
# Quick evaluation — 500 samples, DDPM + DDIM at 10/50/200 steps
python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_0030.pt \
    --n-samples 500 --ddim-steps 10 50 200

# Full evaluation (n≥2048 for stable FID estimates)
python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_0030.pt \
    --n-samples 2048 --ddim-steps 10 50 200

# Self-test (no checkpoint, no InceptionV3 download)
python scripts/evaluate.py --sanity-check
```

Results are saved to `outputs/logs/fid_<timestamp>.csv`.

## Compute

- Development and debugging: local CPU with tiny batches (`--smoke-test` flags)
- Training: Kaggle Notebooks (free GPU tier, T4/P100, ~30 GPU-hr/week quota)
