# Text-to-Image Diffusion Model — From Scratch

A from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch,
extended to support text-conditioned image generation via CLIP embeddings + cross-attention.

No pretrained diffusion models or high-level `diffusers` shortcuts are used for the core
forward/reverse process, U-Net, or sampling logic. The goal is to demonstrate understanding
of the underlying math and engineering, not just API usage.

## Status

**Phase 1 complete · Phase 2 complete — Phase 3 (text conditioning) next.**

## Results

### Phase 1 — Linear schedule, DDPM sampling
Trained 30 epochs on Fashion-MNIST (Kaggle GPU, batch 128).
Loss converged from ~0.078 → ~0.036.
Generated samples are recognizable Fashion-MNIST items (shirts, shoes, bags, dresses).

### Phase 2 — Cosine schedule, DDIM sampling
Trained 30 epochs on Fashion-MNIST with the cosine noise schedule.
Final loss ~0.064 (not directly comparable to Phase 1 — cosine schedule has a different
noise distribution; the model sees more signal at each step, so the loss is higher
but the output quality improves).

**FID evaluation** (n=500, cosine schedule, `scripts/evaluate.py`):

| Schedule | Sampler | Steps | FID ↓ | Time (s) |
|----------|---------|------:|------:|---------:|
| cosine | DDPM (η=0) | 1000 | 339.34 | 301.7 |
| cosine | DDIM (η=0) | 10 | 65.72 | 4.9 |
| cosine | DDIM (η=0) | 50 | 63.58 | 17.0 |

**Key finding:** DDIM-50 achieves better FID than full DDPM-1000 in ~18× less wall time,
confirming the efficiency of deterministic non-Markovian sampling (Song et al. 2020).

> Note: n=500 is below the recommended n≥2048 for stable FID estimates; absolute values
> may be noisy but the trends are clear.  Full n=2048 evaluation planned before Phase 4.

Sample images and loss curve plots are currently on Kaggle and will be added to `docs/`
once all training phases are complete.

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
- [x] Ablation: DDPM-1000 vs DDIM-10/50/200, linear vs cosine (see results above)

### Phase 3 — Text Conditioning (Flowers102, 64×64)
- [ ] CLIP text encoder integration (frozen, pretrained — only this part uses an external model)
- [ ] Cross-attention conditioning in U-Net
- [ ] Classifier-free guidance (CFG)
- [ ] Train on captioned dataset, evaluate guidance scale effect

### Phase 4 — Deliverables
- [ ] Math write-up (forward/reverse process, ELBO derivation, loss simplification)
- [ ] Gradio demo (text prompt → generated image)
- [ ] Comparison with pretrained Stable Diffusion outputs
- [ ] Polished README with results, training curves, sample images

## Repo Structure

```
diffusion-from-scratch/
├── src/
│   ├── models/        # U-Net, attention blocks, time embeddings, CLIP wrapper
│   ├── data/          # Dataset loaders (MNIST, Fashion-MNIST, Flowers102)
│   └── utils/         # Diffusion scheduler, sampling, EMA, FID, visualization
├── configs/           # YAML configs for each phase/experiment
├── scripts/           # train.py, sample.py, evaluate.py
├── notebooks/         # Kaggle/Colab notebooks for GPU training
├── outputs/           # checkpoints/, samples/, logs/  (gitignored)
└── docs/              # math derivations, write-ups
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Phase 1 — linear schedule, DDPM sampling
python scripts/train.py --config configs/phase1_unconditional.yaml

# Phase 2 — cosine schedule, DDIM sampling
python scripts/train.py --config configs/phase2_improved.yaml

# Override device for local CPU debugging
python scripts/train.py --config configs/phase2_improved.yaml --device cpu
```

Checkpoints are saved to `outputs/checkpoints/`, sample grids to `outputs/samples/`, and a live loss CSV + plot to `outputs/logs/` — all every `save_every` / `sample_every` epochs (configured in the YAML).

## Sampling

```bash
# Generate images from a checkpoint
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt

# Override number of samples, grid layout, and output path
python scripts/sample.py --checkpoint outputs/checkpoints/epoch_0030.pt \
    --n-samples 64 --nrow 8 --output my_grid.png

# Self-test without a real checkpoint
python scripts/sample.py --sanity-check
```

## Evaluation (FID)

```bash
# Quick pass — 500 samples, DDPM + DDIM at 10/50/200 steps
python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_0030.pt \
    --n-samples 500 --ddim-steps 10 50 200

# Full evaluation (n≥2048 for stable FID)
python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_0030.pt \
    --n-samples 2048 --ddim-steps 10 50 200

# Self-test (no checkpoint, no InceptionV3 download)
python scripts/evaluate.py --sanity-check
```

Results are saved to `outputs/logs/fid_<timestamp>.csv`.

## Compute Plan

- Development/debugging: local (CPU, tiny batches)
- Actual training: Kaggle Notebooks (free GPU, 30hr/week quota)
